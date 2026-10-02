"""Tabulate VisIT-Bench's original inputs, generations and recorded judgments."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class VisITBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read native task tables and retain exact source records and ordered captions.
        single = pd.read_csv(self.raw_dir / layout['dataset'], dtype=str, keep_default_na=False)
        single['native_record'] = single.to_dict('records')
        single['source_row'], single['source_file'] = single.index, layout['dataset']
        aliases = single.groupby(['image', 'instruction'], sort=False).source_row.agg(list).rename('metadata_rows')
        bank = single.drop_duplicates(['image', 'instruction']).copy()
        if len(single.drop(columns=['native_record', 'source_row', 'source_file']).drop_duplicates()) != len(bank):
            raise ValueError('One image/instruction key has conflicting original metadata')
        bank = bank.merge(aliases, on=['image', 'instruction'], validate='one_to_one')
        bank['captions'] = bank.instruction_conditioned_caption.map(lambda value: [value])
        bank['images'] = bank.image.map(lambda value: [value])
        bank['task_key'] = layout['dataset'] + ':' + bank.source_row.astype(str)
        multi = pd.read_csv(self.raw_dir / layout['multi'], dtype=str, keep_default_na=False)
        multi['native_record'] = multi.to_dict('records')
        multi['source_row'], multi['source_file'] = multi.index, layout['multi']
        multi['images'] = multi.images.map(json.loads)
        multi['captions'] = multi.images_dense_captions.map(json.loads)
        for row in multi.itertuples():
            if not all(isinstance(value, str) for value in row.captions[:len(row.images)]) or not all(pd.isna(value) for value in row.captions[len(row.images):]):
                raise ValueError('Original multi-image caption slots do not match their ordered images')
        multi['captions'] = [row.captions[:len(row.images)] for row in multi.itertuples()]
        multi['metadata_rows'] = multi.source_row.map(lambda value: [value])
        multi['task_key'] = layout['multi'] + ':' + multi.source_row.astype(str)
        tasks = pd.concat([bank, multi], ignore_index=True)
        task_columns = ['task_key', 'instruction', 'instruction_category', 'images', 'captions', 'metadata_rows', 'native_record', 'gpt4_prediction']
        tasks = tasks[task_columns].rename(columns={'native_record': 'task_record'})

        # 2. Load individual human votes and the published paired GPT-4 judgments.
        human = pd.read_csv(self.raw_dir / layout['human'], dtype=str, keep_default_na=False)
        human['native_record'] = human.to_dict('records')
        human = human.assign(source_file=layout['human'], source_row=human.index, protocol='human_pairwise')
        if not (human['model_selection.A'].isin(['TRUE', 'FALSE']) & human['model_selection.B'].isin(['TRUE', 'FALSE']) & human['model_selection.A'].ne(human['model_selection.B'])).all():
            raise ValueError('Original human choices must select exactly one side')
        with ZipFile(self.raw_dir / layout['judgment_archive']) as archive:
            published = pd.DataFrame(json.loads(archive.read(layout['judgment_member'])))
        published['native_record'] = published.astype(object).where(published.notna(), None).to_dict('records')
        published['native_nonfinite_fields'] = published[['A', 'B']].isna().apply(lambda row: row.index[row].tolist(), axis=1)
        published = published.assign(source_file=layout['judgment_archive'], source_row=published.index, protocol='gpt4_pairwise')
        if not (published.engine.eq('gpt-4') & published.evaluated_with_reference.eq(False)).all():
            raise ValueError('A published comparison uses an undeclared judging protocol')
        published['pair_key'] = [json.dumps([r.image_url, r.instruction, sorted([(r.A_model, r.A), (r.B_model, r.B)])], ensure_ascii=False) for r in published.itertuples()]

        # 3. Join explicit later comparison requests to saved judge outputs; make no API calls.
        with ZipFile(self.raw_dir / layout['cache_archive']) as archive:
            cache = pd.read_json(archive.open(layout['cache_member']), lines=True, dtype=False, convert_dates=False)
        cache['cache_row'] = cache.index
        cache = cache.drop_duplicates('query', keep='last')
        parts = []
        for path in sorted(self.raw_dir.glob(layout['queries'])):
            part = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
            part['native_record'] = part.astype(object).where(part.notna(), None).to_dict('records')
            part['native_nonfinite_fields'] = part[['A', 'B']].isna().apply(lambda row: row.index[row].tolist(), axis=1)
            parts.append(part.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=part.index))
        later = pd.concat(parts, ignore_index=True)
        later['pair_key'] = [json.dumps([r.image_url, r.instruction, sorted([(r.A_model, r.A), (r.B_model, r.B)])], ensure_ascii=False) for r in later.itertuples()]
        later = later.loc[~later.pair_key.isin(published.pair_key)].copy()
        later['event_key'] = later.pair_key + later.image_dense_caption.map(lambda value: json.dumps(value, ensure_ascii=False))
        later['source_alias'] = later[['source_file', 'source_row']].to_dict('records')
        later_aliases = later.groupby('event_key', sort=False).source_alias.agg(list).rename('source_aliases')
        later = later.drop_duplicates('event_key').merge(later_aliases, on='event_key', validate='one_to_one').reset_index(drop=True)
        later['comparison_key'] = later.index
        requests = pd.concat([later.assign(order=0), later.assign(order=1)], ignore_index=True)
        requests['query'] = [parameters['request']['template'].format(r.image_dense_caption, r.instruction, r.A if r.order == 0 else r.B, r.B if r.order == 0 else r.A) for r in requests.itertuples()]
        requests = requests.merge(cache, on='query', how='left', validate='many_to_one')
        if requests.response.isna().any():
            raise ValueError('A later source comparison lacks a saved judge response')
        selected_a = requests.response.str.contains('response a is better', case=False, regex=False)
        selected_b = requests.response.str.contains('response b is better', case=False, regex=False)
        requests['winner'] = 'tie'
        requests.loc[selected_a & ~selected_b, 'winner'] = 'A'
        requests.loc[selected_b & ~selected_a, 'winner'] = 'B'
        requests['extraction_query'] = requests.response + parameters['request']['extraction_suffix']
        extraction = cache.set_index('query').response
        requests['extraction_response'] = requests.extraction_query.map(extraction)
        unresolved = selected_a.eq(selected_b)
        if requests.loc[unresolved, 'extraction_response'].isna().any():
            raise ValueError('A source judgment needs an unavailable answer-extraction result')
        parsed_a = requests.extraction_response.str.contains('Final Answer: Response A', regex=False, na=False)
        parsed_b = requests.extraction_response.str.contains('Final Answer: Response B', regex=False, na=False)
        requests.loc[unresolved & parsed_a & ~parsed_b, 'winner'] = 'A'
        requests.loc[unresolved & parsed_b & ~parsed_a, 'winner'] = 'B'
        requests['model_winner'] = [r.A_model if (r.winner == 'A' and r.order == 0) or (r.winner == 'B' and r.order == 1) else r.B_model if r.winner != 'tie' else 'tie' for r in requests.itertuples()]
        requests['judge_record'] = [dict(order=int(r.order), cache_row=int(r.cache_row), query=r.query, response=r.response,
            extraction_response=r.extraction_response if selected_a.iloc[i] == selected_b.iloc[i] else None)
            for i, r in enumerate(requests.itertuples())]
        paired = requests.sort_values(['comparison_key', 'order']).groupby('comparison_key', sort=False).agg(
            auto_evaluation_result=('model_winner', list), cached_judgments=('judge_record', list))
        later = later.merge(paired, on='comparison_key', validate='one_to_one').assign(protocol='gpt4_pairwise')

        # 4. Unpivot comparison sides and append the original correctness records.
        comparisons = pd.concat([human, published, later], ignore_index=True)
        comparisons = comparisons.merge(bank[['image', 'instruction', 'task_key']], left_on=['image_url', 'instruction'], right_on=['image', 'instruction'], how='left', validate='many_to_one')
        if comparisons.task_key.isna().any():
            raise ValueError('A comparison has no exact original image/instruction association')
        vote_a, vote_b = comparisons.auto_evaluation_result.str[0], comparisons.auto_evaluation_result.str[1]
        unanimous = vote_a.eq(vote_b) & vote_a.ne('tie')
        comparisons['grade_a'] = .5
        comparisons.loc[unanimous & vote_a.eq(comparisons.A_model), 'grade_a'] = 1.
        comparisons.loc[unanimous & vote_a.eq(comparisons.B_model), 'grade_a'] = 0.
        human_rows = comparisons.protocol.eq('human_pairwise')
        comparisons.loc[human_rows, 'grade_a'] = comparisons.loc[human_rows, 'model_selection.A'].eq('TRUE').astype(float)
        sides = []
        for side in ['A', 'B']:
            sides.append(comparisons.assign(side=side, subject_key=comparisons[side + '_model'],
                completion=comparisons[side], opponent=comparisons['B_model' if side == 'A' else 'A_model'], response=comparisons.grade_a if side == 'A' else 1. - comparisons.grade_a))
        correctness = pd.concat([single.merge(bank[['image', 'instruction', 'task_key']], on=['image', 'instruction'], validate='many_to_one'), multi], ignore_index=True)
        if not correctness.human_ratings_gpt4_correct.str.lower().isin(['true', 'false']).all():
            raise ValueError('A source correctness value is not Boolean')
        correctness = correctness.assign(protocol='human_correctness', side='', subject_key=labels['reference_alias'],
            completion=correctness.gpt4_prediction, opponent='', response=correctness.human_ratings_gpt4_correct.str.lower().eq('true').astype(float))
        responses = pd.concat(sides + [correctness], ignore_index=True)
        responses = responses.drop(columns=[c for c in task_columns if c in responses and c not in ['task_key', 'native_record']]).merge(tasks, on='task_key', validate='many_to_one')
        caption_mode = responses.subject_key.eq(labels['reference_alias'])
        if not responses.loc[caption_mode, 'completion'].eq(responses.loc[caption_mode, 'gpt4_prediction']).all():
            raise ValueError('The reference alias is not the recorded caption-conditioned GPT-4 output')
        responses['item_key'] = responses.task_key + '|' + responses.protocol + '|' + caption_mode.astype(str)
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key.replace({labels['reference_alias']: labels['reference_model']})
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=model,
            input_scope=labels['caption_input'] if model == labels['reference_alias'] else labels['visual_input'],
            configuration_status=labels['configuration_status']) for model in subjects.subject_key]

        # 5. Preserve actual input bytes, distinguishing visual inputs from grading images.
        items = responses.drop_duplicates('item_key').copy()
        image_bytes = {}
        with ZipFile(self.raw_dir / layout['image_archive']) as archive:
            names = set(archive.namelist())
            for url in sorted({url for images in items.images for url in images}):
                filename = Path(url).name
                member = layout['image_prefix'] + filename
                if member in names:
                    data = archive.read(member)
                    media_type = next(mime for signature, mime in parameters['image_signatures'].items() if data.startswith(bytes.fromhex(signature)))
                    image_bytes[url] = dict(data=data, path='images/' + filename, media_type=media_type)
        attachments, contents, features, criteria, verifiers = [], [], [], [], []
        for row in items.itertuples():
            caption_input = row.subject_key == labels['reference_alias']
            images = [dict(image_bytes[url], role='grading' if caption_input else 'input') for url in row.images if url in image_bytes]
            missing = [url for url in row.images if url not in image_bytes]
            if missing and not caption_input:
                raise ValueError('A vision-language input image is unavailable')
            attachments.append(images)
            contents.append(json.dumps(dict(instruction=row.instruction, instruction_conditioned_captions=row.captions), ensure_ascii=False)
                if caption_input else json.dumps(dict(multimedia_elements=[dict(content_type='text/plain', text=row.instruction)]
                    + [dict(content_type=image['media_type'], location=image['path']) for image in images]), ensure_ascii=False))
            features.append(dict(instruction_category=row.instruction_category, input_scope=labels['caption_input'] if caption_input else labels['visual_input'],
                unavailable_grading_images=str(len(missing))))
            protocol = self.grading['verifiers'][row.protocol]
            criteria.append(dict(rule=protocol['rule'], response_scale=protocol['response_scale']))
            verifiers.append(Judge(judge=protocol['judge'], judged_by=protocol['judged_by'], spec=json.dumps(protocol, sort_keys=True)))
        items['attachments'], items['content'], items['features'] = attachments, contents, features
        items['raw_item_id'] = items.task_key
        items['grading_criterion'], items['verifier'] = criteria, verifiers

        # 6. Keep complete native events and cache evidence; source rows are not new executions.
        responses['response_key'] = responses.index
        responses['test_condition'] = 'source_file=' + responses.source_file + ';source_row=' + responses.source_row.astype(str) + ';side=' + responses.side
        responses['interactors'] = responses.opponent.map(lambda value: 'opponent=' + value if value else None)
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=r.source_file, source_row=int(r.source_row), side=r.side, protocol=r.protocol,
            native_record=r.native_record, native_nonfinite_fields=r.native_nonfinite_fields if isinstance(r.native_nonfinite_fields, list) else [], task_record=r.task_record, task_key=r.task_key, metadata_rows=r.metadata_rows,
            source_aliases=r.source_aliases if isinstance(r.source_aliases, list) else [],
            cached_judgments=r.cached_judgments if isinstance(r.cached_judgments, list) else [],
            unavailable_grading_images=[url for url in r.images if url not in image_bytes], scope=labels['trace_scope']), ensure_ascii=False, allow_nan=False)
            for r in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition', 'interactors']], 'traces': traces}


if __name__ == '__main__':
    VisITBench(__file__).main_from_args()
