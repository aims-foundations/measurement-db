"""Tabulate released MMBench answers with original images and explicit grading."""

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl.utils.escape import unescape
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class MMBenchV11(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, parsing = parameters['layout'], parameters['parsing']
        choices = list(parsing['choices'])
        input_fields = parsing['input_fields'].split()
        stride = int(parsing['circular_stride'])

        # 1. Concatenate native workbooks; identical complete exports are linked copies.
        frames = []
        for path in sorted((self.raw_dir / layout['results']).glob(layout['predictions_glob'])):
            frame = pd.read_excel(path, dtype=object, keep_default_na=False, na_values=[''])
            frame = frame.where(frame.notna(), None)
            if frame['index'].duplicated().any():
                raise ValueError('A prediction workbook repeats an upstream index')
            native = frame.sort_values('index').to_dict('records')
            fingerprint = hashlib.sha256(json.dumps(native, sort_keys=True, ensure_ascii=False,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            frames.append(frame.assign(native_record=frame.to_dict('records'),
                source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index,
                model_label=path.name.split(parsing['filename_marker'])[0], export_hash=fingerprint))
        records = pd.concat(frames, ignore_index=True).astype(object).where(lambda table: table.notna(), None)
        records['source_ref'] = records[['source_file', 'source_row']].to_dict('records')
        records['response_key'] = records.model_label + '::' + records.export_hash + '::' + records['index'].astype(str)
        sources = records.groupby('response_key', sort=False).source_ref.agg(list)
        responses = records.drop_duplicates('response_key').copy()
        responses['source_aliases'] = responses.response_key.map(sources)
        responses['subject_key'] = responses.model_label
        responses['item_key'] = responses['index'].astype(int)
        responses['base_index'] = responses.item_key.mod(stride)

        # 2. Join every recorded input to the task bank and resolve its original image.
        banks = []
        for path in sorted(self.raw_dir.glob(layout['tasks_glob'])):
            frame = pd.read_csv(path, sep='\t', dtype=object, keep_default_na=False, na_values=[''])
            banks.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        bank = pd.concat(banks, ignore_index=True).astype(object).where(lambda table: table.notna(), None)
        bank['item_key'] = bank['index'].astype(int)
        if bank.item_key.duplicated().any():
            raise ValueError('The task bank repeats an upstream index')
        supplied = responses.native_record.map(lambda row: set(row))
        joined = responses.merge(bank[['item_key'] + input_fields + ['answer']],
            on='item_key', how='left', suffixes=('', '_bank'), validate='many_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('A released answer has no corresponding task definition')
        for field in input_fields:
            present = supplied.map(lambda fields: field in fields).to_numpy()
            actual = joined[field].map(lambda value: unescape(str(value)) if value is not None else '')
            expected = joined[field + '_bank'].map(lambda value: str(value) if value is not None else '')
            if not actual.loc[present].eq(expected.loc[present]).all():
                raise ValueError('A recorded input differs from the task bank: ' + field)
        responses['reference'] = joined.answer_bank.to_numpy()
        responses['reference_conflict'] = responses.answer.notna() & responses.answer.ne(responses.reference)
        responses['reference_restored'] = responses.answer.isna()
        images = bank[['item_key', 'image']].set_index('item_key').copy()
        images['source_index'] = images.index
        pending = images.image.str.fullmatch(r'\d+')
        seen = set()
        while pending.any():
            state = tuple(zip(images.index[pending], images.loc[pending, 'image']))
            if state in seen:
                raise ValueError('Cyclic image references in the task bank')
            seen.add(state)
            linked = images.reindex(images.loc[pending, 'image'].astype(int))
            if linked.image.isna().any():
                raise ValueError('A task references an absent source image')
            images.loc[pending, ['image', 'source_index']] = linked[['image', 'source_index']].to_numpy()
            pending = images.image.str.fullmatch(r'\d+')
        image_files = images.drop_duplicates('source_index').copy()
        image_files['bytes'] = image_files.image.map(lambda value: base64.b64decode(value, validate=True))
        image_files['media_type'] = image_files.bytes.map(lambda value:
            parameters['image_types'][Image.open(io.BytesIO(value)).format])
        image_files['path'] = image_files.source_index.map(lambda value: f'images/{value}')
        images = images[['source_index']].merge(image_files[['source_index', 'bytes', 'media_type', 'path']],
            on='source_index', how='left', validate='many_to_one').set_axis(images.index)
        items = bank.merge(images, left_on='item_key', right_index=True, how='left', validate='one_to_one')

        # 3. Keep each presented option ordering, with labels outside the input content.
        prompt = parameters['prompt']
        items['prompt'] = items.hint.map(lambda hint: prompt['hint'].format(hint=hint) if hint is not None else '')
        items['prompt'] += items.question.map(lambda question: prompt['question'].format(question=question))
        option_rows = items[['item_key'] + choices].melt(id_vars='item_key', var_name='letter', value_name='value')
        option_rows = option_rows.dropna(subset=['value'])
        option_rows['text'] = [prompt['choice'].format(letter=row.letter, value=row.value) for row in option_rows.itertuples()]
        option_text = option_rows.groupby('item_key', sort=False).text.agg(''.join)
        items['prompt'] += prompt['options'] + items.item_key.map(option_text) + prompt['suffix']
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type=row.media_type, location=row.path), dict(content_type='text/plain', text=row.prompt)]),
            ensure_ascii=False) for row in items.itertuples()]
        items['attachments'] = [[dict(data=row.bytes, path=row.path, media_type=row.media_type, role='input')]
            for row in items.itertuples()]
        items['raw_item_id'] = items.item_key.astype(str)
        items['features'] = [dict(upstream_index=row['item_key'], base_question_index=row['item_key'] % stride,
            circular_rotation=row['item_key'] // stride, split=row['split'], category=row['category'],
            ability=row['l2-category'], source_file=row['source_file'], source_row=row['source_row'],
            source_image_index=row['source_index']) for row in items.to_dict('records')]
        items['grading_criterion'] = [dict(reference_answer=value, rule=self.grading['rule']) for value in items.answer]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['recorded_answer'], sort_keys=True))

        # 4. Reconstruct only the historical deterministic extraction, with no judge fallback.
        predictions = responses.prediction.fillna('').astype(str)
        words = predictions.str.translate(str.maketrans(parsing['punctuation'], ' ' * len(parsing['punctuation'])))
        word_hits = pd.DataFrame({letter: responses[letter].notna() & words.str.contains(r'(?<!\S)' + letter + r'(?!\S)', regex=True)
            for letter in choices}, index=responses.index)
        inferred = word_hits.idxmax(axis=1).where(word_hits.sum(axis=1).eq(1))
        rejected = pd.Series(False, index=responses.index)
        for message in parameters['refusals'].values():
            rejected |= predictions.str.contains(message, regex=False)
        rejected |= word_hits.sum(axis=1).eq(0) & words.str.contains(r'(?<!\S)' + parsing['rejection_marker'] + r'(?!\S)', regex=True)
        inferred.loc[rejected] = parsing['rejection_marker']
        text_hits = pd.DataFrame({letter: [value is not None and str(value).lower() in text
            for value, text in zip(responses[letter], predictions.str.lower())] for letter in choices}, index=responses.index)
        textual = text_hits.idxmax(axis=1).where(text_hits.sum(axis=1).eq(1))
        inferred = inferred.fillna(textual)
        api_failure = predictions.str.contains(parsing['api_failure'], regex=False)
        inferred.loc[api_failure] = textual.loc[api_failure]
        usable = inferred.notna() & responses.prediction.notna() & ~api_failure & ~responses.reference_conflict
        responses['inferred_option'] = inferred.astype(object).where(inferred.notna(), None)
        responses['response'] = inferred.eq(responses.reference).astype(float).where(usable)
        responses['grade_status'] = 'unresolved_extraction'
        responses.loc[usable, 'grade_status'] = 'derived_historical_comparator'
        responses.loc[responses.reference_conflict, 'grade_status'] = 'conflicting_recorded_reference'
        responses.loc[api_failure, 'grade_status'] = 'api_failure'
        responses.loc[responses.prediction.isna(), 'grade_status'] = 'missing_prediction'

        # 5. Preserve original circular verdicts and supplement caches with their actual scope.
        circular = []
        for path in sorted((self.raw_dir / layout['results']).glob(layout['judgments_glob'])):
            frame = pd.read_excel(path, dtype=object, keep_default_na=False, na_values=[''])
            frame = frame.where(frame.notna(), None)
            saved = read_native_pickle(path.with_suffix('.pkl'))
            if len(frame) != len(saved) or any(saved[int(row['index'])] != {'hit': row['hit'], 'log': row['log']}
                    for row in frame.to_dict('records')):
                raise ValueError('Circular workbook verdicts differ from their native cache')
            circular.append(frame.assign(model_label=path.name.split(parsing['filename_marker'])[0],
                base_index=frame['index'].astype(int), circular_record=frame.to_dict('records'),
                circular_workbook=str(path.relative_to(self.raw_dir)), circular_cache=str(path.with_suffix('.pkl').relative_to(self.raw_dir)),
                circular_source_row=frame.index))
        judged = pd.concat(circular, ignore_index=True)
        base = responses.loc[responses.item_key.eq(responses.base_index), ['model_label', 'base_index', 'prediction']]
        base = base.loc[base.model_label.isin(judged.model_label)]
        match = judged.merge(base, on=['model_label', 'base_index'], how='left', suffixes=('', '_original'), validate='one_to_one')
        if not match.prediction.eq(match.prediction_original).all():
            raise ValueError('A circular verdict does not match its original prediction export')
        judged['circular_evidence'] = judged[['circular_record', 'circular_workbook', 'circular_cache', 'circular_source_row']].to_dict('records')
        responses = responses.merge(judged[['model_label', 'base_index', 'circular_evidence']],
            on=['model_label', 'base_index'], how='left', validate='many_to_one')
        responses['circular_evidence'] = responses.circular_evidence.map(lambda value: value if isinstance(value, dict) else None)
        supplements = []
        for path in sorted((self.raw_dir / layout['results']).glob(layout['supplements_glob'])):
            saved = read_native_pickle(path)
            frame = pd.DataFrame(saved.items(), columns=['item_key', 'cached_prediction'])
            supplements.append(frame.assign(model_label=path.name.split(parsing['filename_marker'])[0],
                supplement_source=str(path.relative_to(self.raw_dir))))
        supplement = pd.concat(supplements, ignore_index=True)
        originals = responses.loc[responses.model_label.isin(supplement.model_label), ['model_label', 'item_key', 'prediction']]
        matched = supplement.merge(originals, on=['model_label', 'item_key'],
            how='left', validate='one_to_one')
        if not matched.cached_prediction.eq(matched.prediction).all():
            raise ValueError('An unassociated supplement prediction needs separate review')
        responses = responses.merge(supplement, on=['model_label', 'item_key'], how='left', validate='many_to_one')

        # 6. Emit linked tables with complete native cells, source aliases and grading evidence.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=value) for value in subjects.subject_key]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(export_hash=row.export_hash, prediction_record=row.native_record,
            source_aliases=row.source_aliases, inferred_option=row.inferred_option, reference_restored=bool(row.reference_restored),
            grade_status=row.grade_status, circular_evidence=row.circular_evidence,
            supplement=dict(file=row.supplement_source, index=row.item_key, prediction=row.cached_prediction)
                if isinstance(row.supplement_source, str) else None), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    MMBenchV11(__file__).main_from_args()
