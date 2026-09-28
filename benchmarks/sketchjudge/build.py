"""Tabulate recorded diagram-grading attempts and their original image inputs."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.build_measurement_tables.load_source_files import read_jsonl_objects


class SketchJudge(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, labels = self.build_parameters, self.build_parameters['labels']
        layout = parameters['layout']

        # 1. Join each recorded attempt to its original question and expert annotation.
        with ZipFile(self.raw_dir / layout['dataset']) as archive:
            master = json.loads(archive.read(layout['master']))
        questions = pd.json_normalize(master['questions'], max_level=0)
        questions['source_question'] = master['questions']
        annotations = pd.json_normalize(master['annotations'], max_level=0)
        annotations['source_annotation'] = master['annotations']
        bank = annotations.merge(questions.drop(columns='category'), on='question_id', validate='many_to_one')
        frames = []
        for path in sorted((self.raw_dir / layout['release'] / 'results').glob('*/*.jsonl')):
            records = read_jsonl_objects(path)
            frame = pd.json_normalize(records, max_level=0).rename(columns={'response': 'model_reply'})
            frame['source_record'] = records
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index,
                setting=path.parent.name, source_model=path.stem))
        attempts = pd.concat(frames, ignore_index=True).merge(bank, on='answer_id', how='left', validate='many_to_one', indicator=True)
        if not attempts._merge.eq('both').all() or not attempts.is_correct.map(type).eq(bool).all():
            raise ValueError('A recorded response lacks an unambiguous expert annotation')
        attempts['prompt_family'] = attempts.source_model.str.extract(parameters['patterns']['prompt_suffix'], expand=False).fillna('baseline')
        attempts['model_key'] = attempts.source_model.str.replace(parameters['patterns']['prompt_suffix'], '', regex=True)
        attempts['model'] = attempts.model_key.map(parameters['models'])
        if attempts.model.isna().any() or not attempts.setting.isin(['WithRef', 'NoRef']).all():
            raise ValueError('An undeclared model or reference setting needs source review')
        attempts['response_key'] = attempts.source_file + '#' + attempts.source_row.astype(str)

        # 2. Resolve the image slots actually recorded by each model call.
        columns = attempts.filter(regex='^' + parameters['patterns']['image_column'] + '$').columns.tolist()
        images = attempts[['response_key', *columns]].melt(id_vars='response_key', var_name='image_slot', value_name='recorded_path').dropna(subset=['recorded_path'])
        images['position'] = images.image_slot.str.removeprefix('image_').astype(int)
        images['source_path'] = images.recorded_path.str.replace('\\', '/', regex=False).str.replace('/+', '/', regex=True).str.extract(parameters['patterns']['image_path'], expand=False)
        if images.source_path.isna().any():
            raise ValueError('A recorded image path does not identify a released diagram')
        images = images.sort_values(['response_key', 'position'])
        ordered = images.groupby('response_key', sort=False).source_path.agg(list)
        attempts['recorded_images'] = attempts.response_key.map(ordered)
        attempts['expected_images'] = [([row.input_image_path] if row.requires_input_image else []) +
            ([row.gt_image_path] if row.setting == 'WithRef' else []) + [row.image_path] for row in attempts.itertuples()]
        attempts['input_matches'] = attempts.recorded_images.eq(attempts.expected_images)
        attempts['input_status'] = attempts.input_matches.map({True: labels['valid_input'], False: labels['conflicting_input']})
        assets = images[['source_path']].drop_duplicates().copy()
        with ZipFile(self.raw_dir / layout['dataset']) as archive:
            assets['data'] = (layout['prefix'] + assets.source_path).map(archive.read)
        images = images.merge(assets, on='source_path', how='left', validate='many_to_one')
        images['attachment'] = [dict(data=row.data, path=row.image_slot + '.png',
            media_type=parameters['media_types']['image'], role=row.image_slot) for row in images.itertuples()]
        attachments = images.groupby('response_key', sort=False).attachment.agg(list)

        # 3. Apply the released parser without repairing malformed model replies.
        decoded = []
        for text in attempts.model_reply:
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                value = None
            decoded.append(value)
        attempts['decoded_reply'] = decoded
        valid = attempts.decoded_reply.map(lambda value: isinstance(value, dict) and isinstance(value.get('is_correct'), bool)
            and ('error_count' not in value or isinstance(value['error_count'], int))
            and ('error_list' not in value or isinstance(value['error_list'], list)))
        attempts['response'] = None
        eligible = valid & attempts.input_matches
        verdicts = attempts.loc[eligible, 'decoded_reply'].map(lambda value: value['is_correct'])
        attempts.loc[eligible, 'response'] = verdicts.eq(attempts.loc[eligible, 'is_correct']).astype(float)
        attempts['grade_status'] = labels['invalid_grade']
        attempts.loc[eligible, 'grade_status'] = labels['valid_grade']
        attempts.loc[~attempts.input_matches, 'grade_status'] = labels['conflicting_input']

        # 4. Keep the recorded model, reference setting and prompt family distinct.
        attempts['subject_key'] = attempts.model_key + '/' + attempts.setting + '/' + attempts.prompt_family
        subjects = attempts.drop_duplicates('subject_key')[['subject_key', 'model', 'model_key', 'setting', 'prompt_family']].copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=row.model_key,
            reference_setting=row.setting, prompt_family=row.prompt_family, reported_temperature='0',
            configuration_status=labels['configuration_status']) for row in subjects.itertuples()]

        # 5. Use complete recorded prompts and image bytes as the item stimulus.
        attempts['item_key'] = attempts.response_key
        items = attempts[['item_key', 'answer_id', 'setting', 'prompt_family', 'query', 'category', 'input_status', 'input_matches', 'is_correct']].copy()
        items['raw_item_id'] = items.answer_id + '/' + items.setting + '/' + items.prompt_family
        items['content'] = items['query']
        items['features'] = [dict(category=row.category, input_association=row.input_status,
            input_scope=labels['input_scope']) for row in items.itertuples()]
        items['attachments'] = items.item_key.map(attachments)
        items['grading_criterion'] = [dict(reference_answer=json.dumps(row.is_correct) if row.input_matches else None,
            rule=self.grading['rule']) for row in items.itertuples()]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['verdict'], sort_keys=True))

        # 6. Preserve full responses and both source claims when an association conflicts.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_record=row.source_record, dataset_archive=layout['dataset'], master_member=layout['master'],
            source_question=row.source_question, source_annotation=row.source_annotation,
            recorded_images=row.recorded_images, expected_images=row.expected_images, input_status=row.input_status,
            grade_status=row.grade_status, grade_scope=labels['grade_scope']), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    SketchJudge(__file__).main_from_args()
