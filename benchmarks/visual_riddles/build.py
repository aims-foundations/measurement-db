#!/usr/bin/env python3
"""Curate Visual Riddles' original images, caption conditions, and human-rated answers."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class VisualRiddles(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Read the native table and expand its literal lists of rated answers.
        riddles = pd.read_csv(self.raw_dir / layout['table'], dtype=str, keep_default_na=False)
        riddles = riddles.rename(columns={'human-caption': 'human_caption'})
        riddles['source_row'] = riddles.index
        if riddles.image_id.duplicated().any():
            raise ValueError('Released image IDs must be unique')
        answers = riddles[['image_id', 'source_row']].assign(
            native_record=riddles['model_rated_answers-open_ended'].map(ast.literal_eval)).explode('native_record', ignore_index=True)
        answers['source_entry'] = answers.groupby('source_row', sort=False).cumcount()
        answers = answers.join(pd.json_normalize(answers.native_record, max_level=0))
        if not answers.type.isin([*parameters['input_modes'], 'human']).all():
            raise ValueError('Unrecognized released answer type')
        responses = answers.loc[answers.type.ne('human')].copy()
        if not responses.human_rating.dropna().map(lambda value: type(value) is bool).all():
            raise ValueError('Human correctness ratings must be boolean or unavailable')
        responses['response_key'] = responses.index
        responses['response'] = responses.human_rating.map({True: 1., False: 0.})
        responses['input_mode'] = responses.type.map(parameters['input_modes'])

        # 2. Distinguish the actual image and human-caption stimuli, retaining full text.
        items = responses[['image_id', 'input_mode']].drop_duplicates().merge(
            riddles, on='image_id', how='left', validate='many_to_one').reset_index(drop=True)
        items['item_key'] = items.index
        items['raw_item_id'] = items.image_id + ':' + items.input_mode
        items['content'] = [row.question if row.input_mode == 'image' else json.dumps(
            dict(question=row.question, human_caption=row.human_caption), ensure_ascii=False)
            for row in items.itertuples()]
        items['features'] = [dict(image_id=row.image_id, input_mode=row.input_mode,
            category=row.category, difficulty_level_index=row.difficulty_level_index) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=answer, rule=self.grading['rule']) for answer in items.ground_truth_answer]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['human'], sort_keys=True), judged_by='human')

        # 3. Attach original image bytes; caption-only models receive source links, not images.
        images = riddles[['image_id', 'file_name']].set_index('image_id')
        images['data'] = [(self.raw_dir / layout['images'] / name).read_bytes() for name in images.file_name]
        assets = images.to_dict('index')
        items['attachments'] = [[dict(data=assets[row.image_id]['data'], path=layout['images'] + '/' + row.file_name,
            media_type='image/jpeg', role='input' if row.input_mode == 'image' else 'source')] for row in items.itertuples()]
        responses = responses.merge(items[['image_id', 'input_mode', 'item_key']],
            on=['image_id', 'input_mode'], how='left', validate='many_to_one')

        # 4. Separate literal model/pipeline configurations without guessing historical settings.
        subjects = responses[['model_name', 'type']].drop_duplicates().reset_index(drop=True)
        subjects['subject_key'] = subjects.index
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.model_name + ' / ' + subjects.type
        subjects['features'] = [dict(source_model=row.model_name, source_type=row.type,
            input_pipeline=parameters['pipelines'][row.type], harness=parameters['labels']['harness']) for row in subjects.itertuples()]
        responses = responses.merge(subjects[['model_name', 'type', 'subject_key']],
            on=['model_name', 'type'], how='left', validate='many_to_one')
        responses['test_condition'] = 'source_type=' + responses.type

        # 5. Keep each complete native answer and its original source position.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['table'], source_row=int(row.source_row),
            source_entry=int(row.source_entry), image_id=row.image_id, native_record=row.native_record),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    VisualRiddles(__file__).main_from_args()
