"""Tabulate released STEM assessment responses, fractional grades and input images."""

import json
import mimetypes
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MultimodalStemAI(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']
        grading = self.grading['verifiers']['hybrid']

        # 1. Load the original question records and the declared individual run channels.
        samples = pd.DataFrame(json.loads((self.raw_dir / layout['records']).read_text()))
        samples['source_row'] = samples.index
        channels = pd.DataFrame(dict(model=self.build_parameters['models'],
            prompt_strategy=self.build_parameters['prompt_strategies'])).rename_axis('channel').reset_index()
        attempts = pd.DataFrame.from_records(samples.llm_responses).rename_axis('source_row').reset_index()
        attempts = attempts.melt(id_vars='source_row', value_vars=channels.channel, var_name='channel', value_name='record')
        attempts = attempts.merge(channels, on='channel', validate='many_to_one')

        # 2. Each source model and prompt strategy defines its recorded subject configuration.
        subjects = channels.rename(columns={'channel': 'subject_key', 'model': 'raw_label'}).copy()
        subjects['features'] = [dict(harness=self.build_parameters['labels']['harness'], source_channel=row.subject_key,
            prompt_strategy=row.prompt_strategy, historical_request_settings='not released') for row in subjects.itertuples()]

        # 3. Keep the full question, ordered images and original grading reference information.
        items = samples.drop(columns='llm_responses').copy()
        items['item_key'] = items.source_row.astype(str)
        items['raw_item_id'] = items.Course_name + ':' + items.Exercise_name
        items['image_paths'] = [list(map(lambda name: str(Path(row.Course_name) / name), row.Question_images)) for row in items.itertuples()]
        items['content'] = [json.dumps(dict(multimedia_elements=[dict(content_type='text/plain',
            text=row.Question)] +
            [dict(content_type=mimetypes.guess_type(name)[0], location=f'question_images/{ordinal}/{name}')
                for ordinal, name in enumerate(row.image_paths, 1)]),
            ensure_ascii=False) for row in items.itertuples()]
        items['features'] = items[['Data_source', 'Course_Category', 'Question_type', 'Language']].to_dict('records')
        # The shared downloader escapes punctuation in local filenames; image locators retain source names.
        items['attachments'] = [[dict(source_path=re.sub(r'[^A-Za-z0-9._/-]', lambda m: f'_x{ord(m[0]):02x}_',
            str(Path(layout['input_images']) / name)), path=f'question_images/{ordinal}/{name}',
            media_type=mimetypes.guess_type(name)[0], role='input') for ordinal, name in enumerate(names, 1)] for names in items.image_paths]
        items['grading_criterion'] = [dict(reference_answer=row.Gold_answer, rule=grading['rule']) for row in items.itertuples()]
        items['verifier'] = [Judge(judge=grading['judge'], spec=json.dumps(dict(**grading['spec'],
            question_type=row.Question_type, extracted_info=row.Extracted_Info), sort_keys=True)) for row in items.itertuples()]

        # 4. Preserve every original score and complete response object without regrading.
        attempts['response_key'] = attempts.source_row.astype(str) + ':' + attempts.channel
        attempts['subject_key'] = attempts.channel
        attempts['item_key'] = attempts.source_row.astype(str)
        attempts['response'] = attempts.record.map(lambda value: value['grade']['grade_score'])
        attempts['trace'] = [json.dumps(dict(source_file=layout['records'], source_row=int(row.source_row),
            channel=row.channel, record=row.record), ensure_ascii=True, allow_nan=False) for row in attempts.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            responses=attempts[['response_key', 'subject_key', 'item_key', 'response']],
            traces=attempts[['response_key', 'trace']])


if __name__ == '__main__':
    MultimodalStemAI(__file__).main_from_args()
