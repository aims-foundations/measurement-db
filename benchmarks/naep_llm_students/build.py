"""Tabulate recorded NAEP choices and preserve the original question documents."""

import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class NAEPLLMStudents(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read every recorded observation without converting literal choices into missing values.
        responses = pd.read_csv(self.raw_dir / layout['responses'], dtype=str, keep_default_na=False)
        responses['source_row'] = responses.index
        responses['native_record'] = responses.drop(columns='source_row').to_dict('records')
        responses['response_key'] = responses.source_row.astype(str)
        keys = ['llm', 'question_id', 'prompt_type', 'student_grade']
        if responses.duplicated(keys).any():
            raise ValueError('Duplicate recorded model/question/prompt/grade observation')
        responses['response'] = pd.to_numeric(responses.is_correct, errors='raise').astype(float)
        if not responses.response.isin([0, 1]).all() or not responses.response.eq(
                responses.predicted_option.eq(responses.gold_option).astype(float)).all():
            raise ValueError('A recorded grade disagrees with the released extracted choice')
        responses['item_key'], responses['subject_key'] = responses.question_id, responses.llm
        enforced = responses.is_grade_enforced.eq('True')
        if not responses.is_grade_enforced.isin(['True', 'False']).all() or not enforced.eq(responses.student_grade.ne('-1')).all():
            raise ValueError('The recorded grade-enforcement fields disagree')
        responses['test_condition'] = [labels['enforced_condition' if row.is_grade_enforced == 'True' else 'default_condition'].format(
            prompt_type=row.prompt_type, student_grade=row.student_grade) for row in responses.itertuples()]

        # 2. Join original NAEP pages to the recorded question identifiers, retaining exact HTML.
        paths = sorted(self.raw_dir.glob(layout['pages']))
        pages = pd.json_normalize([json.loads(path.read_text()) for path in paths], max_level=0)
        pages['source_file'] = [str(path.relative_to(self.raw_dir)) for path in paths]
        pages['item_key'] = pages.questionID.str.removeprefix(parameters['parsing']['question_prefix']).str.strip()
        items = responses[['item_key', 'subject', 'question_grade', 'total_options', 'gold_option']].drop_duplicates()
        if items.item_key.duplicated().any():
            raise ValueError('Conflicting question metadata or reference choices')
        items = items.merge(pages, on='item_key', how='left', validate='one_to_one', indicator=True)
        if not items['_merge'].eq('both').all() or not items.itemHTML.str.strip().ne('').all():
            raise ValueError('A recorded question lacks its complete original NAEP page')
        items['raw_item_id'] = items.item_key
        items['content'] = items.itemHTML
        items['family'] = items.source_file.str.split('/').str[1]
        items['grading_file'] = items.source_file.str.replace('/items/', '/grading/', regex=False).str.replace(r'\.json$', '.html', regex=True)
        guide_text = items.grading_file.map(lambda value: BeautifulSoup((self.raw_dir / value).read_text(), 'html.parser').get_text(' ', strip=True))
        original_keys = guide_text.str.extract(parameters['parsing']['correct_answer_pattern'], expand=False)
        if not original_keys.eq(items.gold_option).all():
            raise ValueError('A released gold choice disagrees with the original NAEP scoring guide')
        items['features'] = [dict(question_subject=row.subject, question_grade=row.question_grade,
            total_options=row.total_options, source_file=row.source_file, grading_source_file=row.grading_file,
            input_scope=labels['input_scope'])
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.gold_option, rule=grading['rule']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['verifiers']['choice'], sort_keys=True)) for _ in items.index]

        # 3. Link all source-page images without claiming they were delivered to the text-only models.
        assets = pd.DataFrame({'source_path': [str(path.relative_to(self.raw_dir)) for path in sorted(self.raw_dir.glob(layout['images']))]})
        assets['data'] = assets.source_path.map(lambda value: (self.raw_dir / value).read_bytes())
        assets['media_type'] = assets.data.map(lambda value: value[:3].hex()).map(parameters['image_media_types'])
        if assets.media_type.isna().any():
            raise ValueError('An upstream source image has an unrecognized encoding')
        links = items[['item_key', 'family', 'content']].copy()
        links['url'] = links.content.map(lambda value: [tag['src'] for tag in BeautifulSoup(value, 'html.parser').find_all('img')])
        links = links.explode('url').dropna(subset='url')
        links['resource_id'] = links.url.str.extract(parameters['parsing']['resource_pattern'], expand=False)
        if links.resource_id.isna().any():
            raise ValueError('An original image reference lacks its public resource identifier')
        links['source_path'] = 'naep/' + links.family + '/images/' + links.resource_id
        links = links.drop_duplicates(['item_key', 'source_path']).merge(assets.drop(columns='data'), on='source_path', how='left', validate='many_to_one')
        if links.media_type.isna().any():
            raise ValueError('A source-document image was not captured')
        links['attachment'] = [dict(source_path=row.source_path, path='source_images/' + row.resource_id,
            media_type=row.media_type, role=labels['image_role']) for row in links.itertuples()]
        grouped = links.groupby('item_key', sort=False).attachment.agg(list)
        items['attachments'] = items.item_key.map(grouped).map(lambda value: value if isinstance(value, list) else [])

        # 4. Preserve model labels, prompting conditions and all released response fields.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key.map(parameters['model_aliases'])
        if subjects.raw_label.isna().any():
            raise ValueError('An upstream model label lacks a declared canonical alias')
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=value) for value in subjects.subject_key]
        page_paths = items.set_index('item_key').source_file
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['responses'], source_row=row.source_row,
            record=row.native_record, question_source_file=page_paths[row.item_key],
            available_output=labels['output_scope']), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier', 'attachments']],
                'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    NAEPLLMStudents(__file__).main_from_args()
