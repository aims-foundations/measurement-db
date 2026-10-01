"""Tabulate MMMU's released predictions and their original multimodal questions."""

import ast
import base64
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from openpyxl.utils.escape import unescape

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MMMUDevVal(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, parsing = parameters['paths'], parameters['parsing']

        # 1. Concatenate maintained exports, keeping every record and its coordinates.
        frames = []
        for path in sorted((self.raw_dir / paths['results']).glob(paths['predictions'])):
            table = pd.read_excel(path, keep_default_na=False)
            frames.append(table.assign(native_record=table.to_dict('records'), source_row=table.index,
                source_file=str(path.relative_to(self.raw_dir)),
                subject_key=path.name.removesuffix(parsing['filename_suffix'])))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda table: table.index)
        if 'id' not in responses:
            responses['id'] = None
        if responses.duplicated(['subject_key', 'index']).any():
            raise ValueError('Duplicate model/question records in maintained exports')
        if set(responses.columns) & {'score', 'hit', 'grade', 'correct', 'accuracy'}:
            raise ValueError('Published grades require an explicit import path; do not discard them')

        # 2. Join the matching historical bank; verify text, options, references and image order.
        items = pd.read_csv(self.raw_dir / paths['questions'], sep='\t', keep_default_na=False)
        letters = list(parsing['letters'])
        fields = list(parameters['question_fields'].values()) + letters
        padding = json.loads(parsing['padding_values'])
        for table in (responses, items):
            table[fields] = table[fields].map(lambda value: unescape(str(value)))
            table[letters] = table[letters].replace(padding, '')
            table['image_names'] = table.image_path.map(
                lambda value: ast.literal_eval(value) if value.startswith('[') else [value])
            table['image_order'] = table.image_names.map(json.dumps)
        items['item_key'] = items['index']
        joined_fields = ['index', 'image_order', *fields]
        responses = responses.merge(items[[*joined_fields, 'item_key', 'id']], on=joined_fields,
            how='left', validate='many_to_one', suffixes=('', '_bank'))
        if responses.item_key.isna().any() or not responses.id.fillna(responses.id_bank).eq(responses.id_bank).all():
            raise ValueError('A released question, reference, identifier or image differs from the task bank')
        items = items.loc[items.item_key.isin(responses.item_key)].copy()

        # 3. Assemble question/option text and attach original images without re-encoding.
        choices = items.melt(id_vars='item_key', value_vars=letters, var_name='letter', value_name='option')
        choices = choices.loc[choices.option.ne('')].copy()
        choices['line'] = choices.letter + '. ' + choices.option + '\n'
        items['text'] = items.question + '\n' + items.item_key.map(choices.groupby('item_key').line.sum()).fillna('')
        items['image_values'] = items.image.map(
            lambda value: ast.literal_eval(value) if value.startswith('[') else [value])
        if not items.image_names.map(len).eq(items.image_values.map(len)).all():
            raise ValueError('Image locators and payloads have different lengths')
        items['attachments'] = [[dict(data=base64.b64decode(value, validate=True),
            path=f'image_{index + 1}.jpg', media_type='image/jpeg', role='input')
            for index, value in enumerate(values)] for values in items.image_values]
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type='text/plain', text=row.text),
            *[dict(content_type=asset['media_type'], location=asset['path']) for asset in row.attachments]]),
            ensure_ascii=False) for row in items.itertuples()]
        items['raw_item_id'] = items.id
        items['features'] = [{name: quote(str(record[column]), safe=' /-._')
            for name, column in parameters['item_features'].items()} for record in items.to_dict('records')]
        items['grading_criterion'] = [dict(reference_answer=answer, rule=self.grading['rule']) for answer in items.answer]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['answer_extraction'], sort_keys=True))

        # 4. Project subjects and ungraded attempts; retain full native outputs in traces.
        subjects = responses[['subject_key']].drop_duplicates().assign(raw_label=lambda table: table.subject_key)
        subjects['features'] = [dict(model_identifier=quote(label, safe=' /-._'),
            harness=parameters['subject']['harness']) for label in subjects.subject_key]
        responses['response'] = None
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            native_record=row.native_record, grade_status='upstream_grade_unavailable'),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': traces,
        }


if __name__ == '__main__':
    MMMUDevVal(__file__).main_from_args()
