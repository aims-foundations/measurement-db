"""Tabulate OCRBench v2's released predictions and their original image inputs."""

import io
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class OCRBenchV2(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters['paths']

        # 1. Load native prediction records and concatenate the image/question shards.
        native = json.loads((self.raw_dir / paths['predictions']).read_text())
        records = pd.json_normalize(native, max_level=0).assign(native_record=native)
        records['source_row'] = records.index
        if any('score' in row for row in native):
            raise ValueError('This ungraded release reader must not discard published scores')
        shards = []
        for path in sorted(self.raw_dir.glob(paths['questions'])):
            shard = pd.json_normalize(pq.read_table(path).to_pylist(), max_level=0)
            shards.append(shard.assign(question_file=str(path.relative_to(self.raw_dir)), question_row=shard.index))
        bank = pd.concat(shards, ignore_index=True)
        if records.id.duplicated().any() or bank.id.duplicated().any() or set(records.id) != set(bank.id):
            raise ValueError('Prediction and question IDs must be unique and match completely')
        bank = bank.rename(columns={column: column + '_bank' for column in bank.columns
            if column not in ['id', 'image', 'question_file', 'question_row']})

        # 2. Validate the join, including the known HF serialization of structured answers.
        records = records.merge(bank, on='id', how='left', validate='one_to_one')
        if not all(records[field].eq(records[field + '_bank']).all() for field in ['question', 'dataset_name', 'type']):
            raise ValueError('A prediction refers to a different question or task type')
        encoded_answers = records.answers.map(lambda values:
            [value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) for value in values])
        expected_bbox = records.native_record.map(lambda row: row.get('bbox'))
        bank_bbox = [row.bbox_list_bank if row.type == parameters['layout']['spotting_type'] else row.bbox_bank
                     for row in records.itertuples()]
        if (not encoded_answers.eq(records.answers_bank).all()
            or expected_bbox.tolist() != bank_bbox
            or records.native_record.map(lambda row: row.get('eval', 'None')).tolist() != records.eval_bank.tolist()
            or records.native_record.map(lambda row: row.get('content')).tolist() != records.content_bank.tolist()):
            raise ValueError('Native grading fields disagree with the image/question release')

        # 3. Keep exact image bytes and native grading inputs in the item definition.
        items = records.assign(item_key=records.id, raw_item_id=records.id.astype(str)).copy()
        items['image_bytes'] = items.image.map(lambda image: image['bytes'])
        if not items.image_bytes.map(lambda data: Image.open(io.BytesIO(data)).format == 'JPEG').all():
            raise ValueError('Expected original JPEG image bytes')
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type='image/jpeg', location='image.jpg'),
            dict(content_type='text/plain', text=question)]), ensure_ascii=False) for question in items.question]
        items['attachments'] = [[dict(data=data, path='image.jpg', media_type='image/jpeg', role='input')]
                                for data in items.image_bytes]
        items['features'] = [dict(dataset_name=quote(row.dataset_name, safe='/'), task_type=row.type)
                             for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=json.dumps(row['answers'], ensure_ascii=False), rule=json.dumps(dict(
            description=self.grading['rule'], native_parameters={key: row[key]
                for key in parameters['grading_fields'].values() if key in row}), ensure_ascii=False)) for row in records.native_record]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['task_metrics'], sort_keys=True))

        # 4. Retain every released attempt. No per-item grades were published.
        subjects = pd.DataFrame([dict(subject_key=0, raw_label=parameters['subject']['label'],
            features=dict(harness=parameters['subject']['harness']))])
        responses = items[['item_key']].assign(response_key=records.id, subject_key=0, response=None)
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=paths['predictions'], source_row=int(row.source_row),
            record=row.native_record, question_file=row.question_file, question_row=int(row.question_row),
            grade_status='upstream_grade_unavailable'), ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses,
            'traces': traces,
        }


if __name__ == '__main__':
    OCRBenchV2(__file__).main_from_args()
