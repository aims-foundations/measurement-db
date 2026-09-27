"""Tabulate the complete released NaturalReasoning questions and ungraded answers."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class NaturalReasoning(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        source_file = parameters['layout']['responses']

        # 1. Load every original JSONL record without guessing types or parsing dates.
        data = pd.read_json(self.raw_dir / source_file, lines=True, dtype=False, convert_dates=False)
        data['source_record'] = data.to_dict('records')
        data['source_row'] = data.index

        # 2. Keep questions separate from references and recorded generated answers.
        items = data.rename(columns={'question': 'content'}).copy()
        items['item_key'] = items.source_row
        items['raw_item_id'] = source_file + ':' + items.source_row.astype(str)
        items['features'] = [parameters['item_features']] * len(items)
        items['grading_criterion'] = [dict(reference_answer=answer or None, rule=grading['rule'])
            for answer in items.reference_answer]
        items['verifier'] = Judge(spec=json.dumps(grading['verifiers']['reference_check'], sort_keys=True),
            judged_by='llm')

        # 3. Expand the released answer lists; no saved correctness verdicts are supplied.
        responses = data.explode('responses', ignore_index=True)
        responses['response_index'] = responses.groupby('source_row', sort=False).cumcount()
        answers = pd.json_normalize(responses.pop('responses'), max_level=0)
        responses = responses.join(answers.rename(columns={'response': 'generated_answer'}))
        responses['item_key'] = responses.source_row
        responses['subject_key'] = responses.response_model
        responses['response_key'] = responses.source_row.astype(str) + ':' + responses.response_index.astype(str)
        responses['response'] = None
        responses['test_condition'] = parameters['labels']['test_condition']

        # 4. Preserve literal model labels and complete native records, including empty references.
        subjects = responses[['subject_key', 'response_model']].drop_duplicates().rename(
            columns={'response_model': 'raw_label'})
        subjects['features'] = [parameters['subject_features']] * len(subjects)
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=source_file, source_row=row.source_row,
            response_index=row.response_index, record=row.source_record,
            grade_status=parameters['labels']['grade_status']), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces}


if __name__ == '__main__':
    NaturalReasoning(__file__).main_from_args()
