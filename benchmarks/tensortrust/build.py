"""Tabulate the released Tensor Trust game attempts and defense-validation records."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TensorTrust(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        grading = self.grading['verifiers']['access_granted']

        # 1. Read each native JSONL table; missing numeric scalars become JSON null.
        sources = []
        for stage, source_file in parameters['records'].items():
            frame = pd.read_json(self.raw_dir / source_file, lines=True,
                convert_dates=False, dtype=parameters['nullable_integers'], precise_float=True)
            frame = frame.astype(object).where(frame.notna(), None)
            frame['source_record'] = frame.to_dict('records')
            frame['source_file'] = source_file
            frame['response_key'] = stage + ':' + frame[parameters['id_columns'][stage]].astype(str)
            frame['user_input'] = frame[parameters['input_columns'][stage]]
            sources.append(frame)
        attempts = pd.concat(sources, ignore_index=True)

        # 2. A subject is the model choice actually recorded by the game.
        subjects = attempts[['llm_choice']].drop_duplicates().rename(columns={'llm_choice': 'subject_key'})
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=parameters['labels']['harness'], recorded_model_choice=choice,
            request_configuration='not recorded per attempt') for choice in subjects.subject_key]

        # 3. The stimulus is the ordered defense/input/defense triple supplied to the game.
        items = attempts[['response_key', 'source_file', 'opening_defense', 'user_input', 'closing_defense']].copy()
        items['item_key'] = items.response_key
        items['raw_item_id'] = items.response_key
        items['content'] = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in
            items[['opening_defense', 'user_input', 'closing_defense']].to_dict('records')]
        items['incomplete'] = items[['opening_defense', 'user_input', 'closing_defense']].isna().any(axis=1)
        # Unknown input fragments must not make unrelated withheld tasks collapse together.
        items['features'] = [dict(unavailable_input_source=row.response_key) if row.incomplete else {}
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=None, rule=grading['rule'])] * len(items)
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['spec'], sort_keys=True))] * len(items)

        # 4. Preserve the published flag and complete native record; do not regrade or drop repeats.
        attempts['subject_key'] = attempts.llm_choice
        attempts['item_key'] = attempts.response_key
        if not attempts.output_is_access_granted.map(lambda value: type(value) is bool).all():
            raise ValueError('The released access-granted flag must be a JSON boolean')
        attempts['response'] = attempts.output_is_access_granted.astype(float)
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, record=row.source_record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=attempts[['response_key', 'subject_key', 'item_key', 'response']],
            traces=attempts[['response_key', 'trace']])


if __name__ == '__main__':
    TensorTrust(__file__).main_from_args()
