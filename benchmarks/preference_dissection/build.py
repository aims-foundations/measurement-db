#!/usr/bin/env python3
"""Tabulate original Preference Dissection choices without reinterpreting them as accuracy."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PreferenceDissection(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters['labels']

        # 1. Read the native comparison records; their positions are the source IDs.
        source_file = parameters['layout']['data']
        pairs = pd.read_parquet(self.raw_dir / source_file).rename_axis('source_row').reset_index()
        pairs['raw_item_id'] = pairs.source_row.astype(str)
        pairs['content'] = [json.dumps(dict(query=query, response_1=first['content'],
            response_2=second['content']), ensure_ascii=False, sort_keys=True)
            for query, first, second in zip(pairs['query'], pairs.response_1, pairs.response_2)]

        # 2. Melt the judge columns into observations, excluding only the human label.
        preferences = pd.json_normalize(pairs.preference_labels.tolist(), max_level=0)
        preferences['source_row'] = pairs.source_row
        responses = preferences.drop(columns=labels['human']).melt(
            id_vars='source_row', var_name='subject_key', value_name='preference_label')
        if not responses.preference_label.isin(parameters['preference_values']).all():
            raise ValueError('Every model preference must be an original response_1/response_2 label')
        responses = responses.merge(pairs[['source_row', 'raw_item_id']], on='source_row', validate='many_to_one')

        # 3. Keep full comparison stimuli and the original model-judge names.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=model)
            for model in subjects.subject_key]
        items = pairs.rename(columns={'source_row': 'item_key'}).copy()
        items['features'] = [dict(input_scope=labels['input_scope']) for _ in items.index]
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['preference'], sort_keys=True))

        # 4. Preserve repeated source records without claiming independent model runs.
        responses['item_key'] = responses.source_row
        responses['response_key'] = responses.subject_key + ':' + responses.raw_item_id
        responses['response'] = responses.preference_label.map(parameters['preference_values']).astype(float)
        responses['test_condition'] = labels['test_condition_prefix'] + responses.raw_item_id

        # 5. Trace only the released choice and its provenance, not unreleased reasoning.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=source_file, source_row=row.source_row,
            source_model_label=row.subject_key, preference_label=row.preference_label), ensure_ascii=False)
            for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    PreferenceDissection(__file__).main_from_args()
