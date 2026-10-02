#!/usr/bin/env python3
"""Tabulate released PKU-SafeRLHF safety annotations and complete original outputs."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class PKUSafeRLHF(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters['labels']

        # 1. Concatenate native comparison tables, keeping every original field.
        files = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            table = pd.read_json(path, lines=True, convert_dates=False, dtype=False)
            table['record'] = table.to_dict('records')
            table['source_file'] = str(path.relative_to(self.raw_dir))
            files.append(table.rename_axis('source_row').reset_index())
        pairs = pd.concat(files, ignore_index=True)
        pairs['pair_key'] = pairs.source_file + ':' + pairs.source_row.astype(str)

        # 2. Stack the two response sides; safety flags are observed grades.
        sides = []
        for side in [0, 1]:
            table = pairs.rename(columns={f'response_{side}_source': 'subject_key',
                f'is_response_{side}_safe': 'safe'})
            sides.append(table[['pair_key', 'source_file', 'source_row', 'prompt',
                'record', 'subject_key', 'safe']].assign(side=side))
        responses = pd.concat(sides, ignore_index=True)
        if not responses.safe.map(type).eq(bool).all():
            raise ValueError('Every safety grade must be an original boolean')

        # 3. Retain model aliases and complete prompts, without annotation leakage.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=model)
            for model in subjects.subject_key]
        items = pairs.drop_duplicates('prompt').rename(columns={
            'prompt': 'content', 'pair_key': 'raw_item_id'}).copy()
        items['item_key'] = items.content
        items['features'] = [dict(input_scope=labels['input_scope']) for _ in items.index]
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = Judge(judge=labels['judge'], judged_by='human',
            spec=json.dumps(self.grading['verifiers']['safety'], sort_keys=True))

        # 4. Identify source annotation occasions, not assumed independent generations.
        responses['response_key'] = responses.pair_key + ':' + responses.side.astype(str)
        responses['item_key'] = responses.prompt
        responses['response'] = responses.safe.astype(float)
        responses['test_condition'] = responses.response_key

        # 5. Preserve both full outputs and all native annotations, including empty text.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            side=row.side, record=row.record), ensure_ascii=False) for row in responses.itertuples()]
        return dict(subjects=subjects,
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            traces=traces)


if __name__ == '__main__':
    PKUSafeRLHF(__file__).main_from_args()
