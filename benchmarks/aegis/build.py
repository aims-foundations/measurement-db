"""Tabulate AEGIS's recorded Mistral responses and original safety annotations."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Aegis(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, protocols = self.build_parameters, self.grading['verifiers']

        # 1. Read original JSON records as tables, preserving their source positions.
        splits = []
        for split, filename in parameters['files'].items():
            records = json.loads((self.raw_dir / filename).read_text())
            frame = pd.json_normalize(records, max_level=0)
            frame['source_record'] = records
            splits.append(frame.assign(source_file=filename, source_row=frame.index, split=split))
        source = pd.concat(splits, ignore_index=True)

        # 2. Select actual recorded responses, including recorded empty strings.
        attempts = source.loc[source.response.notna()].copy()
        if attempts.id.duplicated().any() or not attempts.response_label.isin(parameters['labels']).all():
            raise ValueError('Duplicate source IDs or unknown response safety labels')
        if not attempts.response_label_source.isin(protocols).all() or attempts.prompt.isna().any():
            raise ValueError('A response has no supported original grader or prompt')
        attempts['response'] = attempts.response_label.map(parameters['labels']).astype(float)
        attempts['response_key'] = attempts.source_file + '/' + attempts.source_row.astype(str)

        # 3. Identify the documented generator; do not turn prompts into subjects.
        model = parameters['model']
        subjects = pd.DataFrame([dict(subject_key=model['identifier'], raw_label=model['label'],
            features=dict(source_model_identifier=model['identifier'], harness=model['harness'],
                          source_attribution=model['attribution']))])
        attempts['subject_key'] = model['identifier']

        # 4. Keep prompt text separate from labels and preserve each grading protocol.
        keys = ['prompt', 'response_label_source']
        items = attempts[keys + ['id']].drop_duplicates(keys).reset_index(drop=True)
        items['item_key'], items['raw_item_id'] = items.index, items.id
        items['content'] = items.prompt.map(lambda prompt: json.dumps(dict(prompt=prompt), ensure_ascii=False))
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = [Judge(judged_by=protocols[name]['kind'], spec=json.dumps(protocols[name], sort_keys=True))
            for name in items.response_label_source]
        attempts = attempts.merge(items[keys + ['item_key']], on=keys, validate='many_to_one')

        # 5. Preserve the complete native output and label provenance for every row.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_record=row.source_record), ensure_ascii=False, sort_keys=True, allow_nan=False)
            for row in attempts.itertuples()]
        attempts['test_condition'] = 'upstream_split=' + attempts.split
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier']],
            responses=attempts[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    Aegis(__file__).main_from_args()
