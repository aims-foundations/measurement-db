"""Tabulate released TxBench-PP tasks and their published endpoint verdicts."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TxBenchPP(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout, labels = self.build_parameters['layout'], self.build_parameters['labels']

        # 1. Read the original panel, configuration registry and released tasks.
        index = json.loads((self.raw_dir / layout['index']).read_bytes())
        rows = pd.json_normalize(index['runResults'], max_level=0)
        rows['source_row'], rows['native_record'] = rows.index, index['runResults']
        subjects = pd.json_normalize(index['modelResults'], max_level=0).rename(columns={'id': 'subject_key', 'modelName': 'raw_label'})
        subjects['configuration_record'] = index['modelResults']
        published = pd.json_normalize(index['evals'], max_level=0).rename(columns={'id': 'item_key'})
        records = [json.loads(path.read_bytes()) for path in sorted((self.raw_dir / layout['tasks']).glob('*/eval.json'))]
        items = pd.json_normalize(records, max_level=0).rename(columns={'id': 'item_key', 'task': 'content'})
        items['task_record'] = records
        items = items.merge(published[['item_key', 'prompt']], on='item_key', validate='one_to_one')
        if not items.content.eq(items.prompt).all():
            raise ValueError('The original sources disagree on a released task instruction')

        # 2. Keep recorded verdicts with released instructions and identify configurations.
        rows = rows.loc[rows.task.isin(items.item_key)].copy()
        if rows[['modelId', 'task', 'trialIndex']].duplicated().any() or not rows.passed.map(lambda value: isinstance(value, bool)).all():
            raise ValueError('The native panel has duplicate identities or a non-Boolean endpoint verdict')
        rows = rows.rename(columns={'modelId': 'subject_key', 'task': 'item_key', 'trialIndex': 'trial'})
        rows = rows.merge(subjects[['subject_key', 'configuration_record']], on='subject_key', how='left', validate='many_to_one')
        if rows.configuration_record.isna().any():
            raise ValueError('An original configuration is missing from the model registry')
        rows['response'], rows['response_key'] = rows.passed.astype(float), rows.source_row
        subjects = subjects.loc[subjects.subject_key.isin(rows.subject_key)].copy()
        subjects['features'] = [dict(harness=row.harness, source_configuration=row.subject_key,
            historical_settings=labels['historical_settings']) for row in subjects.itertuples()]
        items['raw_item_id'] = items.item_key
        items['grading_criterion'] = [dict(rule=json.dumps(dict(interpretation=self.grading['rule'], source_grader=grader), sort_keys=True)) for grader in items.grader]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['reported'], sort_keys=True))] * len(items)
        items['features'] = [dict(input_scope=labels['input_scope'], source_data_nodes=json.dumps(nodes)) for nodes in items.data_node]

        # 3. Join explicitly named example runs and their original trajectory previews.
        examples = published[['item_key', 'runs']].explode('runs', ignore_index=True).rename(columns={'runs': 'example_record'})
        examples = examples.join(pd.json_normalize(examples.example_record, max_level=0))
        examples = examples.merge(subjects[['subject_key', 'raw_label', 'harness']],
            left_on=['model', 'harness'], right_on=['raw_label', 'harness'], how='left', validate='many_to_one')
        examples['trial'] = examples.run.str.removeprefix('r').astype(int)
        previews = []
        for path in sorted((self.raw_dir / layout['previews']).glob('*.json')):
            previews.append(dict(trajectoryFile=layout['preview_prefix'] + path.name, native_preview=json.loads(path.read_bytes())))
        examples = examples.merge(pd.DataFrame(previews), on='trajectoryFile', how='left', validate='one_to_one')
        if examples.native_preview.isna().any() or examples.subject_key.isna().any():
            raise ValueError('A linked example lacks its original preview or model configuration')
        for field, column in [('exampleId', 'item_key'), ('runId', 'runId'), ('model', 'model'), ('harness', 'harness'), ('provider', 'provider')]:
            if not examples.native_preview.map(lambda value: value[field]).eq(examples[column]).all():
                raise ValueError('A preview disagrees with its original example identity')
        keys = ['subject_key', 'item_key', 'trial']
        rows = rows.merge(examples[keys + ['example_record', 'native_preview']], on=keys, how='left', validate='one_to_one')
        linked = rows.loc[rows.example_record.notna()]
        if not linked.example_record.map(lambda value: value['passed']).eq(linked.passed).all() or not linked.example_record.map(lambda value: value['cost']).eq(linked.cost).all():
            raise ValueError('A detailed example disagrees with its named panel verdict or cost')

        # 4. Preserve source evidence and both duration fields without filling missing traces.
        rows = rows.merge(items[['item_key', 'task_record']], on='item_key', validate='many_to_one')
        traces = rows[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_index=layout['index'], source_row=int(row.source_row),
            native_record=row.native_record, configuration_record=row.configuration_record, task_record=row.task_record,
            example_record=row.example_record if isinstance(row.example_record, dict) else None,
            native_preview=row.native_preview if isinstance(row.native_preview, dict) else None, scope=labels['trace_scope']),
            ensure_ascii=False, allow_nan=False) for row in rows.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': rows[['response_key', 'subject_key', 'item_key', 'trial', 'response']], 'traces': traces}


if __name__ == '__main__':
    TxBenchPP(__file__).main_from_args()
