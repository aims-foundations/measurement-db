"""Tabulate released scientific tasks and their original endpoint assessments."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ScBenchLong(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout, labels = self.build_parameters['layout'], self.build_parameters['labels']

        # 1. Read the original run index and the four released task definitions.
        index = json.loads((self.raw_dir / layout['index']).read_bytes())
        rows = pd.json_normalize(index['runResults'], max_level=0)
        rows['native_record'] = index['runResults']
        published = pd.json_normalize(index['evals'], max_level=0).rename(columns={'id': 'item_key'})
        records = []
        for path in sorted((self.raw_dir / layout['tasks']).glob('*/eval.json')):
            record = json.loads(path.read_bytes())
            records.append(dict(item_key=record['id'], content=record['task'], data_nodes=record['data_node'],
                vocabulary_path=path.with_name('vocabulary.json')))
        items = pd.DataFrame(records).merge(published, on='item_key', validate='one_to_one')
        if not items.content.eq(items.prompt).all():
            raise ValueError('The released task instructions differ between the original sources')

        # 2. Keep observations with released prompts; all other original rows remain in raw/.
        rows = rows.loc[rows.task.isin(items.item_key)].copy()
        if rows[['modelId', 'task', 'trialIndex']].duplicated().any():
            raise ValueError('The native index repeats an observation identity')
        if not rows.passed.map(lambda value: isinstance(value, bool)).all():
            raise ValueError('A recorded endpoint verdict is not a Boolean')
        rows['response_key'] = rows[['modelId', 'task', 'trialIndex']].apply(lambda row: json.dumps(row.to_list()), axis=1)
        rows['response'] = rows.passed.astype(float)
        rows['item_key'] = rows.task
        rows['subject_key'] = rows.modelId
        rows['trial'] = rows.trialIndex
        subjects = rows[['modelId', 'modelName', 'harness']].drop_duplicates()
        if subjects.modelId.duplicated().any():
            raise ValueError('A native configuration ID has conflicting model or harness labels')
        subjects = subjects.rename(columns={'modelId': 'subject_key', 'modelName': 'raw_label'})
        subjects['features'] = [dict(harness=row.harness, recorded_configuration=row.subject_key,
            historical_settings=labels['historical_settings']) for row in subjects.itertuples()]

        # 3. Preserve the published grading checks separately from the task and input vocabulary.
        rules = items.set_index('item_key').expectedAnswer.map(lambda value: json.dumps(value, sort_keys=True))
        if not rows.expectedAnswer.map(lambda value: json.dumps(value, sort_keys=True)).eq(rows.item_key.map(rules)).all():
            raise ValueError('A native run uses a different grading definition from the released task')
        items['raw_item_id'] = items.item_key
        items['grading_criterion'] = items.item_key.map(rules).map(lambda rule: dict(rule=rule))
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['reported'], sort_keys=True))] * len(items)
        items['features'] = [dict(input_scope=labels['input_scope'], source_data_nodes=json.dumps(nodes)) for nodes in items.data_nodes]
        items['attachments'] = [[dict(source_path=path, path='vocabulary.json', media_type='application/json', role='input')]
                                for path in items.vocabulary_path]

        # 4. Join each explicitly linked preview by its source filename and verify its recorded identity.
        previews = []
        for path in sorted((self.raw_dir / layout['previews']).glob('*.json')):
            record = json.loads(path.read_bytes())
            previews.append(dict(trajectoryFile=layout['preview_source_prefix'] + path.name,
                preview_source=path.relative_to(self.raw_dir).as_posix(), native_preview=record))
        rows = rows.merge(pd.DataFrame(previews), on='trajectoryFile', how='left', validate='many_to_one')
        linked = rows.loc[rows.trajectoryFile.notna()]
        if linked.native_preview.isna().any():
            raise ValueError('An explicitly linked native preview was not captured')
        for field, column in [('exampleId', 'task'), ('model', 'modelName'), ('harness', 'harness')]:
            if not linked.native_preview.map(lambda record: record[field]).eq(linked[column]).all():
                raise ValueError('A native preview disagrees with its index identity')
        traces = rows[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_index=layout['index'], native_record=row.native_record,
            preview_source=row.preview_source if isinstance(row.preview_source, str) else None,
            native_preview=row.native_preview if isinstance(row.native_preview, dict) else None),
            ensure_ascii=False, allow_nan=False) for row in rows.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': rows[['response_key', 'subject_key', 'item_key', 'trial', 'response']], 'traces': traces}


if __name__ == '__main__':
    ScBenchLong(__file__).main_from_args()
