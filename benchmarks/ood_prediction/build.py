#!/usr/bin/env python3
"""Tabulate published target-model labels without treating fold reuse as new trials."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class OODPrediction(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters['labels']

        # 1. Read native split/bucket tables, then explode their prompt lists.
        files = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            table = pd.read_json(path, convert_dates=False).rename_axis('label').reset_index()
            table = table.melt(id_vars='label', var_name='split', value_name='content').explode('content').dropna(subset='content')
            table['source_row'] = table.groupby(['split', 'label']).cumcount()
            table['source_file'] = str(path.relative_to(self.raw_dir))
            table['filename'] = path.stem
            files.append(table)
        occurrences = pd.concat(files, ignore_index=True)
        occurrences = occurrences.join(occurrences.filename.str.extract(parameters['layout']['filename_pattern']))
        if occurrences[['task', 'subject_key', 'setting']].isna().any().any():
            raise ValueError('Every source filename must identify its task, target model and setting')
        if not occurrences.label.isin(parameters['response_values']).all():
            raise ValueError('Every original result must belong to the correct or wrong bucket')

        # 2. Consolidate fold reuse while preserving every original source position.
        keys = ['task', 'subject_key', 'setting', 'content']
        if occurrences.groupby(keys).label.nunique().gt(1).any():
            raise ValueError('Repeated task/model/setting/prompt records have conflicting labels')
        occurrences['source'] = occurrences[['source_file', 'split', 'label', 'source_row']].to_dict('records')
        responses = occurrences.groupby(keys, sort=False).agg(label=('label', 'first'), sources=('source', list)).reset_index()
        responses = responses.rename_axis('response_key').reset_index()

        # 3. Keep target-model subjects and full stimuli under the documented task rules.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=model) for model in subjects.subject_key]
        items = responses.drop_duplicates(['task', 'content']).reset_index(drop=True).rename_axis('item_key').reset_index()
        items['raw_item_id'] = items.task + ':' + items.content.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        items['features'] = [dict(task=task, input_scope=labels['input_scope']) for task in items.task]
        items['grading_criterion'] = [dict(rule=self.grading['verifiers'][task]['criterion']) for task in items.task]
        items['verifier'] = [Judge(judged_by='human', spec=json.dumps(self.grading['verifiers'][task], sort_keys=True))
            if 'verification' in self.grading['verifiers'][task] else ExactMatcher(spec=json.dumps(self.grading['verifiers'][task], sort_keys=True))
            for task in items.task]

        # 4. Join canonical stimulus keys and encode the published binary labels.
        responses = responses.merge(items[['task', 'content', 'item_key']], on=['task', 'content'], validate='many_to_one')
        responses['response'] = responses.label.map(parameters['response_values']).astype(float)
        responses['test_condition'] = 'task=' + responses.task.map(parameters['task_labels']) + ';setting=' + responses.setting

        # 5. Trace label provenance only; the release does not contain generated answers.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(task=row.task, source_model_label=row.subject_key, setting=row.setting,
            prompt=row.content, label=row.label, occurrences=row.sources), ensure_ascii=False) for row in responses.itertuples()]
        return dict(subjects=subjects,
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    OODPrediction(__file__).main_from_args()
