#!/usr/bin/env python3
"""Join TabArena's original test rows and predictions without running its models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TabArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        fold, repeat = int(parameters['paths']['fold']), int(parameters['paths']['repeat'])
        methods = self.raw_dir / parameters['paths']['methods']
        inputs = self.raw_dir / parameters['paths']['inputs']

        # 1. Select the published default configuration and its recorded settings.
        configurations, task_tables = [], []
        for method in parameters['methods']:
            root = methods / method
            default = method + parameters['paths']['config_suffix']
            settings = json.loads((root / 'configs_hyperparameters.json').read_text())[default]
            configurations.append(pd.read_parquet(root / 'configs.parquet').assign(
                method=method, default_config=default, configuration=json.dumps(settings, sort_keys=True)))
            task_tables.append(pd.read_parquet(root / 'task_metadata.parquet'))
        configurations = pd.concat(configurations, ignore_index=True)
        configurations = configurations.loc[
            configurations.framework.eq(configurations.default_config) & configurations.fold.eq(fold)].copy()
        tasks = pd.concat(task_tables, ignore_index=True)[['dataset', 'tid', 'did']].drop_duplicates()
        if tasks.dataset.duplicated().any() or configurations.duplicated(['method', 'dataset']).any():
            raise ValueError('Conflicting dataset IDs or duplicate default configurations')
        configurations = configurations.merge(tasks, on=['dataset', 'tid'], validate='many_to_one')
        configurations['subject_key'] = configurations.method + '/' + configurations.tid.astype(str)

        # 2. Select original test rows; keep every feature and its original precision.
        item_tables = []
        for task in tasks.itertuples(index=False):
            specification = json.loads((inputs / f'data-{task.did}.json').read_text())['data_set_description']
            declaration = json.loads((inputs / f'task-{task.tid}.json').read_text())['task']
            source_data = next(value['data_set'] for value in declaration['input'] if value['name'] == 'source_data')
            target = specification['default_target_attribute']
            if int(source_data['data_set_id']) != task.did or source_data['target_feature'] != target:
                raise ValueError('OpenML task and dataset target differ')
            data = pd.read_parquet(inputs / f'data-{task.did}.parquet')
            splits = pd.DataFrame(arff.loadarff(inputs / f'splits-{task.tid}.arff')[0])
            indices = splits.loc[splits.type.eq(b'TEST') & splits.fold.eq(fold) & splits['repeat'].eq(repeat), 'rowid'].astype('int64')
            if indices.duplicated().any() or indices.empty:
                raise ValueError('Duplicate or absent test row indices')
            selected = data.iloc[indices].reset_index(drop=True)
            excluded = [target]
            for field in ['ignore_attribute', 'row_id_attribute']:
                if specification.get(field):
                    excluded.extend(specification[field].split(','))
            features = selected.drop(columns=excluded).astype(object)
            features = features.where(features.notna(), None)
            content = [json.dumps(dict(dataset=task.dataset, features=row), ensure_ascii=False, allow_nan=False)
                       for row in features.to_dict('records')]
            item_tables.append(pd.DataFrame(dict(dataset=task.dataset, tid=task.tid, did=task.did,
                rowid=indices.to_numpy(), source_position=range(len(indices)), content=content,
                original_target=selected[target].to_numpy(), target=target,
                data_license=specification.get('licence'), data_citation=specification.get('citation'),
                source_url=specification.get('original_data_url'))))
        source_items = pd.concat(item_tables, ignore_index=True)

        # 3. Read native arrays as tables and join by the explicit original row index.
        prediction_tables = []
        for configuration in configurations.itertuples(index=False):
            root = methods / configuration.method / 'model_predictions' / configuration.dataset / str(fold)
            native = json.loads((root / 'metadata.json').read_text())
            if native['dataset'] != configuration.dataset or native['fold'] != fold or native['dtype'] != 'float32':
                raise ValueError('Prediction metadata differs from its dataset/fold')
            if native['models'].count(configuration.framework) != 1:
                raise ValueError('Default model does not identify exactly one native prediction array')
            predictions = np.memmap(root / 'pred-test.dat', dtype=native['dtype'], mode='r',
                shape=tuple(native['pred_test_shape']))[native['models'].index(configuration.framework)]
            labels = pd.read_csv(root / 'label-test.csv.zip', float_precision='round_trip').rename(columns={'Unnamed: 0': 'rowid'})
            target = source_items.loc[source_items.dataset.eq(configuration.dataset), 'target'].iloc[0]
            if labels.columns.tolist() != ['rowid', target] or len(labels) != len(predictions) or not np.isfinite(predictions).all():
                raise ValueError('Native labels/predictions have inconsistent columns, length or numeric values')
            labels = labels.rename(columns={target: 'released_target'}).assign(
                dataset=configuration.dataset, subject_key=configuration.subject_key,
                source_position=range(len(labels)), problem_type=configuration.problem_type,
                prediction=predictions.tolist())
            prediction_tables.append(labels)
        observations = pd.concat(prediction_tables, ignore_index=True).merge(
            source_items[['dataset', 'rowid', 'source_position', 'original_target']],
            on=['dataset', 'rowid', 'source_position'], how='left', validate='many_to_one', indicator=True)
        if not observations['_merge'].eq('both').all() or observations.released_target.isna().any():
            raise ValueError('Prediction row order or source target association is incomplete')
        observations = observations.drop(columns='_merge')
        category = observations.problem_type.ne(parameters['labels']['regression'])
        encoding = observations.loc[category, ['dataset', 'original_target', 'released_target']].drop_duplicates()
        if encoding.duplicated(['dataset', 'original_target']).any() or encoding.duplicated(['dataset', 'released_target']).any():
            raise ValueError('Released class codes do not establish a one-to-one category mapping')
        if not np.allclose(observations.loc[~category, 'original_target'].astype(float),
                           observations.loc[~category, 'released_target'].astype(float), rtol=1e-13, atol=1e-13):
            raise ValueError('Regression targets differ from the original test rows')

        # 4. Derive per-row grades; preserve the native prediction in the trace.
        observations['response'] = np.nan
        binary = observations.problem_type.eq(parameters['labels']['binary'])
        multiclass = observations.problem_type.eq(parameters['labels']['multiclass'])
        regression = observations.problem_type.eq(parameters['labels']['regression'])
        if not (binary | multiclass | regression).all():
            raise ValueError('Unknown source problem type')
        predicted_binary = observations.loc[binary, 'prediction'].astype(float).gt(0.5).astype(int)
        predicted_multiclass = observations.loc[multiclass, 'prediction'].map(np.argmax)
        observations.loc[binary, 'response'] = predicted_binary.eq(observations.loc[binary, 'released_target']).astype(float)
        observations.loc[multiclass, 'response'] = predicted_multiclass.eq(observations.loc[multiclass, 'released_target']).astype(float)
        observations.loc[regression, 'response'] = (
            observations.loc[regression, 'prediction'].astype(float) - observations.loc[regression, 'released_target'].astype(float)) ** 2
        observations['item_key'] = observations.dataset + '/row' + observations.rowid.astype(str)
        observations['response_key'] = observations.subject_key + '/' + observations.item_key
        observations['test_condition'] = f'fold={fold};repeat={repeat}'
        observations['trial'] = 1
        references = observations[['dataset', 'rowid', 'item_key', 'released_target', 'problem_type']].drop_duplicates()
        if references.duplicated(['dataset', 'rowid']).any():
            raise ValueError('Methods disagree about a test-row reference or problem type')
        items = source_items.merge(references, on=['dataset', 'rowid'], validate='one_to_one')
        items['raw_item_id'] = items.item_key
        grading_kind = items.problem_type.map(lambda kind: 'regression' if kind == parameters['labels']['regression'] else 'classification')
        items['grading_criterion'] = [dict(
            reference_answer=json.dumps(dict(original_target=original, released_target=released), ensure_ascii=False, allow_nan=False),
            rule=self.grading['verifiers'][kind]['rule'], response_scale=self.grading['verifiers'][kind]['response_scale'])
            for kind, original, released in zip(grading_kind, items.original_target, items.released_target)]
        items['verifier'] = grading_kind.map(lambda kind: ExactMatcher(spec=json.dumps(self.grading['verifiers'][kind], sort_keys=True)))
        items['features'] = [dict(dataset=row.dataset, openml_task_id=int(row.tid), openml_data_id=int(row.did),
            original_row_index=int(row.rowid), target_name=row.target, problem_type=row.problem_type,
            data_provenance=json.dumps(dict(license=row.data_license, citation=row.data_citation, source_url=row.source_url),
                ensure_ascii=False, sort_keys=True).replace(';', r'\u003b').replace('=', r'\u003d'))
            for row in items.itertuples(index=False)]

        # 5. Keep fitted configurations distinct and retain full native outputs.
        subjects = configurations.copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.method + ' / ' + subjects.dataset
        subjects['features'] = [dict(method=row.method, training_dataset=row.dataset, openml_task_id=int(row.tid),
            default_config=row.framework, outer_fold=fold, outer_repeat=repeat,
            recorded_configuration=row.configuration.replace(';', r'\u003b').replace('=', r'\u003d'),
            configuration_scope=parameters['labels']['configuration_scope']) for row in subjects.itertuples(index=False)]
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(dataset=row.dataset, original_row_index=int(row.rowid),
            source_position=int(row.source_position), prediction=row.prediction, released_target=row.released_target,
            problem_type=row.problem_type), ensure_ascii=False, allow_nan=False) for row in observations.itertuples(index=False)]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=observations[['response_key', 'subject_key', 'item_key', 'response', 'trial', 'test_condition']],
            traces=traces)


if __name__ == '__main__':
    TabArena(__file__).main_from_args()
