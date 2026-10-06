"""Tabulate Image2Struct's original multimodal requests and published metrics."""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Image2Struct(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        metrics = self.grading['verifiers']

        # 1. Read the indexed run files into tables, retaining each native record.
        index = pd.json_normalize(json.loads((self.raw_dir / 'run_specs.json').read_text()), max_level=0)
        runs, parts = [], {name: [] for name in parameters['record_files']}
        for path in sorted((self.raw_dir / 'runs').glob('*/run_spec.json.gz')):
            with gzip.open(path, 'rt') as stream:
                spec = json.load(stream)
            run_key = str(path.parent.relative_to(self.raw_dir))
            runs.append(dict(run_key=run_key, run_spec=spec, name=spec['name']))
            for name, filename in parameters['record_files'].items():
                with gzip.open(path.with_name(filename), 'rt') as stream:
                    records = json.load(stream)
                frame = pd.json_normalize(records, max_level=0)
                parts[name].append(frame.assign(run_key=run_key, **{name + '_record': records}))
        runs = pd.DataFrame(runs)
        if sorted(runs.name) != sorted(index.name):
            raise ValueError('Captured runs do not match the original release index')
        predictions, requests, instances = (pd.concat(parts[name], ignore_index=True)
            for name in ['prediction', 'request', 'instance'])
        instances = instances.rename(columns={'id': 'instance_id'})

        # 2. Join each prediction to its own request, item and run configuration.
        keys = ['run_key', 'instance_id', 'train_trial_index']
        attempts = predictions.merge(requests[keys + ['request', 'request_record']], on=keys,
            how='left', validate='one_to_one').merge(
            instances[['run_key', 'instance_id', 'instance_record', 'references']],
            on=['run_key', 'instance_id'], how='left', validate='many_to_one').merge(
            runs, on='run_key', how='left', validate='many_to_one')
        if attempts[['request', 'instance_record', 'run_spec']].isna().any().any():
            raise ValueError('A published prediction lacks its original request or task definition')
        attempts['domain'] = attempts.name.str.split(':').str[0]
        attempts = attempts.loc[attempts.domain.isin(parameters['included_domains'])].copy()
        attempts['content'] = attempts.request.map(lambda request: json.dumps(
            {key: request[key] for key in parameters['stimulus_fields']}, ensure_ascii=False, sort_keys=True))
        if attempts.content.str.contains('[redacted]', regex=False).any():
            raise ValueError('A selected task has an intentionally redacted stimulus')
        attempts['request_settings'] = attempts.request.map(lambda request: json.dumps(
            {key: value for key, value in request.items() if key not in parameters['stimulus_fields']}, sort_keys=True))

        # 3. Keep recorded model configurations distinct and unpivot the score fields.
        subjects = attempts[['request_settings']].drop_duplicates().reset_index(drop=True)
        subjects['subject_key'] = subjects.index
        subjects['raw_label'] = subjects.request_settings.map(lambda value: json.loads(value)['model'])
        subjects['features'] = subjects.request_settings.map(lambda value: dict(
            **parameters['subject_features'], request_settings=value))
        attempts = attempts.merge(subjects[['request_settings', 'subject_key']],
            on='request_settings', how='left', validate='many_to_one')
        scores = pd.json_normalize(attempts.stats).reindex(columns=metrics)
        attempts = attempts.reset_index(drop=True).join(scores)
        responses = attempts.melt(id_vars=[column for column in attempts if column not in metrics],
            value_vars=list(metrics), var_name='metric', value_name='response')
        # An unconfigured metric is not an attempted, ungraded observation.
        present = [metric in stats for metric, stats in zip(responses.metric, responses.stats)]
        responses = responses.loc[present].reset_index(drop=True)
        responses['response_key'] = responses.index
        responses['metric_spec'] = responses.run_spec.map(lambda value: value['metric_specs'][0])
        responses['references_text'] = responses.references.map(lambda refs: json.dumps(
            [ref['output']['text'] for ref in refs if 'correct' in ref['tags'] and ref['output'].get('text')],
            ensure_ascii=False))
        responses['item_key'] = [json.dumps([row.content, row.references_text, row.metric,
            row.metric_spec, row.run_spec['annotators']], sort_keys=True) for row in responses.itertuples()]

        # 4. Attach original image bytes and make the grading dimension explicit.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.name.str.split(',model=').str[0] + '/' + items.instance_id + '/' + items.metric
        items['attachments'] = items.request.map(lambda request: [dict(source_path=medium['location'],
            path=medium['location'], media_type=medium['content_type'], role='input')
            for medium in request['multimodal_prompt']['media_objects'] if medium.get('location')])
        items['features'] = [dict(domain=row.domain,
            scenario=json.dumps(row.run_spec['scenario_spec'], sort_keys=True)) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + ' Metric: ' + metrics[row.metric]['description'],
            response_scale=metrics[row.metric]['response_scale'],
            **(dict(reference_answer=row.references_text) if row.references_text != '[]' else {}))
            for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(metrics[row.metric],
            metric=row.metric, metric_spec=row.metric_spec, annotators=row.run_spec['annotators']), sort_keys=True))
            for row in items.itertuples()]

        # 5. Link every grade to its complete request, prediction and task record.
        responses['test_condition'] = [json.dumps(dict(source_run=row.name, instance_id=row.instance_id,
            train_trial_index=row.train_trial_index, metric=row.metric), sort_keys=True)
            for row in responses.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_run=row.name, source_directory=row.run_key,
            request=row.request_record, prediction=row.prediction_record, instance=row.instance_record),
            ensure_ascii=False, sort_keys=True, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    Image2Struct(__file__).main_from_args()
