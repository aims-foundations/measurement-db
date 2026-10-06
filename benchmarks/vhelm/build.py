"""Tabulate original VHELM requests, visual inputs and release-declared grades."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class VHELM(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters['paths']
        labels = self.build_parameters['labels']
        perturbation_fields = self.build_parameters['identity']['perturbation_fields'].split(',')
        profiles = self.grading['verifiers']

        # 1. Load the original release schema, configurations, requests and grades.
        with gzip.open(self.raw_dir / paths['schema'], 'rt') as stream:
            schema = json.load(stream)
        groups = pd.json_normalize(schema['run_groups']).reindex(columns=['name', 'environment.main_name']).dropna()
        headline = groups.set_index('name')['environment.main_name'].to_dict()
        run_rows, request_frames, grade_frames = [], [], []
        for path in sorted(self.raw_dir.glob(paths['runs'])):
            with gzip.open(path, 'rt') as stream:
                specification = json.load(stream)
            scenario = specification['name'].split(':')[0]
            if scenario in self.build_parameters['raw_only_scenarios']:
                continue
            metrics = {headline[group] for group in specification['groups'] if group in headline}
            if len(metrics) != 1 or not metrics <= profiles.keys():
                raise ValueError('Each VHELM run must have one declared headline metric')
            run_key = str(path.parent.relative_to(self.raw_dir))
            run_rows.append(dict(run_key=run_key, scenario=scenario, metric=metrics.pop(), specification=specification))
            with gzip.open(path.with_name('scenario_state.json.gz'), 'rt') as stream:
                records = json.load(stream)['request_states']
            frame = pd.json_normalize(records, max_level=0)
            request_frames.append(frame.assign(run_key=run_key, source_row=frame.index + 1, native_record=records))
            with gzip.open(path.with_name('per_instance_stats.json.gz'), 'rt') as stream:
                records = json.load(stream)
            grade_frames.append(pd.json_normalize(records, max_level=0).assign(run_key=run_key))
        runs = pd.DataFrame(run_rows)
        if runs.empty:
            raise ValueError('No supported original VHELM runs were captured')
        requests = pd.concat(request_frames, ignore_index=True)
        grades = pd.concat(grade_frames, ignore_index=True)

        # 2. Match each assessment to its actual input, including its perturbation.
        instances = pd.json_normalize(requests.instance.tolist(), max_level=0)
        requests['instance_id'] = instances.id
        for frame, values in [(requests, instances), (grades, grades)]:
            perturbations = values.get('perturbation', pd.Series(None, index=values.index, dtype=object))
            frame['perturbation_key'] = [json.dumps(
                {key: value[key] for key in perturbation_fields if key in value} if isinstance(value, dict) else None,
                sort_keys=True) for value in perturbations]
        keys = ['run_key', 'instance_id', 'train_trial_index', 'perturbation_key']
        if requests.duplicated(keys).any():
            raise ValueError('Duplicate native VHELM request identities')
        requests = requests.merge(runs, on='run_key', validate='many_to_one')
        grades = grades.explode('stats', ignore_index=True)
        grades['metric_record'] = grades.stats
        grades['metric'] = grades.stats.map(lambda value: value['name']['name'] if isinstance(value, dict) else None)
        grades['response'] = grades.stats.map(lambda value: value.get('mean') if isinstance(value, dict) else None)
        grades = grades.merge(runs[['run_key', 'metric']], on=['run_key', 'metric'], validate='many_to_one')
        if grades.groupby(keys, dropna=False).response.nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting native grades for the same VHELM request')
        grade_records = grades.groupby(keys, sort=False).metric_record.agg(list).rename('metric_records').reset_index()
        grades = grades.drop_duplicates(keys)[keys + ['response']].merge(grade_records, on=keys, validate='one_to_one')
        requests = requests.merge(grades, on=keys, how='outer', validate='one_to_one', indicator=True)
        if requests._merge.eq('right_only').any():
            raise ValueError('A native VHELM grade has no recorded request')
        requests = requests.drop(columns='_merge')
        requests['response'] = pd.to_numeric(requests.response, errors='raise').astype('Float64')
        requests['source_key'] = requests.run_key + '/scenario_state.json.gz:' + requests.source_row.astype(str)

        # 3. Consolidate proven cache copies while retaining all original locations.
        requests['cache_digest'] = [hashlib.sha256(json.dumps([
            row.scenario, row.metric, {**row.native_record, 'result': {
                key: value for key, value in row.native_record['result'].items() if key != 'cached'}}
        ], sort_keys=True, allow_nan=False).encode()).hexdigest() for row in requests.itertuples()]
        requests['cached'] = requests.result.map(lambda value: bool(value.get('cached')))
        has_cached_copy = requests.groupby('cache_digest', sort=False).cached.transform('any')
        requests['response_key'] = requests.source_key
        requests.loc[has_cached_copy, 'response_key'] = (
            'cache:' + requests.cache_digest[has_cached_copy] + ':' + requests.response[has_cached_copy].astype('string').fillna('ungraded'))
        requests['source_alias'] = requests[['run_key', 'source_row', 'cached']].to_dict('records')
        aliases = requests.groupby('response_key', sort=False).source_alias.agg(list).rename('source_aliases').reset_index()
        records = requests.drop_duplicates('response_key').merge(aliases, on='response_key', validate='one_to_one')

        # 4. Describe the actual model configuration separately from the stimulus.
        records['configuration'] = [json.dumps({key: value for key, value in request.items()
            if key not in {'prompt', 'multimodal_prompt'}}, sort_keys=True, allow_nan=False) for request in records.request]
        records['subject_key'] = records.configuration
        subjects = records[['subject_key', 'configuration']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.configuration.map(lambda value: json.loads(value)['model'])
        subjects['features'] = [dict(harness=labels['harness'], release=labels['release'],
            generation_settings=value.replace(';', r'\u003b').replace('=', r'\u003d'))
            for value in subjects.configuration]
        items = records[['response_key', 'instance_id', 'instance', 'request', 'scenario', 'metric']].copy()
        items['item_key'] = items.response_key
        items['raw_item_id'] = items.scenario + '/' + items.instance_id.astype(str)
        items['content'] = [json.dumps(dict(text=request['prompt'],
            multimedia_elements=request['multimodal_prompt']['media_objects']), ensure_ascii=False, allow_nan=False)
            for request in items.request]
        items['features'] = [dict(scenario=row.scenario,
            perturbation=json.dumps(row.instance.get('perturbation'), sort_keys=True)
                .replace(';', r'\u003b').replace('=', r'\u003d')) for row in items.itertuples()]

        # 5. Attach unchanged images and retain the metric-specific reference/scale.
        media = [obj for request in items.request for obj in request['multimodal_prompt']['media_objects'] if obj.get('location')]
        locations = sorted({obj['location'] for obj in media})
        if any(not location.startswith(paths['asset_prefix']) for location in locations):
            raise ValueError('An image has no declared original source folder')
        assets = {location: (self.raw_dir / paths['asset_folder'] / re.sub(
            r'[^A-Za-z0-9._/-]', lambda match: f'_x{ord(match[0]):02x}_',
            location.removeprefix(paths['asset_prefix']))).read_bytes() for location in locations}
        items['attachments'] = [[dict(data=assets[obj['location']], path=obj['location'],
            media_type=obj['content_type'], role='input') for obj in request['multimodal_prompt']['media_objects']
            if obj.get('location')] for request in items.request]
        references = [[reference['output']['text'] for reference in row.instance['references']
                       if 'correct' in reference.get('tags', [])]
                      if profiles[row.metric]['reference_mode'] == 'correct_tagged'
                      else ([row.instance['references'][0]['output']['text']]
                            if profiles[row.metric]['reference_mode'] == 'first' else [])
                      for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=json.dumps(reference, ensure_ascii=False) if reference else None,
            rule=profiles[metric]['criterion'], response_scale=profiles[metric]['response_scale'])
            for reference, metric in zip(references, items.metric)]
        items['verifier'] = [(ExactMatcher if profiles[metric]['verifier_class'] == 'exact_matcher' else Judge)(
            spec=json.dumps(profiles[metric], sort_keys=True)) for metric in items.metric]

        # 6. Keep full native request/result traces, grades and source aliases.
        responses = records.assign(item_key=records.response_key,
            test_condition='release=' + labels['release'] + ';scenario=' + records.scenario + ';metric=' + records.metric)[
            ['response_key', 'subject_key', 'item_key', 'response', 'test_condition']]
        traces = records[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(native_record=row.native_record,
            metric=row.metric, metric_records=row.metric_records if isinstance(row.metric_records, list) else [],
            source_aliases=row.source_aliases), sort_keys=True, ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            responses=responses, traces=traces)


if __name__ == '__main__':
    VHELM(__file__).main_from_args()
