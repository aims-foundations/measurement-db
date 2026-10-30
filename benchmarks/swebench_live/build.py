"""Tabulate original SWE-bench-Live attempts, verdicts and complete rollouts."""

import json
import hashlib
from pathlib import Path
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables
from measurement_db.benchmarks.swebench_live.test_reports import named_checks


class SWEbenchLive(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, labels = self.build_parameters, self.build_parameters['labels']
        release = self.raw_dir / parameters['layout']['release']

        # 1. Flatten both native summary formats into explicit per-instance flags.
        report_paths = sorted([*release.rglob('results.json'), *release.rglob('result.json')])
        reports = pd.json_normalize([json.loads(path.read_text()) for path in report_paths], max_level=0)
        reports['run'] = [str(path.parent.relative_to(release)) for path in report_paths]
        reports['report_file'] = [str(path.relative_to(self.raw_dir)) for path in report_paths]
        flag_columns = [name for name in reports if name.endswith('_ids')]
        reports['report_context'] = [dict((key, value) for key, value in json.loads(path.read_text()).items()
            if key not in flag_columns) for path in report_paths]
        flags = reports.melt(id_vars=['run'], value_vars=flag_columns, var_name='flag', value_name='instance_id')
        flags = flags.explode('instance_id').dropna(subset=['instance_id'])
        flags = flags.groupby(['run', 'instance_id'], sort=False).flag.agg(list).rename('summary_flags').reset_index()

        # 2. Load native predictions, preserving their complete original records.
        frames, prediction_paths = [], sorted(release.rglob('preds.json'))
        for path in prediction_paths:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                frame = pd.Series(payload, name='prediction').rename_axis('instance_id').reset_index()
            else:
                frame = pd.json_normalize(payload, max_level=0)[['instance_id']].assign(prediction=payload)
            frame['run'] = str(path.parent.relative_to(release))
            frame['prediction_file'] = str(path.relative_to(self.raw_dir))
            frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        predictions['prediction_source_key'] = predictions.instance_id
        windows = predictions.run.str.startswith('submissions/windows/win-agent/')
        predictions.loc[windows, 'instance_id'] = predictions.loc[windows, 'instance_id'].replace(parameters['prediction_key_aliases'])
        if predictions.duplicated(['run', 'instance_id']).any():
            raise ValueError('Duplicate native prediction keys require source review')

        # 3. Associate complete rollout files by submission and their explicit task IDs.
        native_runs = sorted(set(reports.run) | set(predictions.run))
        run_pattern = '^(' + '|'.join(re.escape(run) for run in sorted(native_runs, key=len, reverse=True)) + ')/'
        files = pd.DataFrame({'source_file': sorted(str(path.relative_to(release))
            for path in release.rglob('*') if path.is_file())})
        files['run'] = files.source_file.str.extract(run_pattern, expand=False)
        files = files.loc[files.run.notna()].copy()
        files['relative_file'] = files.source_file.str.replace(run_pattern, '', regex=True)
        files['instance_id'] = files.relative_file.str.extract(parameters['patterns']['instance_id'], expand=False)
        files = files.loc[files.instance_id.notna()].copy()
        files['pattern'] = files.relative_file.str.replace(parameters['patterns']['instance_id'], '{instance_id}', regex=True)
        files['role'] = files.pattern.map(parameters['artifact_roles'])
        if files.role.isna().any():
            raise ValueError('An unreviewed source artifact layout needs classification')
        test_reports = files.loc[files.role.eq('evaluation_artifact') &
            files.relative_file.str.endswith(('/report.json', '.report.json', '.results.json'))].copy()
        test_reports['payload'] = test_reports.source_file.map(lambda name: json.loads((release / name).read_text()))
        files = files.loc[files.role.ne('evaluation_artifact')].copy()
        files['content'] = files.source_file.map(lambda name: (release / name).read_bytes().decode('utf-8'))
        files['source_file'] = parameters['layout']['release'] + '/' + files.source_file
        files['artifact'] = files[['source_file', 'role', 'content']].to_dict('records')
        artifacts = files.groupby(['run', 'instance_id'], sort=False).artifact.agg(list).rename('artifacts').reset_index()

        # 4. Unroll item-level website lists, retaining overlapping exports as aliases.
        website = pd.read_json(self.raw_dir / parameters['layout']['reports'], lines=True, convert_dates=False)
        website['source_row'] = website.index
        website = website.loc[website.resolved.map(lambda value: isinstance(value, list))].copy()
        website['website_key'] = website['name'] + '|' + website.date.astype(str)
        website['run'] = website.website_key.map(parameters['website_aliases']).fillna('website/' + website.website_key)
        web_flags = website.melt(id_vars=['run', 'source_row', 'name', 'set', 'total', 'date'],
            value_vars=['resolved', 'applied', 'located'], var_name='flag', value_name='instance_id')
        web_flags = web_flags.explode('instance_id').dropna(subset=['instance_id'])
        web_flags['website_entry'] = [dict(source_file=parameters['layout']['reports'], source_row=int(row.source_row),
            flag=row.flag, name=row.name, subset=row.set, total=int(row.total), date=str(row.date))
            for row in web_flags.itertuples()]
        web = web_flags.groupby(['run', 'instance_id'], sort=False).agg(
            website_flags=('flag', list), website_entries=('website_entry', list)).reset_index()

        # 5. Join evidence for each recorded attempt; missing verdicts remain null.
        attempts = flags.merge(predictions, on=['run', 'instance_id'], how='outer', validate='one_to_one')
        attempts = attempts.merge(artifacts, on=['run', 'instance_id'], how='outer', validate='one_to_one')
        attempts = attempts.merge(web, on=['run', 'instance_id'], how='outer', validate='one_to_one')
        attempts = attempts.merge(reports[['run', 'report_file', 'report_context', 'related_upstream_evaluator_fix']],
            on='run', how='left', validate='many_to_one')
        for column in ['summary_flags', 'website_flags', 'website_entries', 'artifacts']:
            attempts[column] = attempts[column].map(lambda value: value if isinstance(value, list) else [])
        succeeded = attempts.summary_flags.map(lambda flags: bool(set(flags) & {'success_ids', 'resolved_ids'}))
        failed = attempts.summary_flags.map(lambda flags: bool(set(flags) & {'failure_ids', 'unresolved_ids'}))
        web_success = attempts.website_flags.map(lambda flags: 'resolved' in flags)
        if ((succeeded | web_success) & failed).any():
            raise ValueError('Conflicting final verdicts require source review')
        attempts['response'] = (succeeded | web_success).astype(float).where(succeeded | failed | web_success)
        attempts['track'] = attempts.run.str.split('/').str[1].map(parameters['tracks']).fillna('python')
        attempts['evaluator_fix'] = attempts.related_upstream_evaluator_fix.fillna('')
        attempts = attempts.sort_values(['run', 'instance_id']).reset_index(drop=True)
        attempts['response_key'] = attempts.run + '::' + attempts.instance_id
        attempts['subject_key'] = attempts.run

        # 6. Join complete official task definitions, using fallbacks only for absent IDs.
        frames = []
        for filename, track in {**parameters['bank_files'], **parameters['fallback_bank_files']}.items():
            records = pq.read_table(self.raw_dir / filename).to_pylist()
            frame = pd.DataFrame.from_records(records)
            frame['task_record'] = [json.loads(json.dumps(record, default=lambda value: value.isoformat())) for record in records]
            frame['task_source_file'], frame['task_source_row'], frame['track'] = filename, frame.index, track
            frame['fallback'] = filename in parameters['fallback_bank_files']
            frames.append(frame)
        banks = pd.concat(frames, ignore_index=True)
        primary = banks.loc[~banks.fallback]
        fallback = banks.loc[banks.fallback].merge(primary[['track', 'instance_id']],
            on=['track', 'instance_id'], how='left', indicator=True, validate='many_to_one')
        banks = pd.concat([primary, fallback.loc[fallback._merge.eq('left_only')].drop(columns='_merge')], ignore_index=True)
        if banks.duplicated(['track', 'instance_id']).any():
            raise ValueError('Ambiguous official task definitions require source review')
        attempts = attempts.merge(banks[['track', 'instance_id', 'task_record', 'task_source_file', 'task_source_row']],
            on=['track', 'instance_id'], how='left', validate='many_to_one')
        if attempts.task_record.isna().any():
            raise ValueError('A recorded attempt has no official task definition')

        # 7. Register source configurations and task definitions with their grading protocol.
        subjects = attempts[['subject_key', 'run']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.run.str.removeprefix('submissions/').str.removeprefix('website/')
        subjects['features'] = [dict(source_submission=row.run,
            harness=parameters['subject_harnesses'].get(row.raw_label, labels['harness']),
            reasoning_effort=parameters['subject_reasoning_efforts'].get(row.raw_label),
            configuration_scope=labels['configuration_scope'],
            source_readme=parameters['layout']['release'] + '/' + row.run + '/README.md'
                if (release / row.run / 'README.md').is_file() else None)
            for row in subjects.itertuples()]
        items = attempts[['track', 'instance_id', 'evaluator_fix', 'task_record', 'task_source_file', 'task_source_row']].drop_duplicates(
            ['track', 'instance_id', 'evaluator_fix']).reset_index(drop=True)
        items['item_key'] = items.index
        items['raw_item_id'] = items.track + '/' + items.instance_id
        items['testlet_id'] = items.raw_item_id
        items['content'] = items.task_record.map(lambda row: row['problem_statement'])
        items['features'] = [dict(track=row.track, source_instance_id=row.instance_id, repo=row.task_record['repo'],
            base_commit=row.task_record['base_commit'], task_definition_file=row.task_source_file,
            task_definition_row=int(row.task_source_row), task_version_scope=labels['task_version_scope'])
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=record['patch'], rule=json.dumps(dict(
            rule=self.grading['rule'], checks={key: record[key] for key in self.grading['verifiers']['published_report']['task_fields']
                if key in record}), sort_keys=True))
            for record in items.task_record]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(self.grading['verifiers']['published_report'],
            recorded_evaluator_fix=value or None), sort_keys=True)) for value in items.evaluator_fix]
        items['attachments'] = [[dict(path='test-environment.json', role='grading_reference',
            media_type='application/json', data=json.dumps({field: record[field]
                for field in self.grading['verifiers']['published_report']['task_fields'] if field in record},
                sort_keys=True).encode())] for record in items.task_record]
        attempts = attempts.merge(items[['track', 'instance_id', 'evaluator_fix', 'item_key']],
            on=['track', 'instance_id', 'evaluator_fix'], validate='many_to_one')

        # 8. Preserve original predictions and rollout text with explicit source associations.
        attempts = attempts.astype(object).where(attempts.notna(), None)
        # Attach reports only to independently established attempts. Multiple files
        # can be aliases; contradictory named verdicts remain explicitly ungraded.
        by_run, report_evidence = {}, {}
        for report in test_reports.itertuples():
            key = report.run, report.instance_id
            combined = by_run.setdefault(key, {})
            for check, grade in named_checks(report.payload, report.instance_id).items():
                combined[check] = None if check in combined and combined[check] != grade else grade
            report_evidence.setdefault(key, []).append(dict(source_file=report.source_file, record=report.payload))
        definitions, outcomes, seen = [], [], set()
        item_bank = items.set_index('item_key').to_dict('index')
        for row in attempts.itertuples():
            for (group, name), grade in sorted(by_run.get((row.run, row.instance_id), {}).items()):
                component = hashlib.sha256(json.dumps([group, name]).encode()).hexdigest()
                if (row.item_key, component) not in seen:
                    seen.add((row.item_key, component))
                    original = item_bank[row.item_key]
                    definitions.append(dict(parent_item_key=row.item_key, component_id=component,
                        grading_criterion=dict(rule=json.dumps(dict(test_name=name, test_group=group,
                            rule='Released named test verdict; missing checks are not inferred',
                            test_environment_asset='test-environment.json', repo=original['task_record']['repo'],
                            base_commit=original['task_record']['base_commit']), sort_keys=True)),
                        verifier=original['verifier']))
                outcomes.append(dict(parent_response_key=row.response_key, component_id=component, response=grade))
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(kind='published_attempt_record', source_run=row.run, instance_id=row.instance_id,
            summary_file=row.report_file, summary_context=row.report_context, summary_flags=row.summary_flags,
            prediction_file=row.prediction_file, prediction_source_key=row.prediction_source_key,
            prediction=row.prediction, website_entries=row.website_entries,
            artifacts=row.artifacts, test_reports=report_evidence.get((row.run, row.instance_id), []),
            task_source=dict(file=row.task_source_file, row=int(row.task_source_row)),
            log_scope=labels['log_scope']), ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        tables = {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'testlet_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}
        return component_tables(tables, pd.DataFrame(definitions), pd.DataFrame(outcomes))


if __name__ == '__main__':
    SWEbenchLive(__file__).main_from_args()
