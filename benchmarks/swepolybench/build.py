"""Tabulate original SWE-PolyBench attempts and their complete source artifacts."""

import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SWEPolyBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, labels = self.build_parameters, self.build_parameters['labels']
        release = self.raw_dir / parameters['layout']['release']

        # 1. Read complete submission records and original agent predictions.
        metadata_paths = sorted(release.glob('evaluation/*/*/metadata.yaml'))
        submissions = pd.json_normalize([yaml.safe_load(path.read_text()) for path in metadata_paths], max_level=0)
        submissions['run'] = [str(path.parent.relative_to(release)) for path in metadata_paths]
        submissions['bank'] = submissions.run.str.split('/').str[1]
        frames = []
        for path in sorted(release.glob('evaluation/*/*/all_preds.jsonl')):
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            frame = pd.json_normalize(records, max_level=0)
            frame['prediction'] = records
            frame['run'], frame['prediction_file'] = str(path.parent.relative_to(release)), str(path.relative_to(self.raw_dir))
            frames.append(frame[['run', 'instance_id', 'prediction', 'prediction_file']])
        predictions = pd.concat(frames, ignore_index=True)
        if predictions.duplicated(['run', 'instance_id']).any():
            raise ValueError('Repeated prediction IDs require source review')

        # 2. Load verdicts separately from the evidence that a task was attempted.
        result_paths = sorted(release.glob('evaluation/*/*/logs/*_result.json'))
        records = [json.loads(path.read_text()) for path in result_paths]
        results = pd.json_normalize(records, max_level=0)
        results['result'] = records
        results['run'] = [str(path.parent.parent.relative_to(release)) for path in result_paths]
        results['result_file'] = [str(path.relative_to(self.raw_dir)) for path in result_paths]
        if results.duplicated(['run', 'instance_id']).any() or not results.resolved.map(lambda value: isinstance(value, bool)).all():
            raise ValueError('Native result IDs and explicit boolean verdicts require review')

        # 3. Associate complete trajectories with their explicit source task IDs.
        paths = sorted(path for path in release.glob('evaluation/*/*/trajs/**/*') if path.is_file())
        artifacts = pd.DataFrame({'source_file': [str(path.relative_to(self.raw_dir)) for path in paths]})
        artifacts['run'] = [str(path.relative_to(release)).split('/trajs/', 1)[0] for path in paths]
        artifacts['instance_id'] = artifacts.source_file.str.extract(parameters['patterns']['instance_id'], expand=False)
        if artifacts.instance_id.isna().any():
            raise ValueError('An original trajectory has no explicit task association')
        artifacts['content'] = [path.read_bytes().decode('utf-8') for path in paths]
        artifacts['artifact'] = artifacts[['source_file', 'content']].to_dict('records')
        trajectories = artifacts.groupby(['run', 'instance_id'], sort=False).artifact.agg(list).rename('artifacts').reset_index()
        metric_paths = sorted(release.glob('evaluation/*/*/logs/*_metrics.json'))
        metrics = pd.DataFrame({'run': [str(path.parent.parent.relative_to(release)) for path in metric_paths],
            'instance_id': [path.name.removesuffix('_metrics.json') for path in metric_paths],
            'retrieval_metrics': [dict(source_file=str(path.relative_to(self.raw_dir)), content=path.read_bytes().decode('utf-8'))
                for path in metric_paths]})

        # 4. Union actual attempt evidence; default-only result exports remain in raw.
        execution = results.loc[results[['generation', 'patch_applied', 'with_logs']].any(axis=1), ['run', 'instance_id']]
        attempts = pd.concat([predictions[['run', 'instance_id']], trajectories[['run', 'instance_id']], execution]).drop_duplicates()
        attempts = attempts.merge(predictions, on=['run', 'instance_id'], how='left', validate='one_to_one')
        attempts = attempts.merge(trajectories, on=['run', 'instance_id'], how='left', validate='one_to_one')
        attempts = attempts.merge(results[['run', 'instance_id', 'resolved', 'result', 'result_file']],
            on=['run', 'instance_id'], how='left', validate='one_to_one')
        attempts = attempts.merge(metrics, on=['run', 'instance_id'], how='left', validate='one_to_one')
        attempts = attempts.merge(submissions[['run', 'bank']], on='run', how='left', validate='many_to_one')
        attempts['response'] = attempts.resolved.map({True: 1.0, False: 0.0})
        attempts['response_key'] = attempts.run + '::' + attempts.instance_id
        attempts = attempts.sort_values(['run', 'instance_id']).reset_index(drop=True)

        # 5. Join the full or Verified task definition, retaining its grading environment.
        banks = pd.concat([pd.read_csv(self.raw_dir / filename, keep_default_na=False).assign(bank=bank,
            task_source_file=filename) for bank, filename in parameters['task_banks'].items()], ignore_index=True)
        if banks.duplicated(['bank', 'instance_id']).any():
            raise ValueError('Official task keys are not unique')
        attempts = attempts.merge(banks, on=['bank', 'instance_id'], how='left', validate='many_to_one')
        if attempts.problem_statement.isna().any():
            raise ValueError('An attempted task has no official input definition')

        # 6. Register each literal source configuration and grading-aware task.
        subjects = submissions.loc[submissions.run.isin(attempts.run), ['run', 'name']].rename(columns={'run': 'subject_key', 'name': 'raw_label'})
        subjects['features'] = [dict(source_submission=row.subject_key, harness=labels['harness'],
            configuration_scope=labels['configuration_scope'], source_readme=parameters['layout']['release'] + '/' + row.subject_key + '/README.md')
            for row in subjects.itertuples()]
        item_columns = ['bank', 'instance_id', 'repo', 'base_commit', 'language', 'problem_statement', 'patch', 'task_source_file']
        check_fields = self.grading['verifiers']['published_report']['task_fields']
        items = attempts[list(dict.fromkeys(item_columns + check_fields))].drop_duplicates(['bank', 'instance_id']).reset_index(drop=True)
        items['item_key'] = items.index
        items['raw_item_id'] = items.bank + '/' + items.instance_id
        items['content'] = items.problem_statement
        items['features'] = [dict(source_instance_id=row.instance_id, split=row.bank, repo=row.repo,
            base_commit=row.base_commit, language=row.language, task_definition_file=row.task_source_file,
            task_version_scope=labels['task_version_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row['patch'], rule=json.dumps(dict(rule=self.grading['rule'],
            checks={field: row[field] for field in check_fields}), sort_keys=True)) for row in items.to_dict('records')]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['published_report'], sort_keys=True)) for _ in items.index]
        attempts = attempts.merge(items[['bank', 'instance_id', 'item_key']], on=['bank', 'instance_id'], validate='many_to_one')

        # 7. Preserve complete predictions, verdicts, retrieval exports and trajectories.
        attempts = attempts.astype(object).where(attempts.notna(), None)
        traces = attempts[['response_key']].copy()
        records = [dict(kind='published_attempt_record', source_run=row.run, instance_id=row.instance_id,
                prediction_file=row.prediction_file, prediction=row.prediction, result_file=row.result_file, result=row.result,
                artifacts=row.artifacts if isinstance(row.artifacts, list) else [],
                retrieval_metrics=row.retrieval_metrics, task_source=row.task_source_file,
                protocol_scope=labels['task_version_scope']) for row in attempts.itertuples()]
        traces['trace'] = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records]
        attempts['subject_key'] = attempts.run
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    SWEPolyBench(__file__).main_from_args()
