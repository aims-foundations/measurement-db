"""Tabulate released OS-Harm executions and their directly associated safety judgments."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class OSHarm(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout = parameters['layout']
        grading = self.grading['verifiers']['native']

        # 1. Load original executions and directly adjacent native judgment files.
        records = pd.DataFrame([dict(source_file=str(path.relative_to(self.raw_dir)),
            execution_path=str(path.parent.relative_to(self.raw_dir)), record=json.loads(path.read_text()))
            for path in sorted(self.raw_dir.glob(layout['logs']))])
        judgments = pd.DataFrame([dict(execution_path=str(path.parents[4].relative_to(self.raw_dir)),
            judgment_file=str(path.relative_to(self.raw_dir)), judgment=json.loads(path.read_text()))
            for path in sorted(self.raw_dir.glob(layout['judgments']))])
        attempts = records.merge(judgments, on='execution_path', how='outer', validate='one_to_one', indicator=True)
        if not attempts['_merge'].eq('both').all():
            raise ValueError('Every selected execution must have its directly paired released judgment')
        attempts['task'] = attempts.record.map(lambda record: record['task'])
        attempts['configuration'] = attempts.record.map(lambda record: record['params'])
        # Preserve the incomplete source records in raw, without inventing their instructions.
        attempts = attempts.loc[attempts.task.map(lambda task: bool(task['instruction'].strip()))].copy()
        parts = attempts.execution_path.str.split('/')
        attempts['category'] = parts.str[1].map(parameters['categories'])
        attempts['application'], attempts['source_task'] = parts.str[-2], parts.str[-1]

        # 2. Keep literal model names and all recorded inference settings.
        attempts['subject_key'] = attempts.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = attempts[['subject_key', 'configuration']].drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.configuration.map(lambda configuration: configuration['model'])
        subjects['features'] = subjects.configuration.map(lambda configuration: dict(
            source_configuration=configuration, harness=parameters['labels']['harness']))

        # 3. Retain actual recorded task text and injection settings, including source discrepancies.
        # A folder's model label or jailbreak suffix cannot override the execution record.
        attempts['content'] = attempts.task.map(lambda task: json.dumps(task, ensure_ascii=False, sort_keys=True))
        attempts['task_key'] = [json.dumps([row.category, row.application, row.source_task, row.content])
            for row in attempts.itertuples()]
        items = attempts[['task_key', 'category', 'application', 'source_task', 'content']].drop_duplicates('task_key')
        metrics = pd.DataFrame([dict(assessment=name, rule=spec['rule'])
            for name, spec in grading['metrics'].items()])
        items = items.merge(metrics, how='cross')
        items['item_key'] = items.task_key + '#' + items.assessment
        items['raw_item_id'] = items.category + '/' + items.application + '/' + items.source_task + '#' + items.assessment
        items['features'] = [dict(category=row.category, application=row.application,
            source_task=row.source_task, assessment=row.assessment) for row in items.itertuples()]
        items['grading_criterion'] = items.rule.map(lambda rule: dict(rule=rule))
        items['verifier'] = [Judge(judge=grading['judge'], judged_by='llm',
            spec=json.dumps(dict(**grading['verifier'], assessment=name), sort_keys=True)) for name in items.assessment]

        # 4. Preserve full execution logs, captions, trajectory records and native verdicts.
        traces = []
        for row in attempts.itertuples():
            caption_path = Path(row.execution_path) / layout['caption']
            trajectory_path = Path(row.execution_path) / layout['trajectory']
            traces.append(json.dumps(dict(source_file=row.source_file, record=row.record,
                judgment_file=row.judgment_file, judgment=row.judgment,
                caption_file=str(caption_path), caption=json.loads((self.raw_dir / caption_path).read_text()),
                trajectory_file=str(trajectory_path), trajectory=[json.loads(line)
                    for line in (self.raw_dir / trajectory_path).read_text().splitlines() if line.strip()]),
                ensure_ascii=False, allow_nan=False))
        attempts['trace'] = traces

        # 5. Separate safety and task completion; neither assessment is substituted for the other.
        for metric in metrics.assessment:
            if not attempts.judgment.map(lambda value: type(value[metric]) is bool).all():
                raise ValueError('A released assessment is missing or is not a boolean')
            attempts[metric] = attempts.judgment.map(lambda value: float(value[metric]))
        responses = attempts.melt(id_vars=['execution_path', 'subject_key', 'task_key', 'trace'],
            value_vars=list(metrics.assessment), var_name='assessment', value_name='response')
        responses['item_key'] = responses.task_key + '#' + responses.assessment
        responses['response_key'] = responses.execution_path + '#' + responses.assessment
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response']],
            traces=responses[['response_key', 'trace']])


if __name__ == '__main__':
    OSHarm(__file__).main_from_args()
