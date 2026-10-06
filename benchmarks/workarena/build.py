"""Tabulate published WorkArena trajectories and their original human annotations."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class WorkArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']
        grading = self.grading['verifiers']['human']

        # 1. Read the original trajectory files and the complete annotation table.
        paths = sorted(self.raw_dir.glob(layout['trajectories']))
        attempts = pd.DataFrame(dict(source_file=[str(path.relative_to(self.raw_dir)) for path in paths],
            task_id=[path.stem for path in paths], record=[json.loads(path.read_text()) for path in paths]))
        attempts = attempts.join(pd.json_normalize(attempts.record, max_level=0))
        annotations = pd.read_csv(self.raw_dir / layout['annotations'], keep_default_na=False)
        annotations['annotation_row'] = annotations.index
        annotations = annotations.loc[annotations.benchmark.eq(layout['benchmark_filter'])].copy()
        annotations['annotation'] = annotations.drop(columns='annotation_row').to_dict('records')

        # 2. Retain the recorded agent, model settings, flags and package versions.
        attempts['configuration'] = attempts[['agent', 'model', 'model_args', 'flags', 'package_version']].to_dict('records')
        attempts['subject_key'] = attempts.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = attempts[['subject_key', 'model', 'configuration']].drop_duplicates('subject_key').copy()
        # JSON escapes preserve separators inside the schema's k=v feature encoding.
        subjects['features'] = subjects.configuration.map(lambda value: dict(
            source_configuration=json.dumps(value, sort_keys=True).replace(';', r'\u003b').replace('=', r'\u003d'),
            harness=self.build_parameters['labels']['harness']))
        subjects = subjects.rename(columns={'model': 'raw_label'})

        # 3. Use actual task instructions and the recorded seed, not task-name placeholders.
        attempts['item_key'] = [json.dumps([row.task_id, row.seed, row.goal]) for row in attempts.itertuples()]
        items = attempts[['item_key', 'task_id', 'seed', 'goal']].drop_duplicates('item_key').copy()
        items['features'] = items[['task_id', 'seed']].rename(columns={'seed': 'task_seed'}).to_dict('records')
        items['grading_criterion'] = [dict(rule=grading['rule']) for _ in range(len(items))]
        items['verifier'] = [Judge(judge=grading['judge'], judged_by='human',
            spec=json.dumps(grading['spec'], sort_keys=True)) for _ in range(len(items))]
        items = items.rename(columns={'task_id': 'raw_item_id', 'goal': 'content'})

        # 4. Match every rating to its exact agent, experiment and task; retain repeated ratings.
        responses = annotations.merge(attempts, left_on=['model_name', 'exp_name', 'task_id'],
            right_on=['agent', 'experiment', 'task_id'], how='outer', validate='many_to_one', indicator=True)
        if not responses['_merge'].eq('both').all():
            raise ValueError('Every annotation and selected trajectory must have an exact counterpart')
        responses['response'] = responses.trajectory_success.map(grading['values'])
        if responses.response.isna().any():
            raise ValueError('An annotation has an unsupported success label')
        responses['response_key'] = layout['annotations'] + '#' + responses.annotation_row.astype(str)

        # 5. Keep complete native trajectories and annotations, including automatic rewards and errors.
        responses['trace'] = [json.dumps(dict(source_file=row.source_file, record=row.record,
            annotation_file=layout['annotations'], annotation_row=int(row.annotation_row), annotation=row.annotation),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response']],
            traces=responses[['response_key', 'trace']])


if __name__ == '__main__':
    WorkArena(__file__).main_from_args()
