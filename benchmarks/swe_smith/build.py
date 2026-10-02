"""Tabulate published SWE-smith run labels and their complete released conversations."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SweSmith(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']
        grading = self.grading['verifiers']['published']

        # 1. Load the published run tables and task bank, retaining original row locations.
        paths = sorted(self.raw_dir.glob(layout['trajectories']))
        runs = pd.concat([pd.read_parquet(path, columns=[
            'messages', 'instance_id', 'resolved', 'model', 'traj_id']).assign(
                source_file=str(path.relative_to(self.raw_dir)), source_row=lambda frame: range(len(frame)))
            for path in paths], ignore_index=True)
        tasks = pd.concat([pd.read_parquet(path) for path in sorted(self.raw_dir.glob(layout['tasks']))], ignore_index=True)
        runs['source_key'] = runs.source_file + '#' + runs.source_row.astype(str)
        runs['rendering'] = runs.source_file.str.extract(layout['rendering_pattern'], expand=False)
        if not pd.api.types.is_bool_dtype(runs.resolved) or runs.traj_id.isna().any() or runs.traj_id.eq('').any():
            raise ValueError('Native runs require a Boolean verdict and a trajectory ID')
        if runs.groupby('traj_id')[['instance_id', 'model', 'resolved']].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Copies of a trajectory disagree about its task, model or verdict')

        # 2. Recover each task's actual instruction from its first published user message.
        messages = runs[['source_key', 'messages']].assign(messages=runs.messages.map(json.loads)).explode('messages').reset_index(drop=True)
        messages = messages[['source_key']].join(pd.json_normalize(messages.messages, max_level=0))
        instructions = messages.loc[messages.role.eq('user')].drop_duplicates('source_key').copy()
        instructions['text'] = instructions.content.map(lambda value: value if isinstance(value, str)
            else '\n'.join(part['text'] for part in value if isinstance(part.get('text'), str)))
        instructions['content'] = instructions.text.str.extract(layout['instruction_pattern'], expand=False).fillna(instructions.text).str.strip()
        runs = runs.merge(instructions[['source_key', 'content']], on='source_key', how='left', validate='one_to_one')
        if runs.content.isna().any() or runs.content.eq('').any():
            raise ValueError('Every run needs its released task instruction')
        if runs.groupby('traj_id').content.nunique().gt(1).any():
            raise ValueError('Copies of a trajectory contain different task instructions')
        runs['item_key'] = [json.dumps([row.instance_id, row.content]) for row in runs.itertuples()]

        # 3. Keep literal model names and include the published task-specific grading information.
        subjects = runs[['model']].drop_duplicates().rename(columns={'model': 'raw_label'})
        subjects['subject_key'] = subjects.raw_label
        subjects['features'] = [dict(harness=self.build_parameters['labels']['harness']) for _ in range(len(subjects))]
        items = runs[['item_key', 'instance_id', 'content']].drop_duplicates('item_key')
        tasks['published_task'] = json.loads(tasks.to_json(orient='records'))
        items = items.merge(tasks[['instance_id', 'published_task']], on='instance_id', how='left', validate='many_to_one')
        items['published_task'] = items.published_task.map(lambda value: value if isinstance(value, dict) else None)
        items['features'] = [dict(upstream_instance_id=value) for value in items.instance_id]
        items['grading_criterion'] = [dict(rule=grading['rule']) for _ in range(len(items))]
        items['verifier'] = items.published_task.map(lambda task: ExactMatcher(spec=json.dumps(
            dict(**grading['spec'], published_task=task), sort_keys=True, allow_nan=False)))
        items = items.rename(columns={'instance_id': 'raw_item_id'})

        # 4. A provider trajectory ID contributes one observation across its alternate renderings.
        responses = runs.drop_duplicates('traj_id')[['traj_id', 'model', 'item_key', 'resolved']].copy()
        responses = responses.rename(columns={'traj_id': 'response_key', 'model': 'subject_key', 'resolved': 'response'})
        responses['response'] = responses.response.astype(float)

        # 5. Preserve every released conversation; the known misaligned patch column stays in raw.
        runs['source_record'] = runs[['source_file', 'source_row', 'rendering', 'instance_id', 'model', 'resolved', 'messages']].to_dict('records')
        traces = runs.groupby('traj_id', sort=False).source_record.agg(list).reset_index()
        traces['trace'] = [json.dumps(dict(traj_id=row.traj_id, renderings=row.source_record),
            ensure_ascii=True, allow_nan=False) for row in traces.itertuples()]
        traces = traces.rename(columns={'traj_id': 'response_key'})
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response']],
            traces=traces[['response_key', 'trace']])


if __name__ == '__main__':
    SweSmith(__file__).main_from_args()
