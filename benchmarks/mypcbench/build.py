"""Tabulate the task versions and rubric judgments in the official showcase."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MyPCBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Load the complete published showcase, including its historical task definitions.
        paths = sorted(self.raw_dir.glob(layout['trajectories']))
        records = [json.loads(path.read_text()) for path in paths]
        runs = pd.json_normalize(records, max_level=0)
        runs['source_file'] = [str(path.relative_to(self.raw_dir)) for path in paths]
        index = pd.read_json(self.raw_dir / layout['index'])['trajectories']
        declared = pd.json_normalize(index, max_level=0)
        if runs.slug.duplicated().any() or set(runs.slug) != set(declared.slug):
            raise ValueError('The source index and recorded showcase do not match exactly')
        signatures = pd.read_json(self.raw_dir / layout['signatures'])['signatures']
        if not set(pd.json_normalize(signatures).slug).issubset(set(runs.slug)):
            raise ValueError('A family signature names an uncaptured trajectory')

        # 2. Expand native rubric judgments, without substituting newer task definitions.
        outcomes = runs[['slug', 'model_key', 'task_id', 'legacy_id', 'instruction',
                         'category', 'apps', 'source_file', 'rubrics']].explode('rubrics').reset_index(drop=True)
        outcomes = pd.concat([outcomes.drop(columns='rubrics'),
                              pd.json_normalize(outcomes.rubrics, max_level=0)], axis=1)
        outcomes['response'] = pd.to_numeric(outcomes.score, errors='raise').astype(float)
        if not outcomes.response.dropna().isin([0, 1]).all():
            raise ValueError('A recorded rubric score is not binary')
        graded = outcomes.response.notna()
        if not outcomes.loc[graded, 'pass'].eq(outcomes.loc[graded, 'response'].eq(1)).all():
            raise ValueError('The published score and pass flag disagree')
        outcomes['response_key'] = outcomes.slug + '::' + outcomes.id
        outcomes['item_key'] = outcomes.task_id + '::' + outcomes.id
        if outcomes.response_key.duplicated().any() or outcomes.item_key.duplicated().any():
            raise ValueError('The pinned showcase repeats a run or task rubric')

        # 3. Keep requirements and rubric weights in grading, separate from agent instructions.
        items = outcomes.copy()
        items['raw_item_id'] = items.item_key
        items['content'] = items.instruction
        items['features'] = [dict(task_id=row.task_id, legacy_task_id=row.legacy_id,
            rubric_id=row.id, task_category=row.category, task_apps=json.dumps(row.apps),
            definition_scope=labels['definition_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=json.dumps(dict(requirement=row.requirement,
            rubric_id=row.id, weight=row.weight, protocol=grading['rule']), ensure_ascii=False, allow_nan=False))
            for row in items.itertuples()]
        items['verifier'] = [Judge(spec=json.dumps(grading['verifiers']['rubric'], sort_keys=True), judged_by='llm')
                             for _ in items.index]

        # 4. Identify the released actor configuration by its model key, preserving explicit ablations.
        subjects = runs[['model_key', 'model']].drop_duplicates().copy()
        if subjects.model_key.duplicated().any():
            raise ValueError('One recorded model key has conflicting display labels')
        subjects['subject_key'] = subjects.model_key
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(**parameters['subject_features'], source_model_key=row.model_key,
            source_model_label=row.model, declared_tool_variant=labels['cua_only']
            if row.model_key.endswith(labels['cua_suffix']) else labels['unspecified_variant'])
            for row in subjects.itertuples()]

        # 5. Preserve every released trace field and pin its locally captured screenshot bytes.
        steps = runs[['slug', 'steps']].explode('steps').reset_index(drop=True)
        steps = pd.concat([steps[['slug']], pd.json_normalize(steps.steps, max_level=0)], axis=1)
        steps['raw_path'] = labels['site_prefix'] + steps.image
        steps['sha256'] = steps.raw_path.map(lambda value: hashlib.sha256((self.raw_dir / value).read_bytes()).hexdigest())
        steps['bytes'] = steps.raw_path.map(lambda value: (self.raw_dir / value).stat().st_size)
        steps['capture'] = steps[['i', 'image', 'raw_path', 'sha256', 'bytes']].to_dict('records')
        screenshots = steps.groupby('slug', sort=False).capture.agg(list)
        native = dict(zip(runs.slug, records))
        traces = outcomes[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, selected_rubric_id=row.id,
            trajectory=native[row.slug], screenshots=screenshots[row.slug]), ensure_ascii=False, allow_nan=False)
            for row in outcomes.itertuples()]
        outcomes['subject_key'] = outcomes.model_key
        outcomes['test_condition'] = None
        outcomes.loc[outcomes.model_key.str.endswith(labels['cua_suffix']), 'test_condition'] = labels['cua_condition']
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
                'responses': outcomes[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    MyPCBench(__file__).main_from_args()
