"""Tabulate original Reasoning Gym completions without replacing them by averages."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ReasoningGym(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        root = self.raw_dir / parameters['layout']['results']

        # 1. Read run configurations; keep the documented identity conflict raw-only.
        run_paths = sorted(root.glob('*/*/summary.json'))
        runs = pd.json_normalize([json.loads(path.read_text()) for path in run_paths], max_level=0)
        runs['subject_key'] = [str(path.parent.relative_to(root)) for path in run_paths]
        excluded = runs.subject_key.eq(parameters['exclusions']['ambiguous_identity_run'])
        if excluded.sum() != 1:
            raise ValueError('The declared unresolved run no longer matches the original release')
        runs = runs.loc[~excluded].copy()
        names = runs.subject_key.str.split('/').str[-1].str.replace(r'_\d{8}_\d{6}$', '', regex=True)
        if not names.eq(runs.model.str.replace('/', '_', regex=False)).all():
            raise ValueError('Run folder and recorded model identity disagree')
        runs['condition'] = [json.dumps(row, sort_keys=True) for row in
            runs[['subject_key', *parameters['condition_fields']]].to_dict('records')]
        settings = runs.set_index('subject_key').to_dict('index')

        # 2. Flatten question records and join their original run-level settings.
        frames = []
        for run, summary in settings.items():
            for path in sorted((root / run).glob('*/*.json')):
                data = json.loads(path.read_text())
                if (data['name'] != path.stem or data['category'] != path.parent.name
                        or data['system_prompt'] != summary['system_prompt']
                        or data['completions_per_prompt'] != summary['completions_per_prompt']):
                    raise ValueError('Dataset path or generation settings disagree with the native record')
                frame = pd.json_normalize(data['results'], max_level=0)
                if len(frame) != data['total_examples'] or not frame.completions.map(
                        lambda value: isinstance(value, list) and bool(value)).all():
                    raise ValueError('Every question needs its recorded completions and matching count')
                frames.append(frame.assign(subject_key=run, source_file=str(path.relative_to(self.raw_dir)),
                    source_row=frame.index, dataset=data['name'], category=data['category'],
                    difficulty=run.split('/')[0], generator_config=json.dumps(data['config'], sort_keys=True)))
        records = pd.concat(frames, ignore_index=True)
        records = records.merge(runs[['subject_key', 'system_prompt', 'git_hash', 'condition']],
                                on='subject_key', how='left', validate='many_to_one')
        if not records.question.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Every item must retain its complete original question')
        records['item_key'] = records.source_file + ':' + records.source_row.astype(str)

        # 3. Define subjects and items with the complete stimulus and grading identity.
        subjects = runs[['subject_key', 'model']].rename(columns={'model': 'raw_label'})
        subjects['features'] = runs[list(parameters['subject_fields'])].astype(str).assign(
            source_model=runs.model, harness=parameters['labels']['harness']).to_dict('records')
        stimulus = records[['system_prompt', 'question']].to_dict('records')
        criteria = records[['expected_answer', 'dataset', 'generator_config', 'git_hash']].to_dict('records')
        items = records[['item_key']].assign(raw_item_id=records.item_key,
            content=[json.dumps(row, ensure_ascii=False, sort_keys=True) for row in stimulus],
            grading_criterion=[dict(reference_answer=str(row['expected_answer']), rule=self.grading['rule']) for row in criteria],
            verifier=[ExactMatcher(spec=json.dumps({**self.grading['verifiers']['algorithmic'],
                'dataset': row['dataset'], 'configuration': json.loads(row['generator_config']),
                'revision': row['git_hash']}, sort_keys=True)) for row in criteria],
            features=records[['dataset', 'category', 'difficulty']].to_dict('records'))

        # 4. Expand individual completions, preserving fractional scores and exceptions.
        expanded = records.explode('completions', ignore_index=True)
        completions = pd.json_normalize(expanded.completions, max_level=0).reindex(
            columns=['model_answer', 'full_model_response', 'score', 'error'])
        numeric = pd.to_numeric(completions.score, errors='raise')
        if not np.isfinite(numeric).all() or numeric.lt(0).any() or numeric.gt(np.nextafter(1., np.inf)).any():
            raise ValueError('A native reward is non-finite or exceeds its declared scale beyond boundary roundoff')
        scores = numeric.clip(upper=1).astype(object)
        failed_grading = completions.error.notna()
        scores.loc[failed_grading] = None
        expanded['completion_index'] = expanded.groupby('item_key', sort=False).cumcount()
        expanded['response_key'] = expanded.item_key + ':' + expanded.completion_index.astype(str)
        responses = expanded[['response_key', 'subject_key', 'item_key']].assign(response=scores,
            test_condition=expanded.condition)

        # 5. Preserve every original completion and its source position, without clipping text.
        evidence = expanded[['source_file', 'source_row', 'completion_index', 'completions',
                             'mean_score', 'best_score']].rename(columns={'completions': 'original_completion'})
        traces = expanded[['response_key']].assign(
            trace=[json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                   for row in evidence.to_dict('records')])
        return dict(subjects=subjects, items=items, responses=responses, traces=traces)


if __name__ == '__main__':
    ReasoningGym(__file__).main_from_args()
