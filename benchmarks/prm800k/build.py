#!/usr/bin/env python3
"""Tabulate original PRM800K candidates, human ratings and recorded solution prefixes."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class PRM800K(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Load the released annotation tables and retain their original record keys.
        inputs = []
        for path in sorted((self.raw_dir / parameters['layout']['data']).glob('*.jsonl')):
            frame = pd.read_json(path, lines=True, convert_dates=False, dtype=False, precise_float=True).rename_axis('source_row').reset_index()
            inputs.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), phase=path.stem.split('_')[0]))
        records = pd.concat(inputs, ignore_index=True).rename_axis('record_key').reset_index()
        records['annotation'] = records[['labeler', 'timestamp', 'generation', 'is_quality_control_question',
            'is_initial_screening_question']].astype(object).where(pd.notna(records), None).to_dict('records')
        records['label_metadata'] = records.label.map(lambda value: {key: item for key, item in value.items() if key != 'steps'})
        records['question_metadata'] = records.question.map(lambda value: {key: item for key, item in value.items() if key != 'pre_generated_steps'})
        records['reference_answer'] = records.question.map(lambda value: {key: value[key]
            for key in ['ground_truth_solution', 'ground_truth_answer'] if value.get(key) is not None})
        records['reference_answer'] = records.reference_answer.map(lambda value: json.dumps(value,
            ensure_ascii=False, allow_nan=False) if value else None)
        records = records.join(pd.json_normalize(records.question, max_level=0)).join(pd.json_normalize(records.label, max_level=0))

        # 2. Expand the annotation steps and reconstruct only the preceding context.
        steps = records[['record_key', 'steps']].explode('steps').dropna(subset='steps').rename(columns={'steps': 'native_step'})
        steps['step_index'] = steps.groupby('record_key').cumcount()
        steps = steps.merge(records[['record_key', 'phase']], on='record_key', validate='many_to_one')
        first_phase = steps[steps.phase.eq('phase1')].copy()
        first_phase['selected'] = first_phase.native_step.map(lambda step: [step['completions'][step['chosen_completion']]['text']
            if step['chosen_completion'] is not None else (step['human_completion']['text'] if step['human_completion'] is not None else None)])
        first_phase['prefix'] = first_phase.groupby('record_key').selected.transform(lambda rows: rows.cumsum()).map(lambda values: values[:-1])
        if first_phase.prefix.map(lambda values: None in values).any():
            raise ValueError('A phase-1 continuation has an unknown prior step')
        generated = records[['record_key', 'pre_generated_steps']].explode('pre_generated_steps').dropna(subset='pre_generated_steps')
        generated = generated.rename(columns={'pre_generated_steps': 'pregenerated_step'})
        generated['step_index'] = generated.groupby('record_key').cumcount()
        generated['selected'] = generated.pregenerated_step.map(lambda text: [text])
        generated['prefix'] = generated.groupby('record_key').selected.transform(lambda rows: rows.cumsum()).map(lambda values: values[:-1])
        contexts = pd.concat([first_phase[['record_key', 'step_index', 'prefix']],
            generated[['record_key', 'step_index', 'prefix', 'pregenerated_step']]], ignore_index=True)
        contexts = contexts.merge(steps[['record_key', 'step_index', 'native_step']],
            on=['record_key', 'step_index'], how='left', validate='one_to_one')
        contexts['item_key'] = contexts.index

        # 3. Keep all model candidates; add unmatched pre-generated steps with no grade.
        candidates = steps[['record_key', 'step_index']].assign(native_candidate=steps.native_step.map(lambda step: step['completions']))
        candidates = candidates.explode('native_candidate').dropna(subset='native_candidate').reset_index(drop=True)
        candidates['candidate_index'] = candidates.groupby(['record_key', 'step_index']).cumcount()
        candidates = candidates.join(pd.json_normalize(candidates.native_candidate, max_level=0))
        candidates = candidates.rename(columns={'rating': 'response'}).assign(origin='candidate')
        unrated = generated[['record_key', 'step_index', 'pregenerated_step']].rename(columns={'pregenerated_step': 'text'}).merge(
            candidates[['record_key', 'step_index', 'text']].drop_duplicates(), on=['record_key', 'step_index', 'text'],
            how='left', validate='one_to_one', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        unrated = unrated.assign(candidate_index=-1, origin='pregenerated_unrated', response=None, native_candidate=None)
        columns = ['record_key', 'step_index', 'candidate_index', 'origin', 'response', 'native_candidate', 'text']
        responses = pd.concat([candidates[columns], unrated[columns]], ignore_index=True).sort_values(
            ['record_key', 'step_index', 'candidate_index']).reset_index(drop=True)
        responses = responses.merge(contexts[['record_key', 'step_index', 'item_key', 'native_step', 'pregenerated_step']],
            on=['record_key', 'step_index'], validate='many_to_one').merge(records[['record_key', 'source_file', 'source_row',
                'annotation', 'label_metadata', 'question_metadata']], on='record_key', validate='many_to_one')
        responses['response_key'] = responses.index
        responses['subject_key'] = parameters['labels']['subject']
        responses['test_condition'] = responses.source_file + ':row=' + responses.source_row.astype(str)

        # 4. Project known stimuli and their human protocol without inventing a checkpoint.
        items = contexts.merge(records[['record_key', 'source_file', 'source_row', 'phase', 'problem',
            'reference_answer']], on='record_key', validate='many_to_one')
        items = items[items.item_key.isin(responses.item_key)].copy()
        items['raw_item_id'] = items.source_file + ':row=' + items.source_row.astype(str) + ':step=' + items.step_index.astype(str)
        items['content'] = [json.dumps(dict(problem=row.problem, prior_solution_steps=row.prefix), ensure_ascii=False)
            for row in items.itertuples()]
        items['features'] = items[['phase', 'step_index']].to_dict('records')
        items['grading_criterion'] = [dict(rule=self.grading['rule'], reference_answer=None if pd.isna(value) else value)
            for value in items.reference_answer]
        items['verifier'] = [Judge(spec=json.dumps(self.grading['verifiers'][phase], sort_keys=True),
            judge=parameters['labels']['judge'], judged_by='human') for phase in items.phase]
        subjects = pd.DataFrame([dict(subject_key=parameters['labels']['subject'], raw_label=parameters['labels']['subject'],
            features=dict(harness=parameters['labels']['harness']))])

        # 5. Preserve complete candidate records and their annotation provenance.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            step_index=row.step_index, candidate_index=row.candidate_index, origin=row.origin, text=row.text,
            annotation=row.annotation, label_metadata=row.label_metadata, question_metadata=row.question_metadata,
            native_step=row.native_step if isinstance(row.native_step, dict) else None,
            native_candidate=row.native_candidate if isinstance(row.native_candidate, dict) else None,
            pregenerated_step=row.pregenerated_step if isinstance(row.pregenerated_step, str) else None),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    PRM800K(__file__).main_from_args()
