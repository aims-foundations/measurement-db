#!/usr/bin/env python3
"""Tabulate Pxplore's recorded recommendations and every available criterion judgment."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Pxplore(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']
        raw = self.raw_dir / layout['release']

        # 1. Join each released judgment to its complete recorded recommendation.
        inputs, evaluations = [], []
        for path in sorted((raw / layout['evaluations']).glob('*.json')):
            records = json.loads(path.read_text())
            evaluation = pd.json_normalize(records, max_level=0).rename_axis('source_row').reset_index()
            evaluation['native_evaluation'] = records
            evaluations.append(evaluation.assign(source_run=path.stem))
            records = json.loads((raw / layout['inputs'] / path.name).read_text())
            source = pd.json_normalize(records, max_level=0).rename_axis('source_row').reset_index()
            source['native_input'] = records
            inputs.append(source.assign(source_run=path.stem).rename(columns={'course': 'input_course'}))
        calls = pd.concat(evaluations, ignore_index=True).merge(pd.concat(inputs, ignore_index=True),
            on=['source_run', 'source_row'], how='outer', validate='one_to_one', indicator=True)
        if not calls._merge.eq('both').all() or not all(
            a == b for left, right in parameters['associations'].items() for a, b in zip(calls[left], calls[right])):
            raise ValueError('Every judgment must match its recorded course, learner profile and recommendation')
        calls['call_key'] = calls.index
        calls['method'] = calls.source_run.str.split('_').str[0]
        calls['input_mode'] = calls.method.map(parameters['input_modes'])
        if calls.input_mode.isna().any():
            raise ValueError('Unknown released recommendation method')

        # 2. Expand criterion lists and retain both omitted and newly judged criteria.
        components = {}
        for state in ['initial_state', 'next_state']:
            frame = pd.json_normalize(calls[state], max_level=0).reindex(columns=list(parameters['dimensions']))
            frame = frame.assign(call_key=calls.call_key).melt(id_vars='call_key', var_name='dimension', value_name='component')
            frame = frame.explode('component').dropna(subset=['component'])
            frame['criterion'] = frame.component.map(lambda value: value['description']).str.split(
                parameters['parsing']['evidence_marker'], n=1, regex=False).str[0].str.strip()
            if frame.duplicated(['call_key', 'dimension', 'criterion']).any():
                raise ValueError('Ambiguous source criteria cannot be joined silently')
            components[state] = frame.rename(columns={'component': state + '_component'})
        responses = components['initial_state'].merge(components['next_state'],
            on=['call_key', 'dimension', 'criterion'], how='outer', validate='one_to_one', indicator='component_status')
        responses['component_status'] = responses.component_status.astype(str).map(parameters['component_status'])
        responses['native_grade'] = responses.next_state_component.map(lambda value: value.get('is_aligned') if isinstance(value, dict) else None)
        if not responses.native_grade.dropna().map(lambda value: type(value) is bool).all():
            raise ValueError('Alignment grades must be boolean or unavailable')
        responses['response'] = responses.native_grade.map({True: 1., False: 0.})
        responses = responses.merge(calls, on='call_key', how='left', validate='many_to_one')
        responses['response_key'] = responses.index

        # 3. Recover the documented history-only or structured-state model input.
        prompt = (raw / layout['system_prompt']).read_text()
        calls['strategy'] = [dict(interaction_history=[entry['role'] + ': ' + entry['content'] for entry in row.interaction_history])
            if row.input_mode == 'history' else row.student_profile for row in calls.itertuples()]
        calls['content'] = [json.dumps(dict(messages=[dict(role='system', content=prompt), dict(role='user',
            content=json.dumps(dict(recommendation_strategy=row.strategy, candidates=row.recommend_candidates),
                ensure_ascii=False, indent=2))]), ensure_ascii=False, allow_nan=False) for row in calls.itertuples()]
        items = responses.merge(calls[['call_key', 'content']], on='call_key', validate='many_to_one')
        items['criterion_sha256'] = items.criterion.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        items['item_key'] = items.response_key
        items['raw_item_id'] = 's' + items.source_row.astype(str) + ':' + items.input_mode + ':' + items.dimension + ':' + items.criterion_sha256
        items['features'] = items[['source_row', 'input_mode', 'dimension', 'criterion_sha256']].to_dict('records')
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + '\nDimension: ' + row.dimension + '\nCriterion: ' + row.criterion)
            for row in items.itertuples()]
        items['verifier'] = Judge(spec=json.dumps(dict(**self.grading['verifiers']['published'],
            system_prompt=(raw / layout['judge_prompt']).read_text()), ensure_ascii=False, sort_keys=True), judged_by='llm')

        # 4. Keep all literal run configurations separate without guessing checkpoints.
        subjects = calls[['source_run', 'method', 'input_mode']].drop_duplicates().copy()
        subjects['subject_key'] = subjects.source_run
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.source_run
        subjects['features'] = [dict(source_run=row.source_run, method=row.method, input_mode=row.input_mode,
            harness=parameters['labels']['harness']) for row in subjects.itertuples()]
        responses['subject_key'] = responses.source_run
        responses['item_key'] = responses.response_key
        responses['test_condition'] = 'method=' + responses.method + ';src=' + responses.source_run

        # 5. Preserve full source records, including added, omitted and ungraded criteria.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_run=row.source_run, source_row=int(row.source_row),
            source_file=layout['evaluations'] + '/' + row.source_run + '.json',
            input_file=layout['inputs'] + '/' + row.source_run + '.json', dimension=row.dimension, criterion=row.criterion,
            component_status=row.component_status, native_evaluation=row.native_evaluation, native_input=row.native_input),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    Pxplore(__file__).main_from_args()
