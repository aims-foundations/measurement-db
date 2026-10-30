"""Verify rubric observations while preserving ambiguous finance-run associations."""

from collections import Counter, defaultdict
import json

import pandas as pd
import pyarrow.parquet as pq


def verify(directory, root, metadata):
    from .native_result_audits import _bigfinance, _bigfinance_source_records, _check
    from ..build_measurement_tables.response_scales import canonical_response_scale
    native_source = _bigfinance_source_records(directory, metadata)
    tasks, runs, grades, _, _, _ = native_source
    source_runs = {r['source_file'] + ':' + str(r['source_row']): r for group in runs.values() for r in group}
    expected_assessments, ambiguous = defaultdict(list), defaultdict(list)
    expected, assessment_lookup = Counter(), {}
    for key, source in grades.items():
        record = source['record']
        matches = [r for r in runs[key[:3]] if r['record']['final_answer'] == record['final_answer']]
        identifiers = [r['source_file'] + ':' + str(r['source_row']) for r in matches]
        unique = len(matches) == 1
        identity = source['source_file'] + ':' + str(source['source_row'])
        assessment_lookup[identity] = source
        for run in identifiers:
            (expected_assessments if unique else ambiguous)[run].append(source)
        checks = [('final_answer_correct', record['final_answer_correct'])]
        checks += [('rubric:' + str(i), line['earned']) for i, line in enumerate(record['rubric_lines'])]
        for name, grade in checks:
            expected[identity, key[1] + '::' + name, float(grade), identifiers[0] if unique else None] += 1
    attempts, subjects, seen, linked, saved_runs, saved_tasks = {}, {}, set(), {}, {}, {}
    for batch in pq.ParquetFile(root / 'attempts.parquet').iter_batches(batch_size=16):
        for row in batch.to_pylist():
            run = row['raw_attempt_id']
            assert run in source_runs and run not in seen
            seen.add(run)
            original_run = source_runs[run]
            evidence = json.loads(row['trace'])
            saved_runs[run] = {k: evidence[k] for k in ('source_file', 'source_row', 'record')}
            assert saved_runs[run] == original_run
            question = original_run['record']['question_id']
            assert evidence['task_record'] == tasks[question]
            saved_tasks[question] = evidence['task_record']
            ordered = lambda records: sorted(records, key=lambda r: (r['source_file'], r['source_row']))
            assert ordered(evidence['assessments']) == ordered(expected_assessments[run])
            assert ordered(evidence['ambiguous_assessments']) == ordered(ambiguous[run])
            assert subjects.setdefault(original_run['record']['model'], row['subject_id']) == row['subject_id']
            attempts[row['attempt_id']] = run, row['subject_id']
            for assessment in evidence['assessments']:
                key = row['attempt_id'], assessment['record']['judge']
                assert key not in linked
                linked[key] = assessment['source_file'] + ':' + str(assessment['source_row'])
    assert seen == set(source_runs), 'Missing or extra physical executions'
    items = pd.read_parquet(root / 'items.parquet').set_index('item_id').to_dict('index')
    traces = pd.read_parquet(root / 'traces.parquet').set_index('response_id').trace.to_dict()
    actual = Counter()
    prototypes, original_rows = {}, {}
    for response in pd.read_parquet(root / 'responses.parquet').itertuples():
        item = items[response.item_id]
        judge = json.loads(item['verifier'])['judge']
        if pd.isna(response.attempt_id):
            trace = json.loads(traces[response.response_id])
            source = trace['assessment']
            identity = source['source_file'] + ':' + str(source['source_row'])
            assert source == assessment_lookup[identity]
            assert trace['trace_association'] == 'ambiguous_final_answer_match'
            assert len(trace['matching_run_records']) > 1
            run = None
        else:
            run, subject = attempts[response.attempt_id]
            assert subject == response.subject_id
            identity = linked[response.attempt_id, judge]
        record = assessment_lookup[identity]['record']
        if identity in prototypes:
            for field in ['content', 'item_features', 'asset_manifest', 'verifier', 'testlet_id']:
                _check(None if pd.isna(item[field]) else item[field],
                    None if pd.isna(prototypes[identity][field]) else prototypes[identity][field],
                    'BigFinance common complete task ' + field)
        prototypes[identity] = item
        original_rows[identity] = response._asdict()
        assert response.subject_id == subjects[record['model']]
        assert response.trial == record['trial_idx'] + 1 and judge == record['judge']
        assert item['testlet_id'] == record['question_id']
        criterion = json.loads(item['grading_criterion'])
        assert criterion['reference_answer'] == record['reference_answer']
        assert response.test_condition == 'judge=' + judge and pd.isna(response.interactors)
        component = item['raw_item_id'].split('::', 1)[1]
        if component.startswith('rubric:'):
            line = record['rubric_lines'][int(component.split(':')[1])]
            rule = json.loads(criterion['rule'])
            assert rule == dict(metric='rubric_line', criterion=line['text'], points=line['points'],
                rule='Recorded earned flag for this rubric line; points are its original aggregation weight')
        else:
            assert json.loads(criterion['rule']) == dict(metric='final_answer_correct',
                rule=metadata['grading']['verifiers']['measures']['final_answer_correct']['rule'])
        actual[identity, item['raw_item_id'], float(response.response), run] += 1
    assert actual == expected, 'Finance rubric observations or attempt links differ from native evidence'
    parents, rows, restored_traces = {}, [], []
    grading = metadata['grading']['verifiers']
    for identity, item in prototypes.items():
        assessment = assessment_lookup[identity]
        record = assessment['record']
        key = record['model'], record['question_id'], record['trial_idx']
        original_runs = [saved_runs[r['source_file'] + ':' + str(r['source_row'])] for r in runs[key]]
        matches = [i for i, r in enumerate(original_runs) if r['record']['final_answer'] == record['final_answer']]
        task = saved_tasks[key[1]]
        for metric in ['final_answer_correct', 'rubric_fraction']:
            rule = dict(metric=metric, rubric=task['rubric'], interpretation=grading['measures'][metric]['rule'])
            criterion = dict(reference_answer=task['reference_answer'], rule=json.dumps(rule, ensure_ascii=False, sort_keys=True),
                response_scale=json.loads(canonical_response_scale(grading['measures'][metric]['scale'])))
            parent_id = json.dumps([item['content'], item['verifier'], criterion], sort_keys=True)
            parent = dict(item, item_id=parent_id, raw_item_id=key[1], grading_criterion=json.dumps(criterion))
            if parent_id in parents:
                _check(parent, parents[parent_id], 'BigFinance full original task definition')
            parents[parent_id] = parent
            response = dict(original_rows[identity])
            response.pop('Index')
            value = float(record['final_answer_correct']) if metric == 'final_answer_correct' else record['rubric_points_earned']/record['rubric_points_possible']
            response.update(response_id=identity + ':' + metric, item_id=parent_id, attempt_id=None,
                response=value, test_condition='judge=' + record['judge'] + ';metric=' + metric)
            rows.append(response)
            restored_traces.append(dict(response_id=response['response_id'], trace=json.dumps(dict(
                source_file=assessment['source_file'], source_row=assessment['source_row'], metric=metric,
                grade_record=record, task_record=task, run_records=original_runs, matching_run_indices=matches,
                trace_association='unique_final_answer_match' if len(matches) == 1 else 'ambiguous_final_answer_match'))))
    projected = {name: pd.read_parquet(root / (name + '.parquet')) for name in ['subjects', 'benchmarks']}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(rows), traces=pd.DataFrame(restored_traces))
    projected['benchmarks']['response_scale'] = json.dumps(dict(kind='mixed'))
    details = _bigfinance(directory, projected, metadata, source_records=native_source)
    return dict(details, source_attempts=len(source_runs), source_assessments=len(grades),
        source_component_observations=sum(expected.values()), imported_items=len(items),
        source_ambiguous_assessments=len({(r['source_file'], r['source_row']) for matches in ambiguous.values() for r in matches}))
