"""Reconcile MMDocRAG ratings, original questions, subjects and complete answers."""

from collections import Counter
import json
import re
import unicodedata

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='MMDocRAG complete native rating evidence')
    raw = directory / 'raw'
    layout = metadata['build']['parameters']['layout']
    parameters = metadata['build']['parameters']
    protocol = metadata['grading']['verifiers']['release']
    dimensions = {re.sub('[^a-z0-9]', '', name.lower()) for name in protocol['dimensions']}
    bank = {}
    for filename in (layout['preferred_gold'], layout['fallback_gold']):
        for line in (raw / filename).read_text().split('\n'):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('q_id') is not None:
                bank.setdefault(record['q_id'], record)
    item_questions, roster = {}, Counter()
    for item in tables['items'].to_dict('records'):
        task, _, dimension = item['raw_item_id'].rpartition('::')
        _check(dimension in dimensions, True, 'MMDocRAG named native rating dimension')
        qid = int(task.removeprefix('q_id::'))
        original = bank[qid]
        answer = original['answer_short']
        if isinstance(answer, list):
            answer = ', '.join(map(str, answer))
        _check((item['testlet_id'], item['content']), (task, original['question']), 'MMDocRAG complete original question')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=answer, rule='Recorded judge rating for ' + dimension),
            'MMDocRAG original reference and selected dimension')
        _check(json.loads(item['verifier']), {'class': 'judge', 'spec': protocol['spec'], 'judged_by': protocol['judged_by']},
            'MMDocRAG original judge instructions')
        _check(pd.isna(item['asset_manifest']) and pd.isna(item['item_features']), True, 'MMDocRAG no invented input media or attributes')
        item_questions[item['item_id']] = qid
        roster[qid, dimension] += 1
    _check(roster, Counter({(qid, dimension): 1 for qid in bank for dimension in dimensions}), 'MMDocRAG complete question and rating roster')
    originals, traces = attempt_view(tables)
    raw_ids = tables['attempts'].set_index('attempt_id').raw_attempt_id.to_dict()
    actual = {raw_ids[row.attempt_id]: row for row in originals.itertuples()}
    subjects = tables['subjects'].set_index('subject_id').to_dict('index')
    seen, used_subjects, counts = set(), set(), Counter()
    index = 0
    for path in sorted((raw / layout['evaluation_directory']).iterdir()):
        if not path.is_file():
            continue
        match = re.fullmatch(r'(.+?)_(pure-text|multimodal)(?:_response)?_quotes(\d+)(?:_llm-judge\.jsonl|\.jsonl(?:_evaluation\.jsonl)?)', path.name)
        _check(match is not None, True, 'MMDocRAG source evaluation filename')
        model, mode, quotes = match.groups()
        candidates = [f'{model}_{mode}_quotes{quotes}_response.jsonl', f'{model}_{mode}_response_quotes{quotes}.jsonl']
        answers = {}
        for candidate in candidates:
            source = raw / layout['trace_directory'] / candidate
            if source.exists():
                for line in source.read_text().split('\n'):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('q_id') is not None:
                        answers[record['q_id']] = record.get('response')
                break
        records = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        fields = set().union(*(record.keys() for record in records))
        for record in records:
            key, index = str(index), index + 1
            if record.get('q_id') is None or not isinstance(record.get('response'), dict):
                continue
            row = actual[key]
            expected = {field: record.get(field) for field in fields}
            answer = answers.get(record['q_id'])
            evidence = json.loads(traces[row.response_id])
            _check(evidence, dict(evaluation=expected, answer=answer if isinstance(answer, str) else None,
                source_file=path.relative_to(raw).as_posix()), 'MMDocRAG every original judge field and complete attributed answer')
            subject = subjects[row.subject_id]
            label = parameters['subject_aliases'].get(model, model)
            normalize = lambda value: unicodedata.normalize('NFC', value).strip().lower()
            _check(normalize(subject['display_name']), normalize(label), 'MMDocRAG literal recorded model alias')
            variant = parameters['released_variants'].get(subject['display_name'])
            _check(_features(subject['subject_features_extra']), dict(released_variant=variant) if variant else {},
                'MMDocRAG original no-thinking and fine-tuned configurations')
            _check(all(pd.isna(subject[field]) for field in ('access_date', 'harness', 'harness_version', 'reasoning_effort')),
                True, 'MMDocRAG no invented runtime settings')
            _check((item_questions[row.item_id], row.trial, row.test_condition),
                (record['q_id'], 1, mode + '/quotes' + quotes), 'MMDocRAG original question and source condition')
            _check(pd.isna(row.interactors), True, 'MMDocRAG no invented interactors')
            seen.add(key); used_subjects.add(row.subject_id)
            counts['source_answers'] += isinstance(answer, str) and bool(answer)
            counts['source_ungraded'] += pd.isna(row.response)
    _check((seen, used_subjects), (set(actual), set(subjects)), 'MMDocRAG complete native attempt and model coverage')
    return dict(counts, source_responses=len(actual), source_items=len(bank), source_subjects=len(subjects))
