"""Keep individual recorded assertion predictions and their program summaries."""

import csv
import hashlib
import io
import json

import pandas as pd

from measurement_db.build_base import Judge
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


def assertion_rows(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    occurrences = {}
    result = []
    for row in rows:
        statement = row['assert_statement']
        occurrence = occurrences.get(statement, 0)
        occurrences[statement] = occurrence + 1
        identity = hashlib.sha256(json.dumps([statement, occurrence]).encode()).hexdigest()
        result.append((identity, statement, occurrence, {'pass': 1., 'fail': 0.}.get(row['final_result'])))
    return result


def convert(tables, attempts, protocol):
    selected = attempts.task.eq('execution')
    runs = attempts.loc[selected]
    definitions, outcomes, source_rows = {}, [], {}
    for row in runs.itertuples():
        if not isinstance(row.output, str):
            continue
        source_rows[row.response_key] = list(csv.DictReader(io.StringIO(row.output)))
        for position, (identity, statement, occurrence, grade) in enumerate(assertion_rows(row.output)):
            definitions[row.item_key, identity] = dict(parent_item_key=row.item_key, component_id=identity,
                grading_criterion=dict(reference_answer=statement,
                    rule=json.dumps(dict(occurrence=occurrence,
                        rule='Record the parsed assertion prediction: pass=1, fail=0; other labels are unknown. This is the model\'s reported judgment, not an executed test verdict.'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='unordered',
                        meanings={'0': 'The model predicted fail.', '1': 'The model predicted pass.'})),
                verifier=Judge(spec=json.dumps(dict(kind='reported_assertion_prediction', original_protocol=protocol), sort_keys=True)))
            outcomes.append(dict(parent_response_key=row.response_key, component_id=identity, response=grade))
    keys = set(runs.response_key)
    selected_items = set(runs.item_key)
    expanded = component_tables(dict(subjects=tables['subjects'],
        items=tables['items'].loc[tables['items'].item_key.isin(selected_items)],
        responses=tables['responses'].loc[tables['responses'].response_key.isin(keys)],
        traces=tables['traces'].loc[tables['traces'].response_key.isin(keys)]),
        pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))
    remaining = tables['responses'].loc[~tables['responses'].response_key.isin(keys)].copy()
    original_traces = tables['traces'].set_index('response_key').trace.to_dict()
    originals = runs.set_index('response_key').to_dict('index')
    positions = {key: {identity: position for position, (identity, *_rest) in enumerate(assertion_rows(original['output']))}
        for key, original in originals.items() if isinstance(original['output'], str)}
    traces = []
    for row in expanded['responses'].itertuples():
        parent = row.attempt_key
        position = None if row.response_key == parent else positions[parent][json.loads(row.response_key)[1]]
        output = originals[parent]['output'] if position is None else source_rows[parent][position].get('comment')
        traces.append(dict(response_key=row.response_key, trace=json.dumps(dict(source_assessment_key=parent,
            recorded_program_score=originals[parent]['response'], source_assessment_trace=original_traces[parent],
            assertion_row=position, output=output), ensure_ascii=False, allow_nan=False)))
    # The export groups assertion answers by program, but does not establish a
    # shared physical execution. Preserve the association in evidence, not in
    # responses.attempt_id. Other measurement channels retain their old traces.
    return dict(subjects=tables['subjects'],
        items=pd.concat([tables['items'].loc[~tables['items'].item_key.isin(selected_items)], expanded['items']], ignore_index=True),
        responses=pd.concat([remaining, expanded['responses'].drop(columns='attempt_key')], ignore_index=True),
        traces=pd.concat([tables['traces'].loc[~tables['traces'].response_key.isin(keys)], pd.DataFrame(traces)], ignore_index=True))
