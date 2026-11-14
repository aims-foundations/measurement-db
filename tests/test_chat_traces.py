"""Recorded chat variants preserve fields and associations through trace/1."""
from copy import deepcopy
import json

import pytest

from scripts.build_measurement_tables.trace_conversion import convert_trace
from trace_builder_fixtures import build_trace, assert_source_contains
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.chat_trace_audit import audit_messages, recover_messages, recover_taubench



@pytest.mark.parametrize('dialect', ['openai', 'agentdojo'])
def test_calls_blocks_provider_fields_and_empty_outputs_round_trip(dialect):
    if dialect == 'openai':
        call = {'id': '0', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":"α"}', 'signature': 'opaque'}}
        assistant = {'role': 'assistant', 'content': None, 'tool_calls': [call], 'tool_call_id': None}
        tool = {'role': 'tool', 'content': '', 'name': 'lookup', 'tool_call_id': '0', 'tool_calls': None}
    else:
        call = {'id': None, 'function': 'lookup', 'args': {'key': 'α'}, 'placeholder_args': None}
        assistant = {'role': 'assistant', 'content': [{'type': 'text', 'content': 'Checking', 'signature': 'opaque'}], 'tool_calls': [call]}
        tool = {'role': 'tool', 'content': [{'type': 'text', 'content': ''}], 'tool_call': call, 'tool_call_id': None, 'error': None}
    source = [{'role': 'system', 'content': 'Policy'}, {'role': 'user', 'content': 'Task'},
              assistant, tool, assistant, tool, {'role': 'assistant', 'content': 'Done'}]
    before = deepcopy(source)
    value, counts = convert(source)
    validate(value)
    audit_messages(value, source)
    assert source == before
    assert counts['unmatched_tool_results'] == 0
    assert counts['tool_calls'] == counts['tool'] == 2
    assert all(step['arguments'] == '{"key":"α"}' or step['arguments'] == '{"key": "α"}'
               for step in value['steps'] if step['type'] == 'tool')
    if dialect == 'agentdojo':
        assert not any('callId' in step for step in value['steps'])
        assert all('id' not in call for step in value['steps'] for call in step.get('toolCalls', []))


def convert(source, **options):
    result = convert_trace(source, family="chat", **options)
    return result.value, result.counts


def test_task_deduplication_preserves_original_blocks_and_other_user_messages():
    source = [{'role': 'user', 'content': [{'type': 'text', 'content': 'Task', 'extra': False}]},
              {'role': 'assistant', 'content': 'Answer'}, {'role': 'user', 'content': 'Follow up'}]
    value, _ = convert(source, item_content='Task')
    validate(value)
    assert len(value['steps']) == 2
    assert recover_messages(value, item_content='Task') == source


def test_api_user_role_does_not_invent_task_or_simulator_attribution():
    messages = [dict(role='user', content='Now proceed with the plan.')]
    value, _ = convert(messages, item_content='Benchmark task')
    assert value['steps'][0]['label'] == 'user'
    assert 'participant' not in value['steps'][0]
    audit_messages(value, messages)


def test_ambiguous_or_absent_tool_calls_are_reported_without_invented_links():
    call = {'id': None, 'function': 'lookup', 'args': {}}
    value, counts = convert([{'role': 'assistant', 'tool_calls': [call, call]},
                                  {'role': 'tool', 'content': 'result', 'tool_call': call}])
    assert counts['unmatched_tool_results'] == 0
    assert counts['indistinguishable_tool_results'] == 1
    assert 'callId' not in value['steps'][-1]
    _, counts = convert([{'role': 'tool', 'tool_call_id': 'absent', 'content': 'result'}])
    assert counts['unmatched_tool_results'] == 1


def test_tool_argument_key_order_does_not_break_recorded_associations():
    source = [dict(role='assistant', tool_calls=[dict(function='lookup', args={'a': 1, 'b': 2})]),
              dict(role='tool', content='found', tool_call=dict(function='lookup', args={'b': 2, 'a': 1}))]
    value, counts = convert(source)
    assert counts['unmatched_tool_results'] == 0
    audit_messages(value, source)


def test_reasoning_cached_tokens_and_unknown_usage_fields_survive():
    source = [dict(role='assistant', content='Answer', reasoning_content='Reasoning', usage={
        'prompt_tokens': 12, 'completion_tokens': 7,
        'completion_tokens_details': {'reasoning_tokens': 3, 'provider_extension': False},
        'prompt_tokens_details': {'cached_tokens': 8}, 'total_tokens': 19})]
    value, _ = convert(source)
    validate(value)
    assert value['steps'][0]['usage'] == dict(prompt=12, completion=7, reasoning=3, cacheRead=8)
    audit_messages(value, source)


def test_missing_call_id_does_not_choose_between_different_arguments():
    source = [dict(role='assistant', tool_calls=[dict(function='lookup', args={'a': 1}),
                                                dict(function='lookup', args={'a': 2})]),
              dict(role='tool', name='lookup', content='found')]
    _, counts = convert(source)
    assert counts['unmatched_tool_results'] == 1
    assert counts['indistinguishable_tool_results'] == 0


def test_absent_call_arguments_are_not_fabricated_as_json_null():
    with pytest.raises(ValueError, match='args field'):
        convert([dict(role='assistant', tool_calls=[dict(function='lookup')])])


@pytest.mark.parametrize('message', [{'role': 'unknown', 'payload': [0, False, None]}, 'raw message'])
def test_unknown_messages_are_preserved_and_reported(message):
    value, counts = convert([message])
    validate(value)
    assert counts['unrecognized'] == 1
    audit_messages(value, [message])


@pytest.mark.parametrize('mutation', ['content', 'native', 'missing'])
def test_independent_recovery_rejects_corruption(mutation):
    source = [{'role': 'assistant', 'content': 'α\n原文', 'signature': 'original'}]
    value, _ = convert(source)
    if mutation == 'content': value['steps'][0]['content'] = 'changed'
    if mutation == 'native': value['steps'][0]['native']['chat']['source']['signature'] = 'changed'
    if mutation == 'missing': value['steps'] = []
    with pytest.raises(ValueError): audit_messages(value, source)


def hal_source():
    call = {'id': 'c1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
    start = [{'role': 'system', 'content': 'Agent policy'}, {'role': 'user', 'content': 'Help'}]
    action = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
    history = start + [action, {'role': 'tool', 'tool_call_id': 'c1', 'content': 'found'}]

    def log(messages, reply, identity, model='agent-model', second=0):
        return {'inputs': {'model': model, 'messages': messages},
                'output': {'id': identity, 'model': model, 'choices': [{'message': reply}],
                           'usage': {'prompt_tokens': 20, 'completion_tokens': 4}},
                'started_at': f'2026-01-01T00:00:0{second}+00:00', 'ended_at': f'2026-01-01T00:00:0{second+1}+00:00'}
    final = {'role': 'assistant', 'content': 'Booked'}
    last = log(history, final, 'second', second=2)
    return {'configuration': {'agent_args': {'model_name': 'openai/agent-model'}},
            'result': {'reward': 1, 'taken_actions': ['original']},
            'logs': [log([{'role': 'user', 'content': 'Booked'}], {'role': 'assistant', 'content': 'STOP'},
                         'simulator', 'simulator-model', second=4), last, deepcopy(last), log(start, action, 'first')]}


def test_hal_wrappers_simulator_and_recorded_usage_are_not_extra_agent_turns():
    source = hal_source()
    before = deepcopy(source)
    result = build_trace('taubench', source)
    value = result.value
    validate(value)
    assert source == before
    assert result.counts['llm'] == 2
    assert result.counts['unmatched_tool_results'] == 0
    assert_source_contains(value['native']['source'], source)
    assert_source_contains(recover_taubench(value), source)
    assert value['steps'][-1]['type'] == 'message'
    assert value['steps'][-1]['label'] == 'user'
    assert 'participant' not in value['steps'][-1]
    assert value['steps'][-1]['content'] == 'STOP'
    assert [s['usage']['prompt'] for s in value['steps'] if s['type'] == 'llm'] == [20, 20]
    assert value['timing'] == 'recorded'


def test_hal_complete_parallel_proposals_survive_harness_pruning():
    source = hal_source()
    source['logs'][-1]['output']['choices'][0]['message'] = deepcopy(source['logs'][-1]['output']['choices'][0]['message'])
    call = deepcopy(source['logs'][-1]['output']['choices'][0]['message']['tool_calls'][0])
    call['id'] = 'not-executed'
    source['logs'][-1]['output']['choices'][0]['message']['tool_calls'].append(call)
    value = build_trace('taubench', source).value
    assert len(next(s for s in value['steps'] if s.get('toolCalls'))['toolCalls']) == 2
    assert_source_contains(recover_taubench(value), source)


def test_hal_empty_ids_link_to_the_executed_harness_call_without_inventing_ids():
    source = hal_source()
    # Break fixture aliases, as real JSON records do.
    source = json.loads(json.dumps(source))
    for log in source['logs']:
        for message in log['inputs']['messages']:
            for call in message.get('tool_calls') or []:
                call['id'] = ''
            if message.get('role') == 'tool':
                message['tool_call_id'] = ''
    proposal = source['logs'][-1]['output']['choices'][0]['message']['tool_calls']
    proposal[0]['id'] = ''
    second = deepcopy(proposal[0])
    second['function']['arguments'] = '{"other":true}'
    proposal.append(second)
    result = build_trace('taubench', source)
    assert result.counts['unmatched_tool_results'] == 0
    tool = next(s for s in result.value['steps'] if s['type'] == 'tool')
    assert 'callId' not in tool and tool['arguments'] == '{}'
    assert_source_contains(recover_taubench(result.value), source)


def test_hal_outer_wrapper_preserves_policy_role_and_deduplicates_provider_call():
    source = hal_source()
    source['logs'][-1]['op_name'] = 'weave:///project/op/litellm.completion:version'
    provider = deepcopy(source['logs'][-1])
    provider['op_name'] = 'weave:///project/op/openai.chat.completions.create:version'
    provider['inputs']['messages'][0]['role'] = 'user'
    source['logs'].insert(0, provider)
    value = build_trace('taubench', source).value
    assert value['native']['hal_recorded_reply_count'] == 2
    assert 'apiCalls' not in value.get('run', {})
    assert_source_contains(recover_taubench(value), source)


@pytest.mark.parametrize('mutation', ['output', 'missing_call'])
def test_hal_source_audit_rejects_changed_output_or_missing_call(mutation):
    source = hal_source()
    value = build_trace('taubench', source).value
    if mutation == 'output':
        next(step for step in value['steps'] if step.get('content') == 'Booked')['content'] = 'Changed'
    else:
        value['native']['hal_message_sources'][0]['replies'].pop()
    with pytest.raises(ValueError): recover_taubench(value)


def test_distinct_response_ids_with_identical_replies_are_not_merged():
    source = hal_source()
    repeated = deepcopy(source['logs'][1])
    repeated['output']['id'] = 'distinct-recorded-call'
    source['logs'].append(repeated)
    value = build_trace('taubench', source).value
    assert value['native']['hal_recorded_reply_count'] == 3
    assert len(value['native']['hal_message_sources']) == 2
    assert all(len({r['message'] for r in group['replies']}) == len(group['replies'])
               for group in value['native']['hal_message_sources'])
    assert_source_contains(recover_taubench(value), source)


def test_hal_conflicting_histories_remain_separate_recorded_contexts():
    source = hal_source()
    source['logs'][-1]['inputs']['messages'] = deepcopy(source['logs'][-1]['inputs']['messages'])
    source['logs'][-1]['inputs']['messages'][0]['content'] = 'Different policy'
    result = build_trace('taubench', source)
    assert len(result.value['native']['hal_message_sources']) == 2
    assert {s['agent'] for s in result.value['steps']} == {'HAL context 1', 'HAL context 2'}
    assert 'Different policy' in str(result.value['steps'])
    assert_source_contains(recover_taubench(result.value), source)


def test_sierra_native_envelope_and_ungraded_hal_error_remain_distinct():
    source = {'result': {'traj': [{'role': 'user', 'content': 'Customer request'},
                               {'role': 'assistant', 'content': 'Answer'}], 'reward': 0}, 'configuration': {}, 'logs': []}
    value = build_trace('taubench', source).value
    audit_messages(value, source['result']['traj'])
    assert value['steps'][0]['participant'] == 'simulated_user'
    assert value['native']['source']['result']['reward'] == 0
    assert 'traj' not in value['native']['source']['result']
    error = {'result': 'Recorded API failure', 'configuration': {'agent_args': {'model_name': 'model'}}, 'logs': []}
    value = build_trace('taubench', error).value
    validate(value)
    assert value['steps'] == []
    assert value['run']['exception'] == 'Recorded API failure'
    assert 'evaluation' not in value
