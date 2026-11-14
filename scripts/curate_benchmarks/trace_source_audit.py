"""Check displayed fields directly against the retained native trace input.

This module deliberately does not import converters or benchmark builders.
"""
from copy import deepcopy
import json


def _get(value, path):
    for key in path:
        value = value[key]
    return value


def checked_source(value):
    native = value['native']
    original = native['source']
    source = json.loads(original) if native['source_encoding'] == 'json' else original
    records = [(value, native.get('source_fields', []))]
    records += [(step, step.get('native', {}).get('source_fields', [])) for step in value['steps']]
    for step, fields in records:
        for field in fields:
            actual = _get(step, field['target'])
            if field.get('encoding') == 'json':
                actual = json.loads(actual)
            elif 'encoding' in field:
                raise ValueError('Unknown source-field encoding')
            if actual != _get(source, field['source']):
                raise ValueError('Displayed trace field differs from its native source')
    return deepcopy(original), source


def recover_source(value):
    return checked_source(value)[0]


def recover_theagentcompany(value):
    original = value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return original
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    original, source = checked_source(value)
    if value['converter'] == 'text@1':
        if not isinstance(source, str) or len(value['steps']) != 1 or value['steps'][0]['type'] != 'llm':
            raise ValueError('TheAgentCompany text transcript was split into invented turns')
        if value['steps'][0].get('content') != source:
            raise ValueError('TheAgentCompany transcript is not complete')
    elif value['converter'] == 'theagentcompany_roleplay@1':
        expected = []
        for row in source:
            expected.extend(['message', 'llm'] + ['tool'] * len(row['tool_calls']))
        if [s['type'] for s in value['steps']] != expected:
            raise ValueError('TheAgentCompany role-play turns or tool results were lost')
    elif value['converter'] == 'openhands@1':
        covered, anchors, action_steps, identities = [], [], {}, set()
        for step in value['steps']:
            indices = step['native']['openhands']['events']
            if not indices:
                raise ValueError('An OpenHands step has no native event')
            covered.extend(indices)
            anchors.append(min(indices))
            event = source[indices[0]]
            if step['type'] == 'llm':
                for index in indices:
                    if source[index].get('source') != 'agent' or not source[index].get('action'):
                        raise ValueError('A non-agent event was converted to a model turn')
                    action_steps[source[index].get('id')] = step
                response = (event.get('tool_call_metadata') or {}).get('model_response')
                if response and response.get('id'):
                    if response['id'] in identities:
                        raise ValueError('A provider response was displayed as multiple model turns')
                    identities.add(response['id'])
            elif step['type'] == 'tool':
                parent = action_steps.get(event.get('cause'))
                if parent is None:
                    raise ValueError('A displayed tool result has no recorded originating action')
                calls = parent.get('toolCalls') or []
                if not any(c.get('id') == step.get('callId') and c.get('name') == step.get('name')
                           and c['arguments'] == step.get('arguments') for c in calls):
                    raise ValueError('A tool result is associated with a different recorded call')
            elif step['type'] == 'message':
                expected = 'system' if event.get('action') == 'system' else 'user'
                if step.get('label') != expected:
                    raise ValueError('An OpenHands message has incorrect source attribution')
        if sorted(covered) != list(range(len(source))) or anchors != sorted(anchors):
            raise ValueError('OpenHands native events were omitted, duplicated or reordered')
        expected_ids = {e['tool_call_metadata']['model_response']['id'] for e in source
                        if (e.get('tool_call_metadata') or {}).get('model_response', {}).get('id')}
        if identities != expected_ids:
            raise ValueError('OpenHands recorded model-response coverage differs from the source')
    else:
        raise ValueError('Unknown TheAgentCompany trace converter')
    return original


def recover_browsergym(value):
    original = value
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    if value['converter'] != 'browsergym@1':
        raise ValueError('Expected a BrowserGym trace')
    original, source = checked_source(value)
    expected = []
    for number, step in enumerate(source['record']['steps']):
        expected.append((number, 'observation', 'event'))
        if step.get('last_action_error'):
            expected.append((number, 'last_action_error', 'event'))
        if step.get('action') is not None or step.get('reasoning') is not None:
            expected.append((number, 'agent', 'llm'))
    actual = [(s['native']['browsergym']['step'], s['native']['browsergym']['kind'], s['type'])
              for s in value['steps']]
    if actual != expected:
        raise ValueError('BrowserGym observations, errors or agent outputs were omitted or reordered')
    return original


def recover_text(value):
    original = value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return original
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    if value.get('converter') != 'text@1':
        raise ValueError('Expected a text trace')
    if value['native'].get('source_encoding') == 'literal_text':
        import hashlib
        if len(value['steps']) != 1 or value['steps'][0].get('type') != 'llm':
            raise ValueError('A text record must remain one complete output')
        content = value['steps'][0].get('content')
        if not isinstance(content, str) or hashlib.sha256(content.encode('utf-8')).hexdigest() != value['native']['source_sha256']:
            raise ValueError('Recorded text does not match its original source hash')
        return content
    original, source = checked_source(value)
    if not isinstance(source, str) or len(value['steps']) != 1:
        raise ValueError('A text record must remain one complete recorded string')
    if value['steps'][0].get('type') != 'llm' or value['steps'][0].get('content') != source:
        raise ValueError('The complete original answer is not preserved')
    return original


def recover_alpacaeval(value):
    original = value
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    if value.get('converter') != 'pairwise@1':
        raise ValueError('Expected an AlpacaEval pairwise trace')
    original, source = checked_source(value)
    record = source['native_record']
    reference = 'gpt4_1106_preview'
    if reference not in {record['generator_1'], record['generator_2']}:
        raise ValueError('The comparison has no released fixed reference')
    side = 2 if record['generator_1'] == reference else 1
    if len(value['steps']) != 1 or value['steps'][0]['type'] != 'llm':
        raise ValueError('Each pairwise trace must display only its evaluated side')
    step = value['steps'][0]
    if step.get('model') != record[f'generator_{side}'] or step.get('content') != record[f'output_{side}']:
        raise ValueError('The opponent output was substituted for the evaluated model')
    if 'usage' in step or 'cost' in step or 'run' in value:
        raise ValueError('Judge metrics must not be attributed to candidate generation')
    completion = record.get('raw_completion')
    judge = value.get('evaluation', {}).get('judge')
    if completion is None and judge is not None or completion is not None and judge is None:
        raise ValueError('Recorded and unavailable judge outputs must remain distinct')
    return original


def recover_os_harm(value):
    original = value
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    if value.get('converter') != 'os_harm@1':
        raise ValueError('Expected OS-Harm native-format traces')
    original, source = checked_source(value)
    expected = []
    for number, record in enumerate(source['record']['steps']):
        expected.append((number, 'observation', None, 'event'))
        if record.get('response') is not None:
            expected.append((number, 'agent', None, 'llm'))
        expected.extend((number, 'action', i, 'event') for i in range(len(record.get('actions') or [])))
    caption = source.get('caption')
    if isinstance(caption, str) or isinstance(caption, dict) and caption.get('caption') is not None:
        expected.append((None, 'caption', None, 'event'))
    actual = []
    for step in value['steps']:
        origin = step['native']['os_harm']
        actual.append((origin.get('step'), origin['kind'], origin.get('action'), step['type']))
        if step['type'] == 'llm' and any(key in step for key in ['cost', 'usage', 'start', 'end']):
            raise ValueError('Auxiliary captioner/grader metrics were assigned to the solver')
    if actual != expected:
        raise ValueError('OS-Harm observations, replies, actions or captions were omitted or reordered')
    return original


def recover_nyu_ctf(value):
    original = value
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('schema') != 'trace/1':
        return original
    if value.get('converter') != 'nyu_ctf@1':
        raise ValueError('Expected released NYU CTF log formats')
    original, source = checked_source(value)
    expected = [('assessment', None, None, None)]
    expected_turns, actual_turns = [], []
    for log_number, entry in enumerate(source['released_traces']):
        record = entry['source_record']
        expected.append(('file', log_number, None, None))
        if 'messages' in record:
            blocks = [('messages', record['messages'])]
        elif 'history' in record:
            blocks = [('history', record['history'])]
        else:
            blocks = [(key, record.get(key) or []) for key in ['autoprompter', 'planner']]
            blocks += [(f'executors[{i}]', block) for i, block in enumerate(record.get('executors') or [])]
        for component, messages in blocks:
            if component not in {'messages', 'history'} and messages:
                expected.append(('component', log_number, component, None))
            parsed = [m[1] if isinstance(m, list) else m for m in messages]
            typed = component == 'messages' and any(isinstance(m.get('extra'), dict) for m in parsed)
            for index, message in enumerate(parsed):
                role = message.get('role', '').removeprefix('MessageRole.').lower()
                provider_backed = not typed or message.get('extra', {}).get('response', {}).get('data') is not None
                if role == 'assistant' and not message.get('is_demo') and provider_backed:
                    expected_turns.append((log_number, component, index))
                expected.append(('message', log_number, component, index))
                if component == 'history' and message.get('role') == 'assistant' and not message.get('is_demo') and message.get('action') is not None:
                    expected.append(('action', log_number, component, index))
    actual = []
    for step in value['steps']:
        origin = step['native']['nyu_ctf']
        kind, number, component, index = (origin.get(k) for k in ['kind', 'log', 'component', 'index'])
        actual.append((kind, number, component, index))
        if step['type'] == 'llm':
            actual_turns.append((number, component, index))
            if any(k in step for k in ['start', 'end', 'cost', 'usage']):
                raise ValueError('Aggregate or auxiliary metrics became per-turn measurements')
            record = source['released_traces'][number]['source_record']
            if component == 'messages':
                messages = [m[1] if isinstance(m, list) else m for m in record['messages']]
                message = messages[index]
                if message.get('role') != 'assistant' or any(isinstance(m.get('extra'), dict) for m in messages) and message.get('extra', {}).get('response', {}).get('data') is None:
                    raise ValueError('A demonstration or replay copy became a model turn')
            elif component == 'history':
                message = record['history'][index]
                if message.get('is_demo') or message['role'] != 'assistant':
                    raise ValueError('An Enigma demonstration became a model turn')
            else:
                block = record['executors'][int(component[10:-1])] if component.startswith('executors[') else record[component]
                if block[index]['role'].removeprefix('MessageRole.').lower() != 'assistant':
                    raise ValueError('Non-assistant component context became a model turn')
                if any('id' in call for call in step.get('toolCalls', [])):
                    raise ValueError('Component call IDs were invented')
        if component and component not in {'messages', 'history'} and kind == 'message' and step.get('agent') != component:
            raise ValueError('Distinct component histories lost their agent attribution')
    if actual_turns != expected_turns:
        raise ValueError('Recorded model turns were lost or duplicated')
    if actual != expected:
        raise ValueError('Released logs, component boundaries or messages were lost, duplicated or reordered')
    return original
