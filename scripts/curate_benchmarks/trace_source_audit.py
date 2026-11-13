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
