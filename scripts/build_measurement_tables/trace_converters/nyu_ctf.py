"""Project released NYU CTF log formats without inventing a global chronology."""
from collections import Counter

from .source_fields import SourceTrace


def convert(source):
    out = SourceTrace(source, 'nyu_ctf@1', json_encoded=isinstance(source, str))
    counts = Counter()
    assessment = {'type': 'event', 'name': 'published_assessment'}
    out.project(assessment, 'content', ['summary', 'results', out.source['task_id']], json_text=True)
    out.add(assessment, nyu_ctf={'kind': 'assessment'})

    def add(step, log, component, index, path, message, *, model_path=None, calls_path=None):
        if message.get('content') is not None:
            out.project(step, 'output' if step['type'] == 'tool' else 'content', path + ['content'], json_text=True)
        if step['type'] == 'tool' and 'output' not in step:
            out.project(step, 'output', path + ['tool_result', 'result'], json_text=True)
        if step['type'] == 'llm':
            if model_path:
                out.project(step, 'model', model_path)
            if message.get('thought') is not None:
                out.project(step, 'reasoning', path + ['thought'], json_text=True)
            calls = message.get('tool_calls') or []
            if calls_path:
                calls = [message['tool_call']]
            if calls:
                step['toolCalls'] = [{} for _ in calls]
                for ordinal, call in enumerate(calls):
                    prefix = calls_path if calls_path else path + ['tool_calls', ordinal, 'function']
                    out.project(step, ['toolCalls', ordinal, 'name'], prefix + ['name'])
                    out.project(step, ['toolCalls', ordinal, 'arguments'], prefix + ['parsed_args' if calls_path else 'arguments'], json_text=True)
                    if not calls_path and call.get('id') is not None:
                        out.project(step, ['toolCalls', ordinal, 'id'], path + ['tool_calls', ordinal, 'id'])
        if message.get('agent') is not None:
            out.project(step, 'agent', path + ['agent'])
        elif component != 'messages':
            step['agent'] = component
        out.add(step, nyu_ctf={'kind': 'message', 'log': log, 'component': component, 'index': index})
        counts[step['type']] += 1

    for number, log in enumerate(out.source['released_traces']):
        prefix = ['released_traces', number]
        record, base = log['source_record'], prefix + ['source_record']
        boundary = {'type': 'event', 'name': 'released_log_file'}
        out.project(boundary, 'content', prefix + ['source_member'])
        out.add(boundary, nyu_ctf={'kind': 'file', 'log': number})
        counts['released_log_files'] += 1
        if 'messages' in record:
            messages = record['messages']
            # The XML backend logs the raw response, API-formatted replay, and
            # typed parsed response separately. Only the typed object carrying
            # a provider response is a model turn. Null responses are demos.
            typed = any(isinstance((m[1] if isinstance(m, list) else m).get('extra'), dict) for m in messages)
            pending = {}
            for index, entry in enumerate(messages):
                path = base + ['messages', index] + ([1] if isinstance(entry, list) else [])
                message = entry[1] if isinstance(entry, list) else entry
                role, extra = message.get('role'), message.get('extra') or {}
                actual = role == 'assistant' and (not typed or extra.get('response', {}).get('data') is not None)
                if actual:
                    step = {'type': 'llm'}
                    pending = {call['id']: call for call in message.get('tool_calls') or [] if call.get('id') is not None}
                elif role in {'system', 'user'}:
                    step = {'type': 'message', 'label': role, 'content': ''}
                elif role == 'tool' and message.get('tool_call_id') in pending:
                    step = {'type': 'tool'}
                    out.project(step, 'callId', path + ['tool_call_id'])
                    if message.get('name') is not None:
                        out.project(step, 'name', path + ['name'])
                else:
                    step = {'type': 'event', 'name': 'recorded_context'}
                    if role is None and message.get('text') is not None:
                        out.project(step, 'content', path + ['text'], json_text=True)
                model_path = base + ['args', 'model'] if record.get('args', {}).get('model') is not None else None
                add(step, number, 'messages', index, path, message, model_path=model_path)
        elif 'history' in record:
            # History already contains the conversation. Rendering trajectory
            # responses again would count the same generation twice.
            for index, message in enumerate(record['history']):
                path = base + ['history', index]
                role = message['role']
                if message.get('is_demo'):
                    step = {'type': 'event', 'name': 'demonstration'}
                elif role == 'assistant':
                    step = {'type': 'llm'}
                elif role in {'system', 'user'}:
                    step = {'type': 'message', 'label': role, 'content': ''}
                else:
                    raise ValueError('Unknown Enigma history role')
                add(step, number, 'history', index, path, message)
                if role == 'assistant' and not message.get('is_demo') and message.get('action') is not None:
                    action = {'type': 'event', 'name': 'recorded_action'}
                    out.project(action, 'content', path + ['action'], json_text=True)
                    if message.get('agent') is not None:
                        out.project(action, 'agent', path + ['agent'])
                    out.add(action, nyu_ctf={'kind': 'action', 'log': number, 'component': 'history', 'index': index})
                    counts['recorded_actions'] += 1
        elif 'planner' in record:
            components = [(key, base + [key], record.get(key) or [], key + '_model') for key in ['autoprompter', 'planner']]
            components += [(f'executors[{i}]', base + ['executors', i], messages, 'executor_model') for i, messages in enumerate(record.get('executors') or [])]
            for component, path, messages, model_key in components:
                if not messages:
                    continue
                out.add({'type': 'event', 'name': 'agent_history', 'content': component},
                        nyu_ctf={'kind': 'component', 'log': number, 'component': component})
                for index, message in enumerate(messages):
                    role = message['role'].removeprefix('MessageRole.').lower()
                    if role == 'assistant':
                        step = {'type': 'llm'}
                    elif role in {'system', 'user'}:
                        step = {'type': 'message', 'label': role, 'content': ''}
                    elif role == 'observation':
                        step = {'type': 'tool'}
                        if message.get('tool_result', {}).get('name') is not None:
                            out.project(step, 'name', path + [index, 'tool_result', 'name'])
                    elif role == 'knowledge':
                        step = {'type': 'event', 'name': 'retrieved_context'}
                    else:
                        raise ValueError('Unknown component history role')
                    mp = base + [model_key] if record.get(model_key) is not None else None
                    cp = path + [index, 'tool_call'] if message.get('tool_call') is not None else None
                    add(step, number, component, index, path + [index], message, model_path=mp, calls_path=cp)
        else:
            raise ValueError('Unknown released NYU CTF log format')
    return out.value(), counts
