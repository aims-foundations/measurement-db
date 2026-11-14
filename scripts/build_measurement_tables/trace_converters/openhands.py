"""OpenHands event streams to trace/1, using recorded event and provider IDs."""
from collections import Counter

from .source_fields import SourceTrace

CONVERTER = 'openhands@1'
ACTIONS = {'run', 'run_ipython', 'browse', 'browse_interactive', 'read', 'edit',
           'search', 'think', 'finish', 'call_tool_mcp'}
CONTROL = {'recall', 'condense', 'condensation'}


def convert(source):
    out = SourceTrace(source, CONVERTER, json_encoded=isinstance(source, str))
    events = out.source
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise TypeError('OpenHands requires a list of native event objects')
    counts = Counter(events=len(events))
    responses, actions = {}, {}

    def project_text(step, target, path):
        return out.project(step, target, path, json_text=True)

    def llm(event, index):
        metadata = event.get('tool_call_metadata') or {}
        response = metadata.get('model_response')
        if isinstance(response, dict):
            identity = response.get('id')
            if identity and identity in responses:
                previous, original, original_path = responses[identity]
                if original != response:
                    raise ValueError('One OpenHands provider response ID has conflicting contents')
                previous['native']['openhands']['events'].append(index)
                counts['duplicate_response_events'] += 1
                return previous, original_path
            prefix = [index, 'tool_call_metadata', 'model_response']
            message = response['choices'][0]['message']
            step = {'type': 'llm'}
            message_path = prefix + ['choices', 0, 'message']
            for key, target in [('content', 'content'), ('reasoning_content', 'reasoning')]:
                if message.get(key) is not None:
                    project_text(step, target, message_path + [key])
            if response.get('model') is not None:
                out.project(step, 'model', prefix + ['model'])
            calls = message.get('tool_calls') or []
            if calls:
                step['toolCalls'] = [{} for _ in calls]
            for number, call in enumerate(calls):
                call_path = message_path + ['tool_calls', number]
                out.project(step, ['toolCalls', number, 'name'], call_path + ['function', 'name'])
                project_text(step, ['toolCalls', number, 'arguments'], call_path + ['function', 'arguments'])
                if call.get('id'):
                    out.project(step, ['toolCalls', number, 'id'], call_path + ['id'])
            usage = response.get('usage') or {}
            for path, target in [(['prompt_tokens'], 'prompt'), (['completion_tokens'], 'completion'),
                                  (['prompt_tokens_details', 'cached_tokens'], 'cacheRead'),
                                  (['completion_tokens_details', 'reasoning_tokens'], 'reasoning'),
                                  (['cache_creation_input_tokens'], 'cacheWrite')]:
                value = usage
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if value is not None:
                    step.setdefault('usage', {})
                    out.project(step, ['usage', target], prefix + ['usage', *path])
            # Native event timestamps have no timezone in this release. Retain
            # them verbatim; only recorded Unix provider times become UTC times.
            if isinstance(response.get('created'), (int, float)) and not isinstance(response['created'], bool):
                out.project(step, 'start', prefix + ['created'])
            out.add(step, openhands={'events': [index]})
            if identity:
                responses[identity] = (step, response, message_path)
            counts['llm'] += 1
            return step, message_path
        step = {'type': 'llm'}
        args = event.get('args') or {}
        key = 'content' if event.get('action') == 'message' else 'thought'
        if args.get(key) is not None:
            project_text(step, 'content', [index, 'args', key])
        elif event.get('message') is not None:
            project_text(step, 'content', [index, 'message'])
        if event.get('action') in ACTIONS:
            step['toolCalls'] = [{}]
            out.project(step, ['toolCalls', 0, 'name'], [index, 'action'])
            project_text(step, ['toolCalls', 0, 'arguments'], [index, 'args'])
        out.add(step, openhands={'events': [index]})
        counts['llm'] += 1
        return step, None

    for index, event in enumerate(events):
        action, observation = event.get('action'), event.get('observation')
        if action == 'system' or action == 'message' and event.get('source') == 'user':
            step = {'type': 'message', 'label': 'system' if action == 'system' else 'user'}
            path = [index, 'args', 'content'] if 'content' in (event.get('args') or {}) else [index, 'message']
            project_text(step, 'content', path)
            out.add(step, openhands={'events': [index]})
        elif action in CONTROL or observation in CONTROL:
            step = {'type': 'event', 'name': action or observation}
            if event.get('message') is not None:
                project_text(step, 'content', [index, 'message'])
            out.add(step, openhands={'events': [index]})
        elif action in ACTIONS or action == 'message' and event.get('source') == 'agent':
            step, prefix = llm(event, index)
            if event.get('id') is not None:
                if event['id'] in actions:
                    raise ValueError('Duplicated OpenHands action event ID')
                actions[event['id']] = (step, prefix, index)
        elif observation:
            origin = actions.get(event.get('cause'))
            if origin is None:
                step = {'type': 'event', 'name': observation}
                project_text(step, 'content', [index, 'content'] if 'content' in event else [index, 'message'])
                if observation != 'error':
                    counts['unmatched_tool_results'] += 1
            else:
                parent, prefix, action_index = origin
                metadata = events[action_index].get('tool_call_metadata') or {}
                calls = parent.get('toolCalls') or []
                matches = [n for n, c in enumerate(calls) if
                           (not metadata.get('tool_call_id') or c.get('id') == metadata['tool_call_id'])
                           and (not metadata.get('function_name') or c.get('name') == metadata['function_name'])]
                if len(matches) != 1:
                    counts['unmatched_tool_results'] += 1
                    raise ValueError('An OpenHands observation has no unique recorded originating call')
                number = matches[0]
                call_path = prefix + ['tool_calls', number] if prefix else None
                step = {'type': 'tool'}
                project_text(step, 'output', [index, 'content'] if 'content' in event else [index, 'message'])
                if call_path:
                    out.project(step, 'name', call_path + ['function', 'name'])
                    project_text(step, 'arguments', call_path + ['function', 'arguments'])
                    if calls[number].get('id'):
                        out.project(step, 'callId', call_path + ['id'])
                else:
                    out.project(step, 'name', [action_index, 'action'])
                    project_text(step, 'arguments', [action_index, 'args'])
                if observation == 'error' or (event.get('extras') or {}).get('error') is True:
                    project_text(step, 'error', [index, 'content'] if 'content' in event else [index, 'message'])
                if isinstance((event.get('extras') or {}).get('exit_code'), int):
                    out.project(step, 'exitCode', [index, 'extras', 'exit_code'])
                counts['tool'] += 1
            out.add(step, openhands={'events': [index]})
        else:
            out.add({'type': 'event', 'name': 'unrecognized'}, openhands={'events': [index]})
            counts['unrecognized'] += 1
    return out.value(timing='partial' if any('start' in s for s in out.steps) else None), counts
