"""Independent recovery and comparison of chat@1's mapped source fields."""
from copy import deepcopy
import json


def recover_messages(value, *, item_content=None):
    if value.get("converter") != "chat@1":
        raise ValueError("Expected chat@1")
    records = [(step, step["native"]["chat"]) for step in value["steps"]]
    indices = [projection['index'] for _, projection in records]
    if indices != sorted(indices):
        raise ValueError('Chat steps are out of source order')
    records += [({"content": item_content}, projection)
                for projection in value.get("native", {}).get("chat_omitted_tasks", [])]
    output = {}
    for step, projection in records:
        source = deepcopy(projection["source"])
        for field in projection["fields"]:
            target = step
            for key in field["target"]:
                target = target[key]
            if "span" in field:
                start, length = field["span"]
                target = target[start:start + length]
            if field.get("encoding") == "json":
                target = json.loads(target)
            parent = source
            for key in field["source"][:-1]:
                parent = parent[key]
            parent[field["source"][-1]] = deepcopy(target)
        index = projection["index"]
        if index in output:
            raise ValueError("Duplicated chat source index")
        output[index] = source
    if sorted(output) != list(range(len(output))):
        raise ValueError("Missing chat source index")
    return [output[i] for i in range(len(output))]


def audit_messages(value, messages, *, item_content=None):
    if recover_messages(value, item_content=item_content) != messages:
        raise ValueError("Chat trace does not preserve its complete source messages")
    kinds = {'assistant': 'llm', 'tool': 'tool', 'function': 'tool',
             'system': 'message', 'developer': 'message', 'user': 'message'}
    for step in value['steps']:
        source = messages[step['native']['chat']['index']]
        if isinstance(source, dict) and source.get('role') in kinds:
            if step['type'] != kinds[source['role']]:
                raise ValueError('Chat source role was mapped to the wrong step type')
            if source['role'] in {'system', 'developer'} and step.get('label') != 'system':
                raise ValueError('Chat system policy was mapped to the wrong label')


def recover_taubench(value, *, item_content=None):
    """Recover the original envelope and verify every HAL source reference."""
    source = deepcopy(value['native']['source'])
    if 'hal_message_sources' not in value['native']:
        source['result']['traj'] = recover_messages(value, item_content=item_content)
        audit_messages(value, source['result']['traj'], item_content=item_content)
        return source
    messages, represented = [], set()
    groups = value['native']['hal_message_sources']
    for number, group in enumerate(groups, 1):
        log = source['logs'][group['history_log']]
        history = deepcopy(log['inputs']['messages']) + [deepcopy(log['output']['choices'][0]['message'])]
        if group['offset'] != len(messages):
            raise ValueError('HAL context offset differs from its recorded history')
        for reference in group.get('tool_sources', []):
            target = history[reference['message']]
            if target.get('role') != 'tool' or target.get('tool_call_id'):
                raise ValueError('HAL tool reference does not identify an ID-less recorded result')
            call = history[reference['assistant']]['tool_calls'][reference['call']]
            if target.get('name') and target['name'] != call['function']['name']:
                raise ValueError('HAL tool reference points at another recorded function')
            target['tool_call'] = deepcopy(call)
        positions = set()
        for reference in group['replies']:
            original = source['logs'][reference['log']]
            position = reference['message']
            if position != len(original['inputs']['messages']):
                raise ValueError('HAL reply has an incorrect source position')
            if position in positions:
                raise ValueError('Distinct HAL replies were collapsed onto one step')
            positions.add(position)
            output = original['output']
            identity = output.get('id') or ('log', reference['log'])
            if identity in represented:
                raise ValueError('HAL provider call was represented twice')
            represented.add(identity)
            history[position].update({k:deepcopy(v) for k,v in output['choices'][0]['message'].items()
                                      if v is not None and not (k == 'tool_calls' and not v)})
            for key in ['model', 'usage']:
                if output.get(key) is not None:
                    history[position][key] = deepcopy(output[key])
            for key in ['started_at', 'ended_at']:
                if original.get(key) is not None:
                    history[position][key] = original[key]
        if len(groups) > 1:
            for message in history:
                message['agent'] = f'HAL context {number}'
        messages.extend(history)
        if 'simulator_log' in group:
            original = source['logs'][group['simulator_log']]
            if original['inputs']['messages'][-1]['content'] != history[-1].get('content'):
                raise ValueError('HAL simulator reply is associated with the wrong agent reply')
            reply = original['output']['choices'][0]['message']
            message = dict(role='user', content=reply.get('content'))
            if history[-1].get('agent'):
                message['agent'] = history[-1]['agent']
            messages.append(message)
    expected = set()
    model = source['configuration']['agent_args']['model_name'].removeprefix('openai/')
    for index, log in enumerate(source['logs']):
        output, inputs = log.get('output'), log.get('inputs') or {}
        if not isinstance(output, dict) or not isinstance(inputs.get('messages'), list):
            continue
        models = {str(inputs.get('model')).removeprefix('openai/'), str(output.get('model')).removeprefix('openai/')}
        choices = output.get('choices') or []
        if model in models and choices and isinstance(choices[0].get('message'), dict):
            expected.add(output.get('id') or ('log', index))
    if expected != represented or value['native'].get('hal_recorded_reply_count') != len(expected):
        raise ValueError('HAL recorded provider-call coverage differs from its source')
    audit_messages(value, messages, item_content=item_content)
    return source
