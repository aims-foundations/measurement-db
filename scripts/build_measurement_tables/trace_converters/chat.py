"""Recorded chat messages to trace/1, without inventing tool IDs or turns.

The native projection retains every source field that is not displayed. Small
field references replace mapped values, allowing an independent audit to recover
the original messages, including nulls, content blocks and provider extensions.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import json

from ..trace_format import trace

CONVERTER = "chat@1"


def _call(source):
    function = source.get("function")
    if isinstance(function, dict):
        if 'arguments' not in function:
            raise ValueError('A recorded chat tool call has no arguments field')
        name, arguments = function.get("name"), function.get("arguments")
        name_path, args_path = ["function", "name"], ["function", "arguments"]
    else:
        if 'args' not in source:
            raise ValueError('A recorded chat tool call has no args field')
        name, arguments = function, source.get("args")
        name_path, args_path = ["function"], ["args"]
    if not isinstance(name, str) or not name:
        raise ValueError("A chat tool call needs its recorded function name")
    value = {"name": name, "arguments": arguments if isinstance(arguments, str) else
             json.dumps(arguments, ensure_ascii=False, allow_nan=False)}
    if source.get("id") not in (None, ''):
        if not isinstance(source["id"], str):
            raise TypeError("Recorded chat tool IDs must be strings")
        value["id"] = source["id"]
    return value, name_path, args_path


def _time(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("A recorded chat timestamp needs its timezone")
        return parsed.timestamp()
    raise TypeError("Unsupported recorded chat timestamp")


def _same_arguments(left, right):
    if left == right:
        return True
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return False


def convert(messages, *, context=None, user_participant=None, native=None, run=None, model=None):
    """Return (trace, counters); caller owns benchmark envelopes and grading."""
    if not isinstance(messages, list):
        raise TypeError("Chat messages must be a list")
    steps, pending, omitted = [], [], []
    counts = Counter(messages=len(messages))
    first_user, timed, previous_agent = True, False, None
    for index, message in enumerate(messages):
        tool_match = None
        agent = message.get('agent') if isinstance(message, dict) else None
        if agent != previous_agent:
            pending.clear()
            first_user = True
        previous_agent = agent
        residual, fields = deepcopy(message), []

        def project(source_path, target_path, *, encoding=None, span=None):
            parent = residual
            for key in source_path[:-1]:
                parent = parent[key]
            key = source_path[-1]
            if isinstance(parent, dict) and key not in parent:
                return
            if isinstance(parent, list):
                parent[key] = None
            else:
                del parent[key]
            field = {"source": source_path, "target": target_path}
            if encoding:
                field["encoding"] = encoding
            if span is not None:
                field["span"] = span
            fields.append(field)

        def content(target):
            value = message.get("content")
            if isinstance(value, str):
                project(["content"], [target])
                return value
            if value is None:
                return ""
            if not isinstance(value, list):
                raise TypeError("Unsupported chat content")
            parts = []
            offset = 0
            for i, block in enumerate(value):
                if isinstance(block, str):
                    text, path = block, ["content", i]
                elif isinstance(block, dict) and block.get("type") == "text":
                    key = "content" if "content" in block else "text"
                    text, path = block[key], ["content", i, key]
                else:
                    raise ValueError("Unsupported chat content block; add an explicit mapping")
                if not isinstance(text, str):
                    raise TypeError("A chat text block must contain text")
                project(path, [target], span=[offset, len(text)])
                parts.append(text)
                offset += len(text) + 1
            return "\n".join(parts)

        role = message.get("role") if isinstance(message, dict) else None
        if role == "assistant":
            step = {"type": "llm"}
            if model is not None:
                step['model'] = model
            if message.get("content") is not None:
                step["content"] = content("content")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise TypeError("assistant.tool_calls must be a list")
            mapped = []
            for i, call in enumerate(calls):
                value, name_path, args_path = _call(call)
                mapped.append(value)
                project(["tool_calls", i, *name_path], ["toolCalls", i, "name"])
                args = call["function"].get("arguments") if isinstance(call.get("function"), dict) else call.get("args")
                project(["tool_calls", i, *args_path], ["toolCalls", i, "arguments"],
                        encoding=None if isinstance(args, str) else "json")
                if "id" in value:
                    project(["tool_calls", i, "id"], ["toolCalls", i, "id"])
                pending.append(value)
            if mapped:
                step["toolCalls"] = mapped
            counts["tool_calls"] += len(mapped)
            for key in ["model", "reasoning_content"]:
                if isinstance(message.get(key), str):
                    target = "reasoning" if key == "reasoning_content" else key
                    step[target] = message[key]
                    project([key], [target])
            usage = message.get("usage") or {}
            for key, target in [("prompt_tokens", "prompt"), ("completion_tokens", "completion")]:
                if usage.get(key) is not None:
                    step.setdefault("usage", {})[target] = usage[key]
                    project(["usage", key], ["usage", target])
            for detail, key, target in [('completion_tokens_details', 'reasoning_tokens', 'reasoning'),
                                         ('prompt_tokens_details', 'cached_tokens', 'cacheRead')]:
                if isinstance(usage.get(detail), dict) and usage[detail].get(key) is not None:
                    step.setdefault('usage', {})[target] = usage[detail][key]
                    project(['usage', detail, key], ['usage', target])
            for key, target in [("started_at", "start"), ("ended_at", "end")]:
                if message.get(key) is not None:
                    step[target] = _time(message[key])
                    timed = True
            counts["llm"] += 1
        elif role in {"tool", "function"}:
            descriptor = message.get("tool_call")
            supplied = _call(descriptor)[0] if isinstance(descriptor, dict) else None
            call_id = message.get("tool_call_id") or None
            name = message.get("name") or (supplied or {}).get("name")
            matches = [(i, call) for i, call in enumerate(pending)
                       if (call_id is None or call.get("id") == call_id)
                       and (name is None or call.get("name") == name)
                       and (supplied is None or _same_arguments(call["arguments"], supplied["arguments"]))]
            identical = (len(matches) > 1 and call_id is None
                         and all(call.get('name') == matches[0][1].get('name')
                                 and _same_arguments(call['arguments'], matches[0][1]['arguments'])
                                 for _, call in matches))
            if identical:
                # Identical ID-less calls have no individual identity in the
                # source. Preserve their order and report that fact explicitly.
                counts['indistinguishable_tool_results'] += 1
                tool_match = 'indistinguishable'
            call = pending.pop(matches[0][0]) if len(matches) == 1 or identical else None
            if call is None:
                counts["unmatched_tool_results"] += 1
                tool_match = 'unmatched'
            step = {"type": "tool", "output": content("output")}
            for key, value in [("callId", call_id), ("name", name or (call or {}).get("name")),
                               ("arguments", (supplied or call or {}).get("arguments"))]:
                if value is not None:
                    step[key] = value
            if call_id is not None:
                project(["tool_call_id"], ["callId"])
            if message.get("name") is not None:
                project(["name"], ["name"])
            if supplied:
                _, name_path, args_path = _call(descriptor)
                project(["tool_call", *name_path], ["name"])
                args = descriptor.get("function", {}).get("arguments") if isinstance(descriptor.get("function"), dict) else descriptor.get("args")
                project(["tool_call", *args_path], ["arguments"], encoding=None if isinstance(args, str) else "json")
                if descriptor.get("id") is not None and call_id == descriptor["id"]:
                    project(["tool_call", "id"], ["callId"])
            if message.get("error"):
                step["error"] = str(message["error"])
                if isinstance(message["error"], str):
                    project(["error"], ["error"])
            counts["tool"] += 1
        elif role in {"system", "developer", "user"}:
            label = "system" if role in {"system", "developer"} else "user"
            step = {"type": "message", "label": label, "content": content("content")}
            if (role == 'user' and first_user and not user_participant and context is not None
                    and step['content'] == context.item_content):
                step['label'] = 'task'
            if role == "user" and user_participant:
                step["participant"] = user_participant
            if role == "user":
                first_user = False
        else:
            step = {"type": "event", "name": "unrecognized"}
            counts["unrecognized"] += 1
        projection = {"index": index, "source": residual, "fields": fields}
        if tool_match:
            projection['tool_match'] = tool_match
        if agent is not None:
            step['agent'] = agent
            project(['agent'], ['agent'])
        step["native"] = {"chat": projection}
        if (step.get("label") == "task" and context is not None
                and step["content"] == context.item_content):
            omitted.append(projection)
        else:
            steps.append(step)
    rest = deepcopy(native or {})
    if omitted:
        rest["chat_omitted_tasks"] = omitted
    return trace(CONVERTER, steps, timing="recorded" if timed else None, native=rest, run=run), counts
