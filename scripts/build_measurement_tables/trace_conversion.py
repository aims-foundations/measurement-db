"""Explicit source-format converters shared by benchmark builders.

Call ``convert_trace(source, family=...)`` where a builder prepares its traces.
Keep benchmark-specific preparation in build.py; this module owns format parsing.
The build pipeline validates and serializes each ConvertedTrace at its final
observation identity, retaining counters and the original source evidence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from typing import Literal

from .trace_format import (
    ConvertedTrace, as_json_text, compact, to_text, trace, unrecognized, unused,
)

TraceFamily = Literal['mini_swe_agent', 'chat', 'openhands', 'browsergym', 'text', 'pairwise', 'nyu_ctf']

# Source Fields

def _source_fields_get(value, path):
    for key in path:
        value = value[key]
    return value


class SourceTrace:
    def __init__(self, source, converter, *, json_encoded=False):
        self.source = json.loads(source) if json_encoded else source
        self.native = {'source': deepcopy(source), 'source_encoding': 'json' if json_encoded else 'native'}
        self.converter, self.steps = converter, []
        self.sections = {}

    def project(self, step, target, path, *, json_text=False):
        value = _source_fields_get(self.source, path)
        encoding = 'json' if json_text and not isinstance(value, str) else None
        displayed = json.dumps(value, ensure_ascii=False, allow_nan=False) if encoding else value
        keys = [target] if isinstance(target, str) else target
        parent = step
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = displayed
        reference = {'source': path, 'target': keys}
        if encoding:
            reference['encoding'] = encoding
        step.setdefault('native', {}).setdefault('source_fields', []).append(reference)
        return displayed

    def add(self, step, **provenance):
        if provenance:
            step.setdefault('native', {}).update(provenance)
        self.steps.append(step)
        return step

    def project_section(self, name, target, path, *, json_text=False):
        if name not in {'run', 'evaluation'}:
            raise ValueError('Unsupported trace section')
        section = self.sections.setdefault(name, {})
        keys = [target] if isinstance(target, str) else target
        parent = section
        for key in keys[:-1]:
            parent = parent.setdefault(key, {})
        value = self.project(section, keys, path, json_text=json_text)
        references = section.pop('native')['source_fields']
        for reference in references:
            reference['target'] = [name, *reference['target']]
        self.native.setdefault('source_fields', []).extend(references)
        return value

    def value(self, **kwargs):
        if self.sections.keys() & kwargs.keys():
            raise ValueError('A trace section cannot replace projected source fields')
        return trace(self.converter, self.steps, native=self.native, **self.sections, **kwargs)


# Mini Swe Agent

_MINI_SWE_AGENT_CONVERTER = "mini_swe_agent@1"
_MINI_SWE_AGENT_OUTPUT = re.compile(r"\A<returncode>(-?\d+)</returncode>\n<output>\n?(.*)</output>\n?\Z", re.S)
# `extra` keys the mapping reads; everything else in `extra` is kept in `native`.
_MINI_SWE_AGENT_EXTRA_USED = {"response", "cost", "timestamp", "observations", "returncode", "exception_info",
              "exception", "exit_status", "submission", "exception_str", "traceback"}


def _mini_swe_agent_usage(prompt, completion, reasoning=None, cache_read=None, cache_write=None):
    return compact(dict(prompt=prompt, completion=completion, reasoning=reasoning,
                        cacheRead=cache_read, cacheWrite=cache_write)) or None


def _mini_swe_agent_native(**parts):
    return compact({k: v for k, v in parts.items() if v}) or None


def _mini_swe_agent_call(call_id, name, arguments):
    return compact(dict(id=call_id, name=name, arguments=as_json_text(arguments)))


def _mini_swe_agent_chat(m: dict, response: dict, end):
    """Chat-completions turn. Returns (step, start hint)."""
    choices = response.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    usage = response.get("usage") or {}
    step = compact(dict(
        type="llm", model=response.get("model"), content=to_text(m.get("content")),
        reasoning=to_text(message.get("reasoning_content")),
        toolCalls=[_mini_swe_agent_call(c.get("id"), (c.get("function") or {}).get("name"),
                         (c.get("function") or {}).get("arguments")) for c in m.get("tool_calls") or []],
        usage=_mini_swe_agent_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
        end=end))
    rest_choices = []
    for choice in choices:
        if not choice:
            continue
        message_rest = unused(choice.get("message") or {}, {"content", "tool_calls", "reasoning_content", "role"})
        rest_choices.append(compact({**(unused(choice, {"message"}) or {}), "message": message_rest}))
    step["native"] = _mini_swe_agent_native(message=unused(m, {"role", "content", "tool_calls", "extra"}),
                             response=unused(response, {"choices", "model", "created"}),
                             choices=[c for c in rest_choices if c])
    return compact(step), response.get("created")


def _mini_swe_agent_responses(m: dict, end):
    """OpenAI Responses API object. Returns (step, start hint)."""
    text, reasoning, calls, rest_items = [], [], [], []
    for item in m.get("output") or []:
        kind = item.get("type")
        if kind == "function_call":
            calls.append(_mini_swe_agent_call(item.get("call_id"), item.get("name"), item.get("arguments")))
            rest_items.append(unused(item, {"type", "call_id", "name", "arguments"}))
        elif kind == "message":
            text.append(to_text(item.get("content")))
            rest_items.append(unused(item, {"type", "content"}))
        elif kind == "reasoning":
            value = to_text(item.get("summary")) + to_text(item.get("content"))
            if value:
                reasoning.append(value)
            rest_items.append(unused(item, {"type", "summary", "content"}))
        else:
            rest_items.append(item)  # kept whole: not mapped
    usage = m.get("usage") or {}
    step = compact(dict(
        type="llm", model=m.get("model"), content="\n".join(x for x in text if x),
        reasoning="\n".join(reasoning), toolCalls=calls,
        usage=_mini_swe_agent_usage(usage.get("input_tokens"), usage.get("output_tokens"),
                     (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                     (usage.get("input_tokens_details") or {}).get("cached_tokens")),
        end=end if end is not None else m.get("completed_at")))
    step["native"] = _mini_swe_agent_native(response=unused(m, {"object", "output", "model", "created_at", "completed_at",
                                                 "extra"}), output=[x for x in rest_items if x])
    return compact(step), m.get("created_at")


def _mini_swe_agent_anthropic(m: dict, response: dict, end):
    """Anthropic Messages turn (content blocks). Returns (step, start hint)."""
    content = m.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": to_text(content)}]
    text, reasoning, calls = [], [], []
    for block in blocks:
        kind = block.get("type")
        if kind == "tool_use":
            calls.append(_mini_swe_agent_call(block.get("id"), block.get("name"), block.get("input")))
        elif kind == "thinking":
            reasoning.append(block.get("thinking") or "")
        elif kind == "redacted_thinking":
            reasoning.append("[redacted thinking]")
        else:
            text.append(to_text([block]))
    usage = response.get("usage") or {}
    read, write = usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens")
    prompt = usage.get("input_tokens")
    if prompt is not None:  # Anthropic's input_tokens excludes cached tokens
        prompt += (read or 0) + (write or 0)
    step = compact(dict(
        type="llm", model=response.get("model"), content="\n".join(x for x in text if x),
        reasoning="\n".join(x for x in reasoning if x), toolCalls=calls,
        usage=_mini_swe_agent_usage(prompt, usage.get("output_tokens"), None, read, write), end=end))
    step["native"] = _mini_swe_agent_native(message=unused(m, {"role", "content", "extra"}),
                             response=unused(response, {"content", "model"}))
    return compact(step), None


def _mini_swe_agent_tool(call_id, output: str, extra: dict, calls: dict, counts: Counter, native=None):
    match = _MINI_SWE_AGENT_OUTPUT.match(output)
    if match:
        output = match.group(2)
    call = calls.get(call_id)
    if call is None:
        counts["unmatched_tool_results"] += 1
    counts["tool"] += 1
    return compact(dict(
        type="tool", callId=call_id, name=(call or {}).get("name"), arguments=(call or {}).get("arguments"),
        output=output, exitCode=extra.get("returncode"),
        error=to_text(extra.get("exception_info") or extra.get("exception")) or None,
        end=extra.get("timestamp"), native=native)) | {"output": output}


def _mini_swe_agent_recorded_cost(cost, tokens) -> bool:
    """A cost of 0 for a call that used tokens means the harness did not price it
    (e.g. a model missing from its price table), not that it was free."""
    return cost is not None and not (cost == 0 and any(n for n in tokens if isinstance(n, (int, float))))


def _mini_swe_agent(trajectory: dict, evaluation: dict | None = None, native: dict | None = None
            ) -> tuple[dict, Counter]:
    """Convert one mini-swe-agent trajectory (+ its evaluation record) to trace/1."""
    counts: Counter = Counter()
    steps, hints, calls = [], {}, {}
    seen_task = False
    exit_extra: dict = {}
    exit_rest = None

    def add_llm(pair, cost=None, extra_rest=None):
        step, hint = pair
        if _mini_swe_agent_recorded_cost(cost, step.get("usage", {}).values()):
            step["cost"] = cost
        elif cost is not None:
            extra_rest = {**(extra_rest or {}), "cost": cost}
        if extra_rest:
            step["native"] = {**step.get("native", {}), "extra": extra_rest}
        for call in step.get("toolCalls", []):
            if "id" in call:
                calls[call["id"]] = call
        hints[len(steps)] = hint
        steps.append(step)
        counts["llm"] += 1

    for m in trajectory.get("messages") or []:
        counts["messages"] += 1
        role, extra = m.get("role"), m.get("extra") or {}
        extra_rest = unused(extra, _MINI_SWE_AGENT_EXTRA_USED)
        content = m.get("content")
        if m.get("object") == "response":
            add_llm(_mini_swe_agent_responses(m, extra.get("timestamp")), extra.get("cost"), extra_rest)
        elif m.get("type") == "function_call_output":
            steps.append(_mini_swe_agent_tool(m.get("call_id"), to_text(m.get("output")), extra, calls, counts,
                               _mini_swe_agent_native(message=unused(m, {"type", "call_id", "output", "extra"}), extra=extra_rest)))
        elif role == "assistant":
            parse = _mini_swe_agent_chat if "tool_calls" in m or not isinstance(content, list) else _mini_swe_agent_anthropic
            add_llm(parse(m, extra.get("response") or {}, extra.get("timestamp")), extra.get("cost"), extra_rest)
        elif role == "tool":
            steps.append(_mini_swe_agent_tool(m.get("tool_call_id"), to_text(content), extra, calls, counts,
                               _mini_swe_agent_native(message=unused(m, {"role", "content", "tool_call_id", "extra"}),
                                       extra=extra_rest)))
        elif role == "user" and isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            observations = extra.get("observations") or []
            index = 0
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    observation = observations[index] if index < len(observations) else {}
                    index += 1
                    steps.append(_mini_swe_agent_tool(block.get("tool_use_id"), to_text(block.get("content")), observation,
                                       calls, counts, _mini_swe_agent_native(block=unused(block, {"type", "tool_use_id", "content"}),
                                                              observation=unused(observation, _MINI_SWE_AGENT_EXTRA_USED))))
                else:
                    steps.append(compact(dict(type="message", label="harness", content=to_text([block]))))
        elif role in ("system", "user"):
            response = extra.get("response")
            if isinstance(response, dict):
                # The model answered without a tool call; the reply is stored on the harness's
                # format-error message instead of as an assistant message.
                counts["format_error_replies"] += 1
                if response.get("object") == "response":
                    add_llm(_mini_swe_agent_responses(response, None))
                else:
                    message = ((response.get("choices") or [{}])[0] or {}).get("message") or {}
                    add_llm(_mini_swe_agent_chat(message, response, None))
            label = "system" if role == "system" else ("harness" if seen_task else "task")
            seen_task = seen_task or role == "user"
            steps.append(compact(dict(type="message", label=label, content=to_text(content),
                                      native=_mini_swe_agent_native(message=unused(m, {"role", "content", "extra"}),
                                                     extra=extra_rest))) | {"content": to_text(content)})
        elif role == "exit":
            exit_extra = extra
            exit_rest = _mini_swe_agent_native(message=unused(m, {"role", "content", "extra"}), extra=extra_rest,
                                content=content or None)
        else:
            steps.append(unrecognized(m, counts))

    # Timing: each step starts where the previous one ended and ends at its own timestamp.
    previous_end = None
    times = []
    for i, step in enumerate(steps):
        if step["type"] not in ("llm", "tool"):
            continue
        end = step.get("end")
        start = previous_end if previous_end is not None else hints.get(i)
        if end is None:
            end = start
        if start is None:
            start = end
        if start is not None and end is not None and start > end:
            start = end
        if start is not None:
            step["start"] = start
        if end is not None:
            step["end"] = end
            previous_end = end
        times += [t for t in (start, end) if t is not None]

    info = trajectory.get("info") or {}
    stats = info.get("model_stats") or {}
    llm = [s for s in steps if s["type"] == "llm"]
    prompt_tokens = sum(s.get("usage", {}).get("prompt", 0) for s in llm) or None
    completion_tokens = sum(s.get("usage", {}).get("completion", 0) for s in llm) or None
    cost = stats.get("instance_cost")
    cost_recorded = _mini_swe_agent_recorded_cost(cost, (prompt_tokens, completion_tokens))
    run = dict(
        status=exit_extra.get("exit_status") or info.get("exit_status"),
        submission=exit_extra.get("submission"), exception=exit_extra.get("exception_str"),
        traceback=exit_extra.get("traceback"), cost=cost if cost_recorded else None,
        apiCalls=stats.get("api_calls"), promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        durationS=(max(times) - min(times)) if times else None)

    graded = None
    evaluation_rest = None
    if isinstance(evaluation, dict):
        tests = [dict(name=str(t.get("name")), status=str(t.get("status")))
                 for t in evaluation.get("test_results") or []]
        graded = dict(errorCode=evaluation.get("error_code"), errorDetails=evaluation.get("error_details"),
                      passed=sum(t["status"] == "passed" for t in tests), total=len(tests),
                      notRun=sum(t["status"] == "not_run" for t in tests), tests=tests)
        evaluation_rest = unused(evaluation, {"test_results", "error_code", "error_details"})
    if graded is not None:
        graded = {k: v for k, v in graded.items() if v is not None and v != ""}

    rest = _mini_swe_agent_native(
        **(native or {}), trajectory=unused(trajectory, {"messages", "info"}),
        info=unused(info, {"model_stats", "exit_status"}),
        model_stats=unused(stats, {"instance_cost", "api_calls"} if cost_recorded else {"api_calls"}), exit=exit_rest,
        evaluation=evaluation_rest)
    return trace(_MINI_SWE_AGENT_CONVERTER, steps, timing="recorded" if times else None, run=run,
                 evaluation=graded, native=rest), counts


# Chat

_CHAT_CONVERTER = "chat@1"


def _chat_call(source):
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


def _chat_time(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("A recorded chat timestamp needs its timezone")
        return parsed.timestamp()
    raise TypeError("Unsupported recorded chat timestamp")


def _chat_same_arguments(left, right):
    if left == right:
        return True
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return False


def _chat(messages, *, item_content=None, user_participant=None, native=None, run=None, model=None):
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
                value, name_path, args_path = _chat_call(call)
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
                    step[target] = _chat_time(message[key])
                    timed = True
            counts["llm"] += 1
        elif role in {"tool", "function"}:
            descriptor = message.get("tool_call")
            supplied = _chat_call(descriptor)[0] if isinstance(descriptor, dict) else None
            call_id = message.get("tool_call_id") or None
            name = message.get("name") or (supplied or {}).get("name")
            matches = [(i, call) for i, call in enumerate(pending)
                       if (call_id is None or call.get("id") == call_id)
                       and (name is None or call.get("name") == name)
                       and (supplied is None or _chat_same_arguments(call["arguments"], supplied["arguments"]))]
            identical = (len(matches) > 1 and call_id is None
                         and all(call.get('name') == matches[0][1].get('name')
                                 and _chat_same_arguments(call['arguments'], matches[0][1]['arguments'])
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
                _, name_path, args_path = _chat_call(descriptor)
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
            if (role == 'user' and first_user and not user_participant and item_content is not None
                    and step['content'] == item_content):
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
        if (step.get("label") == "task" and item_content is not None
                and step["content"] == item_content):
            omitted.append(projection)
        else:
            steps.append(step)
    rest = deepcopy(native or {})
    if omitted:
        rest["chat_omitted_tasks"] = omitted
    return trace(_CHAT_CONVERTER, steps, timing="recorded" if timed else None, native=rest, run=run), counts


# Openhands

_OPENHANDS_CONVERTER = 'openhands@1'
_OPENHANDS_ACTIONS = {'run', 'run_ipython', 'browse', 'browse_interactive', 'read', 'edit',
           'search', 'think', 'finish', 'call_tool_mcp'}
_OPENHANDS_CONTROL = {'recall', 'condense', 'condensation'}


def _openhands(source):
    out = SourceTrace(source, _OPENHANDS_CONVERTER, json_encoded=isinstance(source, str))
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
        if event.get('action') in _OPENHANDS_ACTIONS:
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
        elif action in _OPENHANDS_CONTROL or observation in _OPENHANDS_CONTROL:
            step = {'type': 'event', 'name': action or observation}
            if event.get('message') is not None:
                project_text(step, 'content', [index, 'message'])
            out.add(step, openhands={'events': [index]})
        elif action in _OPENHANDS_ACTIONS or action == 'message' and event.get('source') == 'agent':
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


# Browsergym

_BROWSERGYM_CONVERTER = 'browsergym@1'


def _browsergym(source):
    out = SourceTrace(source, _BROWSERGYM_CONVERTER, json_encoded=isinstance(source, str))
    record = out.source['record']
    if not isinstance(record.get('steps'), list):
        raise TypeError('BrowserGym requires the recorded step list')
    counts = Counter(source_steps=len(record['steps']))
    for number, native in enumerate(record['steps']):
        prefix = ['record', 'steps', number]
        # These snapshots precede the action in the same saved step. The error
        # field explicitly refers to the previous action, including at the end.
        observation = {'type': 'event', 'name': 'browser_observation'}
        if native.get('url') is not None:
            out.project(observation, 'content', prefix + ['url'])
        out.add(observation, browsergym={'step': number, 'kind': 'observation'})
        counts['observations'] += 1
        if native.get('last_action_error'):
            error = {'type': 'event', 'name': 'last_action_error'}
            out.project(error, 'content', prefix + ['last_action_error'])
            out.add(error, browsergym={'step': number, 'kind': 'last_action_error'})
            counts['action_errors'] += 1
        if native.get('action') is None and native.get('reasoning') is None:
            continue
        step = {'type': 'llm'}
        for key, target in [('action', 'content'), ('reasoning', 'reasoning')]:
            if native.get(key) is not None:
                out.project(step, target, prefix + [key], json_text=True)
        if record.get('model') is not None:
            out.project(step, 'model', ['record', 'model'])
        stats = native.get('stats') or {}
        for key, target in [('input_tokens', 'prompt'), ('output_tokens', 'completion')]:
            if stats.get(key) is not None:
                step.setdefault('usage', {})
                out.project(step, ['usage', target], prefix + ['stats', key])
        if stats.get('cost') is not None:
            if stats['cost'] == 0 and any(step.get('usage', {}).values()):
                counts['zero_cost_with_tokens_omitted'] += 1
            else:
                out.project(step, 'cost', prefix + ['stats', 'cost'])
        out.add(step, browsergym={'step': number, 'kind': 'agent'})
        counts['llm'] += 1
    summary = record.get('summary_info') or {}
    for key, target in [('err_msg', 'exception'), ('stack_trace', 'traceback')]:
        if summary.get(key):
            out.project_section('run', target, ['record', 'summary_info', key], json_text=True)
    return out.value(), counts


# Text

_TEXT_CONVERTER = 'text@1'


def _text(source):
    if not isinstance(source, str):
        raise TypeError('Text traces require the original recorded string')
    # The original input is already the complete content field. Keep its hash
    # for the independent audit instead of storing a second copy of the text.
    native = {'source_encoding': 'literal_text',
              'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest()}
    return trace(_TEXT_CONVERTER, [{'type': 'llm', 'content': source}], native=native), {'llm': 1, 'text_records': 1}


# Pairwise

_PAIRWISE_CONVERTER = 'pairwise@1'


def _pairwise(source, *, reference):
    out = SourceTrace(source, _PAIRWISE_CONVERTER, json_encoded=isinstance(source, str))
    record = out.source['native_record']
    if reference not in {record['generator_1'], record['generator_2']}:
        raise ValueError('A pairwise record does not contain its declared fixed reference')
    # This is the same subject orientation as the benchmark builder. When the
    # generator labels coincide, the fixed bank still occupies the first side.
    side = 2 if record['generator_1'] == reference else 1
    if not isinstance(record[f'output_{side}'], str):
        raise TypeError('The evaluated pairwise output must be its original recorded string')
    step = {'type': 'llm'}
    out.project(step, 'model', ['native_record', f'generator_{side}'])
    out.project(step, 'content', ['native_record', f'output_{side}'], json_text=True)
    out.add(step)
    completion = record.get('raw_completion')
    if completion is not None:
        prefix = ['native_record', 'raw_completion']
        if isinstance(completion, dict) and isinstance(completion.get('message'), dict) and completion['message'].get('content') is not None:
            path = prefix + ['message', 'content']
        elif isinstance(completion, dict) and completion.get('text') is not None:
            path = prefix + ['text']
        else:
            path = prefix
        out.project_section('evaluation', ['judge', 'content'], path, json_text=True)
        if record.get('annotator') is not None:
            out.project_section('evaluation', ['judge', 'participant'], ['native_record', 'annotator'])
    # price_per_example, time_per_example and raw_completion token counts belong
    # to the judge, not to generation of the evaluated model's answer.
    return out.value(), {'llm': 1, 'comparison_sides': 1, 'judge_outputs': int(completion is not None)}


# Nyu Ctf

def _nyu_ctf(source):
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
                    out.project(step, ['toolCalls', ordinal, 'arguments'], prefix + [('parsed_args' if 'parsed_args' in call else 'args') if calls_path else 'arguments'], json_text=True)
                    if not calls_path and call.get('id') is not None:
                        out.project(step, ['toolCalls', ordinal, 'id'], path + ['tool_calls', ordinal, 'id'])
        if message.get('agent') is not None:
            out.project(step, 'agent', path + ['agent'])
        elif component not in {'messages', 'history'}:
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


_FAMILIES = {
    'mini_swe_agent': _mini_swe_agent,
    'chat': _chat,
    'openhands': _openhands,
    'browsergym': _browsergym,
    'text': _text,
    'pairwise': _pairwise,
    'nyu_ctf': _nyu_ctf,
}


def convert_trace(source: object, *, family: TraceFamily, **options) -> ConvertedTrace:
    """Convert one source record, retaining diagnostic counts for build validation.

    Select the recorded format explicitly. Extra options are checked by the
    selected family function; an unknown family or option fails the build.
    Missing traces should remain absent rather than calling this function.
    """
    if family not in _FAMILIES:
        raise ValueError(f'Unsupported trace family: {family!r}')
    value, counts = _FAMILIES[family](source, **options)
    return ConvertedTrace(value, counts)
