"""mini-swe-agent trajectories (programbench) to trace/1.

A trajectory's messages use one of three dialects, depending on the model client of the run:
  * chat completions (LitellmModel):  assistant{tool_calls} + tool{tool_call_id}
  * OpenAI Responses (…ResponseModel): {"object": "response", output: [...]} + function_call_output
  * Anthropic (AnthropicModel):        assistant{content: [tool_use…]} + user{content: [tool_result…]}

Newer agent versions store a reply without a tool call on the harness's format-error message
(`extra.response`), in either the chat or the Responses shape; it becomes a model step followed by
the harness message.

The parsing logic was verified against all 4,965 programbench rows: model steps match the agent's
own `model_stats.api_calls` except for calls the agent never saved, every tool result matches its
call, and no message is unrecognized.
"""
from __future__ import annotations

import re
from collections import Counter

from ..trace_format import as_json_text, compact, to_text, trace, unrecognized, unused

CONVERTER = "mini_swe_agent@1"
OUTPUT = re.compile(r"\A<returncode>(-?\d+)</returncode>\n<output>\n?(.*)</output>\n?\Z", re.S)
# `extra` keys the mapping reads; everything else in `extra` is kept in `native`.
EXTRA_USED = {"response", "cost", "timestamp", "observations", "returncode", "exception_info",
              "exception", "exit_status", "submission", "exception_str", "traceback"}


def _usage(prompt, completion, reasoning=None, cache_read=None, cache_write=None):
    return compact(dict(prompt=prompt, completion=completion, reasoning=reasoning,
                        cacheRead=cache_read, cacheWrite=cache_write)) or None


def _native(**parts):
    return compact({k: v for k, v in parts.items() if v}) or None


def _call(call_id, name, arguments):
    return compact(dict(id=call_id, name=name, arguments=as_json_text(arguments)))


def _chat(m: dict, response: dict, end):
    """Chat-completions turn. Returns (step, start hint)."""
    choices = response.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    usage = response.get("usage") or {}
    step = compact(dict(
        type="llm", model=response.get("model"), content=to_text(m.get("content")),
        reasoning=to_text(message.get("reasoning_content")),
        toolCalls=[_call(c.get("id"), (c.get("function") or {}).get("name"),
                         (c.get("function") or {}).get("arguments")) for c in m.get("tool_calls") or []],
        usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
        end=end))
    rest_choices = []
    for choice in choices:
        if not choice:
            continue
        message_rest = unused(choice.get("message") or {}, {"content", "tool_calls", "reasoning_content", "role"})
        rest_choices.append(compact({**(unused(choice, {"message"}) or {}), "message": message_rest}))
    step["native"] = _native(message=unused(m, {"role", "content", "tool_calls", "extra"}),
                             response=unused(response, {"choices", "model", "created"}),
                             choices=[c for c in rest_choices if c])
    return compact(step), response.get("created")


def _responses(m: dict, end):
    """OpenAI Responses API object. Returns (step, start hint)."""
    text, reasoning, calls, rest_items = [], [], [], []
    for item in m.get("output") or []:
        kind = item.get("type")
        if kind == "function_call":
            calls.append(_call(item.get("call_id"), item.get("name"), item.get("arguments")))
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
        usage=_usage(usage.get("input_tokens"), usage.get("output_tokens"),
                     (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                     (usage.get("input_tokens_details") or {}).get("cached_tokens")),
        end=end if end is not None else m.get("completed_at")))
    step["native"] = _native(response=unused(m, {"object", "output", "model", "created_at", "completed_at",
                                                 "extra"}), output=[x for x in rest_items if x])
    return compact(step), m.get("created_at")


def _anthropic(m: dict, response: dict, end):
    """Anthropic Messages turn (content blocks). Returns (step, start hint)."""
    content = m.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": to_text(content)}]
    text, reasoning, calls = [], [], []
    for block in blocks:
        kind = block.get("type")
        if kind == "tool_use":
            calls.append(_call(block.get("id"), block.get("name"), block.get("input")))
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
        usage=_usage(prompt, usage.get("output_tokens"), None, read, write), end=end))
    step["native"] = _native(message=unused(m, {"role", "content", "extra"}),
                             response=unused(response, {"content", "model"}))
    return compact(step), None


def _tool(call_id, output: str, extra: dict, calls: dict, counts: Counter, native=None):
    match = OUTPUT.match(output)
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


def _recorded_cost(cost, tokens) -> bool:
    """A cost of 0 for a call that used tokens means the harness did not price it
    (e.g. a model missing from its price table), not that it was free."""
    return cost is not None and not (cost == 0 and any(n for n in tokens if isinstance(n, (int, float))))


def convert(trajectory: dict, evaluation: dict | None = None, native: dict | None = None
            ) -> tuple[dict, Counter]:
    """Convert one mini-swe-agent trajectory (+ its evaluation record) to trace/1."""
    counts: Counter = Counter()
    steps, hints, calls = [], {}, {}
    seen_task = False
    exit_extra: dict = {}
    exit_rest = None

    def add_llm(pair, cost=None, extra_rest=None):
        step, hint = pair
        if _recorded_cost(cost, step.get("usage", {}).values()):
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
        extra_rest = unused(extra, EXTRA_USED)
        content = m.get("content")
        if m.get("object") == "response":
            add_llm(_responses(m, extra.get("timestamp")), extra.get("cost"), extra_rest)
        elif m.get("type") == "function_call_output":
            steps.append(_tool(m.get("call_id"), to_text(m.get("output")), extra, calls, counts,
                               _native(message=unused(m, {"type", "call_id", "output", "extra"}), extra=extra_rest)))
        elif role == "assistant":
            parse = _chat if "tool_calls" in m or not isinstance(content, list) else _anthropic
            add_llm(parse(m, extra.get("response") or {}, extra.get("timestamp")), extra.get("cost"), extra_rest)
        elif role == "tool":
            steps.append(_tool(m.get("tool_call_id"), to_text(content), extra, calls, counts,
                               _native(message=unused(m, {"role", "content", "tool_call_id", "extra"}),
                                       extra=extra_rest)))
        elif role == "user" and isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            observations = extra.get("observations") or []
            index = 0
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    observation = observations[index] if index < len(observations) else {}
                    index += 1
                    steps.append(_tool(block.get("tool_use_id"), to_text(block.get("content")), observation,
                                       calls, counts, _native(block=unused(block, {"type", "tool_use_id", "content"}),
                                                              observation=unused(observation, EXTRA_USED))))
                else:
                    steps.append(compact(dict(type="message", label="harness", content=to_text([block]))))
        elif role in ("system", "user"):
            response = extra.get("response")
            if isinstance(response, dict):
                # The model answered without a tool call; the reply is stored on the harness's
                # format-error message instead of as an assistant message.
                counts["format_error_replies"] += 1
                if response.get("object") == "response":
                    add_llm(_responses(response, None))
                else:
                    message = ((response.get("choices") or [{}])[0] or {}).get("message") or {}
                    add_llm(_chat(message, response, None))
            label = "system" if role == "system" else ("harness" if seen_task else "task")
            seen_task = seen_task or role == "user"
            steps.append(compact(dict(type="message", label=label, content=to_text(content),
                                      native=_native(message=unused(m, {"role", "content", "extra"}),
                                                     extra=extra_rest))) | {"content": to_text(content)})
        elif role == "exit":
            exit_extra = extra
            exit_rest = _native(message=unused(m, {"role", "content", "extra"}), extra=extra_rest,
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
    cost_recorded = _recorded_cost(cost, (prompt_tokens, completion_tokens))
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

    rest = _native(
        **(native or {}), trajectory=unused(trajectory, {"messages", "info"}),
        info=unused(info, {"model_stats", "exit_status"}),
        model_stats=unused(stats, {"instance_cost", "api_calls"} if cost_recorded else {"api_calls"}), exit=exit_rest,
        evaluation=evaluation_rest)
    return trace(CONVERTER, steps, timing="recorded" if times else None, run=run,
                 evaluation=graded, native=rest), counts
