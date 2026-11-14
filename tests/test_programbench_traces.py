"""ProgramBench-specific evidence survives the shared trace/1 mapping."""

import copy

import pytest

from trace_builder_fixtures import build_trace

from scripts.curate_benchmarks.native_result_audits import _programbench_trace1


def source(dialect):
    messages = [{"role": "system", "content": "Original system"}, {"role": "user", "content": "Original task"}]
    if dialect == "chat":
        messages += [
            {"role": "assistant", "content": "Checking", "created_at": 100, "completed_at": 102,
             "tool_calls": [{"id": "c1", "type": "function", "extra_content": {"google": {"thought_signature": "opaque signature"}},
                             "function": {"name": "bash", "arguments": '{"command":"ls"}'}}],
             "extra": {"response": {"created": 101, "created_at": 100.5, "completed_at": 101.5,
                                     "model": "fixture", "choices": [{"message": {"role": "assistant"}}]}}},
            {"role": "tool", "tool_call_id": "c1", "content": "full tool output\n"}]
    elif dialect == "anthropic":
        messages += [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Original reasoning", "signature": "opaque signature"},
                {"type": "text", "text": "Checking"}, {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "full tool output\n",
                                             "text": "separate client text", "is_error": False}]}]
    else:
        messages += [
            {"object": "response", "model": "fixture", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Checking"}]},
                {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": '{"command":"ls"}'}]},
            {"type": "function_call_output", "call_id": "c1", "output": "full tool output\n"}]
    return {"source_run": "fixture_run", "task_id": "fixture_task", "published_tests": {"suite/test": True},
            "trajectory": {"messages": messages, "info": {"mini_version": "fixture", "config": {"original": True, "agent": {},
                "model": {"model_name": "fixture"}, "environment": {"image": "fixture", "cwd": "/workspace"}}}},
            "evaluation": {"test_results": [
                {"branch": "suite", "name": "nested/test", "status": "failure", "extra": {"retry": "earlier attempt"}},
                {"branch": "suite", "name": "nested/test", "status": "passed", "extra": {"retry": "final attempt"}}]}}


@pytest.mark.parametrize("dialect", ["chat", "anthropic", "responses"])
def test_dialects_retries_tool_links_and_metadata(dialect):
    native = source(dialect)
    before = copy.deepcopy(native)
    result = build_trace("programbench", native)
    assert native == before
    _programbench_trace1(result.value, native)
    assert result.value["evaluation"]["tests"] == [
        {"name": "suite/nested/test", "status": "failure"}, {"name": "suite/nested/test", "status": "passed"}]
    assert not result.counts["unrecognized"]
    assert not result.counts["unmatched_tool_results"]
    if dialect == "chat":
        client = result.value["native"]["client_metadata"]["2"]
        assert client["created_at"] == 100 and client["response_created"] == 101
        assert client["response_created_at"] == 100.5 and client["response_completed_at"] == 101.5
        assert client["tool_calls"]["0"]["extra_content"]["google"]["thought_signature"] == "opaque signature"
    if dialect == "anthropic":
        assert result.value["native"]["client_metadata"]["2"]["content_blocks"]["0"]["signature"] == "opaque signature"


@pytest.mark.parametrize("graded", [True, False])
def test_empty_published_verdicts_and_missing_grade_remain_distinct(graded):
    native = source("chat")
    native["published_tests"] = {} if graded else None
    native["evaluation"] = {"test_results": [], "error_code": "compile_failed"} if graded else None
    result = build_trace("programbench", native)
    _programbench_trace1(result.value, native)
    assert ("evaluation" in result.value) == graded
    assert ("published_tests" in result.value["native"]) == graded


def test_format_error_retains_response_timestamps_when_step_timing_is_clamped():
    native = source("responses")
    native["trajectory"]["messages"][2]["created_at"] = 9000.111
    native["trajectory"]["messages"].append({
        "role": "user", "content": "A tool call is required.",
        "extra": {"response": {"object": "response", "created_at": 1234.567, "completed_at": 1234.678,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Original rejected reply"}]}]}},
    })
    result = build_trace("programbench", native)
    _programbench_trace1(result.value, native)
    client = result.value["native"]["client_metadata"]["4"]
    assert client["response_created_at"] == 1234.567
    assert client["response_completed_at"] == 1234.678
