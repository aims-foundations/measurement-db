"""trace/1 converters: mapping, schema validity and the cost rule."""
from scripts.build_measurement_tables import trace_format
from scripts.build_measurement_tables.trace_converters import mini_swe_agent


def chat_run(cost, instance_cost, prompt_tokens=120):
    """A two-step chat-completions run: one tool call and its result, then exit."""
    response = {"model": "m-1", "created": 100.0,
                "choices": [{"message": {"role": "assistant"}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 8}}
    return {
        "info": {"model_stats": {"instance_cost": instance_cost, "api_calls": 1}},
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Build it."},
            {"role": "assistant", "content": "Listing files.",
             "tool_calls": [{"id": "c1", "function": {"name": "bash",
                                                      "arguments": '{"command": "ls"}'}}],
             "extra": {"response": response, "cost": cost, "timestamp": 101.0}},
            {"role": "tool", "tool_call_id": "c1",
             "content": "<returncode>0</returncode>\n<output>\nmain.go\n</output>\n",
             "extra": {"returncode": 0, "timestamp": 102.0}},
            {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
        ],
    }


def convert(run):
    value, counts = mini_swe_agent.convert(run, {"test_results": [
        {"name": "t1", "status": "passed"}, {"name": "t2", "status": "failure"}]})
    trace_format.validate(value)
    return value, counts


def test_chat_run_maps_steps_run_and_evaluation():
    value, counts = convert(chat_run(cost=0.25, instance_cost=0.25))
    assert [s["type"] for s in value["steps"]] == ["message", "message", "llm", "tool"]
    assert [s.get("label") for s in value["steps"][:2]] == ["system", "task"]
    llm, tool = value["steps"][2], value["steps"][3]
    assert llm["toolCalls"] == [{"id": "c1", "name": "bash", "arguments": '{"command": "ls"}'}]
    assert llm["cost"] == 0.25 and llm["usage"] == {"prompt": 120, "completion": 8}
    assert (tool["output"], tool["exitCode"], tool["name"]) == ("main.go\n", 0, "bash")
    assert (llm["start"], llm["end"], tool["start"], tool["end"]) == (100.0, 101.0, 101.0, 102.0)
    assert value["run"] == {"status": "Submitted", "cost": 0.25, "apiCalls": 1, "promptTokens": 120,
                            "completionTokens": 8, "durationS": 2.0}
    assert value["evaluation"]["passed"] == 1 and value["evaluation"]["total"] == 2
    assert counts["unmatched_tool_results"] == 0 and counts["unrecognized"] == 0


def test_zero_cost_with_tokens_is_not_recorded_but_kept_in_native():
    value, _ = convert(chat_run(cost=0, instance_cost=0))
    llm = value["steps"][2]
    assert "cost" not in llm and llm["native"]["extra"]["cost"] == 0
    assert "cost" not in value["run"]
    assert value["native"]["model_stats"]["instance_cost"] == 0


def test_cost_is_recorded_unless_zero_with_tokens():
    recorded = mini_swe_agent._recorded_cost
    assert recorded(0.5, (10,))
    assert recorded(0, ())             # nothing says the call used tokens
    assert recorded(0, (None, 0))
    assert not recorded(0, (None, 7))  # used tokens, but priced at 0
    assert not recorded(None, (7,))
