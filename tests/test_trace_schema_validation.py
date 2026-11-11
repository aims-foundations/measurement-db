"""Discriminated step validation retains the complete JSON Schema contract."""

import copy
import json

import jsonschema
import pytest

from scripts.build_measurement_tables import trace_format


def test_optimized_validator_agrees_with_standard_schema_on_steps_and_invalid_fields():
    standard = jsonschema.Draft202012Validator(json.loads(trace_format.SCHEMA_PATH.read_text()))
    values = []
    for step in [
        {"type": "message", "label": "task", "content": "Solve"},
        {"type": "llm", "toolCalls": [{"id": "c1", "arguments": "{}"}], "usage": {"prompt": 12}},
        {"type": "tool", "output": "Complete output"},
        {"type": "event", "name": "original", "native": {"opaque": [1, 2]}},
    ]:
        value = trace_format.trace("fixture@1", [step])
        values.append(value)
        for key, invalid in [("type", "unknown"), ("type", []), ("type", None), ("unknown_field", True),
                             ("usage", {"prompt": "invalid"}), ("toolCalls", [{"id": "missing-arguments"}])]:
            changed = copy.deepcopy(value)
            changed["steps"][0][key] = invalid
            values.append(changed)
        for key in list(step):
            changed = copy.deepcopy(value)
            changed["steps"][0].pop(key)
            values.append(changed)
    for steps in [None, {}, [None], ["llm"], [{"type": "message", "label": "bad", "content": "text"}]]:
        values.append({"schema": "trace/1", "converter": "fixture@1", "steps": steps})
    for value in values:
        expected = standard.is_valid(value)
        if expected:
            trace_format.validate(value)
        else:
            with pytest.raises(jsonschema.ValidationError):
                trace_format.validate(value)
