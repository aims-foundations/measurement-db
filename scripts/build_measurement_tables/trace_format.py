"""Shared helpers for the trace/1 format of traces.parquet (see trace_schema.json).

Builders produce one trace/1 value per response. Converters use these helpers so that every
benchmark writes the same shape: optional fields are omitted when the source lacks them, source
fields a mapping does not use are kept in `native`, and anything unrecognized is kept as an
`unrecognized` event and counted.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SCHEMA = "trace/1"
# Parquet metadata key that marks a traces.parquet file written in this format.
FORMAT_KEY = b"measurement_db.trace_format"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "trace_schema.json"


@dataclass(frozen=True)
class TraceContext:
    """Final observation identity and item text, supplied by the shared builder."""

    benchmark_id: str
    response_id: str
    subject_id: str
    item_id: str
    trial: int
    test_condition: str | None
    interactors: str | None
    item_content: str | None


@dataclass(frozen=True)
class ConvertedTrace:
    """One trace/1 document and the converter's nonnegative accounting counters.

    ``unrecognized`` and ``unmatched_tool_results`` default to zero tolerance in
    StructuredTraceBuild. A reported ``parse_failures`` count always fails a
    build. Missing source traces never reach the converter.
    """

    value: dict
    counts: Mapping[str, int] = field(default_factory=dict)


def compact(value: dict) -> dict:
    """Drop fields the source did not provide (None, empty strings, lists and dicts)."""
    return {k: v for k, v in value.items() if v is not None and v != "" and v != [] and v != {}}


def unused(source: dict, used: set[str]) -> dict | None:
    """The fields of `source` a mapping did not consume, for a step's `native`."""
    rest = {k: v for k, v in source.items() if k not in used}
    return rest or None


def unrecognized(source, counts: Counter, **fields) -> dict:
    """Keep content a converter does not understand, instead of dropping it."""
    counts["unrecognized"] += 1
    return compact(dict(type="event", name="unrecognized", native={"source": source}, **fields))


def to_text(content) -> str:
    """Flatten message content (string, content-block list or None) without losing anything."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(to_text(block.get("content")))
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, dict) and isinstance(block.get("thinking"), str):
                parts.append(block["thinking"])
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def as_json_text(value) -> str:
    """Tool-call arguments are stored as the source string, or as JSON text."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def trace(converter: str, steps: list[dict], *, timing=None, run=None, evaluation=None,
          native=None) -> dict:
    return compact(dict(schema=SCHEMA, converter=converter, timing=timing, steps=steps,
                        run=compact(run or {}) or None,
                        evaluation=compact(evaluation or {}) or None,
                        native=native or None)) | {"steps": steps}


@lru_cache(maxsize=1)
def _validator():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    # The step alternatives have distinct constant type tags. Select the one
    # possible branch before validating it, rather than constructing errors for
    # every other branch on each of millions of steps. All other schema rules
    # still run normally, including the step's required type and allowed tags.
    step = schema["$defs"]["step"]
    branches = {}
    for choice in step["oneOf"]:
        definition = schema["$defs"][choice["$ref"].removeprefix("#/$defs/")]
        tag = definition["properties"]["type"]["const"]
        if tag in branches:
            return jsonschema.Draft202012Validator(schema)
        branches[tag] = choice
    ordinary = jsonschema.Draft202012Validator.VALIDATORS["oneOf"]

    def tagged_step(validator, choices, value, subschema):
        tag = value.get("type") if isinstance(value, dict) else None
        if subschema is step and isinstance(tag, str) and tag in branches:
            yield from validator.descend(value, branches[tag])
        else:
            yield from ordinary(validator, choices, value, subschema)

    validator = jsonschema.validators.extend(jsonschema.Draft202012Validator, {"oneOf": tagged_step})
    return validator(schema)


def validate(value: dict) -> None:
    """Raise jsonschema.ValidationError if `value` is not a valid trace/1 document."""
    _validator().validate(value)


def dumps(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
