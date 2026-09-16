"""Structured grade domains; validation never clips, rounds, or imputes grades."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Real
import re
import sys


_DIRECTIONS = {"higher_is_better", "lower_is_better", "unordered"}
_CATEGORICAL_TYPES = {"binary", "error_presence", "likert_5", "likert_10", "ordinal"}
_CONTINUOUS_TYPES = {"continuous", "continuous_bounded", "continuous_unbounded"}
_NUMBER_KEY = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate response_scale key {key!r}")
        result[key] = value
    return result


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite float64 number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return 0.0 if number == 0 else number


def _meanings(value: object, values: list[float]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("response_scale.meanings must be a nonempty mapping")
    result = {}
    for key, meaning in value.items():
        if not isinstance(key, str) or not _NUMBER_KEY.fullmatch(key):
            raise ValueError("response_scale.meanings keys must be quoted numeric strings")
        number = _number(float(key), "response_scale.meanings key")
        if number not in values:
            raise ValueError(f"response_scale.meanings key {key!r} is not an allowed value")
        # Match the domain's numeric canonicalization, including 1 == 1.0 and -0 == 0.
        canonical_key = str(int(number)) if number.is_integer() else repr(number)
        if canonical_key in result:
            raise ValueError(f"duplicate response_scale.meanings value {canonical_key!r}")
        if not isinstance(meaning, str) or not meaning.strip():
            raise ValueError("response_scale.meanings values must be nonempty strings")
        result[canonical_key] = meaning
    return result


def canonical_response_scale(value: Mapping | str, *, allow_mixed: bool = True) -> str:
    """Accept a mapping/JSON object and return canonical, finite-domain JSON.

    Discrete domains enumerate values. Intervals have inclusive min/max;
    explicit null bounds mean unbounded. A mixed benchmark requires a concrete
    response_scale in each item's grading_criterion. Concrete scales optionally
    declare direction; discrete scales may also annotate known category meanings.
    Omitted/null direction is unknown, not unordered. Unknown direction is omitted
    from canonical JSON so existing unannotated scale identities are unchanged.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError("response_scale must be a structured object, not prose") from exc
    if not isinstance(value, Mapping):
        raise ValueError("response_scale must be a structured object")
    kind = value.get("kind")
    if kind == "discrete":
        if not {"kind", "values"} <= set(value) or set(value) - {"kind", "values", "meanings", "direction"}:
            raise ValueError("discrete response_scale requires kind and values; only meanings and direction are optional")
        raw = value["values"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("response_scale.values must be a nonempty list")
        values = sorted(_number(number, "response_scale.values") for number in raw)
        if len(set(values)) != len(values):
            raise ValueError("response_scale.values must be distinct")
        result = {"kind": kind, "values": values}
        if "meanings" in value:
            result["meanings"] = _meanings(value["meanings"], values)
    elif kind == "interval":
        if not {"kind", "min", "max"} <= set(value) or set(value) - {"kind", "min", "max", "direction"}:
            raise ValueError("interval response_scale requires kind, min, and max; only direction is optional")
        bounds = {
            key: None if value[key] is None else _number(value[key], f"response_scale.{key}")
            for key in ("min", "max")
        }
        if bounds["min"] is not None and bounds["max"] is not None and bounds["min"] > bounds["max"]:
            raise ValueError("response_scale.min must not exceed max")
        result = {"kind": kind, **bounds}
    elif kind == "mixed" and allow_mixed:
        if set(value) != {"kind"}:
            raise ValueError("mixed response_scale requires only kind")
        result = {"kind": kind}
    else:
        allowed = "discrete, interval, or mixed" if allow_mixed else "discrete or interval"
        raise ValueError(f"response_scale.kind must be {allowed}")
    direction = value.get("direction")
    if direction is not None:
        if not isinstance(direction, str) or direction not in _DIRECTIONS:
            raise ValueError("response_scale.direction must be higher_is_better, lower_is_better, unordered, or null")
        result["direction"] = direction
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def resolve_categorical(response_type: str, categorical: bool | None = None) -> bool:
    """Derive unambiguous classifications; require an explicit bool otherwise.

    Numeric fractions/rates may have finite support without being categorical.
    Mixed benchmarks retain the curator's explicit classification.
    """
    if categorical is not None and not isinstance(categorical, bool):
        raise ValueError("categorical must be a bool when supplied")
    expected = (True if response_type in _CATEGORICAL_TYPES else
                False if response_type in _CONTINUOUS_TYPES else None)
    if expected is not None:
        if categorical is not None and categorical != expected:
            raise ValueError(f"categorical must be {str(expected).lower()} for response_type={response_type!r}")
        return expected
    if categorical is None:
        raise ValueError(f"categorical must be explicitly declared for response_type={response_type!r}")
    return categorical


def validate_scale_type(response_type: str, scale: Mapping | str) -> None:
    """Check the declared response type and its explicit domain agree."""
    domain = json.loads(canonical_response_scale(scale))
    kind = domain["kind"]
    values = domain.get("values", [])
    problem = None
    if response_type in {"binary", "error_presence"}:
        if kind != "discrete" or values != [0.0, 1.0]:
            problem = "requires exactly the discrete values 0 and 1"
    elif response_type in {"likert_5", "likert_10"}:
        size = 5 if response_type == "likert_5" else 10
        if kind != "discrete" or len(values) != size:
            problem = f"requires exactly {size} declared discrete categories"
    elif response_type == "ordinal" and kind != "discrete":
        problem = "requires declared discrete categories"
    elif response_type == "continuous" and kind != "interval":
        problem = "requires an explicit interval"
    elif response_type == "continuous_bounded":
        if kind != "interval" or domain["min"] is None or domain["max"] is None:
            problem = "requires an interval with finite min and max"
    elif response_type == "continuous_unbounded":
        if kind != "interval" or (domain["min"] is not None and domain["max"] is not None):
            problem = "requires an interval with at least one null (unbounded) limit"
    if (response_type == "mixed") != (kind == "mixed"):
        problem = "must use kind=mixed if and only if response_type=mixed"
    if problem:
        raise ValueError(f"response_scale for {response_type!r} {problem}")


def item_response_scale(benchmark_scale: Mapping | str, criterion: Mapping | str) -> dict:
    """Resolve one item's domain without duplicating a uniform benchmark scale."""
    domain = json.loads(canonical_response_scale(benchmark_scale))
    criterion = json.loads(criterion) if isinstance(criterion, str) else criterion
    override = criterion.get("response_scale")
    if domain["kind"] == "mixed":
        if override is None:
            raise ValueError("mixed benchmarks require grading_criterion.response_scale on every item")
        return json.loads(canonical_response_scale(override, allow_mixed=False))
    if override is not None:
        raise ValueError("item response_scale is only allowed for a mixed benchmark")
    return domain


def validate_grade(value: object, domain: Mapping | None = None) -> None:
    """Check an author-supplied grade; only explicit None means ungraded.

    Integer-coded categories (including binary) match exactly. For fractional
    categories only, allow floating-point roundoff of eight float64 epsilons;
    values are never changed. Interval boundaries are inclusive and exact.
    """
    if value is None:
        return
    number = _number(value, "response")
    if domain is None:
        return
    if domain["kind"] == "discrete":
        values = domain["values"]
        valid = number in values
        if not valid and not all(float(allowed).is_integer() for allowed in values):
            epsilon = 8 * sys.float_info.epsilon
            valid = any(math.isclose(number, allowed, rel_tol=epsilon, abs_tol=epsilon) for allowed in values)
    elif domain["kind"] == "interval":
        valid = ((domain["min"] is None or number >= domain["min"])
                 and (domain["max"] is None or number <= domain["max"]))
    else:
        raise ValueError("resolve the item's mixed response_scale before validating grades")
    if not valid:
        raise ValueError(f"response {value!r} is outside the declared response_scale {dict(domain)!r}")
