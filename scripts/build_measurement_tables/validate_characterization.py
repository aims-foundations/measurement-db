"""Validate reviewed expectations and compare them with tables and source audits.

This module never downloads, writes outputs, or refreshes expectations. Claims
are measured by each benchmark's independent source audit, not by its builder.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import date, datetime
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
import numpy as np
import pandas as pd

from .validate_benchmark_metadata import _load_yaml_mapping, BenchmarkMetadataError

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "characterization_schema.yaml"
TABLE_NAMES = ("subjects", "items", "benchmarks", "responses", "traces", "assets")


class CharacterizationError(ValueError):
    """A characterization is invalid or differs from observed data."""


@lru_cache(maxsize=1)
def _validator():
    schema = _load_yaml_mapping(SCHEMA_PATH.read_text(), path=SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _finite_numbers(value, path="<root>"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CharacterizationError(f"{path}: mapping keys must be strings")
            _finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise CharacterizationError(f"{path}: numbers must be finite")


def validate_characterization(value):
    _finite_numbers(value)
    errors = sorted(_validator().iter_errors(value), key=lambda e: str(list(e.absolute_path)))
    if errors:
        raise CharacterizationError("; ".join(
            f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in errors
        ))
    return value


def load_characterization(path):
    path = Path(path)
    try:
        value = _load_yaml_mapping(path.read_text(encoding="utf-8"), path=path)
        return validate_characterization(value)
    except (OSError, BenchmarkMetadataError, CharacterizationError) as exc:
        raise CharacterizationError(f"{path}: {exc}") from exc


def _cell(value):
    if isinstance(value, np.ndarray):
        return [_cell(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_cell(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _cell(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def logical_sha256(table):
    """Hash column names and the sorted multiset of canonical row hashes.

All columns, repeated rows, exact strings, list order and asset bytes count.
Row order and Parquet compression do not. Format version 1 fixes this algorithm.
"""
    rows = []
    for row in table.itertuples(index=False, name=None):
        payload = json.dumps([_cell(v) for v in row], ensure_ascii=False,
                             sort_keys=True, separators=(",", ":"), allow_nan=False)
        rows.append(hashlib.sha256(payload.encode("utf-8")).digest())
    header = json.dumps(list(table.columns), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(header.encode("utf-8"))
    for row in sorted(rows):
        digest.update(row)
    return digest.hexdigest()


def characterize_tables(tables):
    """Compute candidates for explicit review; never save them automatically."""
    return {name: {"rows": len(frame), "logical_sha256": logical_sha256(frame)}
            for name, frame in tables.items()}


def check_tables(characterization, tables):
    validate_characterization(characterization)
    expected = characterization["tables"]
    if set(expected) != set(tables):
        raise CharacterizationError(
            f"table set differs: expected {sorted(expected)}, observed {sorted(tables)}")
    for name, observed in characterize_tables(tables).items():
        if observed != expected[name]:
            raise CharacterizationError(
                f"tables.{name}: expected {expected[name]}, observed {observed}")


def _compare(expected, observed, tolerance, path, *, count=False):
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise CharacterizationError(f"{path}: aggregate groups differ")
        for key in expected:
            _compare(expected[key], observed[key], tolerance, f"{path}.{key}")
        return
    if isinstance(observed, (bool, np.bool_)) or not isinstance(observed, Real):
        raise CharacterizationError(f"{path}: observed value must be a finite number")
    if not math.isfinite(observed) or (count and (observed < 0 or observed != int(observed))):
        raise CharacterizationError(f"{path}: invalid observed value {observed!r}")
    if abs(observed - expected) > tolerance:
        raise CharacterizationError(f"{path}: expected {expected}, observed {observed}")


def check_source_claims(characterization, observations):
    """Require every declared claim to be measured; no ignored extra results."""
    validate_characterization(characterization)
    claims = characterization["source_claims"]
    if set(observations) != set(claims):
        raise CharacterizationError(
            f"source claim coverage differs: missing={sorted(set(claims) - set(observations))}, "
            f"unknown={sorted(set(observations) - set(claims))}")
    for name, claim in claims.items():
        _compare(claim["expected"], observations[name], claim.get("absolute_tolerance", 0),
                 f"source_claims.{name}", count=claim["kind"] == "count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--benchmarks-dir", type=Path,
                        help="Require a characterization for every non-template benchmark")
    args = parser.parse_args()
    paths = list(args.paths)
    if args.benchmarks_dir:
        metadata = sorted(p for p in args.benchmarks_dir.glob("*/metadata.yaml")
                          if not p.parent.name.startswith("_"))
        if not metadata:
            parser.error("no benchmark metadata found")
        paths += [p.with_name("characterization.yaml") for p in metadata]
    if not paths:
        parser.error("provide characterization paths or --benchmarks-dir")
    failed = False
    for path in paths:
        try:
            load_characterization(path)
            print(f"OK {path}")
        except CharacterizationError as exc:
            print(f"ERROR {exc}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
