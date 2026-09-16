"""Validate measurement tables against the repository's Parquet schemas."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .define_benchmark_vocabulary import validate_benchmark_release_date
from .response_scales import (
    canonical_response_scale,
    item_response_scale,
    resolve_categorical,
    validate_grade,
    validate_scale_type,
)

from .register_measurements import (
    _RegistrationRows,
    _locked_registration_rows,
    _serialize_verifier,
)
from .hash_measurement_ids import (
    canonical_asset_manifest,
    canonical_grading_criterion,
    item_id_from_content,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS_PATH = _REPO_ROOT / "parquet_schemas.yaml"
with _SCHEMAS_PATH.open() as _schema_file:
    PARQUET_SCHEMAS = yaml.safe_load(_schema_file)


_OBSERVATION_KEY = (
    "subject_id", "item_id", "benchmark_id", "trial", "test_condition", "interactors",
)


def parquet_columns(table: str, *, include_derived: bool = False) -> list[str]:
    """Return schema-ordered columns for a repository Parquet table.

    Registration-time callers omit columns marked ``derived``. Final table
    writers request them with ``include_derived=True``.
    """
    return [
        column["name"]
        for column in PARQUET_SCHEMAS[table]["columns"]
        if include_derived or not column.get("derived")
    ]


def _type_violation(column: pd.Series, declared: str | None) -> str | None:
    """Describe values that do not match a declared schema type, if any."""
    values = column.dropna()
    if declared is None or values.empty:
        return None

    def is_bool(value: object) -> bool:
        return isinstance(value, (bool, np.bool_))

    if declared == "int":
        if pd.api.types.is_integer_dtype(column):
            return None
        valid = values.map(
            lambda value: isinstance(value, (int, np.integer))
            and not is_bool(value)
        )
    elif declared == "float":
        if column.dtype.kind in "iuf":
            return None
        valid = values.map(
            lambda value: isinstance(
                value, (int, float, np.integer, np.floating)
            )
            and not is_bool(value)
        )
    elif declared == "bool":
        if pd.api.types.is_bool_dtype(column):
            return None
        valid = values.map(is_bool)
    elif declared == "string":
        valid = values.map(lambda value: isinstance(value, str))
    elif declared == "binary":
        valid = values.map(lambda value: isinstance(value, bytes))
    elif declared == "list[string]":
        valid = values.map(
            lambda value: isinstance(value, (list, tuple, np.ndarray))
            and all(isinstance(element, str) for element in value)
        )
    else:
        return None

    invalid_count = int((~valid).sum())
    if not invalid_count:
        return None
    example = values[~valid].iloc[0]
    return (
        f"{invalid_count} value(s) not of declared type {declared} "
        f"(e.g. {type(example).__name__}: {str(example)[:80]!r})"
    )


def _row_locations(mask: pd.Series) -> str:
    positions = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False)) + 1
    suffix = "; first 10 shown" if len(positions) > 10 else ""
    return f"at 1-based row positions {positions[:10].tolist()}{suffix}"


def _primary_key_problems(table: str, dataframe: pd.DataFrame) -> list[str]:
    """Check the schema-declared key within this table, without changing rows."""
    schema = PARQUET_SCHEMAS[table]
    keys = schema["primary_key"]
    missing_columns = [key for key in keys if key not in dataframe.columns]
    if missing_columns:
        return [f"primary key missing column(s) {missing_columns}"]
    if dataframe.empty:
        return []
    problems = []
    key_frame = dataframe[keys]
    missing = key_frame.isna().any(axis=1)
    if missing.any():
        problems.append(f"primary key {keys} must be non-null " + _row_locations(missing))

    # Validate before hashing/grouping so malformed values such as lists produce
    # an actionable schema error rather than an unhashable-value exception.
    types = {column["name"]: column["type"] for column in schema["columns"]}
    invalid_types = False
    for key in keys:
        violation = _type_violation(dataframe[key], types[key])
        if violation:
            problems.append(f"primary key {key}: {violation}")
            invalid_types = True
    if invalid_types:
        return problems

    duplicates = ~missing & key_frame.duplicated(keep=False)
    if duplicates.any():
        examples = key_frame.loc[duplicates].drop_duplicates().head(5).to_dict("records")
        problems.append(
            f"primary key {keys} must be unique; duplicate key values {examples} "
            + _row_locations(duplicates)
        )
    return problems


def _observation_integrity_problems(dataframe: pd.DataFrame) -> list[str]:
    """Check trials and observation keys, independently of the primary key."""
    if dataframe.empty:
        return []
    problems = []

    if "trial" in dataframe:
        max_trial = np.iinfo(np.int64).max
        invalid = ~dataframe["trial"].map(
            lambda value: isinstance(value, Integral)
            and not isinstance(value, (bool, np.bool_))
            and 1 <= value <= max_trial
        )
        if invalid.any():
            problems.append(
                "trial must be a positive 1-based integer representable as int64 "
                + _row_locations(invalid)
            )

    if set(_OBSERVATION_KEY) <= set(dataframe.columns):
        invalid_keys = [
            name for name in _OBSERVATION_KEY
            if _type_violation(dataframe[name], "int" if name == "trial" else "string")
        ]
        if invalid_keys:
            problems.append(f"observation key column(s) have invalid types: {invalid_keys}")
        else:
            # DataFrame.duplicated treats missing condition/interactor values as
            # equal, including different pandas/NumPy missing-value sentinels.
            duplicates = dataframe.duplicated(list(_OBSERVATION_KEY), keep=False)
            if duplicates.any():
                problems.append(
                    f"duplicate observation keys {list(_OBSERVATION_KEY)} " + _row_locations(duplicates)
                )
    return problems


def validate_table(
    table: str,
    df: pd.DataFrame,
    *,
    include_derived: bool = False,
    context: str = "",
) -> None:
    """Enforce ``parquet_schemas.yaml`` before writing a table.

    All problems are aggregated into one ``RuntimeError``: missing or
    out-of-order columns, unexpected columns, nulls in non-nullable columns,
    values that do not match their declared type, null/duplicate primary keys,
    and invalid observation identities or trials in responses and traces. ``include_derived``
    selects the final written schema. Every table rejects columns outside its
    canonical schema; no benchmark-specific extension columns are permitted.
    """
    dataframe = df
    if not dataframe.columns.is_unique:
        duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise RuntimeError(
            f"{context + ': ' if context else ''}{table}.parquet violates "
            f"parquet_schemas.yaml — duplicate column names {duplicates}"
        )
    specification = [
        column
        for column in PARQUET_SCHEMAS[table]["columns"]
        if include_derived or not column.get("derived")
    ]
    expected_columns = [column["name"] for column in specification]
    actual_columns = list(dataframe.columns)
    problems: list[str] = []

    missing_columns = [
        column for column in expected_columns if column not in actual_columns
    ]
    if missing_columns:
        problems.append(f"missing column(s) {missing_columns}")
    elif actual_columns[: len(expected_columns)] != expected_columns:
        problems.append(
            f"column order {actual_columns[:len(expected_columns)]} "
            f"!= schema order {expected_columns}"
        )

    extra_columns = [
        column for column in actual_columns if column not in expected_columns
    ]
    if extra_columns:
        problems.append(f"column(s) not in schema: {extra_columns}")

    for column_specification in specification:
        column_name = column_specification["name"]
        if column_name not in dataframe.columns:
            continue
        column = dataframe[column_name]
        if not column_specification.get("nullable", True):
            null_count = int(column.isna().sum())
            if null_count:
                problems.append(
                    f"`{column_name}` is non-nullable but has "
                    f"{null_count} null(s)"
                )
        violation = _type_violation(column, column_specification.get("type"))
        if violation:
            problems.append(f"`{column_name}`: {violation}")

    problems.extend(_primary_key_problems(table, dataframe))
    if table in {"responses", "traces"}:
        problems.extend(_observation_integrity_problems(dataframe))

    if table == "items":
        for field, canonicalize in (
            ("grading_criterion", canonical_grading_criterion),
            ("verifier", _serialize_verifier),
        ):
            if field not in dataframe:
                continue
            for value in dataframe[field].dropna():
                try:
                    canonicalize(value)
                except (TypeError, ValueError) as exc:
                    problems.append(f"invalid {field}: {exc}")

    if table == "responses" and "response" in dataframe:
        values = dataframe["response"].dropna()
        if not _type_violation(values, "float"):
            try:
                finite = np.isfinite(values.to_numpy(dtype="float64"))
            except (OverflowError, ValueError):
                finite = np.array([False])
            if not finite.all():
                problems.append("response must contain only finite grades or nulls")

    if table == "benchmarks" and "release_date" in dataframe:
        for value in dataframe["release_date"].dropna():
            try:
                validate_benchmark_release_date(value)
            except ValueError as exc:
                problems.append(str(exc))

    if table == "benchmarks" and {"response_type", "response_scale"} <= set(dataframe):
        for row in dataframe[["response_type", "response_scale"]].itertuples(index=False):
            try:
                validate_scale_type(row.response_type, row.response_scale)
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))

    if table == "benchmarks" and {"response_type", "categorical"} <= set(dataframe):
        for row in dataframe[["response_type", "categorical"]].itertuples(index=False):
            # Column checks already reject nulls and non-booleans. Normalize
            # NumPy bools only after checking the actual type, never by truthiness.
            if isinstance(row.response_type, str) and isinstance(row.categorical, (bool, np.bool_)):
                try:
                    resolve_categorical(row.response_type, bool(row.categorical))
                except ValueError as exc:
                    problems.append(str(exc))

    if problems:
        location = f"{context}: " if context else ""
        raise RuntimeError(
            f"{location}{table}.parquet violates parquet_schemas.yaml — "
            + "; ".join(problems)
        )


def validate_response_grades(
    responses: pd.DataFrame, response_scale: dict | str, *,
    items: pd.DataFrame | None = None,
) -> None:
    """Check every distinct observed grade against its explicit item domain.

    Pandas missing sentinels represent previously validated ungraded attempts;
    authoring APIs reject NaN and require None before DataFrame construction.
    """
    domain = json.loads(canonical_response_scale(response_scale))
    if domain["kind"] != "mixed":
        if items is not None:
            for criterion in items["grading_criterion"]:
                item_response_scale(domain, criterion)
        groups = [(domain, responses["response"])]
    else:
        if items is None:
            raise ValueError("mixed response_scale requires the items table")
        if items["item_id"].duplicated().any():
            raise ValueError("duplicate item IDs in response_scale lookup")
        domains = {
            row.item_id: item_response_scale(domain, row.grading_criterion)
            for row in items[["item_id", "grading_criterion"]].itertuples(index=False)
        }
        unknown = set(responses["item_id"]) - set(domains)
        if unknown:
            raise ValueError(f"response_scale missing for response item IDs: {sorted(unknown)}")
        groups = (
            (domains[item_id], rows["response"])
            for item_id, rows in responses.groupby("item_id", sort=False)
        )
    for item_domain, values in groups:
        for grade in values.dropna().unique():
            validate_grade(grade, item_domain)


def validate_trace_relations(
    responses: pd.DataFrame,
    traces: pd.DataFrame | None,
    *,
    context: str = "",
) -> None:
    """Validate the zero-or-one trace relationship and repeated join fields."""
    prefix = f"{context}: " if context else ""
    keys = list(_OBSERVATION_KEY)
    for name, table in (("responses", responses), ("traces", traces)):
        if table is None:
            continue
        missing = set(["response_id", *keys]) - set(table.columns)
        if missing:
            raise RuntimeError(f"{prefix}{name} missing relationship columns {sorted(missing)}")
        problems = _primary_key_problems(name, table)
        problems.extend(_observation_integrity_problems(table))
        if problems:
            raise RuntimeError(f"{prefix}{name}: " + "; ".join(problems))
    if traces is None or traces.empty:
        return
    joined = traces[["response_id", *keys]].merge(
        responses[["response_id", *keys]], on="response_id", how="left",
        validate="one_to_one", indicator=True, suffixes=("_trace", "_response"),
    )
    if joined["_merge"].ne("both").any():
        raise RuntimeError(f"{prefix}trace references an absent response_id")
    for name in keys:
        left, right = joined[f"{name}_trace"], joined[f"{name}_response"]
        same = left.eq(right).fillna(False) | (left.isna() & right.isna())
        if not same.all():
            raise RuntimeError(f"{prefix}trace {name} disagrees with its linked response")


def validate_asset_relations(
    items: pd.DataFrame,
    assets: pd.DataFrame | None,
    *,
    benchmark_id: str,
    context: str = "",
    response_scale: dict | str | None = None,
    check_item_ids: bool = True,
) -> set[str]:
    """Validate canonical item manifests against exact asset sidecar bytes.

    Returns the referenced asset IDs. ``assets=None`` represents an absent
    sidecar; an empty schema-correct DataFrame represents an attachment-free
    build before the optional file is omitted. Supply the benchmark response
    scale for version-4 item IDs; omit it when validating historical IDs.
    Dataset relationship checks can disable item-ID recomputation; asset
    ownership, manifests, byte sizes and content hashes are always checked.
    """

    problems: list[str] = []
    if "item_id" not in items.columns:
        problems.append("items.parquet is missing item_id")
    else:
        item_ids = items["item_id"].tolist()
        valid_item_ids = [
            item_id
            for item_id in item_ids
            if isinstance(item_id, str)
            and len(item_id) == 16
            and all(character in "0123456789abcdef" for character in item_id)
        ]
        if len(valid_item_ids) != len(item_ids):
            problems.append(
                "items.parquet item_id values must be 16 lowercase "
                "hexadecimal characters"
            )
        if len(valid_item_ids) != len(set(valid_item_ids)):
            problems.append("items.parquet has duplicate item_id rows")
    if "benchmark_id" not in items.columns:
        problems.append("items.parquet is missing benchmark_id")
    else:
        try:
            item_benchmark_ids = set(items["benchmark_id"])
        except TypeError:
            problems.append("items.parquet has invalid benchmark_id values")
        else:
            if item_benchmark_ids - {benchmark_id}:
                problems.append(
                    "items.parquet must contain exactly benchmark_id "
                    f"{benchmark_id!r}"
                )

    referenced_asset_ids: set[str] = set()
    manifest_rows = (
        items.to_dict(orient="records")
        if "asset_manifest" in items.columns
        else []
    )
    for row_number, item_row in enumerate(manifest_rows, start=1):
        manifest = item_row["asset_manifest"]
        try:
            is_missing = bool(pd.isna(manifest))
        except (TypeError, ValueError):
            is_missing = False
        if is_missing:
            continue
        try:
            if not isinstance(manifest, str):
                raise TypeError("manifest must be a string or null")
            entries = json.loads(manifest)
            if canonical_asset_manifest(entries) != manifest:
                raise ValueError("manifest is not canonical compact JSON")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            problems.append(f"item row {row_number} has invalid asset_manifest ({exc})")
            continue
        referenced_asset_ids.update(str(entry["asset_id"]) for entry in entries)
        if not check_item_ids:
            continue

        required_identity_columns = ("content", "item_features", "verifier")
        if response_scale is not None:
            required_identity_columns += ("grading_criterion",)
        missing_identity_columns = [
            column
            for column in required_identity_columns
            if column not in item_row
        ]
        if missing_identity_columns:
            problems.append(
                f"item row {row_number} with assets is missing identity "
                f"column(s) {missing_identity_columns}"
            )
            continue

        identity_values: dict[str, str | None] = {}
        invalid_identity_columns: list[str] = []
        for column in ("content", "item_features", "verifier", "grading_criterion"):
            value = item_row.get(column)
            try:
                value_is_missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                value_is_missing = False
            if value_is_missing:
                identity_values[column] = None
            elif isinstance(value, str):
                identity_values[column] = value
            else:
                invalid_identity_columns.append(column)
        if invalid_identity_columns:
            problems.append(
                f"item row {row_number} has non-string identity value(s) "
                f"in {invalid_identity_columns}"
            )
            continue

        stored_item_id = item_row.get("item_id")
        try:
            expected_item_id = item_id_from_content(
                benchmark_id,
                identity_values["content"] or "",
                identity_values["item_features"],
                asset_manifest=manifest,
                verifier=identity_values["verifier"],
                grading_criterion=identity_values["grading_criterion"],
                response_scale=response_scale,
            )
        except (TypeError, ValueError) as exc:
            problems.append(f"item row {row_number} item_id does not match: {exc}")
            continue
        if stored_item_id != expected_item_id:
            problems.append(
                f"item row {row_number} item_id does not match its content, "
                "item_features, grading_criterion, verifier, and asset_manifest"
            )

    if assets is None:
        if referenced_asset_ids:
            problems.append(
                f"assets.parquet is absent but {len(referenced_asset_ids)} "
                "asset_id(s) are referenced"
            )
    else:
        try:
            validate_table("assets", assets, context=context)
        except RuntimeError as exc:
            problems.append(str(exc))
        else:
            asset_ids = assets["asset_id"].tolist()
            stored_asset_ids = set(asset_ids)
            if len(asset_ids) != len(stored_asset_ids):
                problems.append("assets.parquet has duplicate asset_id rows")

            missing_asset_ids = referenced_asset_ids - stored_asset_ids
            orphan_asset_ids = stored_asset_ids - referenced_asset_ids
            if missing_asset_ids:
                problems.append(
                    f"assets.parquet is missing {len(missing_asset_ids)} "
                    "referenced asset_id(s)"
                )
            if orphan_asset_ids:
                problems.append(
                    f"assets.parquet has {len(orphan_asset_ids)} orphan asset_id(s)"
                )

            invalid_rows = 0
            for row in assets.itertuples(index=False):
                payload = row.data
                if (
                    row.benchmark_id != benchmark_id
                    or row.byte_size != len(payload)
                    or row.asset_id != hashlib.sha256(payload).hexdigest()
                ):
                    invalid_rows += 1
            if invalid_rows:
                problems.append(
                    f"assets.parquet has {invalid_rows} row(s) with invalid "
                    "owner, byte_size, or SHA-256"
                )

    if problems:
        location = f"{context}: " if context else ""
        raise RuntimeError(location + "asset relation validation failed — " + "; ".join(problems))
    return referenced_asset_ids


def validate_dataset(
    tables: Mapping[str, pd.DataFrame], *,
    expected_benchmark_id: str | None = None, context: str = "",
) -> None:
    """Validate one complete benchmark without changing rows or identifiers.

    Require canonical tables, unique keys, one benchmark, valid foreign keys,
    declared grade domains, matching traces/assets, and exact derived counts.
    Coverage permits only float64 rounding (absolute tolerance 1e-12). Subjects
    and items with no responses remain part of the banks and denominator.
    Responses are required for item-level releases; aggregate/not_released
    releases omit them. Optional empty trace/asset tables are accepted.
    Hash-version policy is separate: this function treats item/response IDs as
    identifiers and does not regenerate them.
    """
    prefix = f"{context}: " if context else ""
    required = {"subjects", "items", "benchmarks"}
    missing = required - set(tables)
    unknown = set(tables) - set(PARQUET_SCHEMAS)
    if missing or unknown:
        raise RuntimeError(f"{prefix}invalid dataset tables: missing={sorted(missing)}, unknown={sorted(unknown)}")
    problems = []
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            problems.append(f"{name} must be a DataFrame")
            continue
        try:
            validate_table(name, frame, include_derived=True)
        except RuntimeError as exc:
            problems.append(str(exc))
    if problems:
        raise RuntimeError(prefix + "dataset validation failed — " + "; ".join(problems))
    benchmarks, subjects, items = (tables[name] for name in ("benchmarks", "subjects", "items"))
    if len(benchmarks) != 1:
        raise RuntimeError(f"{prefix}benchmarks.parquet must contain exactly one benchmark row")
    benchmark = benchmarks.iloc[0]
    benchmark_id = benchmark.benchmark_id
    if expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
        problems.append(f"benchmark_id {benchmark_id!r} does not match folder {expected_benchmark_id!r}")
    for name in ("items", "responses", "traces", "assets"):
        if name in tables:
            wrong = tables[name].benchmark_id.ne(benchmark_id)
            if wrong.any():
                problems.append(f"{name}.benchmark_id must match {benchmark_id!r} " + _row_locations(wrong))
    responses = tables.get("responses")
    traces = tables.get("traces")
    if benchmark.granularity == "item":
        if responses is None or responses.empty:
            problems.append("granularity='item' requires a nonempty responses.parquet")
    elif benchmark.granularity in {"aggregate", "not_released"}:
        if responses is not None:
            problems.append(f"granularity={benchmark.granularity!r} must omit responses.parquet")
    else:
        problems.append(f"invalid granularity {benchmark.granularity!r}")
    if responses is None:
        if traces is not None:
            problems.append("traces.parquet exists without responses.parquet")
        observations = pd.DataFrame(columns=parquet_columns("responses", include_derived=True))
    else:
        observations = responses
        for column, bank in (("subject_id", subjects.subject_id), ("item_id", items.item_id)):
            absent = ~responses[column].isin(bank)
            if absent.any():
                problems.append(f"responses.{column} references absent {column} " + _row_locations(absent))
        try:
            validate_trace_relations(responses, traces)
        except RuntimeError as exc:
            problems.append(str(exc))
    try:
        # This also validates mixed-scale declarations for unobserved items.
        validate_response_grades(observations, benchmark.response_scale, items=items)
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid response scale or grade: {exc}")
    try:
        validate_asset_relations(items, tables.get("assets"), benchmark_id=benchmark_id,
                                 check_item_ids=False)
    except RuntimeError as exc:
        problems.append(str(exc))
    denominator = len(subjects) * len(items)
    observed_pairs = len(observations.loc[observations.response.notna(), ["subject_id", "item_id"]].drop_duplicates())
    expected = {
        "n_subjects": len(subjects), "n_items": len(items),
        "n_responses": len(observations),
        "n_response_values": int(observations.response.nunique(dropna=True)),
        "max_trial": int(observations.trial.max()) if not observations.empty else 0,
        "coverage": observed_pairs / denominator if denominator else 0.0,
        "has_reference_answer": any(json.loads(value).get("reference_answer") is not None
                                    for value in items.grading_criterion),
    }
    for column, value in expected.items():
        actual = benchmark[column]
        matches = pd.notna(actual) and actual == value
        if column == "coverage" and pd.notna(actual):
            matches = math.isfinite(float(actual)) and math.isclose(
                float(actual), value, rel_tol=0, abs_tol=1e-12,
            )
        if not matches:
            problems.append(f"benchmarks.{column}={actual!r}; expected {value!r} from the dataset")
    if problems:
        raise RuntimeError(prefix + "dataset validation failed — " + "; ".join(problems))


_SUBJECTS_COLUMNS = parquet_columns("subjects")
_ITEMS_COLUMNS = parquet_columns("items")
_BENCHMARKS_COLUMNS = parquet_columns("benchmarks")
_FINAL_BENCHMARKS_COLUMNS = parquet_columns(
    "benchmarks", include_derived=True
)
_DERIVED_BENCHMARK_COLUMNS = set(_FINAL_BENCHMARKS_COLUMNS) - set(
    _BENCHMARKS_COLUMNS
)


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    )


def _materialize(
    rows_by_id: dict[str, dict], columns: list[str]
) -> pd.DataFrame:
    """Build one schema-ordered table from registered rows.

    Starting with an all-object empty table preserves registry column ordering
    while avoiding repeated DataFrame concatenation during registration. The
    shared writer enforces physical Parquet types independently of pandas inference.
    """
    dataframe = _empty(columns)
    if rows_by_id:
        dataframe = pd.concat(
            [
                dataframe,
                pd.DataFrame(list(rows_by_id.values()), columns=columns),
            ],
            ignore_index=True,
        )
    return dataframe


def ensure_unique_trials(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Renumber colliding trials within a response's primary-key group.

    Existing trial numbers are preserved when already distinct. If collisions
    occur within ``(subject_id, item_id, test_condition, interactors)``, every
    row in that group is numbered 1, 2, 3, ... in input order. ``interactors``
    participates only when present, so pre-migration tables remain supported.

    Returns the resulting DataFrame and the number of changed trial values.
    The caller receives the original object when no collision exists and a
    copy when renumbering is necessary.
    """
    dataframe = df
    if dataframe.empty:
        return dataframe, 0
    group_columns = ["subject_id", "item_id", "test_condition"]
    if "interactors" in dataframe.columns:
        group_columns.append("interactors")

    keys = dataframe[group_columns].copy()
    for column in group_columns[2:]:
        keys[column] = keys[column].fillna("\x00NULL")

    collisions = keys.assign(
        trial=dataframe["trial"].values
    ).duplicated(keep=False)
    if not collisions.any():
        return dataframe, 0

    grouped_collisions = collisions.groupby(
        [keys[column] for column in group_columns], sort=False
    )
    needs_renumbering = grouped_collisions.transform("any").to_numpy()

    dataframe = dataframe.copy()
    renumbered_trials = (
        keys.groupby(group_columns, sort=False)
        .cumcount()
        .to_numpy()[needs_renumbering]
        + 1
    )
    renumbered_count = int(
        (
            dataframe.loc[needs_renumbering, "trial"].to_numpy()
            != renumbered_trials
        ).sum()
    )
    dataframe.loc[needs_renumbering, "trial"] = renumbered_trials
    return dataframe, renumbered_count


def _validated_registration_tables(
    rows: _RegistrationRows,
    context: str,
    *,
    expected_benchmark_id: str | None = None,
    require_complete: bool = False,
) -> dict[str, pd.DataFrame]:
    """Materialize and validate all current registration tables.

    ``rows`` must come from ``_locked_registration_rows`` so validation always
    sees the dictionaries currently bound after any call to ``reload()``.
    """
    if expected_benchmark_id is not None:
        registered_benchmarks = set(rows.benchmarks)
        expected_benchmarks = {expected_benchmark_id}
        if registered_benchmarks != expected_benchmarks:
            raise RuntimeError(
                f"{context}: this benchmark folder must register exactly "
                f"benchmark_id {expected_benchmark_id!r}; registered "
                f"{sorted(registered_benchmarks)!r}"
            )

    if rows.subjects:
        unmapped_subjects = sorted(
            row["display_name"]
            for row in rows.subjects.values()
            if row["normalized_name"] is None
        )
        if unmapped_subjects:
            sys.stderr.write(
                f"⚠ {len(unmapped_subjects)} subject name(s) not in "
                "scripts/build_measurement_tables/map_model_registry.json "
                "(normalized_name/provider left null):\n"
                + "".join(f"    {name}\n" for name in unmapped_subjects)
                + "  -> map them per the curation guide's model-registry "
                "phase, then re-run build.py\n"
            )

    benchmark_is_complete = require_complete or any(
        _DERIVED_BENCHMARK_COLUMNS.intersection(row)
        for row in rows.benchmarks.values()
    )
    benchmark_columns = (
        _FINAL_BENCHMARKS_COLUMNS
        if benchmark_is_complete
        else _BENCHMARKS_COLUMNS
    )
    table_specifications = (
        ("subjects", rows.subjects, _SUBJECTS_COLUMNS, False),
        ("items", rows.items, _ITEMS_COLUMNS, False),
        (
            "benchmarks",
            rows.benchmarks,
            benchmark_columns,
            benchmark_is_complete,
        ),
    )
    tables: dict[str, pd.DataFrame] = {}
    for name, registered_rows, columns, include_derived in table_specifications:
        if not registered_rows:
            continue
        if name == "benchmarks" and include_derived:
            # This complete row replaces the historical read/annotate/rewrite
            # pass, so infer its final dtypes directly before the single write.
            table = pd.DataFrame(list(registered_rows.values()), columns=columns)
        else:
            table = _materialize(registered_rows, columns)
        validate_table(
            name,
            table,
            include_derived=include_derived,
            context=context,
        )
        tables[name] = table

    if "items" in tables:
        wrong_owner = sorted(
            set(tables["items"]["benchmark_id"].dropna())
            - set(rows.benchmarks)
        )
        if wrong_owner:
            raise RuntimeError(
                f"{context}: items.parquet references unregistered "
                f"benchmark_id value(s) {wrong_owner}"
            )
        for row in tables["items"].itertuples(index=False):
            try:
                scale = rows.benchmarks[row.benchmark_id]["response_scale"]
                expected_item_id = item_id_from_content(
                    row.benchmark_id,
                    row.content if pd.notna(row.content) else (
                        "" if pd.notna(row.asset_manifest) else f"raw:{row.raw_item_id}"
                    ),
                    row.item_features if pd.notna(row.item_features) else None,
                    asset_manifest=row.asset_manifest if pd.notna(row.asset_manifest) else None,
                    verifier=row.verifier, grading_criterion=row.grading_criterion,
                    response_scale=scale,
                )
                if row.item_id != expected_item_id:
                    raise ValueError("item_id does not match its grading protocol and effective response_scale")
            except ValueError as exc:
                raise RuntimeError(f"{context}: item {row.item_id}: {exc}") from exc
    return tables


def validate_registrations(
    context: str = "", *, expected_benchmark_id: str | None = None
) -> None:
    """Preflight pending registration tables without writing files.

    ``expected_benchmark_id`` enforces the one-benchmark-per-folder contract
    used by ``build_base.BenchmarkBuild``.
    """
    with _locked_registration_rows() as rows:
        _validated_registration_tables(
            rows,
            context,
            expected_benchmark_id=expected_benchmark_id,
        )
