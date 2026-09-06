"""Validate measurement tables against the repository's Parquet schemas."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .register_measurements import (
    _RegistrationRows,
    _locked_registration_rows,
)
from .hash_measurement_ids import (
    canonical_asset_manifest,
    item_id_from_content,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS_PATH = _REPO_ROOT / "parquet_schemas.yaml"
with _SCHEMAS_PATH.open() as _schema_file:
    PARQUET_SCHEMAS = yaml.safe_load(_schema_file)


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
        if pd.api.types.is_numeric_dtype(
            column
        ) and not pd.api.types.is_bool_dtype(column):
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


def validate_table(
    table: str,
    df: pd.DataFrame,
    *,
    include_derived: bool = False,
    allow_extra: bool = False,
    context: str = "",
) -> None:
    """Enforce ``parquet_schemas.yaml`` before writing a table.

    All problems are aggregated into one ``RuntimeError``: missing or
    out-of-order columns, unexpected columns, nulls in non-nullable columns,
    and values that do not match their declared type. ``include_derived``
    selects the final written schema; ``allow_extra`` permits response-table
    extensions after the canonical columns.
    """
    dataframe = df
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
    if extra_columns and not allow_extra:
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

    if problems:
        location = f"{context}: " if context else ""
        raise RuntimeError(
            f"{location}{table}.parquet violates parquet_schemas.yaml — "
            + "; ".join(problems)
        )


def validate_asset_relations(
    items: pd.DataFrame,
    assets: pd.DataFrame | None,
    *,
    benchmark_id: str,
    context: str = "",
) -> set[str]:
    """Validate canonical item manifests against exact asset sidecar bytes.

    Returns the referenced asset IDs. ``assets=None`` represents an absent
    sidecar; an empty schema-correct DataFrame represents an attachment-free
    build before the optional file is omitted.
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
            if item_benchmark_ids != {benchmark_id}:
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

        missing_identity_columns = [
            column
            for column in ("content", "item_features", "verifier")
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
        for column in ("content", "item_features", "verifier"):
            value = item_row[column]
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
        expected_item_id = item_id_from_content(
            benchmark_id,
            identity_values["content"] or "",
            identity_values["item_features"],
            asset_manifest=manifest,
            verifier=identity_values["verifier"],
        )
        if stored_item_id != expected_item_id:
            problems.append(
                f"item row {row_number} item_id does not match its content, "
                "item_features, verifier, and asset_manifest"
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

    Starting with an all-object empty table preserves the historical column
    ordering and Parquet dtypes while avoiding repeated DataFrame concatenation
    during registration.
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

    benchmark_is_complete = any(
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
