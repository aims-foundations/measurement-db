"""
AndroidWorld audit (processed).

response_matrix.csv: nonempty CSV; first column named Agent with unique nonempty agent id per row; remaining headers are canonical task_id strings (116 AndroidWorld tasks), unique; column set and order match task_metadata.csv task_id rows; every score cell is finite and in {0, 1} (1 = pass).

task_metadata.csv: columns task_id, primary_app, all_apps, task_type, validation_method, max_steps; one row per task; task_id unique and nonempty; primary_app and all_apps nonempty strings; task_type in {TC, IR}; validation_method nonempty; max_steps a positive integer (string or int in CSV).

leaderboard_summary.csv: columns rank, release_date, source, model_type, open_source, model_size, model, screen_repr, success_rate, num_trials, pass_at_k in that order; each row has finite success_rate in [0, 100]; source and model nonempty. Optional: for each matrix Agent, if exactly one leaderboard row has source equal to the agent name (after stripping whitespace; case-sensitive), that row's success_rate/100 matches the agent's row mean pass rate in the matrix (within tolerance).

Cross-file: task column names of response_matrix.csv equal the task_id sequence in task_metadata.csv (same order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import bad_pct_suffix, path_errors_if_missing, processed_dir_from_script  # noqa: E402

PROC = processed_dir_from_script(__file__)
RESPONSE_MATRIX = PROC / "response_matrix.csv"
TASK_METADATA = PROC / "task_metadata.csv"
LEADERBOARD_SUMMARY = PROC / "leaderboard_summary.csv"

ID_COL = "Agent"
RTOL = 1e-5
ATOL = 0.02  # leaderboard aggregate vs matrix mean (percentage points as fraction)

EXPECTED_LB_COLS = [
    "rank",
    "release_date",
    "source",
    "model_type",
    "open_source",
    "model_size",
    "model",
    "screen_repr",
    "success_rate",
    "num_trials",
    "pass_at_k",
]

EXPECTED_META_COLS = [
    "task_id",
    "primary_app",
    "all_apps",
    "task_type",
    "validation_method",
    "max_steps",
]

TASK_TYPES = frozenset({"TC", "IR"})


def _task_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != ID_COL]


def _audit_response_matrix(df: pd.DataFrame) -> list[str]:
    label = "response_matrix.csv"
    errs: list[str] = []
    n = len(df)
    if n == 0:
        return [f"{label}: empty table"]

    if ID_COL not in df.columns:
        return [f"{label}: must have {ID_COL!r} column"]

    bad_id = df[ID_COL].isna() | (df[ID_COL].astype(str).str.strip() == "")
    if bad_id.any():
        errs.append(
            f"{label}: {ID_COL} must be non-empty"
            + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df[ID_COL].duplicated(keep=False)
    if dup.any():
        errs.append(
            f"{label}: duplicate {ID_COL} values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    task_cols = _task_columns(df)
    if not task_cols:
        errs.append(f"{label}: no task columns after {ID_COL!r}")
        return errs

    if len(set(task_cols)) != len(task_cols):
        errs.append(f"{label}: task column headers must be unique")

    bad_task = [t for t in task_cols if not str(t).strip()]
    if bad_task:
        errs.append(f"{label}: empty task column name(s)")

    w = df[task_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n_cells = w.size
    if np.isinf(w).any():
        errs.append(f"{label}: score cells must not be infinite")
    nan_ct = int(np.isnan(w).sum())
    if nan_ct:
        pct = 100.0 * nan_ct / n_cells
        errs.append(
            f"{label}: expected dense 0/1 matrix — missing/NaN cells: {nan_ct:,} / {n_cells:,} ({pct:.2f}%)"
        )
    fin = np.isfinite(w)
    oob = fin & ~np.isclose(w, 0.0) & ~np.isclose(w, 1.0)
    if oob.any():
        pct = 100.0 * int(oob.sum()) / n_cells
        errs.append(
            f"{label}: scores must be 0 or 1 — invalid: {int(oob.sum()):,} / {n_cells:,} ({pct:.2f}%)"
        )

    return errs


def _audit_task_metadata(meta: pd.DataFrame, task_cols: list[str]) -> list[str]:
    label = "task_metadata.csv"
    errs: list[str] = []

    if list(meta.columns) != EXPECTED_META_COLS:
        errs.append(
            f"{label}: columns expected {EXPECTED_META_COLS}, got {list(meta.columns)}"
        )
        return errs

    n = len(meta)
    if n == 0:
        errs.append(f"{label}: empty table")
        return errs

    if n != len(task_cols):
        errs.append(
            f"{label}: expected {len(task_cols)} rows (one per matrix task column), got {n}"
        )

    tids = meta["task_id"].astype(str)
    if tids.str.strip().eq("").any():
        errs.append(f"{label}: task_id must be non-empty")

    dup = meta["task_id"].duplicated(keep=False)
    if dup.any():
        errs.append(f"{label}: duplicate task_id ({int(dup.sum())} rows)")

    meta_order = list(meta["task_id"].astype(str))
    if meta_order != [str(c) for c in task_cols]:
        errs.append(
            f"{label}: task_id row order must match response_matrix task columns left-to-right"
        )

    for col in ("primary_app", "all_apps", "validation_method"):
        bad = meta[col].isna() | (meta[col].astype(str).str.strip() == "")
        if bad.any():
            errs.append(
                f"{label}: {col!r} must be non-empty"
                + bad_pct_suffix(int(bad.sum()), n)
            )

    bad_type = ~meta["task_type"].astype(str).isin(TASK_TYPES)
    if bad_type.any():
        errs.append(
            f"{label}: task_type must be one of {sorted(TASK_TYPES)}"
            + bad_pct_suffix(int(bad_type.sum()), n)
        )

    steps = pd.to_numeric(meta["max_steps"], errors="coerce")
    bad_steps = steps.isna() | (steps < 1) | (steps % 1 != 0)
    if bad_steps.any():
        errs.append(
            f"{label}: max_steps must be a positive integer"
            + bad_pct_suffix(int(bad_steps.sum()), n)
        )

    return errs


def _audit_leaderboard_summary(lb: pd.DataFrame) -> list[str]:
    label = "leaderboard_summary.csv"
    errs: list[str] = []

    if list(lb.columns) != EXPECTED_LB_COLS:
        errs.append(
            f"{label}: columns expected {EXPECTED_LB_COLS}, got {list(lb.columns)}"
        )
        return errs

    if len(lb) == 0:
        errs.append(f"{label}: empty table")
        return errs

    sr = pd.to_numeric(lb["success_rate"], errors="coerce")
    if sr.isna().any():
        errs.append(
            f"{label}: success_rate must be numeric"
            + bad_pct_suffix(int(sr.isna().sum()), len(lb))
        )
    else:
        if ((sr < 0) | (sr > 100)).any():
            errs.append(f"{label}: success_rate must be in [0, 100]")

    bad_src = lb["source"].isna() | (lb["source"].astype(str).str.strip() == "")
    if bad_src.any():
        errs.append(
            f"{label}: source must be non-empty"
            + bad_pct_suffix(int(bad_src.sum()), len(lb))
        )

    bad_model = lb["model"].isna() | (lb["model"].astype(str).str.strip() == "")
    if bad_model.any():
        errs.append(
            f"{label}: model must be non-empty"
            + bad_pct_suffix(int(bad_model.sum()), len(lb))
        )

    return errs


def _audit_matrix_vs_leaderboard(
    matrix_df: pd.DataFrame,
    task_cols: list[str],
    lb: pd.DataFrame,
) -> list[str]:
    """When source exactly equals Agent (strip only), success_rate should match row mean."""
    errs: list[str] = []
    w = matrix_df[task_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    agents = matrix_df[ID_COL].astype(str).str.strip().tolist()
    sources = lb["source"].astype(str).str.strip()
    rates = pd.to_numeric(lb["success_rate"], errors="coerce")

    for i, agent in enumerate(agents):
        mask = sources == agent
        idx = lb.index[mask]
        if len(idx) == 0:
            continue
        if len(idx) > 1:
            continue
        row_mean = float(np.nanmean(w[i]))
        expected_pct = float(rates.loc[idx[0]])
        if not np.isclose(row_mean, expected_pct / 100.0, rtol=RTOL, atol=ATOL):
            errs.append(
                f"leaderboard_summary vs response_matrix: agent {agent!r} row mean {row_mean:.4f} "
                f"does not match success_rate {expected_pct}% (tolerance {ATOL})"
            )
    return errs


def main() -> int:
    paths = [RESPONSE_MATRIX, TASK_METADATA, LEADERBOARD_SUMMARY]
    pe = path_errors_if_missing(paths)
    if pe:
        for e in pe:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    all_errs: list[str] = []

    try:
        matrix_df = pd.read_csv(RESPONSE_MATRIX)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: response_matrix.csv: cannot read ({exc!r})", file=sys.stderr)
        return 1

    all_errs.extend(_audit_response_matrix(matrix_df))
    task_cols = _task_columns(matrix_df)

    try:
        # keep_default_na=False: literal "n/a" in primary_app/all_apps must stay a string
        meta = pd.read_csv(TASK_METADATA, keep_default_na=False)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: task_metadata.csv: cannot read ({exc!r})", file=sys.stderr)
        return 1

    all_errs.extend(_audit_task_metadata(meta, task_cols))

    try:
        lb = pd.read_csv(LEADERBOARD_SUMMARY, keep_default_na=False)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: leaderboard_summary.csv: cannot read ({exc!r})", file=sys.stderr)
        return 1

    all_errs.extend(_audit_leaderboard_summary(lb))

    if not all_errs and task_cols:
        all_errs.extend(_audit_matrix_vs_leaderboard(matrix_df, task_cols, lb))

    if all_errs:
        for e in all_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_agents = len(matrix_df)
    n_tasks = len(task_cols)
    print(
        f"OK: AndroidWorld — {n_agents} agents × {n_tasks} tasks; "
        "task_metadata aligned; leaderboard_summary valid; "
        "per-task agents match leaderboard rates where source names align."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
