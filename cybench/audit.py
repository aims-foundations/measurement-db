"""
Audit Cybench processed outputs:
  - processed/response_matrix.csv           (unguided, binary)
  - processed/response_matrix_subtask_guided.csv  (subtask-guided, binary)
  - processed/response_matrix_subtask_scores.csv  (fractional subtask scores)
  - processed/task_metadata.csv
  - processed/leaderboard_aggregate.csv

Task names must exactly match the canonical ``TASKS`` list in
``01_build_response_matrix.py``. Matrix models must match ``MODELS_PAPER``.
Binary matrix cells must be exactly ``0`` or ``1``. Subtask score cells must
be ``X`` or a valid ``N/D`` fraction (non-negative numerator, positive
denominator, numerator <= denominator). Task metadata columns and value sets
must match canonical definitions.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import bad_pct_suffix, path_errors_if_missing, processed_dir_from_script  # noqa: E402

EXPECTED_TASK_METADATA_COLUMNS = [
    "task_name",
    "task_path",
    "first_solve_time",
    "fst_minutes",
    "category",
    "competition",
    "competition_full",
]

EXPECTED_LEADERBOARD_COLUMNS = [
    "model",
    "tasks_evaluated",
    "unguided_pct_solved",
    "flag_success_count",
    "subtask_guided_pct_solved",
    "subtask_pct_solved",
    "fst_unguided",
    "fst_subtask",
    "notes",
]

VALID_CATEGORIES = frozenset({"Reverse", "Forensics", "Web", "Crypto", "Pwn", "Misc"})
VALID_COMPETITIONS = frozenset({"HTB", "GLA", "S23", "S22", "HKC"})

_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")


def _load_build_script() -> object:
    """Load ``01_build_response_matrix.py`` to extract canonical task/model lists."""
    build_path = Path(__file__).resolve().parent / "01_build_response_matrix.py"
    spec = importlib.util.spec_from_file_location("cybench_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load build script: {build_path}")
    mod = importlib.util.module_from_spec(spec)
    # Suppress the script's main() from running on import
    sys.modules["cybench_build"] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_canonical(mod: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (task_names, model_names) from the build module."""
    task_names = getattr(mod, "TASK_NAMES", None)
    models = getattr(mod, "MODELS_PAPER", None)
    if not task_names or not models:
        raise RuntimeError(
            "01_build_response_matrix.py must define non-empty TASK_NAMES and MODELS_PAPER"
        )
    return tuple(task_names), tuple(models)


# ---------------------------------------------------------------------------
# Binary matrix audit
# ---------------------------------------------------------------------------

def _audit_binary_matrix(
    df: pd.DataFrame,
    *,
    label: str,
    expected_tasks: tuple[str, ...],
    expected_models: tuple[str, ...],
    errors: list[str],
) -> None:
    """Audit a binary (0/1) response matrix CSV."""
    n = len(df)
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    if df.columns[0] != "task_name":
        errors.append(
            f"{label}: first column must be 'task_name', got {df.columns[0]!r}"
            + bad_pct_suffix(n, n)
        )
        return

    model_cols = list(df.columns[1:])
    if tuple(model_cols) != expected_models:
        missing = sorted(set(expected_models) - set(model_cols))
        extra = sorted(set(model_cols) - set(expected_models))
        errors.append(
            f"{label}: model columns mismatch "
            f"(missing {missing!r}, extra {extra!r})"
            + bad_pct_suffix(n, n)
        )

    # task_name checks
    bad_name = df["task_name"].isna() | (df["task_name"].astype(str).str.strip() == "")
    if bad_name.any():
        errors.append(
            f"{label}: task_name must be non-empty"
            + bad_pct_suffix(int(bad_name.sum()), n)
        )

    dup = df["task_name"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate task_name values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    actual_tasks = tuple(df["task_name"].tolist())
    if actual_tasks != expected_tasks:
        missing = [t for t in expected_tasks if t not in set(actual_tasks)]
        extra = [t for t in actual_tasks if t not in set(expected_tasks)]
        errors.append(
            f"{label}: task_name set does not match canonical TASKS "
            f"(missing {missing[:5]!r}{'…' if len(missing) > 5 else ''}, "
            f"extra {extra[:5]!r}{'…' if len(extra) > 5 else ''})"
            + bad_pct_suffix(len(missing) + len(extra), len(expected_tasks))
        )

    # Binary cell checks (0 or 1 exactly)
    present_models = [c for c in model_cols if c in df.columns]
    if not present_models:
        return

    n_cells = n * len(present_models)
    bad_empty = 0
    bad_not_01 = 0
    for c in present_models:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(arr)
        bad_empty += int((~finite).sum())
        bad_not_01 += int((finite & (arr != 0.0) & (arr != 1.0)).sum())

    if bad_empty:
        pct = 100.0 * bad_empty / n_cells
        errors.append(
            f"{label}: matrix cells must be numeric and finite"
            f" — malformed cells: {bad_empty:,} / {n_cells:,} ({pct:.2f}%)"
        )
    if bad_not_01:
        pct = 100.0 * bad_not_01 / n_cells
        errors.append(
            f"{label}: matrix cells must be exactly 0 or 1"
            f" — malformed cells: {bad_not_01:,} / {n_cells:,} ({pct:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Subtask scores matrix audit
# ---------------------------------------------------------------------------

def _is_valid_score_cell(val: object) -> bool:
    """Return True if val is 'X' or a valid 'N/D' fraction (0 <= N <= D, D > 0)."""
    s = str(val).strip()
    if s == "X":
        return True
    m = _FRACTION_RE.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return den > 0 and 0 <= num <= den
    return False


def _audit_subtask_scores_matrix(
    df: pd.DataFrame,
    *,
    expected_tasks: tuple[str, ...],
    expected_models: tuple[str, ...],
    errors: list[str],
) -> None:
    label = "response_matrix_subtask_scores.csv"
    n = len(df)
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    if df.columns[0] != "task_name":
        errors.append(
            f"{label}: first column must be 'task_name', got {df.columns[0]!r}"
            + bad_pct_suffix(n, n)
        )
        return

    model_cols = list(df.columns[1:])
    if tuple(model_cols) != expected_models:
        missing = sorted(set(expected_models) - set(model_cols))
        extra = sorted(set(model_cols) - set(expected_models))
        errors.append(
            f"{label}: model columns mismatch "
            f"(missing {missing!r}, extra {extra!r})"
            + bad_pct_suffix(n, n)
        )

    bad_name = df["task_name"].isna() | (df["task_name"].astype(str).str.strip() == "")
    if bad_name.any():
        errors.append(
            f"{label}: task_name must be non-empty"
            + bad_pct_suffix(int(bad_name.sum()), n)
        )

    dup = df["task_name"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate task_name values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    actual_tasks = tuple(df["task_name"].tolist())
    if actual_tasks != expected_tasks:
        missing = [t for t in expected_tasks if t not in set(actual_tasks)]
        extra = [t for t in actual_tasks if t not in set(expected_tasks)]
        errors.append(
            f"{label}: task_name set does not match canonical TASKS "
            f"(missing {missing[:5]!r}{'…' if len(missing) > 5 else ''}, "
            f"extra {extra[:5]!r}{'…' if len(extra) > 5 else ''})"
            + bad_pct_suffix(len(missing) + len(extra), len(expected_tasks))
        )

    # Score cell validation: must be 'X' or 'N/D'
    present_models = [c for c in model_cols if c in df.columns]
    n_cells = n * len(present_models)
    bad_cells = 0
    bad_examples: list[str] = []
    for c in present_models:
        for val in df[c]:
            if not _is_valid_score_cell(val):
                bad_cells += 1
                if len(bad_examples) < 5:
                    bad_examples.append(repr(str(val)))

    if bad_cells:
        preview = ", ".join(bad_examples)
        extra = " …" if bad_cells > 5 else ""
        pct = 100.0 * bad_cells / n_cells
        errors.append(
            f"{label}: cells must be 'X' or 'N/D' fraction (e.g. '2/5') "
            f"— bad cells: {bad_cells:,} / {n_cells:,} ({pct:.2f}%); "
            f"examples: {preview}{extra}"
        )


# ---------------------------------------------------------------------------
# Task metadata audit
# ---------------------------------------------------------------------------

def _audit_task_metadata(
    df: pd.DataFrame,
    *,
    expected_tasks: tuple[str, ...],
    errors: list[str],
) -> None:
    label = "task_metadata.csv"
    n = len(df)

    if list(df.columns) != EXPECTED_TASK_METADATA_COLUMNS:
        errors.append(
            f"{label}: columns mismatch; "
            f"expected {EXPECTED_TASK_METADATA_COLUMNS}, got {list(df.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    bad_name = df["task_name"].isna() | (df["task_name"].astype(str).str.strip() == "")
    if bad_name.any():
        errors.append(
            f"{label}: task_name must be non-empty"
            + bad_pct_suffix(int(bad_name.sum()), n)
        )

    dup = df["task_name"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate task_name values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    actual_tasks = tuple(df["task_name"].tolist())
    if actual_tasks != expected_tasks:
        missing = [t for t in expected_tasks if t not in set(actual_tasks)]
        extra = [t for t in actual_tasks if t not in set(expected_tasks)]
        errors.append(
            f"{label}: task_name set does not match canonical TASKS "
            f"(missing {missing[:5]!r}{'…' if len(missing) > 5 else ''}, "
            f"extra {extra[:5]!r}{'…' if len(extra) > 5 else ''})"
            + bad_pct_suffix(len(missing) + len(extra), len(expected_tasks))
        )

    bad_cat = df["category"].isna() | ~df["category"].astype(str).isin(VALID_CATEGORIES)
    if bad_cat.any():
        bad_vals = df.loc[bad_cat, "category"].unique().tolist()[:5]
        errors.append(
            f"{label}: category must be one of {sorted(VALID_CATEGORIES)}; "
            f"unexpected: {bad_vals!r}"
            + bad_pct_suffix(int(bad_cat.sum()), n)
        )

    bad_comp = df["competition"].isna() | ~df["competition"].astype(str).isin(VALID_COMPETITIONS)
    if bad_comp.any():
        bad_vals = df.loc[bad_comp, "competition"].unique().tolist()[:5]
        errors.append(
            f"{label}: competition must be one of {sorted(VALID_COMPETITIONS)}; "
            f"unexpected: {bad_vals!r}"
            + bad_pct_suffix(int(bad_comp.sum()), n)
        )

    bad_path = df["task_path"].isna() | (df["task_path"].astype(str).str.strip() == "")
    if bad_path.any():
        errors.append(
            f"{label}: task_path must be non-empty"
            + bad_pct_suffix(int(bad_path.sum()), n)
        )

    bad_fst = df["first_solve_time"].isna() | (df["first_solve_time"].astype(str).str.strip() == "")
    if bad_fst.any():
        errors.append(
            f"{label}: first_solve_time must be non-empty"
            + bad_pct_suffix(int(bad_fst.sum()), n)
        )

    fst_mins = pd.to_numeric(df["fst_minutes"], errors="coerce")
    bad_fst_mins = fst_mins.isna() | (fst_mins < 0)
    if bad_fst_mins.any():
        errors.append(
            f"{label}: fst_minutes must be a non-negative number"
            + bad_pct_suffix(int(bad_fst_mins.sum()), n)
        )


# ---------------------------------------------------------------------------
# Leaderboard aggregate audit
# ---------------------------------------------------------------------------

def _audit_leaderboard_aggregate(df: pd.DataFrame, errors: list[str]) -> None:
    label = "leaderboard_aggregate.csv"
    n = len(df)

    if list(df.columns) != EXPECTED_LEADERBOARD_COLUMNS:
        errors.append(
            f"{label}: columns mismatch; "
            f"expected {EXPECTED_LEADERBOARD_COLUMNS}, got {list(df.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    bad_model = df["model"].isna() | (df["model"].astype(str).str.strip() == "")
    if bad_model.any():
        errors.append(
            f"{label}: model must be non-empty"
            + bad_pct_suffix(int(bad_model.sum()), n)
        )

    dup = df["model"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate model values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    tasks_eval = pd.to_numeric(df["tasks_evaluated"], errors="coerce")
    bad_tasks = tasks_eval.isna() | (tasks_eval <= 0)
    if bad_tasks.any():
        errors.append(
            f"{label}: tasks_evaluated must be a positive number"
            + bad_pct_suffix(int(bad_tasks.sum()), n)
        )

    unguided_pct = pd.to_numeric(df["unguided_pct_solved"], errors="coerce")
    bad_pct = unguided_pct.isna() | (unguided_pct < 0) | (unguided_pct > 100)
    if bad_pct.any():
        errors.append(
            f"{label}: unguided_pct_solved must be in [0, 100]"
            + bad_pct_suffix(int(bad_pct.sum()), n)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    proc = processed_dir_from_script(__file__)
    paths = {
        "response_matrix.csv": proc / "response_matrix.csv",
        "response_matrix_subtask_guided.csv": proc / "response_matrix_subtask_guided.csv",
        "response_matrix_subtask_scores.csv": proc / "response_matrix_subtask_scores.csv",
        "task_metadata.csv": proc / "task_metadata.csv",
        "leaderboard_aggregate.csv": proc / "leaderboard_aggregate.csv",
    }
    missing_errs = path_errors_if_missing(paths.values())
    if missing_errs:
        for e in missing_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        mod = _load_build_script()
        expected_tasks, expected_models = _get_canonical(mod)
    except Exception as exc:
        print(
            f"ERROR: cannot load canonical task/model lists from build script: {exc}",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []

    # Binary matrices
    for name in ("response_matrix.csv", "response_matrix_subtask_guided.csv"):
        try:
            df = pd.read_csv(paths[name])
        except Exception as exc:
            errors.append(f"{name}: cannot read CSV ({exc!r})")
            continue
        _audit_binary_matrix(
            df,
            label=name,
            expected_tasks=expected_tasks,
            expected_models=expected_models,
            errors=errors,
        )

    # Subtask scores matrix
    try:
        df_scores = pd.read_csv(paths["response_matrix_subtask_scores.csv"])
    except Exception as exc:
        errors.append(f"response_matrix_subtask_scores.csv: cannot read CSV ({exc!r})")
        df_scores = None

    if df_scores is not None:
        _audit_subtask_scores_matrix(
            df_scores,
            expected_tasks=expected_tasks,
            expected_models=expected_models,
            errors=errors,
        )

    # Task metadata
    try:
        df_meta = pd.read_csv(paths["task_metadata.csv"])
    except Exception as exc:
        errors.append(f"task_metadata.csv: cannot read CSV ({exc!r})")
        df_meta = None

    if df_meta is not None:
        _audit_task_metadata(df_meta, expected_tasks=expected_tasks, errors=errors)

    # Leaderboard aggregate
    try:
        df_lb = pd.read_csv(paths["leaderboard_aggregate.csv"])
    except Exception as exc:
        errors.append(f"leaderboard_aggregate.csv: cannot read CSV ({exc!r})")
        df_lb = None

    if df_lb is not None:
        _audit_leaderboard_aggregate(df_lb, errors=errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(
        f"OK: Cybench — response_matrix.csv, response_matrix_subtask_guided.csv, "
        f"response_matrix_subtask_scores.csv, task_metadata.csv, leaderboard_aggregate.csv "
        f"passed checks "
        f"({len(expected_tasks)} canonical tasks, {len(expected_models)} paper models)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
