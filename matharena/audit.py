"""
MathArena audit (processed). Every file matching response_matrix_*.csv in processed.

response_matrix_*.csv: readable nonempty CSV; first column model_name or model_attempt with unique nonempty values per row; task column names unique integers with no duplicate ids after parsing; score cells finite from 0 through 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import (  # noqa: E402
    bad_pct_suffix,
    parse_task_column_id,
    processed_dir_from_script,
)

SCORE_MIN = 0.0
SCORE_MAX = 1.0
ROW_ID_COLUMNS = ("model_name", "model_attempt")


def _audit_response_matrix(path: Path) -> list[str]:
    label = path.name
    errors: list[str] = []

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return [f"{label}: cannot read CSV ({exc!r})"]

    n = len(df)
    if n == 0:
        return [f"{label}: empty table"]

    first = df.columns[0]
    if first not in ROW_ID_COLUMNS:
        errors.append(
            f"{label}: first column must be one of {ROW_ID_COLUMNS}, got {first!r}"
            + bad_pct_suffix(n, n)
        )
        return errors

    id_col = first

    bad_id = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    if bad_id.any():
        errors.append(
            f"{label}: {id_col} must be non-empty"
            + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate {id_col} values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    task_cols = [c for c in df.columns if c != id_col]
    if not task_cols:
        errors.append(f"{label}: no task/score columns after {id_col!r}")
        return errors

    n_task = len(task_cols)
    task_ok = True
    if len(set(task_cols)) != n_task:
        n_dup_hdr = n_task - len(set(task_cols))
        pct = 100.0 * n_dup_hdr / n_task
        errors.append(
            f"{label}: task column headers must be unique"
            f" — duplicate headers: {n_dup_hdr:,} / {n_task:,} columns ({pct:.2f}%)"
        )
        task_ok = False

    bad_task_names: list[str] = []
    parsed_ids: list[int] = []
    for c in task_cols:
        try:
            parsed_ids.append(parse_task_column_id(c))
        except (TypeError, ValueError):
            bad_task_names.append(repr(c))

    if bad_task_names:
        preview = ", ".join(bad_task_names[:5])
        extra = f" (+{len(bad_task_names) - 5} more)" if len(bad_task_names) > 5 else ""
        errors.append(
            f"{label}: task column names must be integers (task ids)"
            f" — invalid: {preview}{extra}"
        )
        task_ok = False

    if task_ok and len(set(parsed_ids)) != n_task:
        n_dup_id = n_task - len(set(parsed_ids))
        pct = 100.0 * n_dup_id / n_task
        errors.append(
            f"{label}: parsed task ids must be unique (e.g. not both '1' and '01')"
            f" — duplicate ids: {n_dup_id:,} / {n_task:,} columns ({pct:.2f}%)"
        )
        task_ok = False

    if not task_ok:
        return errors

    n_cells = n * len(task_cols)
    bad_empty = 0
    bad_oob = 0
    for c in task_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(arr)
        bad_empty += int((~finite).sum())
        oob_mask = finite & ((arr < SCORE_MIN) | (arr > SCORE_MAX))
        bad_oob += int(oob_mask.sum())

    if bad_empty:
        pct = 100.0 * bad_empty / n_cells
        errors.append(
            f"{label}: score cells must be non-empty and finite"
            f" — malformed cells: {bad_empty:,} / {n_cells:,} ({pct:.2f}%)"
        )
    if bad_oob:
        pct = 100.0 * bad_oob / n_cells
        errors.append(
            f"{label}: score cells must be in [{SCORE_MIN}, {SCORE_MAX}]"
            f" — malformed cells: {bad_oob:,} / {n_cells:,} ({pct:.2f}%)"
        )

    return errors


def main() -> int:
    proc = processed_dir_from_script(__file__)
    files = sorted(proc.glob("response_matrix_*.csv"))
    if not files:
        print(f"ERROR: No response_matrix_*.csv under {proc}", file=sys.stderr)
        return 1

    all_errs: list[str] = []
    for f in files:
        all_errs.extend(_audit_response_matrix(f))

    if all_errs:
        for e in all_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"OK: MathArena — {len(files)} response_matrix_*.csv file(s) passed checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
