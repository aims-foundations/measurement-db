"""
AlpacaEval audit (processed).

response_matrix.csv: nonempty CSV; first column named Model with unique nonempty model id per row; remaining column headers are item indices (integers 0..n-1, unique); each score cell is missing (NaN) or finite in {0, 1} (1 = evaluated model wins vs reference).

response_matrix_preference.csv: same row/column layout as response_matrix.csv; each cell is missing (NaN) or a finite float in [1, 2] (AlpacaEval judge preference); every defined preference aligns with the binary matrix (1 iff preference > 1.5, else 0).

item_metadata.csv: columns item_idx, instruction, dataset, n_models_scored, mean_preference, mean_win_rate; item_idx unique and contiguous 0..N-1 with N equal to the number of item columns in the matrices; instruction and dataset nonempty strings; aggregates numeric and consistent with the preference and binary matrices (per-item counts/means within tolerance).

model_summary.csv: columns model, binary_win_rate, mean_preference, n_items_scored, official_win_rate, lc_win_rate, avg_length, mode; model unique and nonempty; binary_win_rate in [0, 1]; mean_preference in [1, 2] when finite; n_items_scored positive integer matching non-missing preference count per model; binary_win_rate and mean_preference match row means from the matrices (within tolerance). Leaderboard fields may be missing.
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
    path_errors_if_missing,
    processed_dir_from_script,
)

PROC = processed_dir_from_script(__file__)
RESPONSE_MATRIX = PROC / "response_matrix.csv"
RESPONSE_PREF = PROC / "response_matrix_preference.csv"
ITEM_METADATA = PROC / "item_metadata.csv"
MODEL_SUMMARY = PROC / "model_summary.csv"

PREF_MIN = 1.0
PREF_MAX = 2.0
WIN_THRESHOLD = 1.5
# Row-level aggregates in item_metadata / model_summary are rounded in the builder.
RTOL = 1e-5
ATOL = 1e-4


def _task_columns(df: pd.DataFrame, id_col: str) -> list[str]:
    return [c for c in df.columns if c != id_col]


def _audit_matrix_layout(
    path: Path,
    *,
    id_col: str,
    label: str,
) -> tuple[list[str], pd.DataFrame | None, list[str]]:
    """Return (errors, dataframe or None if unreadable, task column names)."""
    errs: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return ([f"{label}: cannot read CSV ({exc!r})"], None, [])

    n = len(df)
    if n == 0:
        return ([f"{label}: empty table (no rows)"], None, [])

    if id_col not in df.columns:
        errs.append(f"{label}: must have a {id_col!r} column")
        return (errs, None, [])

    bad_id = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    if bad_id.any():
        errs.append(
            f"{label}: {id_col} must be non-empty"
            + bad_pct_suffix(int(bad_id.sum()), n)
        )

    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        errs.append(
            f"{label}: duplicate {id_col} values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    task_cols = _task_columns(df, id_col)
    if not task_cols:
        errs.append(f"{label}: no item columns after {id_col!r}")
        return (errs, df, task_cols)

    if len(set(task_cols)) != len(task_cols):
        errs.append(f"{label}: item column headers must be unique")

    parsed: list[int] = []
    bad_names: list[str] = []
    for c in task_cols:
        try:
            parsed.append(parse_task_column_id(c))
        except (TypeError, ValueError):
            bad_names.append(repr(c))
    if bad_names:
        preview = ", ".join(bad_names[:5])
        extra = f" (+{len(bad_names) - 5} more)" if len(bad_names) > 5 else ""
        errs.append(
            f"{label}: item column names must be integers (item_idx)"
            f" — invalid: {preview}{extra}"
        )

    if parsed and len(set(parsed)) != len(parsed):
        errs.append(
            f"{label}: parsed item indices must be unique (e.g. not both '1' and '01')"
        )

    if parsed:
        expected = list(range(len(parsed)))
        if sorted(parsed) != expected:
            errs.append(
                f"{label}: item columns must be exactly 0..{len(parsed) - 1} (sorted contiguous)"
            )

    return (errs, df, task_cols)


def _audit_binary_cells(
    w_num: np.ndarray,
    label: str,
) -> list[str]:
    errs: list[str] = []
    n_cells = w_num.size
    bad_inf = 0
    bad_oob = 0
    arr = w_num
    finite = np.isfinite(arr)
    bad_inf += int(np.isinf(arr).sum())
    oob = finite & ~np.isclose(arr, 0.0) & ~np.isclose(arr, 1.0)
    bad_oob += int(oob.sum())
    if bad_inf:
        pct = 100.0 * bad_inf / n_cells
        errs.append(
            f"{label}: score cells must be missing (NaN) or finite"
            f" — infinite values: {bad_inf:,} / {n_cells:,} ({pct:.2f}%)"
        )
    if bad_oob:
        pct = 100.0 * bad_oob / n_cells
        errs.append(
            f"{label}: defined scores must be 0 or 1"
            f" — invalid cells: {bad_oob:,} / {n_cells:,} ({pct:.2f}%)"
        )
    return errs


def _audit_preference_cells(
    p_num: np.ndarray,
    label: str,
) -> list[str]:
    errs: list[str] = []
    n_cells = p_num.size
    bad_inf = 0
    bad_oob = 0
    arr = p_num
    finite = np.isfinite(arr)
    bad_inf += int(np.isinf(arr).sum())
    oob = finite & ((arr < PREF_MIN - 1e-6) | (arr > PREF_MAX + 1e-6))
    bad_oob += int(oob.sum())
    if bad_inf:
        pct = 100.0 * bad_inf / n_cells
        errs.append(
            f"{label}: cells must be missing (NaN) or finite floats"
            f" — infinite values: {bad_inf:,} / {n_cells:,} ({pct:.2f}%)"
        )
    if bad_oob:
        pct = 100.0 * bad_oob / n_cells
        errs.append(
            f"{label}: defined preferences must be in [{PREF_MIN}, {PREF_MAX}]"
            f" — out of range: {bad_oob:,} / {n_cells:,} ({pct:.2f}%)"
        )
    return errs


def _models_match(win_df: pd.DataFrame, pref_df: pd.DataFrame, id_col: str) -> list[str]:
    a = win_df[id_col].astype(str).values
    b = pref_df[id_col].astype(str).values
    if len(a) != len(b):
        return [
            "response_matrix.csv and response_matrix_preference.csv: row counts differ"
        ]
    if not (a == b).all():
        return [
            "response_matrix.csv and response_matrix_preference.csv: Model column order or values differ row-by-row"
        ]
    return []


def _columns_match(win_cols: list[str], pref_cols: list[str], id_col: str) -> list[str]:
    w = [c for c in win_cols if c != id_col]
    p = [c for c in pref_cols if c != id_col]
    if w != p:
        return [
            "response_matrix.csv and response_matrix_preference.csv: item column headers differ"
        ]
    return []


def _audit_win_pref_consistency(
    win_df: pd.DataFrame,
    task_cols: list[str],
    id_col: str,
    w: np.ndarray,
    p: np.ndarray,
) -> list[str]:
    errs: list[str] = []
    w_ok = np.isfinite(w)
    p_ok = np.isfinite(p)
    miss_mismatch = w_ok ^ p_ok
    if miss_mismatch.any():
        i, j = np.argwhere(miss_mismatch)[0]
        mid = win_df.iloc[i][id_col]
        col = task_cols[j]
        errs.append(
            f"model {mid!r} item {col}: binary and preference missingness mismatch"
        )
        return errs
    expected = np.where(p > WIN_THRESHOLD, 1.0, 0.0)
    both = w_ok & p_ok
    if both.any() and not np.allclose(w[both], expected[both], rtol=0, atol=0):
        diff = both & ~np.isclose(w, expected, rtol=0, atol=0)
        i, j = np.argwhere(diff)[0]
        mid = win_df.iloc[i][id_col]
        col = task_cols[j]
        errs.append(
            f"model {mid!r} item {col}: binary {w[i, j]} inconsistent with preference {p[i, j]} "
            f"(expected {expected[i, j]} for threshold {WIN_THRESHOLD})"
        )
    return errs


def _audit_item_metadata(
    meta: pd.DataFrame,
    task_cols: list[str],
    id_col: str,
    p_mat: np.ndarray,
    w_mat: np.ndarray,
) -> list[str]:
    errs: list[str] = []
    required = [
        "item_idx",
        "instruction",
        "dataset",
        "n_models_scored",
        "mean_preference",
        "mean_win_rate",
    ]
    for col in required:
        if col not in meta.columns:
            errs.append(f"item_metadata.csv: missing column {col!r}")
    if errs:
        return errs

    n_items = len(task_cols)
    if len(meta) != n_items:
        errs.append(
            f"item_metadata.csv: expected {n_items} rows (one per item column), got {len(meta)}"
        )

    idx = pd.to_numeric(meta["item_idx"], errors="coerce")
    if idx.isna().any():
        errs.append("item_metadata.csv: item_idx must be numeric")
    elif list(idx.astype(int).sort_values().values) != list(range(n_items)):
        errs.append(
            "item_metadata.csv: item_idx must be a permutation of 0..N-1 matching matrix columns"
        )

    bad_instr = meta["instruction"].isna() | (meta["instruction"].astype(str).str.strip() == "")
    if bad_instr.any():
        errs.append(
            "item_metadata.csv: instruction must be non-empty"
            + bad_pct_suffix(int(bad_instr.sum()), len(meta))
        )

    bad_ds = meta["dataset"].isna() | (meta["dataset"].astype(str).str.strip() == "")
    if bad_ds.any():
        errs.append(
            "item_metadata.csv: dataset must be non-empty"
            + bad_pct_suffix(int(bad_ds.sum()), len(meta))
        )

    # Per-item consistency with matrices (column labels are item_idx 0..n-1)
    col_n_scored = np.isfinite(p_mat).sum(axis=0)
    col_mean_pref = np.nanmean(p_mat, axis=0)
    col_mean_win = np.nanmean(w_mat, axis=0)

    meta_by_idx = meta.set_index("item_idx", drop=False)
    for k, c in enumerate(task_cols):
        j = parse_task_column_id(c)
        if j not in meta_by_idx.index:
            errs.append(f"item_metadata.csv: missing row for item_idx {j}")
            if len(errs) >= 25:
                return errs
            continue
        row = meta_by_idx.loc[j]
        n_scored = int(col_n_scored[k])
        if int(row["n_models_scored"]) != n_scored:
            errs.append(
                f"item_metadata.csv item_idx {j}: n_models_scored {row['n_models_scored']} != {n_scored} from preference matrix"
            )
            if len(errs) >= 25:
                return errs
        m_pref = float(col_mean_pref[k])
        if not np.isclose(float(row["mean_preference"]), m_pref, rtol=RTOL, atol=ATOL):
            errs.append(
                f"item_metadata.csv item_idx {j}: mean_preference mismatch (metadata vs matrix)"
            )
            if len(errs) >= 25:
                return errs
        m_win = float(col_mean_win[k])
        if not np.isclose(float(row["mean_win_rate"]), m_win, rtol=RTOL, atol=ATOL):
            errs.append(
                f"item_metadata.csv item_idx {j}: mean_win_rate mismatch (metadata vs matrix)"
            )
            if len(errs) >= 25:
                return errs

    return errs


def _audit_model_summary(
    summary: pd.DataFrame,
    win_df: pd.DataFrame,
    task_cols: list[str],
    id_col: str,
    p_mat: np.ndarray,
    w_mat: np.ndarray,
) -> list[str]:
    errs: list[str] = []
    expected_cols = [
        "model",
        "binary_win_rate",
        "mean_preference",
        "n_items_scored",
        "official_win_rate",
        "lc_win_rate",
        "avg_length",
        "mode",
    ]
    if list(summary.columns) != expected_cols:
        errs.append(
            f"model_summary.csv: columns expected {expected_cols}, got {list(summary.columns)}"
        )
        return errs

    models_wm = set(win_df[id_col].astype(str))
    models_sm = set(summary["model"].astype(str))
    if models_wm != models_sm:
        only_w = models_wm - models_sm
        only_s = models_sm - models_wm
        errs.append(
            "model_summary.csv: model set must match response_matrix Model column"
            f" — only in matrix: {sorted(only_w)[:5]}{'...' if len(only_w) > 5 else ''}"
            f" only in summary: {sorted(only_s)[:5]}{'...' if len(only_s) > 5 else ''}"
        )
        return errs

    dup = summary["model"].duplicated(keep=False)
    if dup.any():
        errs.append(
            f"model_summary.csv: duplicate model entries ({int(dup.sum())} rows)"
        )

    bad_m = summary["model"].isna() | (summary["model"].astype(str).str.strip() == "")
    if bad_m.any():
        errs.append("model_summary.csv: model must be non-empty")

    row_n = np.isfinite(p_mat).sum(axis=1)
    row_mean_p = np.nanmean(p_mat, axis=1)
    row_mean_w = np.nanmean(w_mat, axis=1)
    model_to_i = {str(win_df.iloc[i][id_col]): i for i in range(len(win_df))}

    for r in summary.itertuples(index=False):
        m = str(r.model)
        if m not in model_to_i:
            continue
        i = model_to_i[m]
        n_scored = int(row_n[i])
        if int(r.n_items_scored) != n_scored:
            errs.append(
                f"model_summary.csv model {m!r}: n_items_scored {r.n_items_scored} != {n_scored}"
            )
            if len(errs) >= 25:
                return errs
        bwr = float(row_mean_w[i])
        if not np.isclose(float(r.binary_win_rate), bwr, rtol=RTOL, atol=ATOL):
            errs.append(
                f"model_summary.csv model {m!r}: binary_win_rate mismatch vs matrix row mean"
            )
            if len(errs) >= 25:
                return errs
        mp = float(row_mean_p[i])
        if not np.isclose(float(r.mean_preference), mp, rtol=RTOL, atol=ATOL):
            errs.append(
                f"model_summary.csv model {m!r}: mean_preference mismatch vs matrix row mean"
            )
            if len(errs) >= 25:
                return errs

        if not (0.0 - 1e-9 <= float(r.binary_win_rate) <= 1.0 + 1e-9):
            errs.append(
                f"model_summary.csv model {m!r}: binary_win_rate out of [0, 1]"
            )
        if not (PREF_MIN - 1e-6 <= float(r.mean_preference) <= PREF_MAX + 1e-6):
            errs.append(
                f"model_summary.csv model {m!r}: mean_preference out of [{PREF_MIN}, {PREF_MAX}]"
            )

    return errs


def main() -> int:
    paths = [RESPONSE_MATRIX, RESPONSE_PREF, ITEM_METADATA, MODEL_SUMMARY]
    path_errs = path_errors_if_missing(paths)
    if path_errs:
        for e in path_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    id_col = "Model"
    all_errs: list[str] = []

    e_win, win_df, win_tasks = _audit_matrix_layout(RESPONSE_MATRIX, id_col=id_col, label="response_matrix.csv")
    all_errs.extend(e_win)
    if win_df is None:
        for e in all_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    e_pref, pref_df, pref_tasks = _audit_matrix_layout(
        RESPONSE_PREF, id_col=id_col, label="response_matrix_preference.csv"
    )
    all_errs.extend(e_pref)
    if pref_df is None:
        for e in all_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    all_errs.extend(_models_match(win_df, pref_df, id_col))
    all_errs.extend(_columns_match(list(win_df.columns), list(pref_df.columns), id_col))

    task_cols = win_tasks
    w_num = win_df[task_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    p_num = pref_df[task_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    all_errs.extend(_audit_binary_cells(w_num, "response_matrix.csv"))
    all_errs.extend(_audit_preference_cells(p_num, "response_matrix_preference.csv"))

    if not all_errs:
        all_errs.extend(_audit_win_pref_consistency(win_df, task_cols, id_col, w_num, p_num))

    meta = pd.read_csv(ITEM_METADATA)
    all_errs.extend(_audit_item_metadata(meta, task_cols, id_col, p_num, w_num))

    summary = pd.read_csv(MODEL_SUMMARY)
    all_errs.extend(_audit_model_summary(summary, win_df, task_cols, id_col, p_num, w_num))

    if all_errs:
        for e in all_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_models = len(win_df)
    n_items = len(task_cols)
    print(
        f"OK: AlpacaEval — {n_models} models × {n_items} items; "
        f"binary/preferences consistent (threshold {WIN_THRESHOLD}); "
        "item_metadata and model_summary match matrices."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
