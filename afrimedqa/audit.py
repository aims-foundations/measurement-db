"""
AfriMed-QA audit (processed). Built by 01_build_response_matrix.py.

response_matrix.csv: sample_id unique; model columns unique; each cell 0 or 1 unless allow-missing flag allows blank cells.
task_metadata.csv: fixed schema; row count and item_id order match response_matrix sample_id.
model_summary.csv: fixed schema; model set matches matrix columns; n_items_evaluated equals n_correct plus n_incorrect; accuracy and coverage match matrix; source_dataset and prompt_type in allowed sets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import bad_pct_suffix, path_errors_if_missing, processed_dir_from_script  # noqa: E402

EXPECTED_TASK_METADATA_COLUMNS = [
    "item_id",
    "question",
    "answer",
    "specialty",
    "country",
    "region_specific",
    "question_type",
]

EXPECTED_MODEL_SUMMARY_COLUMNS = [
    "model",
    "accuracy",
    "n_items_evaluated",
    "n_correct",
    "n_incorrect",
    "coverage",
    "source_dataset",
    "prompt_type",
]

SOURCE_DATASET_OK = frozenset(
    {"afrimedqa-v2", "afrimedqa-v1", "afrimedqa-v2.5", "medqa", "unknown", ""}
)
PROMPT_TYPE_OK = frozenset({"base", "instruct", ""})


def _norm_cell_str(val: object) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _classify_model_cell(val: object) -> str:
    """Return ``missing`` | ``ok`` | ``bad`` for one response-matrix model cell."""
    if isinstance(val, str) and val.strip() == "":
        return "missing"
    if pd.isna(val):
        return "missing"
    num = pd.to_numeric(val, errors="coerce")
    if pd.isna(num):
        return "bad"
    fv = float(num)
    if fv == 0.0 or fv == 1.0:
        return "ok"
    return "bad"


def _audit_response_matrix(
    df: pd.DataFrame,
    errors: list[str],
    *,
    allow_missing: bool,
) -> tuple[list[str], int]:
    """Return (model_column_names, n_rows). On hard failure, model list may be partial."""
    label = "response_matrix.csv"
    n = len(df)
    if n == 0:
        errors.append(f"{label}: empty table")
        return [], 0

    id_col = df.columns[0]
    if id_col != "sample_id":
        errors.append(
            f"{label}: first column must be named 'sample_id', got {id_col!r}"
            + bad_pct_suffix(n, n)
        )
        return [], n

    model_cols = [c for c in df.columns if c != "sample_id"]
    if not model_cols:
        errors.append(f"{label}: no model columns after sample_id" + bad_pct_suffix(n, n))
        return [], n

    if len(set(model_cols)) != len(model_cols):
        errors.append(f"{label}: duplicate model column names" + bad_pct_suffix(n, n))

    sid = df["sample_id"]
    bad_sid = sid.isna() | (sid.astype(str).str.strip() == "")
    if bad_sid.any():
        errors.append(
            f"{label}: sample_id must be non-empty"
            + bad_pct_suffix(int(bad_sid.sum()), n)
        )

    dup = sid.duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate sample_id values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    for mc in model_cols:
        if _norm_cell_str(mc) == "":
            errors.append(f"{label}: empty model column name")

    n_cells = n * len(model_cols)
    n_missing = 0
    n_bad = 0
    for c in model_cols:
        for v in df[c]:
            t = _classify_model_cell(v)
            if t == "missing":
                n_missing += 1
            elif t == "bad":
                n_bad += 1

    if not allow_missing and n_missing:
        pct = 100.0 * n_missing / n_cells
        errors.append(
            f"{label}: every model cell must be 0 or 1 (no blank/NaN/empty-string cells); "
            f"found {n_missing:,} empty/missing / {n_cells:,} ({pct:.2f}%) "
            f"— pass --allow-missing if sparse evaluation is expected"
        )

    if n_bad:
        pct = 100.0 * n_bad / n_cells
        errors.append(
            f"{label}: non-missing cells must be exactly 0 or 1"
            f" — malformed cells: {n_bad:,} / {n_cells:,} ({pct:.2f}%)"
        )

    return model_cols, n


def _audit_task_metadata(
    meta: pd.DataFrame,
    *,
    sample_ids: pd.Series,
    n_matrix_rows: int,
    errors: list[str],
) -> None:
    label = "task_metadata.csv"
    n = len(meta)
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    if list(meta.columns) != EXPECTED_TASK_METADATA_COLUMNS:
        errors.append(
            f"{label}: columns mismatch; expected {EXPECTED_TASK_METADATA_COLUMNS}, "
            f"got {list(meta.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    if n != n_matrix_rows:
        errors.append(
            f"{label}: row count {n:,} != response_matrix row count {n_matrix_rows:,}"
        )

    iid = meta["item_id"]
    bad_iid = iid.isna() | (iid.astype(str).str.strip() == "")
    if bad_iid.any():
        errors.append(
            f"{label}: item_id must be non-empty"
            + bad_pct_suffix(int(bad_iid.sum()), n)
        )

    dup = iid.duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate item_id values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    if n == n_matrix_rows and len(sample_ids) == n:
        misaligned = (meta["item_id"].astype(str).values != sample_ids.astype(str).values).sum()
        if misaligned:
            errors.append(
                f"{label}: item_id order must match response_matrix sample_id rows"
                f" — mismatched rows: {misaligned:,} / {n:,}"
            )


def _audit_model_summary(
    summary: pd.DataFrame,
    *,
    model_cols: list[str],
    response_df: pd.DataFrame,
    n_rows: int,
    errors: list[str],
) -> None:
    label = "model_summary.csv"
    n = len(summary)
    if n == 0:
        errors.append(f"{label}: empty table")
        return

    if list(summary.columns) != EXPECTED_MODEL_SUMMARY_COLUMNS:
        errors.append(
            f"{label}: columns mismatch; expected {EXPECTED_MODEL_SUMMARY_COLUMNS}, "
            f"got {list(summary.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    models_in_summary = summary["model"].astype(str)
    bad_m = models_in_summary.isna() | (models_in_summary.str.strip() == "")
    if bad_m.any():
        errors.append(
            f"{label}: model must be non-empty"
            + bad_pct_suffix(int(bad_m.sum()), n)
        )

    dup = summary["model"].duplicated(keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate model values"
            + bad_pct_suffix(int(dup.sum()), n)
        )

    set_summary = frozenset(summary["model"].astype(str))
    set_matrix = frozenset(model_cols)
    if model_cols and set_summary != set_matrix:
        only_s = sorted(set_summary - set_matrix)
        only_m = sorted(set_matrix - set_summary)
        errors.append(
            f"{label}: model names must match response_matrix columns exactly "
            f"(only in summary: {only_s[:6]!r}{'…' if len(only_s) > 6 else ''}, "
            f"only in matrix: {only_m[:6]!r}{'…' if len(only_m) > 6 else ''})"
        )

    for col in ("n_items_evaluated", "n_correct", "n_incorrect"):
        s = pd.to_numeric(summary[col], errors="coerce")
        if s.isna().any():
            errors.append(
                f"{label}: {col} must be numeric"
                + bad_pct_suffix(int(s.isna().sum()), n)
            )

    for _, row in summary.iterrows():
        m = str(row["model"])
        ni = int(row["n_items_evaluated"])
        nc = int(row["n_correct"])
        ninc = int(row["n_incorrect"])
        if ni != nc + ninc:
            errors.append(
                f"{label}: model {m!r}: n_items_evaluated ({ni}) != "
                f"n_correct ({nc}) + n_incorrect ({ninc})"
            )

        acc = row["accuracy"]
        cov = row["coverage"]
        if ni > 0:
            exp_acc = nc / ni
            if pd.isna(acc) or not np.isclose(float(acc), exp_acc, rtol=0.0, atol=1e-3):
                errors.append(
                    f"{label}: model {m!r}: accuracy {acc!r} != n_correct/n_items "
                    f"({exp_acc:.6f}) within tolerance"
                )
            if n_rows > 0:
                exp_cov = ni / n_rows
                if pd.isna(cov) or not np.isclose(float(cov), exp_cov, rtol=0.0, atol=1e-3):
                    errors.append(
                        f"{label}: model {m!r}: coverage {cov!r} != n_items/n_rows "
                        f"({exp_cov:.6f}) within tolerance"
                    )
        else:
            if not pd.isna(acc):
                errors.append(
                    f"{label}: model {m!r}: accuracy should be NaN when n_items_evaluated==0"
                )

        sd = _norm_cell_str(row["source_dataset"])
        if sd not in SOURCE_DATASET_OK:
            errors.append(
                f"{label}: model {m!r}: source_dataset {sd!r} not in allowed set"
            )

        pt = _norm_cell_str(row["prompt_type"])
        if pt not in PROMPT_TYPE_OK:
            errors.append(
                f"{label}: model {m!r}: prompt_type {pt!r} not in allowed set"
            )

    # Reconcile counts with the matrix column (when model exists in frame)
    if model_cols and n_rows > 0:
        for _, row in summary.iterrows():
            m = str(row["model"])
            if m not in response_df.columns:
                continue
            col = response_df[m]
            s = pd.to_numeric(col, errors="coerce")
            n_eval = int(s.notna().sum())
            n_cor = int((s == 1).sum())
            n_inc = int((s == 0).sum())
            ni = int(row["n_items_evaluated"])
            nc = int(row["n_correct"])
            nii = int(row["n_incorrect"])
            if (ni, nc, nii) != (n_eval, n_cor, n_inc):
                errors.append(
                    f"{label}: model {m!r}: summary counts (eval={ni}, cor={nc}, inc={nii}) "
                    f"!= matrix (eval={n_eval}, cor={n_cor}, inc={n_inc})"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="AfriMed-QA processed CSV audit.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow NaN in response_matrix (sparse); default requires dense 0/1.",
    )
    args = parser.parse_args()

    proc = processed_dir_from_script(__file__)
    paths = {
        "response_matrix.csv": proc / "response_matrix.csv",
        "task_metadata.csv": proc / "task_metadata.csv",
        "model_summary.csv": proc / "model_summary.csv",
    }
    missing_errs = path_errors_if_missing(paths.values())
    if missing_errs:
        for e in missing_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    errors: list[str] = []

    try:
        response_df = pd.read_csv(paths["response_matrix.csv"], low_memory=False)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: response_matrix.csv: cannot read CSV ({exc!r})", file=sys.stderr)
        return 1

    model_cols, n_rows = _audit_response_matrix(
        response_df, errors, allow_missing=args.allow_missing
    )

    try:
        meta_df = pd.read_csv(paths["task_metadata.csv"], low_memory=False)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: task_metadata.csv: cannot read CSV ({exc!r})", file=sys.stderr)
        return 1

    sample_ids = (
        response_df["sample_id"]
        if "sample_id" in response_df.columns
        else pd.Series(dtype=object)
    )
    _audit_task_metadata(meta_df, sample_ids=sample_ids, n_matrix_rows=n_rows, errors=errors)

    try:
        summary_df = pd.read_csv(paths["model_summary.csv"], low_memory=False)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: model_summary.csv: cannot read CSV ({exc!r})", file=sys.stderr)
        return 1

    _audit_model_summary(
        summary_df,
        model_cols=model_cols,
        response_df=response_df,
        n_rows=n_rows,
        errors=errors,
    )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    mode = "sparse cells allowed" if args.allow_missing else "dense matrix (no missing cells)"
    print(
        "OK: AfriMed-QA — response_matrix.csv, task_metadata.csv, model_summary.csv passed checks "
        f"({n_rows:,} items × {len(model_cols)} models; {mode})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
