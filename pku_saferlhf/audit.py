"""
PKU-SafeRLHF audit (processed).

safety_summary.csv: column order EXPECTED_SUMMARY_COLUMNS in this script; config default; split train or test; better_response_id and safer_response_id 0 or 1; is_response_*_safe boolean or 0 or 1; response_*_len positive; prompt_source nonempty; response_*_severity_level 0 through 3; pair_idx is 0 through n minus 1 once per split; response_0_source and response_1_source values not audited.
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

EXPECTED_SUMMARY_COLUMNS = [
    "config",
    "split",
    "pair_idx",
    "response_0_len",
    "response_1_len",
    "prompt_source",
    "response_0_source",
    "response_1_source",
    "is_response_0_safe",
    "is_response_1_safe",
    "response_0_severity_level",
    "response_1_severity_level",
    "better_response_id",
    "safer_response_id",
]

SAFETY_SUMMARY = processed_dir_from_script(__file__) / "safety_summary.csv"


def _audit_safety_summary(df: pd.DataFrame, errors: list[str]) -> None:
    n = len(df)
    if n == 0:
        errors.append("safety_summary.csv: empty table")
        return

    if list(df.columns) != EXPECTED_SUMMARY_COLUMNS:
        errors.append(
            f"safety_summary.csv columns mismatch; expected {EXPECTED_SUMMARY_COLUMNS}, got {list(df.columns)}"
            + bad_pct_suffix(n, n)
        )
        return

    bad_cfg = df["config"] != "default"
    if bad_cfg.any():
        vals = df.loc[bad_cfg, "config"].unique().tolist()
        errors.append(
            f"Expected all config=='default'; found {vals!r}"
            + bad_pct_suffix(int(bad_cfg.sum()), n)
        )

    bad_split = ~df["split"].isin(["train", "test"])
    if bad_split.any():
        errors.append(
            "split must be only 'train' or 'test'"
            + bad_pct_suffix(int(bad_split.sum()), n)
        )

    for col in ("better_response_id", "safer_response_id"):
        s = pd.to_numeric(df[col], errors="coerce")
        arr = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(arr)
        bad_empty = int((~finite).sum())
        bad_val = int((finite & (arr != 0.0) & (arr != 1.0)).sum())
        if bad_empty:
            errors.append(
                f"{col} must be non-empty and numeric"
                + bad_pct_suffix(bad_empty, n)
            )
        if bad_val:
            errors.append(
                f"{col} must be exactly 0 or 1"
                + bad_pct_suffix(bad_val, n)
            )

    for col in ("is_response_0_safe", "is_response_1_safe"):
        bad = ~df[col].isin([True, False, 0, 1])
        if bad.any():
            errors.append(
                f"{col} must be boolean (or 0/1)" + bad_pct_suffix(int(bad.sum()), n)
            )

    for col in ("response_0_len", "response_1_len"):
        bad = df[col] <= 0
        if bad.any():
            errors.append(
                f"{col} must be positive for all rows" + bad_pct_suffix(int(bad.sum()), n)
            )

    bad_prompt_src = df["prompt_source"].isna() | (
        df["prompt_source"].astype(str).str.strip() == ""
    )
    if bad_prompt_src.any():
        errors.append(
            "prompt_source must be non-empty"
            + bad_pct_suffix(int(bad_prompt_src.sum()), n)
        )

    for col in ("response_0_severity_level", "response_1_severity_level"):
        s = pd.to_numeric(df[col], errors="coerce")
        arr = np.asarray(s, dtype=np.float64)
        finite = np.isfinite(arr)
        ok = finite & (
            (arr == 0.0) | (arr == 1.0) | (arr == 2.0) | (arr == 3.0)
        )
        bad = ~ok
        if bad.any():
            errors.append(
                f"{col} must be exactly 0, 1, 2, or 3"
                + bad_pct_suffix(int(bad.sum()), n)
            )

    n_bad_pair = 0
    pair_parts: list[str] = []
    for split, g in df.groupby("split"):
        m = len(g)
        if set(g["pair_idx"].tolist()) != set(range(m)):
            n_bad_pair += m
            pair_parts.append(f"{split!r} ({m:,} rows)")
    if n_bad_pair:
        errors.append(
            "pair_idx must be exactly 0..n-1 once per split; bad splits: "
            + ", ".join(pair_parts)
            + bad_pct_suffix(n_bad_pair, n)
        )


def main() -> int:
    errs = path_errors_if_missing([SAFETY_SUMMARY])
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    df = pd.read_csv(SAFETY_SUMMARY)
    errors: list[str] = []
    _audit_safety_summary(df, errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"OK: safety_summary.csv — {len(df):,} rows passed checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
