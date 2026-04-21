"""
UltraFeedback audit (raw). From 01_build stream_and_extract.

extracted_scores.csv: columns model, prompt_id, mean_score, n_aspects in that order; model and prompt_id nonempty; mean_score numeric 1 through 5; n_aspects integer 4; no duplicate model and prompt_id pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import bad_pct_suffix, path_errors_if_missing, raw_dir_from_script  # noqa: E402

EXPECTED_COLUMNS = ["model", "prompt_id", "mean_score", "n_aspects"]

# Per-aspect GPT-4 ratings are 1–5; mean must stay in that range.
SCORE_MIN, SCORE_MAX = 1.0, 5.0

# UltraFeedback GPT-4 annotations use four aspects per completion.
EXPECTED_N_ASPECTS = 4

EXTRACTED_SCORES = raw_dir_from_script(__file__) / "extracted_scores.csv"


def _audit_extracted_scores(df: pd.DataFrame, errors: list[str]) -> None:
    n = len(df)

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"extracted_scores.csv columns mismatch; expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
            + bad_pct_suffix(max(n, 1), max(n, 1))
        )
        return

    if n == 0:
        errors.append("extracted_scores.csv: empty table")
        return

    bad_model = df["model"].isna() | (df["model"].astype(str).str.strip() == "")
    if bad_model.any():
        errors.append(
            "model must be non-empty" + bad_pct_suffix(int(bad_model.sum()), n)
        )

    bad_pid = df["prompt_id"].isna() | (df["prompt_id"].astype(str).str.strip() == "")
    if bad_pid.any():
        errors.append(
            "prompt_id must be non-empty" + bad_pct_suffix(int(bad_pid.sum()), n)
        )

    score = pd.to_numeric(df["mean_score"], errors="coerce")
    bad_score_nan = score.isna()
    if bad_score_nan.any():
        errors.append(
            "mean_score must be numeric" + bad_pct_suffix(int(bad_score_nan.sum()), n)
        )
    bad_score_range = score.notna() & ((score < SCORE_MIN) | (score > SCORE_MAX))
    if bad_score_range.any():
        errors.append(
            f"mean_score must be in [{SCORE_MIN}, {SCORE_MAX}]"
            + bad_pct_suffix(int(bad_score_range.sum()), n)
        )

    nas = pd.to_numeric(df["n_aspects"], errors="coerce")
    valid_n = nas.notna() & (nas % 1 == 0) & (nas == EXPECTED_N_ASPECTS)
    if (~valid_n).any():
        errors.append(
            f"n_aspects must be exactly {EXPECTED_N_ASPECTS} (integer)"
            + bad_pct_suffix(int((~valid_n).sum()), n)
        )

    dup = df.duplicated(["model", "prompt_id"], keep=False)
    if dup.any():
        errors.append(
            "duplicate (model, prompt_id) rows (pivot would aggregate; expect one row per pair)"
            + bad_pct_suffix(int(dup.sum()), n)
        )


def main() -> int:
    errs = path_errors_if_missing([EXTRACTED_SCORES])
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    df = pd.read_csv(EXTRACTED_SCORES)
    errors: list[str] = []
    _audit_extracted_scores(df, errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"OK: extracted_scores.csv — {len(df):,} rows passed checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
