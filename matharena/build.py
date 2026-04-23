"""
Build MathArena long-form responses from per-model per-problem output data.

Data sources:
  MathArena HuggingFace datasets (https://huggingface.co/MathArena):
  - 27 output datasets covering AIME 2025/2026, HMMT, BRUMO, CMIMC, SMT,
    APEX, ArXivMath, Kangaroo, IMO, IMC, USAMO, Putnam, Miklos competitions
  - Each dataset has ~62 models evaluated 4 times per problem
  - Final-answer competitions have a `correct` boolean field
  - Proof-based competitions have `points_judge_1/2` fields (0-7 scale)

Items are scoped per-competition. Each problem within a competition becomes a
distinct item_id (disambiguated by the competition name prefix in the
`raw_item_id`); multiple attempts on the same (model, item) are encoded as
distinct `trial` values via `ensure_unique_trials`.

GitHub: https://github.com/eth-sri/matharena
Paper: "MathArena: Evaluating LLMs on Uncontaminated Math Competitions" (NeurIPS D&B 2025)

Outputs:
  - responses.parquet  # long-form
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'MathArena: uncontaminated math competitions (27 datasets); final-answer accuracy + proof-judge scores across multiple attempts per problem.',
    'testing_condition': 'test_condition encodes the competition (e.g. "competition=aime_2025"). Multiple attempts per (model, problem) become distinct trials.',
    'paper_url': 'https://arxiv.org/abs/2505.23281',
    'data_source_url': 'https://github.com/eth-sri/matharena',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'unknown',
    'citation': """@misc{balunović2026matharenaevaluatingllmsuncontaminated,
      title={MathArena: Evaluating LLMs on Uncontaminated Math Competitions},
      author={Mislav Balunović and Jasper Dekoninck and Ivo Petrov and Nikola Jovanović and Martin Vechev},
      year={2026},
      eprint={2505.23281},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.23281},
}""",
    'tags': ['reasoning'],
}


import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

FINAL_ANSWER_DATASETS = {
    "aime_2025": "MathArena/aime_2025_outputs",
    "aime_2025_I": "MathArena/aime_2025_I_outputs",
    "aime_2025_II": "MathArena/aime_2025_II_outputs",
    "aime_2026": "MathArena/aime_2026_outputs",
    "aime_2026_I": "MathArena/aime_2026_I_outputs",
    "hmmt_feb_2025": "MathArena/hmmt_feb_2025_outputs",
    "hmmt_feb_2026": "MathArena/hmmt_feb_2026_outputs",
    "hmmt_nov_2025": "MathArena/hmmt_nov_2025_outputs",
    "brumo_2025": "MathArena/brumo_2025_outputs",
    "cmimc_2025": "MathArena/cmimc_2025_outputs",
    "smt_2025": "MathArena/smt_2025_outputs",
    "apex_2025": "MathArena/apex_2025_outputs",
    "apex_shortlist": "MathArena/apex_shortlist_outputs",
    "arxivmath_1225": "MathArena/arxivmath-1225_outputs",
    "arxivmath_0126": "MathArena/arxivmath-0126_outputs",
    "arxivmath_0226": "MathArena/arxivmath-0226_outputs",
    "kangaroo_2025_1_2": "MathArena/kangaroo_2025_1-2_outputs",
    "kangaroo_2025_3_4": "MathArena/kangaroo_2025_3-4_outputs",
    "kangaroo_2025_5_6": "MathArena/kangaroo_2025_5-6_outputs",
    "kangaroo_2025_7_8": "MathArena/kangaroo_2025_7-8_outputs",
    "kangaroo_2025_9_10": "MathArena/kangaroo_2025_9-10_outputs",
    "kangaroo_2025_11_12": "MathArena/kangaroo_2025_11-12_outputs",
}

PROOF_DATASETS = {
    "usamo_2025": "MathArena/usamo_2025_outputs",
    "imo_2025": "MathArena/imo_2025_outputs",
    "imc_2025": "MathArena/imc_2025_outputs",
    "putnam_2025": "MathArena/putnam_2025_outputs",
    "miklos_2025": "MathArena/miklos_2025_outputs",
}


def download_dataset(dataset_id, comp_name):
    """Download a HuggingFace dataset and cache as parquet in raw/."""
    cache_path = RAW_DIR / f"{comp_name}.parquet"
    if cache_path.exists():
        print(f"  [cached] {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading {dataset_id} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train")
        keep_cols = [
            "problem_idx", "model_name", "idx_answer", "correct",
            "gold_answer", "parsed_answer", "problem", "problem_statement",
            "cost", "input_tokens", "output_tokens",
            "points_judge_1", "max_points_judge_1",
            "points_judge_2", "max_points_judge_2",
        ]
        available_cols = [c for c in keep_cols if c in ds.column_names]
        ds_small = ds.select_columns(available_cols)
        df = ds_small.to_pandas()
        df.to_parquet(cache_path, index=False)
        print(f"  Saved to {cache_path} ({len(df)} rows, "
              f"{os.path.getsize(cache_path)/1024:.1f} KB)")
        return df
    except Exception as e:
        print(f"  ERROR downloading {dataset_id}: {e}")
        return None


def _problem_content(row) -> str | None:
    """Best-effort problem text."""
    for key in ("problem", "problem_statement", "gold_answer"):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            s = str(v).strip()
            if s:
                return s[:4000]
    return None


def _build_rows_final_answer(df: pd.DataFrame, comp_name: str, bench_id: str) -> list[dict]:
    """Emit one row per (model, problem, attempt) with binary correctness."""
    if df is None or "correct" not in df.columns:
        print(f"  SKIP {comp_name}: no 'correct' column")
        return []

    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

    # Register items once per (comp, problem_idx)
    item_ids: dict = {}
    item_gold: dict = {}
    # Pick a representative row per problem for content/gold_answer
    problem_keys = df["problem_idx"].drop_duplicates().tolist()
    for pid in problem_keys:
        sample_row = df[df["problem_idx"] == pid].iloc[0].to_dict()
        content = _problem_content(sample_row)
        gold = sample_row.get("gold_answer")
        gold_str = None if gold is None or (isinstance(gold, float) and pd.isna(gold)) else str(gold)
        iid = register_item(
            benchmark_id=bench_id,
            raw_item_id=f"{comp_name}::{pid}",
            content=content,
            correct_answer=gold_str,
        )
        item_ids[pid] = iid
        item_gold[pid] = gold_str

    rows = []
    for rec in df.itertuples(index=False):
        corr = getattr(rec, "correct", None)
        if corr is None or (isinstance(corr, float) and pd.isna(corr)):
            continue
        subj = resolve_subject(str(rec.model_name))
        pid = rec.problem_idx
        iid = item_ids.get(pid)
        if iid is None:
            continue
        rows.append({
            "subject_id": subj,
            "item_id": iid,
            "benchmark_id": bench_id,
            "trial": 1,  # overwritten by ensure_unique_trials
            "test_condition": f"competition={comp_name}",
            "response": float(corr),
            "correct_answer": item_gold.get(pid),
            "trace": None,
        })
    print(f"  {comp_name}: {len(rows):,} rows (final-answer)")
    return rows


def _build_rows_proof(df: pd.DataFrame, comp_name: str, bench_id: str) -> list[dict]:
    """Emit one row per (model, problem, attempt) with normalized judge score."""
    if df is None:
        return []

    has_j1 = "points_judge_1" in df.columns
    has_j2 = "points_judge_2" in df.columns
    if not has_j1 and not has_j2:
        print(f"  SKIP {comp_name}: no points columns")
        return []

    df = df.copy()
    if has_j1 and has_j2:
        p1 = pd.to_numeric(df["points_judge_1"], errors="coerce").fillna(0)
        p2 = pd.to_numeric(df["points_judge_2"], errors="coerce").fillna(0)
        m1 = pd.to_numeric(df.get("max_points_judge_1", pd.Series([7] * len(df))),
                           errors="coerce").fillna(7)
        m2 = pd.to_numeric(df.get("max_points_judge_2", pd.Series([7] * len(df))),
                           errors="coerce").fillna(7)
        df["norm_points"] = ((p1 / m1) + (p2 / m2)) / 2.0
    elif has_j1:
        p1 = pd.to_numeric(df["points_judge_1"], errors="coerce").fillna(0)
        m1 = pd.to_numeric(df.get("max_points_judge_1", pd.Series([7] * len(df))),
                           errors="coerce").fillna(7)
        df["norm_points"] = p1 / m1
    else:
        p2 = pd.to_numeric(df["points_judge_2"], errors="coerce").fillna(0)
        m2 = pd.to_numeric(df.get("max_points_judge_2", pd.Series([7] * len(df))),
                           errors="coerce").fillna(7)
        df["norm_points"] = p2 / m2
    df["norm_points"] = df["norm_points"].clip(0, 1)

    item_ids: dict = {}
    item_gold: dict = {}
    for pid in df["problem_idx"].drop_duplicates().tolist():
        sample_row = df[df["problem_idx"] == pid].iloc[0].to_dict()
        content = _problem_content(sample_row)
        gold = sample_row.get("gold_answer")
        gold_str = None if gold is None or (isinstance(gold, float) and pd.isna(gold)) else str(gold)
        iid = register_item(
            benchmark_id=bench_id,
            raw_item_id=f"{comp_name}::{pid}",
            content=content,
            correct_answer=gold_str,
        )
        item_ids[pid] = iid
        item_gold[pid] = gold_str

    rows = []
    for rec in df.itertuples(index=False):
        pts = getattr(rec, "norm_points", None)
        if pts is None or (isinstance(pts, float) and pd.isna(pts)):
            continue
        subj = resolve_subject(str(rec.model_name))
        pid = rec.problem_idx
        iid = item_ids.get(pid)
        if iid is None:
            continue
        rows.append({
            "subject_id": subj,
            "item_id": iid,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": f"competition={comp_name}",
            "response": float(pts),
            "correct_answer": item_gold.get(pid),
            "trace": None,
        })
    print(f"  {comp_name}: {len(rows):,} rows (proof)")
    return rows


def main():
    print("MathArena long-form builder")
    print("=" * 65)

    bench_id = get_benchmark_id(
        "matharena",
        name="MathArena",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
    )

    all_rows: list[dict] = []

    print(f"\n  Processing {len(FINAL_ANSWER_DATASETS)} final-answer competitions...")
    for comp_name, dataset_id in sorted(FINAL_ANSWER_DATASETS.items()):
        df = download_dataset(dataset_id, comp_name)
        if df is None:
            continue
        all_rows.extend(_build_rows_final_answer(df, comp_name, bench_id))

    print(f"\n  Processing {len(PROOF_DATASETS)} proof-based competitions...")
    for comp_name, dataset_id in sorted(PROOF_DATASETS.items()):
        df = download_dataset(dataset_id, comp_name)
        if df is None:
            continue
        all_rows.extend(_build_rows_proof(df, comp_name, bench_id))

    cols = [
        "subject_id", "item_id", "benchmark_id", "trial",
        "test_condition", "response", "correct_answer", "trace",
    ]
    out_df = pd.DataFrame(all_rows, columns=cols)
    out_df = ensure_unique_trials(out_df)
    out_df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(out_df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")

    if out_df.empty:
        print("  WARNING: no rows produced")
        return

    print(f"\n  subjects: {out_df['subject_id'].nunique()}")
    print(f"  items:    {out_df['item_id'].nunique()}")
    print(f"  rows:     {len(out_df):,}")
    print(f"  max trial: {int(out_df['trial'].max())}")


if __name__ == "__main__":
    main()
