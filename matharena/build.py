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
    'description': 'MathArena: uncontaminated math competitions (27 datasets); final-answer accuracy + proof-judge per-criterion rubric scores across multiple attempts per problem.',
    'testing_condition': 'Binary-verdict competitions have test_condition=None; partial-credit (proof) competitions emit one row per (judge, criterion) with test_condition=f"judge={k};criterion={title}" and response=points/max_points in [0,1]. Multiple attempts per (model, problem) become distinct trials.',
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
    'modality': ['text'],
    'domain': ['mathematics'],
    'response_type': 'mixed',
    'response_scale': 'binary for AIME family (per-attempt correct); continuous fraction points/max for rubric comps (USAMO/IMO/IMC/Putnam/Miklos, per-criterion)',
    'categorical': False,
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
        df = pd.read_parquet(cache_path)
        # Back-fill the ``problem`` column when a legacy cache lacks it. Problem
        # text never changes upstream per (comp, problem_idx), so we only fetch
        # a thin per-problem supplement and cache it alongside.
        if "problem" not in df.columns and "problem_idx" in df.columns:
            problems = _load_problem_supplement(dataset_id, comp_name)
            if problems is not None:
                df = df.merge(problems, on="problem_idx", how="left")
        return df

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


def download_outputs_dataset(dataset_id, comp_name):
    """Download the full ``_outputs`` dataset (includes per-criterion rubric details)
    and cache as ``{comp}_outputs.parquet`` in raw/.

    Unlike :func:`download_dataset`, this keeps ``grading_details_judge_*`` and
    ``problem`` so we can emit per-criterion rows for partial-credit comps.
    """
    cache_path = RAW_DIR / f"{comp_name}_outputs.parquet"
    if cache_path.exists():
        print(f"  [cached] {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading {dataset_id} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train")
        keep_cols = [
            "problem_idx", "model_name", "idx_answer",
            "problem", "problem_statement",
            "points_judge_1", "max_points_judge_1", "grading_details_judge_1",
            "points_judge_2", "max_points_judge_2", "grading_details_judge_2",
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


def _load_problem_supplement(dataset_id: str, comp_name: str) -> "pd.DataFrame | None":
    """Return a tiny (problem_idx, problem) table for one competition.

    Cached at ``raw/problems/{comp_name}.parquet`` so re-runs don't re-hit HF.
    """
    supp_dir = RAW_DIR / "problems"
    supp_dir.mkdir(parents=True, exist_ok=True)
    supp_path = supp_dir / f"{comp_name}.parquet"
    if supp_path.exists():
        try:
            return pd.read_parquet(supp_path)
        except Exception:
            pass
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train")
        cols = ds.column_names
        if "problem_idx" not in cols:
            return None
        text_col = None
        for c in ("problem", "problem_statement"):
            if c in cols:
                text_col = c
                break
        if text_col is None:
            return None
        ds_small = ds.select_columns(["problem_idx", text_col])
        df = ds_small.to_pandas().drop_duplicates(subset=["problem_idx"]).reset_index(drop=True)
        if text_col != "problem":
            df = df.rename(columns={text_col: "problem"})
        df.to_parquet(supp_path, index=False)
        print(f"  Supplement cached: {supp_path.name} ({len(df)} problems)")
        return df
    except Exception as e:
        print(f"  WARN: problem supplement fetch failed for {comp_name}: {e}")
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


def _register_items_for_comp(df: pd.DataFrame, comp_name: str, bench_id: str):
    """Register one item per (comp, problem_idx). Returns (item_ids, item_gold)."""
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
    return item_ids, item_gold


def _build_rows_final_answer(df: pd.DataFrame, comp_name: str, bench_id: str) -> list[dict]:
    """Binary-verdict competition: one row per (model, problem, idx_answer).

    ``trial = idx_answer + 1``, ``test_condition = None``, ``response`` is 0.0/1.0
    from the upstream ``correct`` boolean.
    """
    if df is None or "correct" not in df.columns:
        print(f"  SKIP {comp_name}: no 'correct' column")
        return []

    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

    item_ids, item_gold = _register_items_for_comp(df, comp_name, bench_id)

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
        idx_answer = getattr(rec, "idx_answer", 0)
        try:
            trial = int(idx_answer) + 1
        except (TypeError, ValueError):
            trial = 1
        rows.append({
            "subject_id": subj,
            "item_id": iid,
            "benchmark_id": bench_id,
            "trial": trial,  # ensure_unique_trials may bump if collisions remain
            "test_condition": None,
            "response": float(corr),
            "correct_answer": item_gold.get(pid),
            "trace": None,
        })
    print(f"  {comp_name}: {len(rows):,} rows (final-answer, per-attempt)")
    return rows


def _build_rows_proof(df: pd.DataFrame, comp_name: str, bench_id: str) -> list[dict]:
    """Rubric competition: one row per (model, problem, idx_answer, judge, criterion).

    The upstream ``_outputs`` dataset attaches a ``grading_details_judge_{k}``
    list to every (model, problem, attempt), where each element is a dict with
    ``{title, grading_scheme_desc, max_points, points, desc}``. We emit one row
    per criterion with::

        test_condition = f"judge={k};criterion={title}"
        response       = points / max_points  (clipped to [0, 1])

    Criteria with NaN/0 ``max_points`` or NaN ``points`` are skipped; we never
    invent criteria when upstream returns an empty list. Missing judge columns
    (e.g. IMC/Putnam/Miklos have only judge 1) are handled gracefully — we
    iterate whatever ``grading_details_judge_*`` columns are present.
    The item identity is still one-per-problem; the rubric criterion lives in
    ``test_condition``, NOT in the item.
    """
    if df is None:
        return []

    # Discover which judges' grading_details are present.
    judge_keys = []
    for k in (1, 2):
        if f"grading_details_judge_{k}" in df.columns:
            judge_keys.append(k)
    if not judge_keys:
        print(f"  SKIP {comp_name}: no grading_details_judge_* columns")
        return []

    item_ids, item_gold = _register_items_for_comp(df, comp_name, bench_id)

    def _num(x):
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    rows = []
    n_empty_details = 0
    n_skipped_criteria = 0
    for rec in df.itertuples(index=False):
        subj = resolve_subject(str(rec.model_name))
        pid = rec.problem_idx
        iid = item_ids.get(pid)
        if iid is None:
            continue
        idx_answer = getattr(rec, "idx_answer", 0)
        try:
            trial = int(idx_answer) + 1
        except (TypeError, ValueError):
            trial = 1
        for k in judge_keys:
            details = getattr(rec, f"grading_details_judge_{k}", None)
            if details is None:
                continue
            # Upstream sometimes returns a numpy array rather than a list.
            try:
                iterable = list(details)
            except TypeError:
                continue
            if len(iterable) == 0:
                n_empty_details += 1
                continue
            for crit in iterable:
                if crit is None or not hasattr(crit, "get"):
                    # Tolerate dict-like; numpy arrays of dicts expose .get.
                    try:
                        crit = dict(crit)
                    except Exception:
                        n_skipped_criteria += 1
                        continue
                title = crit.get("title")
                max_pts = _num(crit.get("max_points"))
                pts = _num(crit.get("points"))
                if max_pts is None or max_pts == 0.0 or pts is None:
                    n_skipped_criteria += 1
                    continue
                r = pts / max_pts
                if r < 0.0:
                    r = 0.0
                elif r > 1.0:
                    r = 1.0
                title_str = "" if title is None else str(title)
                rows.append({
                    "subject_id": subj,
                    "item_id": iid,
                    "benchmark_id": bench_id,
                    "trial": trial,  # ensure_unique_trials may bump if collisions remain
                    "test_condition": f"judge={k};criterion={title_str}",
                    "response": float(r),
                    "correct_answer": item_gold.get(pid),
                    "trace": None,
                })
    extra = ""
    if n_empty_details or n_skipped_criteria:
        extra = f" (skipped {n_skipped_criteria} criteria, {n_empty_details} empty detail lists)"
    print(f"  {comp_name}: {len(rows):,} rows (rubric, per-criterion × per-judge){extra}")
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
        modality=INFO.get("modality"),
        domain=INFO.get("domain"),
        response_type=INFO.get("response_type"),
        response_scale=INFO.get("response_scale"),
        categorical=INFO.get("categorical"),
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
        # Partial-credit comps use the ``_outputs`` dataset, which includes the
        # full per-criterion ``grading_details_judge_*`` lists we need for
        # rubric emission. It also includes ``problem``, so no separate
        # problem-text supplement is required here.
        df = download_outputs_dataset(dataset_id, comp_name)
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
