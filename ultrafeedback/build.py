"""Build UltraFeedback long-form responses from GPT-4 multi-aspect ratings.

Data source:
  - openbmb/UltraFeedback on HuggingFace: 17 models x ~64K prompts, GPT-4
    multi-aspect ratings (1-5) on four aspects: helpfulness, honesty,
    truthfulness, instruction_following.

Each (model, prompt, aspect) yields one long-form row with
``test_condition = f"aspect={aspect}"``, so a single prompt registers as
one item shared across the four aspect conditions.

Output:
  - raw/extracted_scores_per_aspect.csv  # cached per-aspect extraction
  - responses.parquet                     # long-form responses
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'UltraFeedback GPT-4 multi-aspect ratings (1-5) from 17 models on ~64K prompts.',
    'testing_condition': 'Four aspects (helpfulness/honesty/truthfulness/instruction_following); aspect encoded in test_condition.',
    'paper_url': 'https://arxiv.org/abs/2310.01377',
    'data_source_url': 'https://huggingface.co/datasets/openbmb/UltraFeedback',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{cui2024ultrafeedbackboostinglanguagemodels,
      title={UltraFeedback: Boosting Language Models with Scaled AI Feedback},
      author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Bingxiang He and Wei Zhu and Yuan Ni and Guotong Xie and Ruobing Xie and Yankai Lin and Zhiyuan Liu and Maosong Sun},
      year={2024},
      eprint={2310.01377},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.01377},
}""",
    'tags': ['reward-model', 'preference'],
    'modality': ['text'],
    'domain': ['preference'],
    'response_type': 'likert_5',
    'response_scale': '{1, 2, 3, 4, 5}',
    'categorical': True,
    'release_date': '2023-10',
}


import os
import sys
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
TRACES_PATH = _BENCHMARK_DIR / "traces.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)

# Canonical aspect names in UltraFeedback's annotations.
ASPECTS = ("helpfulness", "honesty", "truthfulness", "instruction_following")


def _rating_from(aspect_data) -> float | None:
    if isinstance(aspect_data, dict):
        val = aspect_data.get("Rating", aspect_data.get("rating"))
    elif isinstance(aspect_data, (int, float)):
        val = aspect_data
    else:
        return None
    if val is None:
        return None
    try:
        val = float(val)
    except (ValueError, TypeError):
        return None
    if 1.0 <= val <= 5.0:
        return val
    return None


def stream_per_aspect() -> pd.DataFrame:
    """Stream UltraFeedback and extract (model, prompt_id, aspect, rating) rows."""
    cache_path = RAW_DIR / "extracted_scores_per_aspect.csv"

    if cache_path.exists() and cache_path.stat().st_size > 1000:
        cached = pd.read_csv(cache_path)
        # Honor cache only if the trace column was captured; older caches
        # pre-date trace extraction and must be rebuilt.
        if "response_text" in cached.columns:
            print(f"  using cached extraction: {cache_path}")
            return cached
        print("  cached extraction lacks response_text; re-extracting...")

    print("  streaming openbmb/UltraFeedback from HuggingFace ...")
    from datasets import load_dataset

    ds = load_dataset(
        "openbmb/UltraFeedback",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    records = []
    n_processed = n_skipped = 0

    for item in ds:
        prompt_text = item.get("instruction") or item.get("prompt") or ""
        if "id" in item and item["id"]:
            prompt_id = str(item["id"])
        else:
            prompt_id = (item.get("source", "") + "_" + str(n_processed))

        completions = item.get("completions", []) or []
        if not completions:
            n_skipped += 1
            continue

        for comp in completions:
            model = comp.get("model", "")
            if not model:
                continue
            resp_text = comp.get("response")
            if not isinstance(resp_text, str):
                resp_text = None
            annotations = comp.get("annotations", {}) or comp.get("ratings", {}) or {}
            for aspect, aspect_data in annotations.items():
                canonical = aspect.strip().lower().replace("-", "_").replace(" ", "_")
                if canonical not in ASPECTS:
                    continue
                rating = _rating_from(aspect_data)
                if rating is None:
                    continue
                records.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "aspect": canonical,
                    "rating": rating,
                    "response_text": resp_text,
                })

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"    processed {n_processed:,} prompts, {len(records):,} rows")

    print(f"  total prompts: {n_processed:,}; skipped: {n_skipped:,}; rows: {len(records):,}")

    df = pd.DataFrame(records)
    # Truncate prompt_text in the cache to keep it manageable.
    if "prompt_text" in df.columns:
        df["prompt_text"] = df["prompt_text"].astype(str).str.slice(0, 4000)
    if "response_text" in df.columns:
        df["response_text"] = df["response_text"].astype(str).str.slice(0, 8000)
    df.to_csv(cache_path, index=False)
    print(f"  cached: {cache_path}")
    return df


def build_long_form(scores_df: pd.DataFrame) -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "ultrafeedback",
        name="UltraFeedback",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
        modality=INFO.get("modality"),
        domain=INFO.get("domain"),
        response_type=INFO.get("response_type"),
        response_scale=INFO.get("response_scale"),
        categorical=INFO.get("categorical"),
        paper_url=INFO.get("paper_url"),
        release_date=INFO.get("release_date"),
    )

    if scores_df.empty:
        df = pd.DataFrame()
        df.to_parquet(RESPONSES_PATH, index=False)
        registry_save(CONTRIB_DIR)
        return df

    # Build a prompt_id -> content map so we register each item once.
    content_map: dict[str, str | None] = {}
    for pid, text in scores_df[["prompt_id", "prompt_text"]].itertuples(index=False):
        if pid in content_map:
            continue
        t = (str(text).strip() if pd.notna(text) else "")
        content_map[pid] = t if t else None

    has_resp = "response_text" in scores_df.columns

    rows = []
    subject_cache: dict[str, str] = {}
    item_cache: dict[str, str] = {}
    for rec in scores_df.itertuples(index=False):
        model = rec.model
        if model not in subject_cache:
            subject_cache[model] = resolve_subject(model)
        subj = subject_cache[model]

        pid = rec.prompt_id
        if pid not in item_cache:
            item_cache[pid] = register_item(
                benchmark_id=bench_id,
                raw_item_id=str(pid),
                content=content_map.get(pid),
            )
        item = item_cache[pid]

        raw_trace = getattr(rec, "response_text", None) if has_resp else None
        if isinstance(raw_trace, float) and pd.isna(raw_trace):
            raw_trace = None
        elif isinstance(raw_trace, str) and not raw_trace.strip():
            raw_trace = None

        rows.append({
            "subject_id": subj,
            "item_id": item,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": f"aspect={rec.aspect}",
            "response": float(rec.rating),
            "correct_answer": None,
            "trace": raw_trace,
        })

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)

    trace_cols = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace"]
    traces = df.loc[df["trace"].notna(), trace_cols].copy()

    resp = df.copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(resp):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    if len(traces) > 0:
        traces.to_parquet(TRACES_PATH, index=False)
        print(f"  wrote {TRACES_PATH.name} ({len(traces):,} rows)")
    return df


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (empty)")
        return
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    for cond, g in df.groupby("test_condition"):
        print(f"    {cond}: {len(g):,} rows, mean={g['response'].mean():.3f}")


def main():
    print("UltraFeedback long-form builder")
    print("=" * 60)
    scores_df = stream_per_aspect()
    df = build_long_form(scores_df)
    print_stats(df)


if __name__ == "__main__":
    main()
