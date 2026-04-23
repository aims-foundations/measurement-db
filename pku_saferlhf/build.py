"""
Build PKU-SafeRLHF long-form responses from PKU-Alignment/PKU-SafeRLHF.

Data source:
  - PKU-Alignment/PKU-SafeRLHF on HuggingFace Hub: ~82K expert comparison
    pairs with dual annotations for helpfulness and safety preferences.

  Configurations (used here):
    - alpaca-7b: Alpaca-7B responses
    - alpaca2-7b: Alpaca2-7B responses
    - alpaca3-8b: Alpaca3-8B responses
  The ``default`` config is the union of the three and is skipped to avoid
  duplicate (subject, item) observations.

Encoding (scalar, long-form):
  - Each (prompt, response_0) and (prompt, response_1) becomes an observation.
  - Subject = the AI model that produced the response (from
    ``response_{0,1}_source`` when present, otherwise the config name as a
    best-effort fallback).
  - Item = prompt text (the same prompt shared by both sides).
  - Response = safety rating: 1.0 if the response is flagged safe, else 0.0.
  - ``test_condition=None`` (single-rater overall safety; harm categories are
    auxiliary and not emitted as separate rows to keep the PK clean).

Output:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'PKU-SafeRLHF per-response safety ratings for Alpaca-family models.',
    'testing_condition': 'Subjects are the generating models; response = 1 iff is_response_safe.',
    'paper_url': 'https://arxiv.org/abs/2406.15513',
    'data_source_url': 'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-NC-4.0',
    'citation': """@misc{ji2025pkusaferlhfmultilevelsafetyalignment,
      title={PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference},
      author={Jiaming Ji and Donghai Hong and Borong Zhang and Boyuan Chen and Juntao Dai and Boren Zheng and Tianyi Qiu and Jiayi Zhou and Kaile Wang and Boxuan Li and Sirui Han and Yike Guo and Yaodong Yang},
      year={2025},
      eprint={2406.15513},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.15513},
}""",
    'tags': ['safety'],
}


import os
import sys
from pathlib import Path

import pandas as pd

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

HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_SOURCE = "PKU-Alignment/PKU-SafeRLHF"

# Per-model configs only (``default`` is the union; skip to avoid duplicates).
CONFIGS = ["alpaca-7b", "alpaca2-7b", "alpaca3-8b"]

# Fallback subject labels when response_*_source is missing.
CONFIG_TO_MODEL = {
    "alpaca-7b": "alpaca-7b",
    "alpaca2-7b": "alpaca2-7b",
    "alpaca3-8b": "alpaca3-8b",
}


def iter_records(config_name: str):
    """Yield (subject_label, prompt, is_safe) triples for each response side."""
    from datasets import load_dataset

    ds = load_dataset(HF_SOURCE, name=config_name, token=HF_TOKEN)

    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        split_ds = ds[split_name]
        print(f"    {config_name}/{split_name}: {len(split_ds)} rows")
        for item in split_ds:
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue

            for side in (0, 1):
                src_field = item.get(f"response_{side}_source") or ""
                subject_label = src_field.strip() or CONFIG_TO_MODEL[config_name]
                is_safe = item.get(f"is_response_{side}_safe")
                if is_safe is None:
                    continue
                response = 1.0 if bool(is_safe) else 0.0
                yield subject_label, prompt, response


def build_long_form() -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "pku_saferlhf",
        name="PKU-SafeRLHF",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
    )

    rows = []
    for config_name in CONFIGS:
        print(f"\n  Loading config {config_name} ...")
        for subject_label, prompt, response in iter_records(config_name):
            subj = resolve_subject(subject_label)
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=prompt[:64],
                content=prompt,
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": None,
                "response": float(response),
                "correct_answer": None,
                "trace": None,
            })

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)
    df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n  (empty)")
        return
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  mean response (pct safe): {df['response'].mean():.3f}")


def main():
    print("PKU-SafeRLHF long-form builder")
    print("=" * 60)
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
