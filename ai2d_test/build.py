"""
Build AI2D_TEST long-form responses from VLMEval/OpenVLMRecords.

Data source:
  - HuggingFace dataset: VLMEval/OpenVLMRecords
  - Per-model xlsx files at mmeval/{model}/{model}_AI2D_TEST.xlsx
  - Science diagram understanding MCQ benchmark

Score format:
  - Binary 0/1: whether extracted answer letter matches ground truth
  - Rows skipped if answer cannot be parsed

Outputs:
  - raw/{model}_AI2D_TEST.xlsx: Downloaded xlsx files
  - responses.parquet   # long-form
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'Build AI2D_TEST response matrix from VLMEval/OpenVLMRecords',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/1603.07396',
    'data_source_url': 'https://allenai.org/data/diagrams',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-SA-4.0',
    'citation': """@misc{kembhavi2016diagramworthdozenimages,
      title={A Diagram Is Worth A Dozen Images},
      author={Aniruddha Kembhavi and Mike Salvato and Eric Kolve and Minjoon Seo and Hannaneh Hajishirzi and Ali Farhadi},
      year={2016},
      eprint={1603.07396},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1603.07396},
}""",
    'tags': ['vision-language'],
    'modality': ['text', 'image'],
    'domain': ['science'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2016-03',
}


import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

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

REPO_ID = "VLMEval/OpenVLMRecords"
BENCHMARK_SUFFIX = "AI2D_TEST"


def list_all_models():
    """List all model directories in the repo."""
    items = list(list_repo_tree(REPO_ID, path_in_repo="mmeval", repo_type="dataset"))
    models = []
    for item in items:
        name = item.path.replace("mmeval/", "")
        if "/" not in name:
            models.append(name)
    return sorted(models)


def find_models_for_benchmark(all_models):
    """Find which models have results for this benchmark."""
    models_with_bench = []
    for model in all_models:
        try:
            items = list(list_repo_tree(REPO_ID, path_in_repo=f"mmeval/{model}", repo_type="dataset"))
            filenames = [item.path.split("/")[-1] for item in items]
            target = f"{model}_{BENCHMARK_SUFFIX}.xlsx"
            if target in filenames:
                models_with_bench.append(model)
        except Exception:
            pass
    return models_with_bench


def extract_answer_letter(prediction_text, choices=("A", "B", "C", "D")):
    """Extract the answer letter from a model's free-text prediction."""
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip()

    if pred.upper() in choices:
        return pred.upper()

    m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-D])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\*\*([A-D])\b", pred)
    if m:
        return m.group(1).upper()

    m = re.match(r"^[(\s]*([A-D])[.):\s]", pred)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", pred)
    if m:
        return m.group(1).upper()

    return None


def score_mcq(row):
    """Score a multiple choice question."""
    pred_letter = extract_answer_letter(str(row["prediction"]))
    if pred_letter is None:
        return np.nan
    return 1 if pred_letter == row["answer"] else 0


def _item_content(row) -> str | None:
    """Build item content from question + MCQ options if present."""
    question = row.get("question")
    if question is None or (isinstance(question, float) and pd.isna(question)):
        return None
    parts = [str(question).strip()]
    opts = []
    for letter in ("A", "B", "C", "D"):
        val = row.get(letter)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            opts.append(f"{letter}: {val}")
    if opts:
        parts.append("\n".join(opts))
    return "\n\n".join(parts)[:4000]


def main():
    print("AI2D_TEST long-form builder")
    print("=" * 60)

    bench_id = get_benchmark_id(
        "ai2d_test",
        name="AI2D_TEST",
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

    print("Listing all models in VLMEval/OpenVLMRecords...")
    all_models = list_all_models()
    print(f"Found {len(all_models)} models total")

    print(f"Scanning for {BENCHMARK_SUFFIX}...")
    models_with_bench = find_models_for_benchmark(all_models)
    print(f"Found {len(models_with_bench)} models with {BENCHMARK_SUFFIX}")

    cols = [
        "subject_id", "item_id", "benchmark_id", "trial",
        "test_condition", "response", "correct_answer", "trace",
    ]

    if not models_with_bench:
        print("  No models found — writing empty responses.parquet")
        df = pd.DataFrame(columns=cols)
        df.to_parquet(RESPONSES_PATH, index=False)
        registry_save(CONTRIB_DIR)
        return

    item_id_by_index: dict[int, str] = {}
    item_answer_by_index: dict[int, str | None] = {}
    all_scores: dict[str, pd.Series] = {}
    all_preds: dict[str, pd.Series] = {}
    failed_models = []

    for i, model in enumerate(models_with_bench):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Downloading {i + 1}/{len(models_with_bench)}: {model}")

        try:
            fpath = f"mmeval/{model}/{model}_{BENCHMARK_SUFFIX}.xlsx"
            local_path = hf_hub_download(REPO_ID, fpath, repo_type="dataset")

            raw_dest = RAW_DIR / f"{model}_{BENCHMARK_SUFFIX}.xlsx"
            if not raw_dest.exists():
                shutil.copy2(local_path, raw_dest)

            df = pd.read_excel(local_path)

            if len(df) == 0:
                failed_models.append((model, "empty dataframe"))
                continue

            if "index" not in df.columns:
                df["index"] = range(len(df))

            df["score"] = df.apply(score_mcq, axis=1)

            for _, row in df.iterrows():
                idx = int(row["index"])
                if idx in item_id_by_index:
                    continue
                content = _item_content(row)
                gold = row.get("answer")
                gold_str = None if gold is None or (isinstance(gold, float) and pd.isna(gold)) else str(gold)
                iid = register_item(
                    benchmark_id=bench_id,
                    raw_item_id=f"ai2d_test_{idx}",
                    content=content,
                    correct_answer=gold_str,
                )
                item_id_by_index[idx] = iid
                item_answer_by_index[idx] = gold_str

            all_scores[model] = df.set_index("index")["score"]
            # Preserve the raw prediction column as the per-row trace.
            all_preds[model] = df.set_index("index")["prediction"]

        except Exception as e:
            failed_models.append((model, str(e)[:100]))

    if not all_scores:
        print("  No successful downloads — writing empty responses.parquet")
        df = pd.DataFrame(columns=cols)
        df.to_parquet(RESPONSES_PATH, index=False)
        registry_save(CONTRIB_DIR)
        return

    rows = []
    for model, scores in all_scores.items():
        subj = resolve_subject(model)
        preds = all_preds.get(model)
        for idx, score in scores.items():
            if pd.isna(score):
                continue
            idx_int = int(idx)
            iid = item_id_by_index.get(idx_int)
            if iid is None:
                continue
            trace = None
            if preds is not None:
                raw = preds.get(idx_int)
                if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                    raw_s = str(raw)
                    trace = raw_s[:8000] if raw_s else None
            rows.append({
                "subject_id": subj,
                "item_id": iid,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": None,
                "response": float(score),
                "correct_answer": item_answer_by_index.get(idx_int),
                "trace": trace,
            })

    out_df = pd.DataFrame(rows, columns=cols)
    out_df = ensure_unique_trials(out_df)

    # Split: responses.parquet keeps trace=None; traces.parquet holds the trace column.
    resp = out_df[cols].copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)

    traces_path = _BENCHMARK_DIR / "traces.parquet"
    traces = out_df.loc[out_df["trace"].notna(), [
        "subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace",
    ]]
    if len(traces) > 0:
        traces.to_parquet(traces_path, index=False)

    registry_save(CONTRIB_DIR)

    print(f"  wrote {RESPONSES_PATH.name} ({len(out_df):,} rows)")
    if len(traces) > 0:
        print(f"  wrote {traces_path.name} ({len(traces):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")

    if failed_models:
        print(f"\n  Failed models ({len(failed_models)}):")
        for m, e in failed_models[:10]:
            print(f"    {m}: {e}")

    if not out_df.empty:
        print(f"\n  subjects: {out_df['subject_id'].nunique()}")
        print(f"  items:    {out_df['item_id'].nunique()}")
        print(f"  rows:     {len(out_df):,}")


if __name__ == "__main__":
    main()
