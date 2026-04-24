#!/usr/bin/env python3
"""Build MMLU-Pro long-form responses (models x questions, binary correct).

Data sources:
  1. Per-question model outputs from TIGER-AI-Lab/MMLU-Pro GitHub eval_results/
     (~48 models x 12,032 questions, binary correct/incorrect from regex-
     extracted model answer vs. gold letter).
  2. Per-category leaderboard from TIGER-Lab/mmlu_pro_leaderboard_submission
     (249+ models x 14 categories; aggregate accuracy, continuous).

Source 1 rows carry no ``test_condition``. Source 2 rows use
``test_condition = "source=leaderboard;category=<Category>"`` so they never
collide with source 1 per-question rows.

Outputs:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'MMLU-Pro — 12K-question challenge MMLU variant across 14 subjects; per-question binary correctness + leaderboard per-category aggregates.',
    'testing_condition': 'Per-question rows (source 1) are binary. Leaderboard rows (source 2) are aggregate accuracies with test_condition=source=leaderboard;category=X.',
    'paper_url': 'https://arxiv.org/abs/2406.01574',
    'data_source_url': 'https://github.com/TIGER-AI-Lab/MMLU-Pro',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{wang2024mmluprorobustchallengingmultitask,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01574},
}""",
    'tags': ['reasoning'],
    'modality': ['text'],
    'domain': ['general', 'reasoning'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2024-06',
}


import io
import json
import re
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
EVAL_RESULTS_DIR = RAW_DIR / "eval_results"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)

LEADERBOARD_CATEGORIES = [
    "Biology", "Business", "Chemistry", "Computer Science", "Economics",
    "Engineering", "Health", "History", "Law", "Math",
    "Philosophy", "Physics", "Psychology", "Other",
]


def download():
    """Download per-question zips from GitHub API + leaderboard CSV."""
    print("  fetching eval_results zip listing...")
    api_url = "https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/contents/eval_results"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        listing = json.loads(urllib.request.urlopen(req, timeout=30).read())
        zips = [f for f in listing if f["name"].endswith(".zip")]
        print(f"  {len(zips)} zips listed")
        for z in zips:
            dest = EVAL_RESULTS_DIR / z["name"]
            if dest.exists():
                continue
            try:
                urllib.request.urlretrieve(z["download_url"], dest)
                time.sleep(0.3)
            except Exception as e:
                print(f"    skip {z['name']}: {e}")
    except Exception as e:
        print(f"  WARNING: GitHub listing failed: {e}; using existing data")

    lb_path = RAW_DIR / "leaderboard_results.csv"
    if not lb_path.exists():
        try:
            url = (
                "https://huggingface.co/datasets/TIGER-Lab/mmlu_pro_leaderboard_submission"
                "/resolve/main/leaderboard_results.csv"
            )
            urllib.request.urlretrieve(url, lb_path)
            print("  leaderboard CSV downloaded")
        except Exception as e:
            print(f"  leaderboard download failed: {e}")


def extract_model_name(zip_filename: str) -> str:
    name = zip_filename
    name = re.sub(r"^model_outputs_", "", name)
    name = re.sub(r"_\d+shots\.json\.zip$", "", name)
    name = re.sub(r"_\d+shots\.zip$", "", name)
    name = re.sub(r"_\d+-shots\.zip$", "", name)
    name = re.sub(r"_\d+shots_\d+_\d+_\d+\.zip$", "", name)
    name = re.sub(r"\.zip$", "", name)
    name = re.sub(r"\.json$", "", name)
    return name


def extract_answer(pred_text) -> str | None:
    if pred_text is None:
        return None
    s = str(pred_text).strip()
    if len(s) == 1 and s in "ABCDEFGHIJ":
        return s
    m = re.search(r"answer is\s*\(?([A-J])\)?", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-J])\b", s)
    if matches:
        return matches[-1]
    m = re.search(r"^([A-J])[.\s:]", s)
    if m:
        return m.group(1)
    return None


def iter_per_question_records():
    """Yield (model_label, question_id, content, answer, correct) tuples."""
    zips = sorted(EVAL_RESULTS_DIR.glob("*.zip"))
    for zf_path in zips:
        model_name = extract_model_name(zf_path.name)
        try:
            with zipfile.ZipFile(zf_path) as zf:
                json_files = [
                    n for n in zf.namelist()
                    if n.endswith(".json") and not n.startswith("__MACOSX")
                ]
                if not json_files:
                    continue
                with zf.open(json_files[0]) as f:
                    data = json.load(f)
            if not isinstance(data, list):
                continue
        except Exception as e:
            print(f"    skip {zf_path.name}: {e}")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id")
            if qid is None:
                continue
            gold = str(item.get("answer", "")).strip().upper()
            pred = item.get("pred")
            extracted = extract_answer(pred)
            correct = 1 if (extracted is not None and extracted == gold) else 0

            # Build item content from question + options (if present)
            q_text = item.get("question", "")
            options = item.get("options", [])
            if isinstance(options, list) and options:
                opts = "\n".join(
                    f"{chr(ord('A') + i)}: {o}"
                    for i, o in enumerate(options[:10])
                )
                content = f"{q_text}\n\n{opts}"
            elif q_text:
                content = q_text
            else:
                content = None

            # Model output text: prefer `model_outputs` (full generation), fall
            # back to `pred` (which may be the extracted letter or raw text).
            trace_text = item.get("model_outputs")
            if trace_text is None:
                trace_text = pred
            if trace_text is not None and not isinstance(trace_text, str):
                trace_text = str(trace_text)
            if isinstance(trace_text, str):
                trace_text = trace_text.strip() or None
            if isinstance(trace_text, str) and len(trace_text) > 8000:
                trace_text = trace_text[:8000]

            yield {
                "model": model_name,
                "qid": qid,
                "content": content,
                "answer": gold,
                "correct": correct,
                "trace": trace_text,
            }


def build_long_form():
    bench_id = get_benchmark_id(
        "mmlupro",
        name="MMLU-Pro",
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

    rows = []

    # Source 1: per-question binary
    print("[mmlupro] Source 1: per-question eval_results")
    n_q = 0
    for rec in iter_per_question_records():
        subj = resolve_subject(rec["model"])
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=f"q:{rec['qid']}",
            content=rec["content"],
            correct_answer=rec["answer"] or None,
        )
        rows.append({
            "subject_id": subj,
            "item_id": item,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": None,
            "response": float(rec["correct"]),
            "correct_answer": rec["answer"] or None,
            "trace": rec.get("trace"),
        })
        n_q += 1
    print(f"  parsed {n_q} per-question records")

    # Source 2: leaderboard per-category
    print("[mmlupro] Source 2: leaderboard")
    lb_path = RAW_DIR / "leaderboard_results.csv"
    n_lb = 0
    if lb_path.exists():
        try:
            lb = pd.read_csv(lb_path)
            for cat in LEADERBOARD_CATEGORIES:
                if cat in lb.columns:
                    lb[cat] = pd.to_numeric(lb[cat], errors="coerce")
            if "Overall" in lb.columns:
                lb["Overall"] = pd.to_numeric(lb["Overall"], errors="coerce")
                lb = lb.sort_values("Overall", ascending=False).drop_duplicates(
                    subset="Models", keep="first"
                )

            for cat in LEADERBOARD_CATEGORIES:
                if cat not in lb.columns:
                    continue
                content = f"leaderboard:category={cat}"
                item = register_item(
                    benchmark_id=bench_id,
                    raw_item_id=content,
                    content=content,
                )
                for _, r in lb.iterrows():
                    model = r.get("Models")
                    score = r.get(cat)
                    if pd.isna(score) or not isinstance(model, str):
                        continue
                    # Leaderboard accuracies are in [0, 1] already, but some are 0-100; normalize.
                    s = float(score)
                    if s > 1.0:
                        s = s / 100.0
                    subj = resolve_subject(model)
                    rows.append({
                        "subject_id": subj,
                        "item_id": item,
                        "benchmark_id": bench_id,
                        "trial": 1,
                        "test_condition": f"source=leaderboard;category={cat}",
                        "response": s,
                        "correct_answer": None,
                        "trace": None,
                    })
                    n_lb += 1
        except Exception as e:
            print(f"  leaderboard parse failed: {e}")
    print(f"  parsed {n_lb} leaderboard records")

    cols = ["subject_id", "item_id", "benchmark_id", "trial",
            "test_condition", "response", "correct_answer", "trace"]
    df = pd.DataFrame(rows, columns=cols)
    df = ensure_unique_trials(df)

    # Split traces into a sidecar so responses.parquet stays small.
    traces = df.loc[df["trace"].notna(), [
        "subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace",
    ]].copy()

    resp = df.copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)

    if len(traces) > 0:
        traces.to_parquet(_BENCHMARK_DIR / "traces.parquet", index=False)

    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(resp):,} rows)")
    if len(traces) > 0:
        print(f"  wrote traces.parquet ({len(traces):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n  (empty DataFrame)")
        return
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  response mean: {df['response'].mean():.3f}")


def main():
    print(f"[mmlupro] building from {_BENCHMARK_DIR}")
    download()
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
