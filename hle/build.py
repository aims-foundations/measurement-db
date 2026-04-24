"""Build HLE (Humanity's Last Exam) long-form responses from per-item judged data.

Data sources:
  1. supaihq/hle (GitHub) — judged_hle_pro.json
     Per-question judged results for up to 19 models on 1,369 questions.

  2. deepwriter-ai/hle-gemini-3-0 (GitHub) — questions_and_answer_hle_gem3pro.csv
     Per-question results for Gemini 3 Pro on 878 questions. Merged in for
     the Gemini-3-Pro-Preview subject only (the supaihq data covers other
     models). Questions not already in supaihq get Gemini-3-only rows.

HLE has 2,500 total questions; this covers ~1,792.

Outputs:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': "Humanity's Last Exam — 2500-question frontier benchmark; per-item binary correct/incorrect from supaihq judged data + deepwriter Gemini-3 data.",
    'testing_condition': 'Binary per-question correctness from supaihq judge + deepwriter CSV score field.',
    'paper_url': 'https://arxiv.org/abs/2501.14249',
    'data_source_url': 'https://github.com/supaihq/hle',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{phan2026humanitysexam,
      title={Humanity's Last Exam},
      author={Long Phan and Alice Gatti and Ziwen Han and Nathaniel Li and Josephina Hu and Hugh Zhang and Chen Bo Calvin Zhang and Mohamed Shaaban and John Ling and Sean Shi and Michael Choi and others},
      year={2026},
      eprint={2501.14249},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.14249},
}""",
    'tags': ['reasoning'],
    'modality': ['text'],
    'domain': ['general', 'reasoning'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2025-01',
}


import csv
import json
import sys
import urllib.request
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

HLE_DOWNLOADS = {
    "judged_hle_pro.json": (
        "https://raw.githubusercontent.com/supaihq/hle/main/judged_hle_pro.json"
    ),
    "questions_and_answer_hle_gem3pro.csv": (
        "https://raw.githubusercontent.com/deepwriter-ai/hle-gemini-3-0/main"
        "/questions_and_answer_hle_gem3pro.csv"
    ),
}

MODEL_RENAMES = {
    "main": "Sup-AI-Ensemble",
    "alibaba/qwen3-max": "Qwen3-Max",
    "alibaba/qwen3-next-80b-a3b-thinking": "Qwen3-Next-80B-A3B-Thinking",
    "alibaba/qwen3-vl-thinking": "Qwen3-VL-Thinking",
    "anthropic/claude-opus-4.5": "Claude-Opus-4.5",
    "anthropic/claude-sonnet-4.5": "Claude-Sonnet-4.5",
    "deepseek/deepseek-v3.2-exp-thinking": "DeepSeek-V3.2-Exp-Thinking",
    "deepseek/deepseek-v3.2-thinking": "DeepSeek-V3.2-Thinking",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
    "google/gemini-3-pro-preview": "Gemini-3-Pro-Preview",
    "minimax/minimax-m2": "MiniMax-M2",
    "mistral/magistral-medium": "Mistral-Magistral-Medium",
    "mistral/mistral-large": "Mistral-Large",
    "moonshotai/kimi-k2-thinking-turbo": "Kimi-K2-Thinking-Turbo",
    "openai/gpt-5-pro": "GPT-5-Pro",
    "openai/gpt-5.1": "GPT-5.1",
    "xai/grok-4": "Grok-4",
    "zai/glm-4.6": "GLM-4.6",
}


def download():
    for fname, url in HLE_DOWNLOADS.items():
        dest = RAW_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  downloading {fname}...")
        req = urllib.request.Request(url, headers={"User-Agent": "hle-builder"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  saved {fname} ({dest.stat().st_size/1024:.0f} KB)")


def _trim(s, n=8000):
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return None
    return s[:n]


def load_supaihq():
    """Return ({qid: {model_name: (0/1, trace_or_None)}}, {qid: correct_answer})."""
    path = RAW_DIR / "judged_hle_pro.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    answers = {}
    for qid, qdata in data.items():
        judge_response = qdata.get("judge_response", {})
        raw_resps = qdata.get("response", {}) or {}
        out[qid] = {}
        for model_name, judgment in judge_response.items():
            if isinstance(judgment, dict) and "correct" in judgment:
                correct_str = str(judgment.get("correct", "")).strip().lower()
                score = 1 if correct_str == "yes" else 0
                # Prefer the full response text; fall back to the judge's
                # extracted model_answer.
                trace_text = _trim(raw_resps.get(model_name))
                if trace_text is None:
                    trace_text = _trim(judgment.get("model_answer"))
                out[qid][model_name] = (score, trace_text)
                # Each judgment contains the gold answer; they agree across models.
                if qid not in answers:
                    ca = judgment.get("correct_answer")
                    if ca is not None:
                        answers[qid] = _trim(ca, n=4000)
    return out, answers


def load_deepwriter():
    """Return ({qid: (0/1, trace_or_None)}, {qid: correct_answer}) for Gemini-3-Pro-Preview."""
    path = RAW_DIR / "questions_and_answer_hle_gem3pro.csv"
    if not path.exists():
        return {}, {}
    out = {}
    answers = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("id", "").strip()
            if not qid or qid == "Totals:":
                continue
            try:
                score = int(float(row["score"]))
            except (ValueError, KeyError, TypeError):
                continue
            # deepwriter only surfaces the parsed letter, not the full CoT.
            trace_text = _trim(row.get("extracted_answer"))
            out[qid] = (score, trace_text)
            ca = row.get("answer")
            if ca:
                answers[qid] = _trim(ca, n=4000)
    return out, answers


def build_long_form():
    bench_id = get_benchmark_id(
        "hle",
        name="Humanity's Last Exam",
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

    supaihq, supaihq_answers = load_supaihq()
    deepwriter, deepwriter_answers = load_deepwriter()
    # Combined ground-truth lookup; supaihq answers take precedence where both exist.
    answers = dict(deepwriter_answers)
    answers.update(supaihq_answers)
    print(f"  supaihq questions: {len(supaihq)}")
    print(f"  deepwriter questions: {len(deepwriter)}")

    rows = []
    for qid, model_results in supaihq.items():
        content = f"hle:{qid}"
        correct = answers.get(qid)
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=qid,
            content=content,
            correct_answer=correct,
        )
        for raw_model, payload in model_results.items():
            score, trace_text = payload
            display = MODEL_RENAMES.get(raw_model, raw_model)
            subj = resolve_subject(display)
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": None,
                "response": float(score),
                "correct_answer": correct,
                "trace": trace_text,
            })

    # Add Gemini-3 rows from deepwriter, avoiding double-counting for qids
    # already present in supaihq's Gemini-3-Pro-Preview results.
    gemini_display = "Gemini-3-Pro-Preview"
    gemini_subj = resolve_subject(gemini_display)
    existing_gemini = {
        qid for qid, mr in supaihq.items()
        if "google/gemini-3-pro-preview" in mr or gemini_display in mr
    }
    added = 0
    for qid, payload in deepwriter.items():
        if qid in existing_gemini:
            continue
        score, trace_text = payload
        content = f"hle:{qid}"
        correct = answers.get(qid)
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=qid,
            content=content,
            correct_answer=correct,
        )
        rows.append({
            "subject_id": gemini_subj,
            "item_id": item,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": None,
            "response": float(score),
            "correct_answer": correct,
            "trace": trace_text,
        })
        added += 1
    print(f"  added {added} Gemini-3 rows from deepwriter")

    cols = ["subject_id", "item_id", "benchmark_id", "trial",
            "test_condition", "response", "correct_answer", "trace"]
    df = pd.DataFrame(rows, columns=cols)
    df = ensure_unique_trials(df)

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
    print(f"[hle] building from {_BENCHMARK_DIR}")
    download()
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
