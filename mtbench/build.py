"""Build MT-Bench long-form responses from GPT-4 single-answer judgment data.

Data sources:
  - https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl
    34 models x 80 questions x 2 turns, GPT-4 absolute scores 1-10.
  - https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/question.jsonl
    The 80 questions with both turns of each conversation.

Output:
  - raw/gpt-4_single.jsonl, raw/question.jsonl
  - responses.parquet   # long-form: (subject_id, item_id, trial, response, trace)
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions
"""

INFO = {
    'description': 'MT-Bench GPT-4 single-answer judgment scores (34 models, 80 questions, 2 turns).',
    'testing_condition': 'Each turn is registered as a distinct item; turn 2 is a follow-up to turn 1.',
    'paper_url': 'https://arxiv.org/abs/2306.05685',
    'data_source_url': 'https://huggingface.co/spaces/lmsys/mt-bench',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-4.0',
    'citation': """@misc{zheng2023judgingllmasajudgemtbenchchatbot,
      title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2306.05685},
}""",
    'tags': ['preference', 'pairwise'],
    'modality': ['text'],
    'domain': ['preference'],
    'response_type': 'likert_10',
    'response_scale': '{1, 2, ..., 10}',
    'categorical': True,
}


import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"

RAW_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)

JUDGMENT_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/model_judgment/gpt-4_single.jsonl"
)
QUESTIONS_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/question.jsonl"
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  saved {dest.name} ({len(data):,} bytes)")
    return dest


def load_questions(path: Path) -> dict[tuple[str, int], str]:
    """Map (question_id, turn) -> prompt text."""
    out: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = str(rec["question_id"])
        for i, turn_text in enumerate(rec.get("turns", []), start=1):
            out[(qid, i)] = turn_text
    return out


def parse_judgments(jsonl_path: Path) -> pd.DataFrame:
    """Extract (model, question_id, turn, score) from the judgment JSONL."""
    records = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        model = rec.get("model_id") or rec.get("model") or ""
        qid = rec.get("question_id", "")
        turn = rec.get("turn", 1)
        score = rec.get("score", rec.get("rating"))

        if score is None or score == -1:
            m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", rec.get("judgment", ""))
            if m:
                score = float(m.group(1))

        if model and qid is not None and score is not None and score != -1:
            try:
                records.append({
                    "model": model,
                    "question_id": str(qid),
                    "turn": int(turn),
                    "score": float(score),
                })
            except (ValueError, TypeError):
                continue

    print(f"  parsed {len(records):,} judgment records")
    return pd.DataFrame(records)


def build_long_form(judgments: pd.DataFrame, questions: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Write responses.parquet + registry contribs. One row per (model, question, turn)."""
    bench_id = get_benchmark_id(
        "mtbench",
        name="MT-Bench",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
        modality=INFO.get("modality"),
        domain=INFO.get("domain"),
        response_type=INFO.get("response_type"),
        response_scale=INFO.get("response_scale"),
        categorical=INFO.get("categorical"),
    )

    rows = []
    missing_content = 0
    for rec in judgments.itertuples(index=False):
        subj = resolve_subject(rec.model)
        content = questions.get((rec.question_id, rec.turn))
        if content is None:
            missing_content += 1
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=f"{rec.question_id}_turn{rec.turn}",
            content=content,
        )
        rows.append({
            "subject_id": subj,
            "item_id": item,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": None,
            "response": rec.score,
            "correct_answer": None,
            "trace": None,
        })

    if missing_content:
        print(f"  WARNING: {missing_content} judgments had no matching question prompt")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)
    df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    n_subjects = df["subject_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"\n  subjects: {n_subjects}")
    print(f"  items:    {n_items}")
    print(f"  rows:     {len(df):,}")
    print(f"  response mean/std: {df['response'].mean():.2f} / {df['response'].std():.2f}")


def main():
    print(f"[mtbench] building from {_BENCHMARK_DIR}")

    judgment_path = _download(JUDGMENT_URL, RAW_DIR / "gpt-4_single.jsonl")
    questions_path = _download(QUESTIONS_URL, RAW_DIR / "question.jsonl")

    judgments = parse_judgments(judgment_path)
    questions = load_questions(questions_path)
    print(f"  loaded {len(questions)} question prompts")

    df = build_long_form(judgments, questions)
    print_stats(df)


if __name__ == "__main__":
    main()
