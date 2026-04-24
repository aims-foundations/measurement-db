"""Build LiveCodeBench long-form responses from submissions + leaderboard JSONs.

Data sources:
  1. Primary: LiveCodeBench/submissions repo eval_all.json files.
     Each per-problem entry carries ``graded_list`` = list of booleans per
     sampled generation (length 1, 5, or 10 depending on the submission).
     We emit one row per sample with binary response; downstream pass@k is
     ``.groupby([subject, item]).response.mean()``.
  2. Supplementary: LiveCodeBench/livecodebench.github.io leaderboard
     performances_generation.json + v5.json (fills coverage gaps).
     Leaderboard-only entries expose only a continuous ``pass@1`` float —
     we keep those as a single aggregate row with
     test_condition="source=leaderboard;aggregate=pass@1".

Problem sets are nested: 713 v3, 880 v5, 1055 v6.

Output:
  - responses.parquet   # long-form: (subject_id, item_id, trial, response, trace)
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions
"""

INFO = {
    'description': 'LiveCodeBench per-sample binary outcomes from submissions; leaderboard-only rows remain aggregate pass@1.',
    'testing_condition': 'submissions-sourced rows emit one row per sampled generation with binary response; leaderboard-only rows (no per-sample data upstream) keep aggregate pass@1 with test_condition="source=leaderboard;aggregate=pass@1".',
    'paper_url': 'https://arxiv.org/abs/2403.07974',
    'data_source_url': 'https://github.com/LiveCodeBench/submissions',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-4.0',
    'citation': """@misc{jain2024livecodebenchholisticcontaminationfree,
      title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
      author={Naman Jain and King Han and Alex Gu and Wen-Ding Li and Fanjia Yan and Tianjun Zhang and Sida Wang and Armando Solar-Lezama and Koushik Sen and Ion Stoica},
      year={2024},
      eprint={2403.07974},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2403.07974},
}""",
    'tags': ['coding'],
    'modality': ['text'],
    'domain': ['software_engineering'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2024-03',
}


import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
SUBMISSIONS_DIR = f"{RAW_DIR}/submissions"
LEADERBOARD_DIR = f"{RAW_DIR}/livecodebench.github.io/build"
PUBLIC_TESTS_PATH = _BENCHMARK_DIR / "raw" / "public_tests.parquet"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
TRACES_PATH = _BENCHMARK_DIR / "traces.parquet"

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)


def download():
    """Download raw data from external sources."""
    raw_path = Path(RAW_DIR)
    raw_path.mkdir(parents=True, exist_ok=True)

    repos = [
        ("https://github.com/LiveCodeBench/submissions.git",
         raw_path / "submissions"),
        ("https://github.com/LiveCodeBench/livecodebench.github.io.git",
         raw_path / "livecodebench.github.io"),
    ]
    for url, target in repos:
        if not target.exists():
            print(f"Cloning {url}...")
            subprocess.run(["git", "clone", url, str(target)], check=True)
        else:
            print(f"{target.name} already cloned, pulling latest...")
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=False)


SKIP_MODELS = {
    "DeepSeek-V3 copy",
    "O1-2024-12-17 (Low)_prompt_old",
}


def load_submissions():
    """Load per-problem per-sample binary outcomes + per-problem content.

    Returns a per-model dict of the form
      {"results": {qid: [bool, bool, ...]}, "n_problems": int,
       "source": "submissions", "per_sample": True}
    whenever a ``graded_list`` is available on the entry. When the raw file
    only has a scalar pass@1 (rare), we fall back to a one-element list.
    """
    models = {}
    item_content = {}
    if not os.path.isdir(SUBMISSIONS_DIR):
        return models, item_content

    dirs = sorted([
        d for d in os.listdir(SUBMISSIONS_DIR)
        if os.path.isdir(os.path.join(SUBMISSIONS_DIR, d)) and d != ".git"
    ])

    for d in dirs:
        if d in SKIP_MODELS:
            print(f"  Skipping {d} (duplicate)")
            continue

        model_dir = os.path.join(SUBMISSIONS_DIR, d)
        json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
        if not json_files:
            print(f"  Skipping {d} (no data files)")
            continue

        best_file = None
        best_n = 0
        for f in json_files:
            fpath = os.path.join(model_dir, f)
            try:
                with open(fpath) as fp:
                    data = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            if len(data) > best_n:
                best_n = len(data)
                best_file = (f, data)

        if best_file is None:
            continue
        _, data = best_file

        results = {}
        traces_by_qid: dict = {}
        has_any_list = False
        for entry in data:
            qid = entry["question_id"]
            graded = entry.get("graded_list")
            if isinstance(graded, list) and len(graded) > 0:
                # Per-sample binary list available.
                results[qid] = [bool(g) for g in graded]
                has_any_list = True
                # Per-sample raw model output text (aligned with graded_list).
                outputs = entry.get("output_list")
                if isinstance(outputs, list) and outputs:
                    per_sample = []
                    for out in outputs:
                        if isinstance(out, str):
                            per_sample.append(out[:8000])
                        elif out is None:
                            per_sample.append(None)
                        else:
                            per_sample.append(str(out)[:8000])
                    traces_by_qid[qid] = per_sample
            else:
                # Fallback: only pass@1 scalar — wrap as single-sample float
                # (will later be emitted as aggregate leaderboard-style).
                p1_key = "pass@1" if "pass@1" in entry else "pass1"
                p1 = entry.get(p1_key)
                if p1 is None:
                    continue
                results[qid] = float(p1)

            if qid not in item_content:
                title = entry.get("question_title", "") or ""
                content = entry.get("question_content", "") or ""
                text = f"{title}\n{content}" if content else title
                text = text.strip()
                if len(text) >= 10:
                    item_content[qid] = text[:4000]

        models[d] = {
            "results": results,
            "traces": traces_by_qid,
            "n_problems": len(results),
            "source": "submissions",
            "per_sample": has_any_list,
        }

    return models, item_content


def _truncate(s: str | None, limit: int = 4000) -> str | None:
    """Truncate ``s`` to ``limit`` chars; leave a truncation marker."""
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    if not s:
        return None
    if len(s) <= limit:
        return s
    suffix = "\n...[truncated]"
    return s[: max(0, limit - len(suffix))] + suffix


def load_public_tests() -> dict[str, str]:
    """Return {question_id -> json.dumps(public_test_cases)}.

    The LiveCodeBench HF dataset ``livecodebench/code_generation_lite``
    carries a ``public_test_cases`` field on each task (JSON-serialized list
    of {input, output, testtype} dicts). We flatten every release_version
    config we can load and combine them. Result cached under
    ``raw/public_tests.parquet``.
    """
    if PUBLIC_TESTS_PATH.exists():
        df = pd.read_parquet(PUBLIC_TESTS_PATH)
        mapping = dict(zip(df["question_id"], df["public_test_cases"]))
        print(f"  Loaded {len(mapping)} public_test_cases from cache")
        return mapping

    mapping: dict[str, str] = {}
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: 'datasets' unavailable; correct_answer will be None",
              file=sys.stderr)
        return mapping

    # Iterate available release_versions; each newer release nests earlier
    # ones, so we iterate from smallest to largest to let later ones win on
    # ties.  Not every version is guaranteed to load — skip failures.
    for version_tag in (
        "release_v1", "release_v2", "release_v3", "release_v4",
        "release_v5", "release_v6", "release_latest",
    ):
        try:
            ds = load_dataset(
                "livecodebench/code_generation_lite",
                version_tag=version_tag,
                split="test",
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"  skip {version_tag}: {exc}", file=sys.stderr)
            continue

        for r in ds:
            qid = r.get("question_id")
            if not qid:
                continue
            pub = r.get("public_test_cases")
            if pub is None:
                continue
            # public_test_cases upstream is already a JSON string; keep as-is
            # if so, else serialize.
            if isinstance(pub, str):
                val = pub
            else:
                try:
                    val = json.dumps(pub, ensure_ascii=False)
                except (TypeError, ValueError):
                    continue
            if val:
                mapping[str(qid)] = val

    if mapping:
        PUBLIC_TESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"question_id": list(mapping.keys()),
             "public_test_cases": list(mapping.values())}
        )
        df.to_parquet(PUBLIC_TESTS_PATH, index=False)
        print(f"  Cached {len(mapping)} public_test_cases to "
              f"{PUBLIC_TESTS_PATH.name}")

    return mapping


def load_leaderboard_json(path, label):
    with open(path) as f:
        data = json.load(f)

    models = {}
    for perf in data["performances"]:
        model = perf["model"]
        qid = perf["question_id"]
        p1 = perf["pass@1"]
        if p1 > 1.0:
            p1 = p1 / 100.0
        models.setdefault(model, {"results": {}, "source": label, "per_sample": False})["results"][qid] = p1

    for m in models:
        models[m]["n_problems"] = len(models[m]["results"])
    return models


def build_long_form(all_models, item_content, public_tests):
    bench_id = get_benchmark_id(
        "livecodebench",
        name="LiveCodeBench",
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
    missing_content = 0
    for model_name, data in all_models.items():
        subj = resolve_subject(model_name)
        source = data.get("source", "")
        traces_by_qid = data.get("traces", {}) or {}
        for qid, val in data["results"].items():
            content = item_content.get(qid)
            if content is None:
                missing_content += 1
            correct = _truncate(public_tests.get(str(qid)))
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=str(qid),
                content=content,
                correct_answer=correct,
            )

            if isinstance(val, list):
                # Per-sample binary from submissions graded_list.
                per_sample_traces = traces_by_qid.get(qid)
                for trial_idx, graded in enumerate(val, start=1):
                    trace_text = None
                    if isinstance(per_sample_traces, list) and trial_idx - 1 < len(per_sample_traces):
                        trace_text = per_sample_traces[trial_idx - 1]
                    rows.append({
                        "subject_id": subj,
                        "item_id": item,
                        "benchmark_id": bench_id,
                        "trial": trial_idx,
                        "test_condition": "source=submissions",
                        "response": 1.0 if bool(graded) else 0.0,
                        "correct_answer": correct,
                        "trace": trace_text,
                    })
            else:
                # Leaderboard-only or scalar-only fallback — aggregate pass@1.
                try:
                    pf = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isnan(pf):
                    continue
                tc = (
                    "source=leaderboard;aggregate=pass@1"
                    if source.startswith("leaderboard")
                    else "source=submissions;aggregate=pass@1"
                )
                rows.append({
                    "subject_id": subj,
                    "item_id": item,
                    "benchmark_id": bench_id,
                    "trial": 1,
                    "test_condition": tc,
                    "response": pf,
                    "correct_answer": correct,
                    "trace": None,
                })

    if missing_content:
        print(f"  NOTE: {missing_content} rows had no per-item content (leaderboard-only problems)")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)

    resp_cols = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition",
                 "response", "correct_answer", "trace"]
    resp = df[resp_cols].copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)

    traces = df.loc[df["trace"].notna(), [
        "subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace",
    ]]
    if len(traces) > 0:
        traces.to_parquet(TRACES_PATH, index=False)
        print(f"  wrote {TRACES_PATH.name} ({len(traces):,} rows)")
    else:
        print("  (no traces extracted — upstream doesn't publish model outputs)")

    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(resp):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  max trial: {df['trial'].max()}")
    print(f"  mean response: {df['response'].mean():.3f}")


def main():
    download()
    print("=== Loading submissions repo data ===")
    sub_models, item_content = load_submissions()
    print(f"Loaded {len(sub_models)} models from submissions repo")
    print(f"Collected content for {len(item_content)} items")

    print("\n=== Loading leaderboard data (v6) ===")
    lb_v6_path = f"{LEADERBOARD_DIR}/performances_generation.json"
    lb_v6 = load_leaderboard_json(lb_v6_path, "leaderboard_v6") if os.path.exists(lb_v6_path) else {}
    print(f"Loaded {len(lb_v6)} models from leaderboard v6")

    print("\n=== Loading leaderboard data (v5) ===")
    lb_v5_path = f"{LEADERBOARD_DIR}/v5.json"
    lb_v5 = load_leaderboard_json(lb_v5_path, "leaderboard_v5") if os.path.exists(lb_v5_path) else {}
    print(f"Loaded {len(lb_v5)} models from leaderboard v5")

    all_models = dict(sub_models)
    for lb, lb_label in [(lb_v6, "v6"), (lb_v5, "v5")]:
        for model, data in lb.items():
            if model not in all_models:
                all_models[model] = data
            elif data["n_problems"] > all_models[model]["n_problems"]:
                all_models[model] = data

    all_qids = set()
    for data in all_models.values():
        all_qids.update(data["results"].keys())
    print(f"\nTotal unique problems: {len(all_qids)}")
    print(f"Total models: {len(all_models)}")

    print("\n=== Loading public test cases (ground truth) ===")
    public_tests = load_public_tests()
    print(f"Loaded public_test_cases for {len(public_tests)} question_ids")

    df = build_long_form(all_models, item_content, public_tests)
    print_stats(df)


if __name__ == "__main__":
    main()
