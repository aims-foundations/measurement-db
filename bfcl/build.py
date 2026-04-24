"""Build BFCL long-form responses from BFCL-Result score files.

The BFCL-Result repo stores per-(model, category) score files as NDJSON:
  - Line 0: {"accuracy": ..., "correct_count": ..., "total_count": ...}
  - Lines 1+: failed task entries whose task id lives in entry["prompt"]["id"]

Strategy: for each model x category, load the score file, collect failed task IDs,
and mark all tasks in the model's own result file as pass (1) unless failed.

Uses the 2024-12-29 snapshot (92 models, 22 categories).

Output:
  - responses.parquet   # long-form: (subject_id, item_id, trial, response, trace)
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions
"""

INFO = {
    'description': """Build a per-task binary response matrix and extract per-task error types from BFCL score files""",
    'testing_condition': '',
    'paper_url': 'https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html',
    'data_source_url': 'https://github.com/HuanzhiMao/BFCL-Result',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'Apache-2.0',
    'citation': '@misc{patil2025bfcl,\n  title={BFCL},\n  year={2025},\n  howpublished={\\url{https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html}},\n}',
    'tags': ['coding'],
    'modality': ['text'],
    'domain': ['tool_use'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2024-09',
}


import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
SCORE_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "score")
RESULT_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "result")
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
TRACES_PATH = _BENCHMARK_DIR / "traces.parquet"

# Pinned commit of ShishirPatil/gorilla that carries the canonical BFCL_v3
# task files (before the v4 rename in Aug 2025). Fetched once and cached.
# Two pinned gorilla commits: the post-package commit holds the bulk of the
# v3 task files (under bfcl_eval/data/), and the pre-retirement commit still
# carries the executable (exec_*) and rest JSONs before they were dropped
# from the leaderboard. Short 10-char SHAs are used because the full 40-char
# SHA returns 404 from raw.githubusercontent for this repo.
GORILLA_COMMIT = "c15b2a1516"
GORILLA_BASE = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    f"{GORILLA_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/data"
)
GORILLA_EXEC_COMMIT = "11cb4fe244"
GORILLA_EXEC_BASE = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    f"{GORILLA_EXEC_COMMIT}/berkeley-function-call-leaderboard/data"
)
BFCL_V3_FILES = [
    "BFCL_v3_irrelevance.json",
    "BFCL_v3_java.json",
    "BFCL_v3_javascript.json",
    "BFCL_v3_live_irrelevance.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_live_parallel.json",
    "BFCL_v3_live_parallel_multiple.json",
    "BFCL_v3_live_relevance.json",
    "BFCL_v3_live_simple.json",
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_long_context.json",
    "BFCL_v3_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
    "BFCL_v3_simple.json",
]
BFCL_V3_EXEC_FILES = [
    "BFCL_v3_exec_simple.json",
    "BFCL_v3_exec_multiple.json",
    "BFCL_v3_exec_parallel.json",
    "BFCL_v3_exec_parallel_multiple.json",
    "BFCL_v3_rest.json",
]

# Canonical ``possible_answer`` files. Upstream BFCL_v3 stores the expected
# function-call JSON for each task under ``possible_answer/``. Categories
# without a possible_answer file (irrelevance/relevance detection, exec_*,
# multi_turn) have no per-task answer — the judge uses behavior/execution
# outcomes instead.
BFCL_V3_POSSIBLE_ANSWER_FILES = [
    "BFCL_v3_java.json",
    "BFCL_v3_javascript.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_live_parallel.json",
    "BFCL_v3_live_parallel_multiple.json",
    "BFCL_v3_live_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
    "BFCL_v3_simple.json",
]

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)


def download():
    """Download raw data from external sources."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone_dir = RAW_DIR / "BFCL-Result"
    if not clone_dir.exists():
        print("Cloning BFCL-Result repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/HuanzhiMao/BFCL-Result.git", str(clone_dir)],
            check=True,
        )
    else:
        print("BFCL-Result repo already exists, pulling latest...")
        subprocess.run(
            ["git", "-C", str(clone_dir), "pull", "--ff-only"],
            check=False,
        )


def download_v3_task_files():
    """Cache the canonical BFCL_v3 task JSONs under raw/bfcl_v3/.

    Pulls the main v3 files from the packagerization commit and the
    retired-from-leaderboard exec_*/rest files from an earlier commit. Also
    caches the per-task ``possible_answer`` JSONs (expected function-call
    ground truth) under raw/bfcl_v3/possible_answer/.
    """
    dest_dir = RAW_DIR / "bfcl_v3"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for fname, base_url in (
        [(f, GORILLA_BASE) for f in BFCL_V3_FILES]
        + [(f, GORILLA_EXEC_BASE) for f in BFCL_V3_EXEC_FILES]
    ):
        dest = dest_dir / fname
        if dest.exists() and dest.stat().st_size > 100:
            continue
        url = f"{base_url}/{fname}"
        print(f"  Fetching {fname}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
        except Exception as e:
            print(f"    WARNING: failed to fetch {fname}: {e}")

    # possible_answer/ sits alongside data/ in gorilla. Try the packaged-layout
    # base first (bfcl_eval/data/possible_answer), then the pre-package layout
    # (data/possible_answer).
    pa_dir = dest_dir / "possible_answer"
    pa_dir.mkdir(parents=True, exist_ok=True)
    for fname in BFCL_V3_POSSIBLE_ANSWER_FILES:
        dest = pa_dir / fname
        if dest.exists() and dest.stat().st_size > 2:
            continue
        fetched = False
        for base in (f"{GORILLA_BASE}/possible_answer",
                     f"{GORILLA_EXEC_BASE}/possible_answer"):
            url = f"{base}/{fname}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                fetched = True
                break
            except Exception:
                continue
        if not fetched:
            print(f"    NOTE: no possible_answer for {fname} (upstream omits it)")

    return dest_dir


def _question_to_text(q) -> str | None:
    """Flatten a BFCL ``question`` field (list[list[dict|str]]) to a string."""
    if q is None:
        return None
    if isinstance(q, str):
        return q.strip() or None
    parts: list[str] = []

    def _walk(obj):
        if isinstance(obj, dict):
            role = obj.get("role")
            content = obj.get("content")
            if isinstance(content, str) and content.strip():
                if role:
                    parts.append(f"[{role}] {content}")
                else:
                    parts.append(content)
        elif isinstance(obj, list):
            for x in obj:
                _walk(x)
        elif isinstance(obj, str) and obj.strip():
            parts.append(obj)

    _walk(q)
    if not parts:
        return None
    return "\n".join(parts)[:8000]


def load_task_content(task_files_dir: Path) -> dict[str, str]:
    """Return {task_id: prompt_text} from canonical BFCL_v3 files + score fallback.

    Primary source: the pinned gorilla snapshot under raw/bfcl_v3/. For
    categories without a canonical question file (exec_*, rest), we fall back
    to scraping ``prompt.question`` from the embedded BFCL-Result score
    entries (best-effort; only failed tasks have embedded prompts, so
    coverage for those categories will be partial).
    """
    content: dict[str, str] = {}

    for fname in BFCL_V3_FILES + BFCL_V3_EXEC_FILES:
        fpath = task_files_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = str(rec.get("id", ""))
                if not tid:
                    continue
                text = _question_to_text(rec.get("question"))
                if text:
                    content[tid] = text

    # Fallback: scan all BFCL-Result score files for embedded prompts. This
    # recovers prompts for exec_* and rest categories (failed entries only).
    if os.path.isdir(SCORE_DIR):
        for model_name in os.listdir(SCORE_DIR):
            mdir = os.path.join(SCORE_DIR, model_name)
            if not os.path.isdir(mdir):
                continue
            for fname in os.listdir(mdir):
                if not fname.endswith("_score.json"):
                    continue
                fpath = os.path.join(mdir, fname)
                try:
                    with open(fpath) as f:
                        for i, line in enumerate(f):
                            if i == 0:
                                continue  # header
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            p = entry.get("prompt")
                            if not isinstance(p, dict):
                                continue
                            tid = str(p.get("id", ""))
                            if not tid or tid in content:
                                continue
                            text = _question_to_text(p.get("question"))
                            if text:
                                content[tid] = text
                except OSError:
                    continue

    return content


def _truncate(s: str | None, limit: int = 4000) -> str | None:
    """Truncate ``s`` to at most ``limit`` chars with a marker."""
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


def load_possible_answers(task_files_dir: Path) -> dict[str, str]:
    """Return {task_id: json.dumps(expected_fn_calls)} from possible_answer/.

    Each line in a possible_answer JSONL carries ``{"id": ..., "ground_truth": ...}``
    where ground_truth is a list/nested-list of expected function calls.
    Categories without a possible_answer (exec_*, irrelevance/relevance,
    multi_turn) are judged on behavior and have no per-task answer string —
    those tasks return ``None``.
    """
    pa_dir = task_files_dir / "possible_answer"
    if not pa_dir.exists():
        return {}
    out: dict[str, str] = {}
    for fname in BFCL_V3_POSSIBLE_ANSWER_FILES:
        fpath = pa_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 2:
            continue
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid = str(rec.get("id", ""))
                    if not tid:
                        continue
                    gt = rec.get("ground_truth")
                    if gt is None:
                        continue
                    try:
                        val = json.dumps(gt, ensure_ascii=False)
                    except (TypeError, ValueError):
                        val = str(gt)
                    if val:
                        out[tid] = val
        except OSError:
            continue
    return out


def load_score_file(path):
    """Load an NDJSON score file. Returns (header_dict, set_of_failed_task_ids).

    Score file format:
      Line 0: {"accuracy": ..., "correct_count": ..., "total_count": ...}
      Lines 1+: failed entries where the actual task ID is in entry["prompt"]["id"]
                (NOT entry["id"] which is a numeric line index)
    """
    with open(path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    header = lines[0]
    failed_ids = set()
    for entry in lines[1:]:
        task_id = entry.get("prompt", {}).get("id")
        if task_id is None:
            task_id = str(entry.get("id", ""))
        failed_ids.add(str(task_id))
    return header, failed_ids


def get_all_task_ids_from_result(result_dir, model_name, category):
    """Return all task IDs for a (model, category) from the result file."""
    result_file = os.path.join(result_dir, model_name, f"BFCL_v3_{category}_result.json")
    if not os.path.exists(result_file):
        return None
    ids = []
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ids.append(str(entry.get("id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def load_model_traces(result_dir, model_name, category):
    """Return {task_id: trace_text} from a (model, category) result file.

    Each line of the result JSONL holds ``{"id": ..., "result": ..., ...}``
    where ``result`` is the raw model output (function-call text / JSON). We
    serialize it back to a string so it's a faithful trace.
    """
    result_file = os.path.join(result_dir, model_name, f"BFCL_v3_{category}_result.json")
    if not os.path.exists(result_file):
        return {}
    traces: dict[str, str] = {}
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = str(entry.get("id", ""))
            if not tid:
                continue
            raw = entry.get("result")
            if raw is None:
                continue
            if isinstance(raw, str):
                text = raw
            else:
                try:
                    text = json.dumps(raw, ensure_ascii=False)
                except (TypeError, ValueError):
                    text = str(raw)
            if not text:
                continue
            traces[tid] = text[:8000]
    return traces


def build_long_form(model_dirs, categories, category_task_ids, task_content=None,
                    possible_answers=None):
    """Write long-form responses.parquet + registry contribs."""
    bench_id = get_benchmark_id(
        "bfcl",
        name="BFCL",
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
    task_content = task_content or {}
    possible_answers = possible_answers or {}

    rows = []
    for model_idx, model_name in enumerate(model_dirs):
        subj = resolve_subject(model_name)
        model_score_dir = os.path.join(SCORE_DIR, model_name)

        for cat in categories:
            if cat not in category_task_ids:
                continue
            local_ids, _ = category_task_ids[cat]

            score_path = os.path.join(model_score_dir, f"BFCL_v3_{cat}_score.json")
            if not os.path.exists(score_path):
                # Subject didn't attempt this category -> skip (unobserved)
                continue

            _, failed_ids = load_score_file(score_path)

            model_result_ids = get_all_task_ids_from_result(RESULT_DIR, model_name, cat)
            task_iter = model_result_ids if model_result_ids is not None else local_ids
            traces_for_cat = load_model_traces(RESULT_DIR, model_name, cat)

            for tid in task_iter:
                raw_id = f"{cat}::{tid}"
                correct = _truncate(possible_answers.get(tid))
                item = register_item(
                    benchmark_id=bench_id,
                    raw_item_id=raw_id,
                    content=task_content.get(tid),
                    correct_answer=correct,
                )
                response = 0.0 if tid in failed_ids else 1.0
                rows.append({
                    "subject_id": subj,
                    "item_id": item,
                    "benchmark_id": bench_id,
                    "trial": 1,
                    "test_condition": None,
                    "response": float(response),
                    "correct_answer": correct,
                    "trace": traces_for_cat.get(tid),
                })

        if (model_idx + 1) % 10 == 0:
            print(f"  Processed {model_idx + 1}/{len(model_dirs)} models")

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
    n_subjects = df["subject_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"\n  subjects: {n_subjects}")
    print(f"  items:    {n_items}")
    print(f"  rows:     {len(df):,}")
    print(f"  pass rate: {df['response'].mean():.3f}")


def main():
    download()
    print("\nFetching canonical BFCL_v3 task files from gorilla@{}...".format(
        GORILLA_COMMIT[:10]
    ))
    task_files_dir = download_v3_task_files()
    task_content = load_task_content(task_files_dir)
    print(f"  loaded {len(task_content)} task prompts")

    possible_answers = load_possible_answers(task_files_dir)
    print(f"  loaded {len(possible_answers)} possible_answer ground truths")

    model_dirs = sorted([
        d for d in os.listdir(SCORE_DIR)
        if os.path.isdir(os.path.join(SCORE_DIR, d))
    ])
    print(f"Found {len(model_dirs)} models with score files")

    sample_model = model_dirs[0]
    score_files = sorted([
        f for f in os.listdir(os.path.join(SCORE_DIR, sample_model))
        if f.endswith("_score.json")
    ])
    categories = [
        f.replace("BFCL_v3_", "").replace("_score.json", "")
        for f in score_files
    ]
    print(f"Found {len(categories)} categories: {categories}")

    print("\nDiscovering task IDs from result files...")
    ref_model = "gpt-4o-2024-11-20-FC"
    if ref_model not in model_dirs:
        ref_model = model_dirs[0]

    category_task_ids = {}
    total_tasks = 0
    for cat in categories:
        ids = get_all_task_ids_from_result(RESULT_DIR, ref_model, cat)
        if ids is None:
            for alt_model in model_dirs:
                ids = get_all_task_ids_from_result(RESULT_DIR, alt_model, cat)
                if ids is not None:
                    print(f"    -> Using {alt_model} for {cat}")
                    break
        if ids is not None:
            global_ids = [f"{cat}::{tid}" for tid in ids]
            category_task_ids[cat] = (ids, global_ids)
            total_tasks += len(ids)
            print(f"  {cat}: {len(ids)} tasks")
        else:
            print(f"  {cat}: SKIPPED (no result file found)")

    print(f"\nTotal task items across categories: {total_tasks}")

    df = build_long_form(model_dirs, categories, category_task_ids, task_content,
                         possible_answers)
    print_stats(df)


if __name__ == "__main__":
    main()
