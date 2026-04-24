"""
Build RewardBench long-form responses from per-model per-item evaluation scores.

Data sources:
  - allenai/reward-bench-results (HuggingFace): eval-set-scores/{org}/{model}.json
    files holding per-item binary outcomes on the 2,985-item RewardBench eval set.

Encoding (scalar, long-form):
  - Subject = reward model (``{org}/{model}``).
  - Item = ``{subset}:{raw_id}`` pair — content is not exposed per-item by the
    results repo, so the item registry stores the surrogate id only.
  - test_condition = ``subset={...}`` (23 RewardBench subsets: chat /
    chat_hard / safety / reasoning families).
  - response = 1.0 if the model correctly ranked chosen > rejected, else 0.0.

Output:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'RewardBench per-(reward-model, item) correctness on chosen>rejected ranking.',
    'testing_condition': 'test_condition=subset=...; response=1 iff reward model picks the chosen response.',
    'paper_url': 'https://arxiv.org/abs/2403.13787',
    'data_source_url': 'https://huggingface.co/datasets/allenai/reward-bench',
    'subject_type': 'reward_model',
    'item_type': 'response_pair',
    'license': 'ODC-BY',
    'citation': """@misc{lambert2024rewardbenchevaluatingrewardmodels,
      title={RewardBench: Evaluating Reward Models for Language Modeling},
      author={Nathan Lambert and Valentina Pyatkin and Jacob Morrison and LJ Miranda and Bill Yuchen Lin and Khyathi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2403.13787},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.13787},
}""",
    'tags': ['reasoning'],
    'modality': ['text'],
    'domain': ['reward_modeling'],
    'response_type': 'binary',
    'response_scale': '{0, 0.5, 1} (0.5 = judge tie, <1% of rows)',
    'categorical': True,
    'release_date': '2024-03',
}


import json
import os
import sys
import time
import urllib.error
import urllib.request
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

HF_RESULTS_REPO = "allenai/reward-bench-results"
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"
REFERENCE_MODEL_PATH = "eval-set-scores/openai/gpt-4o-2024-05-13.json"


def download_file(url: str, dest_path: str, retries: int = 3, delay: float = 1.0) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    FAILED after {retries} attempts: {e}")
                return False
    return False


def enumerate_model_files() -> list[str]:
    """List all JSON files under eval-set-scores/ via HF API."""
    print("Enumerating model files from HuggingFace API ...")
    tree_url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/eval-set-scores"
    req = urllib.request.Request(tree_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        orgs_data = json.loads(resp.read())
    orgs = [d["path"] for d in orgs_data if d["type"] == "directory"]
    print(f"  Found {len(orgs)} organizations")

    all_files = []
    for org_path in orgs:
        url = f"{HF_API_BASE}/{HF_RESULTS_REPO}/tree/main/{org_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files_data = json.loads(resp.read())
        files = [
            f["path"] for f in files_data
            if f["type"] == "file" and f["path"].endswith(".json")
        ]
        all_files.extend(files)

    print(f"  Found {len(all_files)} total JSON files")
    return all_files


def download_all_models(all_files: list[str]) -> None:
    print(f"\nDownloading {len(all_files)} model files ...")
    downloaded = skipped = failed = 0
    for i, fpath in enumerate(all_files):
        rel_path = fpath.replace("eval-set-scores/", "")
        dest_path = str(RAW_DIR / rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            skipped += 1
            continue
        url = f"{HF_RESOLVE_BASE}/{HF_RESULTS_REPO}/resolve/main/{fpath}"
        if download_file(url, dest_path):
            downloaded += 1
        else:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{len(all_files)}")
            time.sleep(0.3)
    print(f"  Downloaded: {downloaded}, Cached: {skipped}, Failed: {failed}")


def _extract_user_prompt(text_chosen) -> str | None:
    """Pull the user prompt out of a `text_chosen` entry.

    Format: list of {"role": "user" | "assistant", "content": str} — we take
    the first user message. Fallback to the raw string if it's already a str.
    """
    if isinstance(text_chosen, str):
        return text_chosen.strip() or None
    if isinstance(text_chosen, list):
        for msg in text_chosen:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
        # No explicit 'user' role: fall back to the first non-empty content.
        for msg in text_chosen:
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return None


def _extract_assistant_response(text_field) -> str | None:
    """Pull the assistant response text out of a `text_chosen`/`text_rejected` entry."""
    if isinstance(text_field, str):
        return text_field.strip() or None
    if isinstance(text_field, list):
        for msg in text_field:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
        # Fallback: last non-empty content (assistant usually comes last).
        for msg in reversed(text_field):
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return None


def get_reference_item_data() -> tuple[list[str], list[str], list[str | None], list[str | None]]:
    """Load the reference model to obtain canonical (raw_id, subset, prompt, chosen_trace) ordering.

    The reference JSON carries ``text_chosen`` aligned with ``id`` and
    ``subset``, so we can derive the per-item prompt (content) without a
    second download. We additionally extract the assistant-role content
    from ``text_chosen`` to use as a per-item trace (shared across all
    reward-model subjects that score this item).
    """
    ref_rel = REFERENCE_MODEL_PATH.replace("eval-set-scores/", "")
    ref_local = RAW_DIR / ref_rel
    if not (ref_local.exists() and ref_local.stat().st_size > 100):
        url = f"{HF_RESOLVE_BASE}/{HF_RESULTS_REPO}/resolve/main/{REFERENCE_MODEL_PATH}"
        print(f"  Downloading reference model from {url}")
        ref_local.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, str(ref_local))
    with open(ref_local) as f:
        data = json.load(f)
    raw_ids = [str(x) for x in data["id"]]
    subsets = list(data["subset"])
    text_chosen = data.get("text_chosen") or [None] * len(raw_ids)
    contents = [_extract_user_prompt(tc) for tc in text_chosen]
    chosen_responses = [_extract_assistant_response(tc) for tc in text_chosen]
    return raw_ids, subsets, contents, chosen_responses


def build_long_form() -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "rewardbench",
        name="RewardBench",
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

    print("STEP 1: Enumerating model files")
    all_files = enumerate_model_files()

    print("\nSTEP 2: Downloading model JSON files")
    download_all_models(all_files)

    print("\nSTEP 3: Extracting reference item IDs")
    ref_raw_ids, ref_subsets, ref_contents, ref_chosen_traces = get_reference_item_data()
    n_items = len(ref_raw_ids)
    n_with_content = sum(1 for c in ref_contents if c)
    n_with_trace = sum(1 for t in ref_chosen_traces if t)
    print(f"  {n_items} items across {len(set(ref_subsets))} subsets")
    print(f"  {n_with_content}/{n_items} items have extractable prompt text")
    print(f"  {n_with_trace}/{n_items} items have extractable chosen-response trace")

    # Pre-register items (one per (subset, raw_id) combo). For RewardBench
    # the correct_answer is the upstream-labeled "chosen" assistant response
    # (the one a correct reward model should rank higher).
    def _cap(s: str | None, n: int = 4000) -> str | None:
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None
        return s[:n]

    item_ids: list[str] = []
    item_correct: list[str | None] = []
    for raw_id, subset, content, chosen_trace in zip(
        ref_raw_ids, ref_subsets, ref_contents, ref_chosen_traces
    ):
        raw_item_id = f"{subset}:{raw_id}"
        correct = _cap(chosen_trace)
        iid = register_item(
            benchmark_id=bench_id,
            raw_item_id=raw_item_id,
            content=content,
            correct_answer=correct,
        )
        item_ids.append(iid)
        item_correct.append(correct)

    print("\nSTEP 4: Building long-form rows")
    rows = []
    skipped = 0
    for fpath in all_files:
        rel_path = fpath.replace("eval-set-scores/", "")
        local_path = RAW_DIR / rel_path
        if not local_path.exists():
            skipped += 1
            continue
        try:
            with open(local_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        results = data.get("results")
        if results is None or len(results) != n_items:
            skipped += 1
            continue

        file_subsets = data.get("subset")
        if file_subsets and file_subsets != ref_subsets:
            skipped += 1
            continue

        model_name = data.get("model") or rel_path.replace(".json", "")
        subj = resolve_subject(model_name)

        for idx, raw in enumerate(results):
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            rows.append({
                "subject_id": subj,
                "item_id": item_ids[idx],
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": f"subset={ref_subsets[idx]}",
                "response": val,
                "correct_answer": item_correct[idx],
                "trace": ref_chosen_traces[idx],
            })

    if skipped:
        print(f"  skipped {skipped} result files")

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
        print("\n  (empty)")
        return
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  mean response: {df['response'].mean():.3f}")


def main():
    print("RewardBench long-form builder")
    print("=" * 60)
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
