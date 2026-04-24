#!/usr/bin/env python3
"""
Build the SWE-bench Verified long-form responses from the experiments repo.

Reads results.json files downloaded from:
  https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/

Each results.json has the structure:
  {
    "no_generation": [...],   # instance IDs with no generated patch
    "no_logs": [...],         # instance IDs with no execution logs
    "resolved": [...]         # instance IDs that were successfully resolved
  }

Outputs:
  - responses.parquet   # long-form: (subject_id, item_id, trial, response, trace)
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions
"""

INFO = {
    'description': 'Build the SWE-bench Verified response matrix from the experiments repo',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2310.06770',
    'data_source_url': 'https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/',
    'subject_type': 'agent',
    'item_type': 'issue',
    'license': 'MIT',
    'citation': """@misc{jimenez2024swebenchlanguagemodelsresolve,
      title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
      author={Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
      year={2024},
      eprint={2310.06770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.06770},
}""",
    'tags': ['coding'],
    'modality': ['text'],
    'domain': ['software_engineering'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2023-10',
}


import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# Paths
_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw" / "results_json"
INSTANCE_CONTENT_PATH = _BENCHMARK_DIR / "raw" / "instance_content.parquet"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)


def download():
    """Download raw data from external sources."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SWE-bench submission results via GitHub API...")
    api_url = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        submissions = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print(f"WARNING: GitHub API call failed: {e}. Using existing raw data if available.")
        submissions = []

    downloaded = 0
    for sub in submissions:
        if sub.get("type") != "dir":
            continue
        name = sub["name"]
        out_file = RAW_DIR / f"{name}.json"
        if out_file.exists():
            continue
        results_url = (
            f"https://raw.githubusercontent.com/SWE-bench/experiments/main/"
            f"evaluation/verified/{name}/results/results.json"
        )
        try:
            req2 = urllib.request.Request(results_url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req2, timeout=15).read()
            out_file.write_bytes(data)
            downloaded += 1
            if downloaded % 20 == 0:
                time.sleep(1)
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"Downloaded {downloaded} new results files. "
          f"Total: {len(list(RAW_DIR.glob('*.json')))}")


def _truncate_patch(p: str | None, limit: int = 4000) -> str | None:
    """Truncate a reference patch string to at most ``limit`` chars.

    If the patch is longer than ``limit``, we reserve room for a
    ``...[truncated]`` suffix so downstream readers know content is cut.
    """
    if p is None:
        return None
    s = str(p)
    if not s:
        return None
    if len(s) <= limit:
        return s
    suffix = "\n...[truncated]"
    return s[: max(0, limit - len(suffix))] + suffix


def fetch_instance_content() -> tuple[dict[str, str], dict[str, str]]:
    """Load {instance_id -> problem_statement} and {instance_id -> patch}.

    Caches the parsed dataset to ``raw/instance_content.parquet`` so rebuilds
    are fully offline after the first run. The reference patch is the
    upstream ``patch`` field (the canonical git-diff ground truth) and is
    used as ``correct_answer`` on the registered item.
    """
    if INSTANCE_CONTENT_PATH.exists():
        df = pd.read_parquet(INSTANCE_CONTENT_PATH)
        content_map = dict(zip(df["instance_id"], df["problem_statement"]))
        patch_map: dict[str, str] = {}
        if "patch" in df.columns:
            patch_map = {
                r["instance_id"]: r["patch"]
                for _, r in df.iterrows()
                if isinstance(r["patch"], str) and r["patch"]
            }
        print(f"  Loaded {len(content_map)} problem_statements "
              f"(+{len(patch_map)} patches) from cache")
        return content_map, patch_map

    print("  Downloading SWE-bench Verified problem_statements from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: 'datasets' library not available; content will be None",
              file=sys.stderr)
        return {}, {}

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    rows = [
        {
            "instance_id": r["instance_id"],
            "problem_statement": r["problem_statement"] or "",
            "patch": r.get("patch") or "",
        }
        for r in ds
    ]
    df = pd.DataFrame(rows)
    INSTANCE_CONTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INSTANCE_CONTENT_PATH, index=False)
    content_map = dict(zip(df["instance_id"], df["problem_statement"]))
    patch_map = {
        r["instance_id"]: r["patch"]
        for _, r in df.iterrows()
        if isinstance(r["patch"], str) and r["patch"]
    }
    print(f"  Cached {len(content_map)} problem_statements "
          f"(+{len(patch_map)} patches) to {INSTANCE_CONTENT_PATH.name}")
    return content_map, patch_map


def load_results(results_dir: Path):
    """Load all results.json files and return {model_name: set_of_resolved_ids}."""
    model_results = {}
    all_instance_ids = set()

    for fpath in sorted(results_dir.glob("*.json")):
        model_name = fpath.stem
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Skipping {fpath.name}: {e}", file=sys.stderr)
            continue

        if "resolved" not in data:
            print(f"WARNING: Skipping {fpath.name}: no 'resolved' key", file=sys.stderr)
            continue

        resolved = set(data["resolved"])
        model_results[model_name] = resolved

        for key in data:
            if isinstance(data[key], list):
                all_instance_ids.update(data[key])

    return model_results, sorted(all_instance_ids)


def build_long_form(model_results, instance_ids, instance_content, instance_patches):
    """Build long-form responses.parquet + registry contribs."""
    bench_id = get_benchmark_id(
        "swebench",
        name="SWE-bench Verified",
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

    missing_content = 0
    rows = []
    for model_name, resolved_set in sorted(model_results.items()):
        subj = resolve_subject(model_name)
        for iid in instance_ids:
            # SWE-bench results.json enumerates every instance in some bucket
            # (resolved, no_generation, no_logs, ...). Cells are observed 0/1.
            content = instance_content.get(iid)
            if content is None or content == "":
                missing_content += 1
                content = None
            # Reference patch (git-diff) is the canonical ground truth.
            correct = _truncate_patch(instance_patches.get(iid))
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=str(iid),
                content=content,
                correct_answer=correct,
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": None,
                "response": 1.0 if iid in resolved_set else 0.0,
                "correct_answer": correct,
                "trace": None,
            })

    if missing_content:
        print(f"  NOTE: {missing_content} rows had no problem_statement "
              f"(upstream instance not in Verified HF dataset)")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)
    df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  mean response (resolve rate): {df['response'].mean():.4f}")


def main():
    download()
    print(f"Loading results from: {RAW_DIR}")
    model_results, instance_ids = load_results(RAW_DIR)

    print(f"Found {len(model_results)} models")
    print(f"Found {len(instance_ids)} unique instance IDs")

    print("\nFetching per-instance problem_statements...")
    instance_content, instance_patches = fetch_instance_content()

    df = build_long_form(model_results, instance_ids, instance_content, instance_patches)
    print_stats(df)


if __name__ == "__main__":
    main()
