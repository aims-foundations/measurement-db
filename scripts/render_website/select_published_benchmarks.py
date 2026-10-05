"""Filter benchmarks in the HuggingFace repo (which is also what is currently displayed
on the website) to those with enough subjects, items, and released responses.

Keeps a slug if its benchmarks.parquet has n_subjects >= 30 and n_items >= 50,
and its responses.parquet (or legacy response.parquet) exists. Slugs in INCLUDED
are kept without meeting either condition; slugs in EXCLUDED are dropped regardless.
The curated list is written to website/content/generated/published-benchmarks.json,
which is used by generate_benchmark_gallery.py.
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import list_repo_files

#HuggingFace repo is "aims-foundations/measurement-db".
REPO = "aims-foundations/measurement-db"
MIN_SUBJECTS = 30
MIN_ITEMS = 50

# Present on HuggingFace but deliberately not displayed.
EXCLUDED = {
    "benger", "doris_mae", "hivmedqa", "tdd_bench_verified",  # Zenodo rejects access
    "diagram_understanding",                                   # raw source unreachable
    "tabicl",                                                  # complex package installation
    "appworld", "autoelicit", "mathconstruct", "sciarena",
    "fantastic_bugs", "reeval",
    "agc_bench",
    "live_agent_risk",
    "cross_care",    # not released yet
    "alpacaeval",    #Response is a continuous soft win-probability
    "gaia"
}

# Displayed regardless of the subject/item thresholds and of a missing
# responses table. Still has to exist in the repo, and EXCLUDED wins.
INCLUDED = {
    "tau_voice",
    "tau2_bench",
    "deepswe",
    "terminal_bench_2_1",
    "programbench",
    "agents_last_exam", "osworld_v2", "frontieror", "arc_agi_3"
}

assert not (INCLUDED & EXCLUDED), f"in both INCLUDED and EXCLUDED: {INCLUDED & EXCLUDED}"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "website" / "content" / "generated" / "published-benchmarks.json"

# One API call, so existence checks don't cost a request per slug.
repo_files = set(list_repo_files(REPO, repo_type="dataset"))
# list of benchmark slugs derived from the HuggingFace repo, excluding those in EXCLUDED.
displayed = sorted({f.split("/")[0] for f in repo_files if "/" in f} - EXCLUDED)

published_benchmarks = []
for benchmark in displayed:
    if benchmark in INCLUDED:
        published_benchmarks.append(benchmark)
        continue

    if not any(f"{benchmark}/{name}" in repo_files
               for name in ("responses.parquet", "response.parquet")):
        continue

    df = pd.read_parquet(f"hf://datasets/{REPO}/{benchmark}/benchmarks.parquet")
    if df["n_subjects"].iloc[0] >= MIN_SUBJECTS and df["n_items"].iloc[0] >= MIN_ITEMS:
        published_benchmarks.append(benchmark)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(published_benchmarks, indent=2) + "\n")
