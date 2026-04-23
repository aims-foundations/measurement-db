"""
Build AfriMed-QA long-form responses from per-model per-item evaluation results.

Data source:
  - GitHub: intron-innovation/AfriMed-QA, results/ directory
  - Each model has a subdirectory with CSV files for different datasets/settings
  - MCQ CSV files contain: sample_id, question, answer, preds, correct (binary 0/1)
  - We focus on MCQ tasks with base-prompt, 0-shot evaluation

AfriMed-QA overview:
  - Medical QA benchmark for African healthcare contexts
  - Multiple dataset versions: v1 (3000 MCQs), v2 (3910 MCQs), v2.5 (289 MCQs)
  - Questions span 20+ medical specialties; contributors from 16 African countries

Strategy:
  - Select best eval per model: base-prompt > instruct, v2 > v1 > v2.5
  - Observation `test_condition` captures `source=<dataset>|prompt=<base|instruct>`
  - Items are shared across models by sample_id; their content/ground-truth come
    from the richest available CSV (or the phase_2 raw file, when present).

Outputs:
  - responses.parquet  # long-form
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': 'AfriMed-QA: medical QA for African healthcare contexts; binary correctness on MCQ items.',
    'testing_condition': 'test_condition captures evaluation source dataset + prompt (e.g. "source=afrimedqa-v2|prompt=base").',
    'paper_url': 'https://arxiv.org/abs/2411.15640',
    'data_source_url': 'https://github.com/intron-innovation/AfriMed-QA',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-NC-SA-4.0',
    'citation': """@misc{olatunji2025afrimedqapanafricanmultispecialtymedical,
      title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
      author={Tobi Olatunji and Charles Nimo and Abraham Owodunni and Tassallah Abdullahi and Emmanuel Ayodele and Mardhiyah Sanni and Chinemelu Aka and Folafunmi Omofoye and Foutse Yuehgoh and Timothy Faniran and Bonaventure F. P. Dossou and Moshood Yekini and Jonas Kemp and Katherine Heller and Jude Chidubem Omeke and Chidi Asuzu MD and Naome A. Etori and Aimérou Ndiaye and Ifeoma Okoh and Evans Doe Ocansey and Wendy Kinara and Michael Best and Irfan Essa and Stephen Edward Moore and Chris Fourie and Mercy Nyamewaa Asiedu},
      year={2025},
      eprint={2411.15640},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={https://doi.org/10.18653/v1/2025.acl-long.96},
      url={https://arxiv.org/abs/2411.15640},
}""",
    'tags': ['multilingual'],
}


import csv
import os
import re
import subprocess
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

REPO_URL = "https://github.com/intron-innovation/AfriMed-QA.git"
REPO_DIR = RAW_DIR / "AfriMed-QA"
RESULTS_DIR = REPO_DIR / "results"
DATA_DIR = REPO_DIR / "data"


def clone_repo():
    if REPO_DIR.is_dir() and RESULTS_DIR.is_dir():
        print(f"  Already cloned: {REPO_DIR}")
        return
    print(f"  Cloning {REPO_URL} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  ERROR: git clone failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Cloned to: {REPO_DIR}")


def classify_csv(filename):
    fname = filename.lower()
    if not fname.endswith(".csv"):
        return None
    if "mcq" not in fname:
        return None

    if "afrimed-qa-v2.5" in fname or "afrimed-qa-v2-5" in fname:
        dataset = "afrimedqa-v2.5"
    elif "afrimed-qa-v2" in fname:
        dataset = "afrimedqa-v2"
    elif "afrimed-qa-v1" in fname or "afrimed-qa_" in fname:
        dataset = "afrimedqa-v1"
    elif "medqa" in fname:
        dataset = "medqa"
    else:
        dataset = "unknown"

    if "instruct-prompt" in fname or "instruct_prompt" in fname or "instruct_0shot" in fname:
        prompt = "instruct"
    else:
        prompt = "base"

    shot_match = re.search(r"(\d+)[_-]?shot", fname)
    shots = int(shot_match.group(1)) if shot_match else 0
    return {"dataset": dataset, "prompt": prompt, "shots": shots}


def read_mcq_csv(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"    WARNING: Could not read {filepath}: {e}")
        return None
    if "" in df.columns or "Unnamed: 0" in df.columns:
        idx_col = "" if "" in df.columns else "Unnamed: 0"
        df = df.drop(columns=[idx_col], errors="ignore")
    if "sample_id" not in df.columns or "correct" not in df.columns:
        return None
    result = df[["sample_id"]].copy()
    result["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    result = result.dropna(subset=["sample_id"])
    return result


def read_mcq_csv_full(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        return None
    if "" in df.columns or "Unnamed: 0" in df.columns:
        idx_col = "" if "" in df.columns else "Unnamed: 0"
        df = df.drop(columns=[idx_col], errors="ignore")
    if "sample_id" not in df.columns:
        return None
    return df


def discover_evaluations():
    evaluations = []
    for model_dir_name in sorted(os.listdir(RESULTS_DIR)):
        model_path = RESULTS_DIR / model_dir_name
        if not model_path.is_dir():
            continue
        for csv_file in sorted(os.listdir(model_path)):
            info = classify_csv(csv_file)
            if info is None:
                continue
            filepath = model_path / csv_file
            # Check header
            try:
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    if "correct" not in header or "sample_id" not in header:
                        continue
            except (StopIteration, Exception):
                continue
            try:
                with open(filepath, "r") as fcount:
                    count_reader = csv.reader(fcount)
                    next(count_reader)
                    n_items = sum(1 for _ in count_reader)
            except Exception:
                n_items = 0
            evaluations.append({
                "model_dir": model_dir_name,
                "dataset": info["dataset"],
                "prompt": info["prompt"],
                "shots": info["shots"],
                "filepath": str(filepath),
                "filename": csv_file,
                "n_items": n_items,
            })
    print(f"  Found {len(evaluations)} MCQ evaluation files")
    return evaluations


def select_primary_evaluations(evaluations):
    """Select the best evaluation file for each model."""
    candidates = [
        e for e in evaluations
        if e["prompt"] == "base" and e["shots"] == 0 and e["dataset"].startswith("afrimedqa")
    ]

    model_candidates: dict = {}
    for e in candidates:
        model_candidates.setdefault(e["model_dir"], []).append(e)

    dataset_priority = {"afrimedqa-v2": 0, "afrimedqa-v1": 1, "afrimedqa-v2.5": 2}
    selected = []
    for model, cands in sorted(model_candidates.items()):
        cands.sort(key=lambda e: (dataset_priority.get(e["dataset"], 99), -e["n_items"]))
        selected.append(cands[0])

    covered = {e["model_dir"] for e in selected}
    instruct = [
        e for e in evaluations
        if e["model_dir"] not in covered
        and e["dataset"].startswith("afrimedqa")
        and e["shots"] == 0
    ]
    instruct_by_model: dict = {}
    for e in instruct:
        instruct_by_model.setdefault(e["model_dir"], []).append(e)
    for model, cands in sorted(instruct_by_model.items()):
        cands.sort(key=lambda e: (dataset_priority.get(e["dataset"], 99), -e["n_items"]))
        selected.append(cands[0])
        covered.add(model)

    # Models whose only afrimedqa eval is under an "unknown" dataset tag
    unknown_candidates: dict = {}
    for e in evaluations:
        if (e["model_dir"] not in covered
                and e["dataset"] == "unknown"
                and e["prompt"] == "base"
                and e["shots"] == 0
                and e["n_items"] >= 2800):
            unknown_candidates.setdefault(e["model_dir"], []).append(e)
    for model, cands in sorted(unknown_candidates.items()):
        cands.sort(key=lambda e: -e["n_items"])
        selected.append(cands[0])

    print(f"  Selected {len(selected)} model evaluations total")
    return selected


def clean_model_name(model_dir_name):
    name_map = {
        "jsl-med-llama-8b": "JSL-MedLlama-3-8B-v2.0",
        "mistral-7b": "Mistral-7B-Instruct-v0.2",
        "phi3-mini-4k": "Phi-3-mini-4k-instruct",
        "Mistral-7B-Instruct-v02": "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v03": "Mistral-7B-Instruct-v0.3",
        "Meditron-7B-FT": "Meditron-7B",
        "PMC-LLAMA-7B-FT": "PMC-LLaMA-7B",
    }
    return name_map.get(model_dir_name, model_dir_name)


def _load_item_registry(bench_id: str, selected_evals):
    """Register each distinct sample_id exactly once.

    Content preference (richest first):
      1. phase_2 raw CSV (question_clean + answer_options)
      2. Any results CSV row carrying a question/specialty etc.
    """
    # Collect all sample_ids + pick the richest results CSV for fallback content.
    all_ids: set = set()
    richest_results = None
    richest_n = -1
    for ev in selected_evals:
        fdf = read_mcq_csv_full(ev["filepath"])
        if fdf is None:
            continue
        all_ids.update(fdf["sample_id"].dropna().astype(str).tolist())
        # prefer files with 'specialty' column, else largest
        has_spec = "specialty" in fdf.columns
        score = ev["n_items"] + (10 ** 6 if has_spec else 0)
        if score > richest_n:
            richest_results = fdf
            richest_n = score

    phase2_path = REPO_DIR / "data" / "afri_med_qa_15k_v2.5_phase_2_15275.csv"
    phase2 = None
    if phase2_path.exists():
        try:
            phase2 = pd.read_csv(phase2_path, low_memory=False)
            phase2 = phase2.set_index("sample_id")
        except Exception:
            phase2 = None

    results_by_id: dict = {}
    if richest_results is not None:
        for _, row in richest_results.iterrows():
            sid = row.get("sample_id")
            if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                continue
            results_by_id[str(sid)] = row

    item_ids: dict = {}
    item_gold: dict = {}
    for sid in sorted(all_ids):
        content = None
        gold = None
        if phase2 is not None and sid in phase2.index:
            prow = phase2.loc[sid]
            if isinstance(prow, pd.DataFrame):
                prow = prow.iloc[0]
            parts = []
            q = prow.get("question_clean") if "question_clean" in prow else prow.get("question")
            if q is not None and not (isinstance(q, float) and pd.isna(q)):
                parts.append(str(q))
            opts = prow.get("answer_options")
            if opts is not None and not (isinstance(opts, float) and pd.isna(opts)):
                parts.append(str(opts)[:1500])
            if parts:
                content = "\n\n".join(parts)[:4000]
            ca = prow.get("correct_answer")
            if ca is not None and not (isinstance(ca, float) and pd.isna(ca)):
                gold = str(ca)
        if content is None and sid in results_by_id:
            rrow = results_by_id[sid]
            q = rrow.get("question")
            if q is None or (isinstance(q, float) and pd.isna(q)):
                q = rrow.get("question_x") or rrow.get("question_y")
            if q is not None and not (isinstance(q, float) and pd.isna(q)):
                content = str(q)[:4000]
            if gold is None:
                ca = rrow.get("answer") or rrow.get("correct_answer")
                if ca is not None and not (isinstance(ca, float) and pd.isna(ca)):
                    gold = str(ca)
        iid = register_item(
            benchmark_id=bench_id,
            raw_item_id=str(sid),
            content=content,
            correct_answer=gold,
        )
        item_ids[str(sid)] = iid
        item_gold[str(sid)] = gold
    return item_ids, item_gold


def main():
    print("AfriMed-QA long-form builder")
    print("=" * 60)

    clone_repo()

    bench_id = get_benchmark_id(
        "afrimedqa",
        name="AfriMed-QA",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
    )

    evaluations = discover_evaluations()
    selected_evals = select_primary_evaluations(evaluations)

    cols = [
        "subject_id", "item_id", "benchmark_id", "trial",
        "test_condition", "response", "correct_answer", "trace",
    ]

    if not selected_evals:
        print("  No evaluations selected — writing empty responses.parquet")
        df = pd.DataFrame(columns=cols)
        df.to_parquet(RESPONSES_PATH, index=False)
        registry_save(CONTRIB_DIR)
        return

    item_ids, item_gold = _load_item_registry(bench_id, selected_evals)
    print(f"  Registered {len(item_ids):,} items")

    rows = []
    for ev in selected_evals:
        df = read_mcq_csv(ev["filepath"])
        if df is None:
            continue
        model_name = clean_model_name(ev["model_dir"])
        subj = resolve_subject(model_name)
        cond = f"source={ev['dataset']}|prompt={ev['prompt']}"
        for _, r in df.iterrows():
            sid = str(r["sample_id"])
            score = r["correct"]
            if pd.isna(score):
                continue
            iid = item_ids.get(sid)
            if iid is None:
                # Item wasn't registered (missing from any full CSV); register lazily
                iid = register_item(
                    benchmark_id=bench_id,
                    raw_item_id=sid,
                    content=None,
                )
                item_ids[sid] = iid
            rows.append({
                "subject_id": subj,
                "item_id": iid,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": cond,
                "response": float(score),
                "correct_answer": item_gold.get(sid),
                "trace": None,
            })

    out_df = pd.DataFrame(rows, columns=cols)
    out_df = ensure_unique_trials(out_df)
    out_df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(out_df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")

    if out_df.empty:
        print("  WARNING: no rows produced")
        return

    print(f"\n  subjects: {out_df['subject_id'].nunique()}")
    print(f"  items:    {out_df['item_id'].nunique()}")
    print(f"  rows:     {len(out_df):,}")


if __name__ == "__main__":
    main()
