#!/usr/bin/env python3
"""Build Cybench long-form responses from paper Tables 10-12.

Data sources:
  - Paper: "Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks
    of Language Models" (ICLR 2025). arXiv:2408.08926
    Tables 10-12: Per-task results for 8 models (single attempt, structured bash agent).
    Extracted programmatically using pdfplumber.
  - GitHub repo: https://github.com/andyzorigin/cybench
    task_list.txt and challenge metadata.

Outputs:
  - responses.parquet   # long-form: (subject_id, item_id, trial, response, trace)
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions

Item-level: each of the 40 CTF challenges is one item (shared across modes).
Observation-level: the run mode lives in test_condition:
  - "mode=unguided"
  - "mode=subtask_guided"
  - "mode=subtask_fractional"

Requirements:
  pip install pdfplumber
"""

INFO = {
    'description': 'Cybench CTF agent benchmark: 8 models x 40 tasks, unguided + subtask-guided modes.',
    'testing_condition': 'mode in {unguided, subtask_guided, subtask_fractional} encoded as test_condition. unguided and subtask_guided are already per-run binary (one attempt per model-task in the paper). subtask_fractional is an aggregate fraction of subtasks solved per task — the paper publishes only the summed score (e.g., "2/5"), not per-subtask binary, so we cannot split it into per-subtask rows.',
    'paper_url': 'https://arxiv.org/abs/2408.08926',
    'data_source_url': 'https://github.com/andyzorigin/cybench',
    'subject_type': 'agent',
    'item_type': 'ctf_challenge',
    'license': 'Apache-2.0',
    'citation': """@misc{zhang2025cybenchframeworkevaluatingcybersecurity,
      title={Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models},
      author={Andy K. Zhang and Neil Perry and Riya Dulepet and Joey Ji and Celeste Menders and Justin W. Lin and Eliot Jones and Gashon Hussein and Samantha Liu and Donovan Jasper and Pura Peetathawatchai and Ari Glenn and Vikram Sivashankar and Daniel Zamoshchin and Leo Glikbarg and Derek Askaryar and Mike Yang and Teddy Zhang and Rishi Alluri and Nathan Tran and Rinnara Sangpisit and Polycarpos Yiorkadjis and Kenny Osele and Gautham Raghupathi and Dan Boneh and Daniel E. Ho and Percy Liang},
      year={2025},
      eprint={2408.08926},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2408.08926},
}""",
    'tags': ['agent'],
    'modality': ['text'],
    'domain': ['cybersecurity'],
    'response_type': 'fraction',
    'response_scale': 'per-task k/N for subtask_fractional mode; binary for unguided/subtask_guided',
    'categorical': False,
}


import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber is required. Install with: pip install pdfplumber")
    sys.exit(1)


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

ARXIV_ID = "2408.08926"
PDF_URL = f"https://arxiv.org/pdf/{ARXIV_ID}"
PDF_PATH = RAW_DIR / "cybench_paper.pdf"


def download():
    """Download raw data from external sources."""
    clone_dir = RAW_DIR / "cybench_repo"
    if not clone_dir.exists():
        print("Cloning cybench repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/andyzorigin/cybench.git", str(clone_dir)],
            check=True,
        )
    else:
        print("cybench repo already cloned, pulling latest...")
        subprocess.run(
            ["git", "-C", str(clone_dir), "pull", "--ff-only"],
            check=False,
        )

    download_pdf()


# ===========================================================================
# Task list in exact order from the paper (Tables 10-12)
# ===========================================================================
TASKS = [
    ("Loot Stash",                    "00:02", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Very Easy] LootStash"),
    ("Urgent",                        "00:04", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] Urgent"),
    ("Flag Command",                  "00:06", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Very Easy] Flag Command"),
    ("Packed Away",                   "00:06", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Very Easy] PackedAway"),
    ("It Has Begun",                  "00:07", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] It Has Begun"),
    ("Dynastic",                      "00:09", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic"),
    ("Primary Knowledge",             "00:11", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Primary Knowledge"),
    ("Crushing",                      "00:20", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Easy] Crushing"),
    ("Permuted",                      "00:42", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Hard] Permuted"),
    ("Labyrinth Linguist",            "00:43", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Easy] Labyrinth Linguist"),
    ("Partial Tenacity",              "01:05", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Medium] Partial Tenacity"),
    ("Delulu",                        "01:15", "Pwn",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/pwn/[Very Easy] Delulu"),
    ("Data Siege",                    "01:18", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Medium] Data Siege"),
    ("Were Pickle Phreaks Revenge",   "02:00", "Misc",       "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/misc/[Medium] Were Pickle Phreaks Revenge"),
    ("Unbreakable",                   "02:03", "Misc",       "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/misc/[Easy] Unbreakable"),
    ("Lock Talk",                     "02:12", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Medium] LockTalk"),
    ("Flecks of Gold",                "06:08", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Hard] FlecksOfGold"),
    ("Skilift",                       "00:05", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/intro/skilift"),
    ("Glacier Exchange",              "00:09", "Web",        "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/web/GlacierExchange"),
    ("Missing Bits",                  "00:09", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/MissingBits"),
    ("SLCG",                          "00:29", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/SLCG"),
    ("RPGO",                          "00:45", "Reverse",    "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/rev/rpgo"),
    ("SOP",                           "00:51", "Reverse",    "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/rev/sop"),
    ("Avatar",                        "01:24", "Misc",       "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/misc/avatar"),
    ("Walking to the Seaside",        "02:13", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/WalkingToTheSeaSide"),
    ("Shuffled AES",                  "02:39", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/shuffled-aes"),
    ("Eval Me",                       "00:11", "Forensics",  "S23", "benchmark/project-sekai-ctf/sekaictf-2023/forensics/eval-me"),
    ("Noisy CRC",                     "00:35", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/noisy-crc"),
    ("Network Tools",                 "00:42", "Pwn",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/pwn/network-tools"),
    ("Noisier CRC",                   "01:25", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/noisier-crc"),
    ("Chunky",                        "01:45", "Web",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/web/chunky"),
    ("Just Another Pickle Jail",      "04:04", "Misc",       "S23", "benchmark/project-sekai-ctf/sekaictf-2023/misc/just-another-pickle-jail"),
    ("Frog WAF",                      "05:30", "Web",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/web/frog-waf"),
    ("Randsubware",                   "05:56", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/randsubware"),
    ("Failproof",                     "01:00", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/failproof"),
    ("Ezmaze",                        "03:24", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/ezmaze"),
    ("Diffecient",                    "07:34", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/diffecient"),
    ("Robust CBC",                    "24:54", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/robust-cbc"),
    ("Back to the Past",              "00:32", "Web",        "HKC", "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/22-back-to-the-past"),
    ("MOTP",                          "00:52", "Web",        "HKC", "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/05-motp"),
]

TASK_NAMES = [t[0] for t in TASKS]

# 8 models from the paper, structured bash, single attempt
MODELS_PAPER = [
    "GPT-4o",
    "OpenAI o1-preview",
    "Claude 3 Opus",
    "Claude 3.5 Sonnet",
    "Mixtral 8x22b Instruct",
    "Gemini 1.5 Pro",
    "Llama 3 70b Chat",
    "Llama 3.1 405B Instruct",
]

COMP_FULL = {
    "HTB": "HackTheBox Cyber Apocalypse 2024",
    "GLA": "GlacierCTF 2023",
    "S23": "SekaiCTF 2023",
    "S22": "SekaiCTF 2022",
    "HKC": "HKCert CTF 2022",
}


# ===========================================================================
# PDF downloading and table extraction
# ===========================================================================

def download_pdf():
    """Download paper PDF from arXiv if not already present."""
    if PDF_PATH.exists():
        print(f"  PDF already exists: {PDF_PATH}")
        return
    print(f"  Downloading {PDF_URL} ...")
    req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    PDF_PATH.write_bytes(data)
    print(f"  Saved {len(data)} bytes to {PDF_PATH}")


def _normalize_task_name(name):
    """Normalize task name by removing spaces, dashes, apostrophes for matching."""
    return re.sub(r'[\s\-\'"]', '', name).lower()


def _build_name_map():
    return {_normalize_task_name(t): t for t in TASK_NAMES}


def _find_table_pages(pdf):
    """Find which PDF pages contain Tables 10, 11, 12 as headings."""
    table_pages = {}
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        for line in text.split('\n'):
            line_nospace = line.replace(' ', '')
            for tnum in (10, 11, 12):
                if tnum not in table_pages and re.match(
                    rf'^Table\s*{tnum}\s*[:.]', line_nospace
                ):
                    table_pages[tnum] = i
    return table_pages


def extract_table_10_11(pdf, page_idx):
    """Extract binary (✓/X) table from a page (Table 10 or 11)."""
    page = pdf.pages[page_idx]
    text = page.extract_text()
    results = {}
    for line in text.split('\n'):
        m = re.match(
            r'(.+?)\s+(\d{2}:\d{2})\s+([WRCFPM])\s+(HTB|GLA|S23|S22|HKC)\s+(.*)',
            line
        )
        if m and 'SuccessCount' not in line:
            task_raw = m.group(1)
            vals = [
                1 if ch == '✓' else 0
                for ch in m.group(5).strip().split()
                if ch in ('✓', 'X')
            ]
            if len(vals) == 8:
                results[_normalize_task_name(task_raw)] = vals
    return results


def extract_table_12(pdf, page_idx):
    """Extract fractional subtask scores from Table 12."""
    page = pdf.pages[page_idx]
    text = page.extract_text()
    results = {}
    for line in text.split('\n'):
        m = re.match(
            r'(.+?)\s+(\d{2}:\d{2}:\d{2})\s+([WRCFPM])\s+(HTB|GLA|S23|S22|HKC)\s+(.*)',
            line
        )
        if m and 'SumofScores' not in line:
            task_raw = m.group(1)
            vals = [
                t for t in m.group(5).strip().split()
                if t == 'X' or '/' in t
            ]
            if len(vals) == 8:
                results[_normalize_task_name(task_raw)] = vals
    return results


def extract_all_tables(pdf_path):
    """Extract Tables 10, 11, 12 from the Cybench paper PDF."""
    pdf = pdfplumber.open(pdf_path)
    table_pages = _find_table_pages(pdf)
    name_map = _build_name_map()

    if 10 not in table_pages or 11 not in table_pages or 12 not in table_pages:
        pdf.close()
        raise RuntimeError(
            f"Could not find all tables in PDF. Found pages: {table_pages}. "
            "The PDF format may have changed."
        )

    print(f"  Table 10 (unguided) on page {table_pages[10]}")
    print(f"  Table 11 (subtask-guided) on page {table_pages[11]}")
    print(f"  Table 12 (subtask scores) on page {table_pages[12]}")

    raw_t10 = extract_table_10_11(pdf, table_pages[10])
    raw_t11 = extract_table_10_11(pdf, table_pages[11])
    raw_t12 = extract_table_12(pdf, table_pages[12])
    pdf.close()

    def map_names(raw_dict, table_label):
        mapped = {}
        for norm_name, vals in raw_dict.items():
            if norm_name in name_map:
                mapped[name_map[norm_name]] = vals
            else:
                print(f"  WARNING: {table_label}: unmapped task '{norm_name}'")
        return mapped

    unguided = map_names(raw_t10, "Table 10")
    guided = map_names(raw_t11, "Table 11")
    subtask = map_names(raw_t12, "Table 12")

    for label, data in [("Table 10", unguided), ("Table 11", guided), ("Table 12", subtask)]:
        missing = [t for t in TASK_NAMES if t not in data]
        if missing:
            raise RuntimeError(
                f"{label}: Missing {len(missing)} tasks: {missing}. "
                "PDF extraction may have failed."
            )
        if len(data) != 40:
            raise RuntimeError(f"{label}: Expected 40 tasks, got {len(data)}")

    return unguided, guided, subtask


def fraction_to_float(frac_str):
    """Convert '2/5' to 0.4, 'X' to 0.0."""
    if frac_str == "X":
        return 0.0
    parts = frac_str.split("/")
    return int(parts[0]) / int(parts[1])


# ===========================================================================
# Item content
# ===========================================================================

def _task_content_map():
    """Build task_name -> content string, using repo metadata.json where available."""
    repo_dir = RAW_DIR / "cybench_repo"
    task_meta = {}
    if repo_dir.exists():
        difficulty_re = re.compile(r"^\[.*?\]\s*(.+)$")
        numeric_prefix_re = re.compile(r"^\d+[-_]")

        def norm(s: str) -> str:
            s = difficulty_re.sub(r"\1", s)
            s = numeric_prefix_re.sub("", s)
            return re.sub(r"[^a-z0-9]", "", s.lower())

        for meta_file in repo_dir.rglob("metadata.json"):
            task_dir = meta_file.parent.parent
            try:
                with open(meta_file) as f:
                    task_meta[norm(task_dir.name)] = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    else:
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

    content_map = {}
    for task_name, _fst, cat, comp, _path in TASKS:
        meta = task_meta.get(norm(task_name))
        if meta:
            cats = ", ".join(meta.get("categories", [])) or cat
            diff = meta.get("difficulty", "")
            prompt = meta.get("easy_prompt") or meta.get("hard_prompt") or ""
            content = f"[{cats}] difficulty={diff}"
            if prompt:
                content += f" | {prompt[:1500]}"
        else:
            content = f"[{cat}] {COMP_FULL[comp]}: {task_name}"
        content_map[task_name] = content
    return content_map


# ===========================================================================
# Long-form builder
# ===========================================================================

def build_long_form(unguided, guided, subtask):
    """Build the long-form responses.parquet."""
    bench_id = get_benchmark_id(
        "cybench",
        name="Cybench",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
        modality=INFO.get("modality"),
        domain=INFO.get("domain"),
        response_type=INFO.get("response_type"),
        response_scale=INFO.get("response_scale"),
        categorical=INFO.get("categorical"),
    )

    content_map = _task_content_map()

    rows = []
    for task_name in TASK_NAMES:
        content = content_map.get(task_name)
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=task_name,
            content=content,
        )
        for mode, data in [
            ("unguided", unguided),
            ("subtask_guided", guided),
        ]:
            vals = data[task_name]
            for m_idx, model in enumerate(MODELS_PAPER):
                subj = resolve_subject(model)
                rows.append({
                    "subject_id": subj,
                    "item_id": item,
                    "benchmark_id": bench_id,
                    "trial": 1,
                    "test_condition": f"mode={mode}",
                    "response": float(vals[m_idx]),
                    "correct_answer": None,
                    "trace": None,
                })
        # Subtask fractional scores
        vals = subtask[task_name]
        for m_idx, model in enumerate(MODELS_PAPER):
            subj = resolve_subject(model)
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": "mode=subtask_fractional",
                "response": float(fraction_to_float(vals[m_idx])),
                "correct_answer": None,
                "trace": None,
            })

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)
    df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df):
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  conditions: {sorted(df['test_condition'].dropna().unique())}")
    for cond, g in df.groupby("test_condition"):
        print(f"    {cond}: {len(g):,} rows, mean={g['response'].mean():.3f}")


def main():
    print(f"[cybench] building from {_BENCHMARK_DIR}")
    download()
    print("\nExtracting Tables 10-12 from PDF")
    unguided, guided, subtask = extract_all_tables(PDF_PATH)
    print(f"  Extracted: {len(unguided)} unguided, {len(guided)} guided, "
          f"{len(subtask)} subtask entries")

    df = build_long_form(unguided, guided, subtask)
    print_stats(df)


if __name__ == "__main__":
    main()
