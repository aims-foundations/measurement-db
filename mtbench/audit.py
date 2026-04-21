"""
MT-Bench audit (processed).

response_matrix.csv: nonempty; Model column nonempty per row; at least one task column; task column names parse as ints listed in item_content item_id; scores numeric from 1 through 10.
item_content.csv: item_id and content columns; item_id unique; Presidio on content needs presidio-analyzer and en_core_web_sm; IGNORED_FLAGS for manual false positives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2]
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from audit.utils import (  # noqa: E402
    parse_task_column_id,
    path_errors_if_missing,
    presidio_scan_content_column,
    processed_dir_from_script,
)

PROC = processed_dir_from_script(__file__)
RESPONSE_MATRIX = PROC / "response_matrix.csv"
ITEM_CONTENT = PROC / "item_content.csv"

PRESIDIO_THRESHOLD = 0.35

# Presidio ignore rules: (item_id, entity_type, literals). A hit is ignored only if the span text,
# after normalization (whitespace + casefold), equals one of the literals.
# Built from item_content.csv prompts + Presidio noise at score ~0.85 (terminals/2.txt ~958–1044).
IGNORED_FLAGS: list[tuple[int, str, tuple[str, ...]]] = [
    (81, "LOCATION", ("Hawaii",)),  # Hawaii travel blog prompt
    (88, "DATE_TIME", ("one morning",)),  # generic "one morning"
    (91, "LOCATION", ("Mars",)),  # Mars (roleplay)
    (91, "PERSON", ("Elon Musk",)),  # Elon Musk (public figure)
    (92, "PERSON", ("Sheldon",)),  # Sheldon (fictional)
    (98, "PERSON", ("Tony Stark", "Stark")),  # Tony Stark (fictional); separate span "Stark"
    (100, "DATE_TIME", ("100-years-old",)),  # "100-years-old" tree roleplay
    (103, "PERSON", ("Thomas",)),  # Thomas — first name only
    (104, "PERSON", ("David",)),  # David — first name only
    (105, "PERSON", ("Alice", "Bert", "Cheryl", "David", "Enid")),  # first names
    (109, "DATE_TIME", ("One morning",)),  # "One morning" shadow puzzle
    (109, "PERSON", ("Suresh",)),  # Suresh — first name in reasoning puzzle
    (
        112,
        "DATE_TIME",
        ("the first year", "the second year", "the two years"),
    ),  # first/second year investment wording
    (119, "PERSON", ("Benjamin",)),  # Benjamin — math word problem
    (122, "NRP", ("C++",)),  # C++ misclassified as nationality
    (124, "PERSON", ("str2",)),  # str2 — code identifier
    (131, "DATE_TIME", ("Nov. 18, 2019", "2022")),  # movie review dates in extraction task
    (132, "DATE_TIME", ("19th-century",)),  # "19th-century" in categorization task
    (132, "LOCATION", ("Russia",)),  # Russia — literature / history context
    (132, "PERSON", ("Leo Tolstoy",)),  # Leo Tolstoy (public figure / literature)
    (133, "DATE_TIME", ("the year", "year", "1997")),  # Harry Potter extraction template
    (133, "PERSON", ("J.K. Rowling", "Harry Potter", "Harry")),  # author and fictional character
    (134, "DATE_TIME", ("2021", "the same year")),  # synthetic company table
    (134, "LOCATION", ("wi",)),  # "wi" — fragment false positive
    (
        134,
        "PERSON",
        ("Amy Williams", "Mark Thompson", "Sarah Johnson", "James Smith"),
    ),  # fictional CEOs
    (135, "LOCATION", ("Copenhagen", "Denmark")),  # capitals JSON task
    (135, "PERSON", ("Eldoria",)),  # fictional city name in prompt
    (
        136,
        "LOCATION",
        ("Amazon River", "Brazil", "Colombia", "Peru"),
    ),  # counting task
    (136, "PERSON", ("li",)),  # "li" — fragment false positive
    (137, "DATE_TIME", ("Yesterday",)),  # news article
    (137, "LOCATION", ("Berlin",)),  # Berlin
    (137, "PERSON", ("Adamson Emerson", "Dieter Zetsche")),  # fictional or public figures
    (139, "DATE_TIME", ("3/4)x^3 - e^(2x",)),  # math misread as date
    (139, "PERSON", ("m(c^2",)),  # math misread as person
    (
        140,
        "DATE_TIME",
        (
            "each month",
            "the year 2022",
            "2022-01-01,150.02,155.28,148.50,153.80,15678900",
            "2022-01-02,154.32,157.25,153.48,156.25,19874500",
            "2022-02-01,160.50,163.28,159.50,161.80,14326700",
            "2022-02-02,161.80,164.25,161.30,163.90,17689200",
            "2022-03",
            "2022-03-02,167.00,169.85,165",
            "2022-03-01",
        ),
    ),  # stock OHLC rows / "each month" / 2022
    (140, "PHONE_NUMBER", ("164.25", "168.35", "169.85")),  # stock decimals misread as phone
    (142, "LOCATION", ("Earth",)),  # Earth — satellite orbit STEM
    (143, "LOCATION", ("Earth",)),  # Earth — photosynthesis STEM
    (147, "LOCATION", ("the Vegona River",)),  # fictional bridge prompt
    (147, "PERSON", ("Vega",)),  # city name misread as person
    (150, "LOCATION", ("Alps", "Rhine River", "Western Europe")),  # geography STEM
    (153, "LOCATION", ("US", "China")),  # antitrust comparison essay
    (154, "DATE_TIME", ("45 minutes", "3 days")),  # lesson plan durations
    (154, "LOCATION", ("China", "Britain")),  # history / geography in prompt
    (159, "LOCATION", ("Japan",)),  # Japan — business etiquette prompt
]


def main() -> int:
    path_errs = path_errors_if_missing([RESPONSE_MATRIX, ITEM_CONTENT])
    if path_errs:
        for e in path_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    df = pd.read_csv(RESPONSE_MATRIX)
    items = pd.read_csv(ITEM_CONTENT)
    errors: list[str] = []

    if len(df) == 0:
        errors.append("response_matrix.csv is empty (no rows)")

    for col in ("item_id", "content"):
        if col not in items.columns:
            errors.append(f"item_content.csv must have a {col!r} column")

    if "item_id" in items.columns:
        dup_items = items["item_id"].duplicated(keep=False)
        if dup_items.any():
            errors.append(
                f"item_content.csv: duplicate item_id values ({int(dup_items.sum())} rows)"
            )

    task_cols: list[str] = []
    if "Model" not in df.columns:
        errors.append("response_matrix.csv must have a 'Model' column")
    else:
        bad_model = df["Model"].isna() | (df["Model"].astype(str).str.strip() == "")
        if bad_model.any():
            errors.append(f"{int(bad_model.sum())} empty model name(s)")
        task_cols = [c for c in df.columns if c != "Model"]

    if "Model" in df.columns and not task_cols:
        errors.append(
            "response_matrix.csv must have at least one task id column (besides Model)"
        )

    known: set[int] = set()
    if "item_id" in items.columns:
        known = set(items["item_id"].astype(int))
    bad_names: list[str] = []
    missing: list[int] = []
    score_errs: list[str] = []

    for c in task_cols:
        try:
            tid = parse_task_column_id(c)
        except (TypeError, ValueError):
            bad_names.append(str(c))
            continue
        if known and tid not in known:
            missing.append(tid)
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            score_errs.append(f"{c!r}: {int(s.isna().sum())} NaN/non-numeric")
        elif ((s < 1) | (s > 10)).any():
            score_errs.append(f"{c!r}: {int(((s < 1) | (s > 10)).sum())} value(s) outside [1, 10]")

    if bad_names:
        errors.append(f"Non-integer task id column(s): {bad_names}")
    if missing:
        errors.append(f"Task id(s) not in item_content.csv: {sorted(set(missing))}")
    errors.extend(score_errs)

    if "item_id" in items.columns and "content" in items.columns:
        pii_errs, n_ignored_pii = presidio_scan_content_column(
            items,
            score_threshold=PRESIDIO_THRESHOLD,
            ignored_flags=IGNORED_FLAGS,
        )
        errors.extend(pii_errs)
    else:
        n_ignored_pii = 0

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    ok_pii = f"Presidio: no actionable PII ({PRESIDIO_THRESHOLD=})"
    if n_ignored_pii:
        ok_pii += f", {n_ignored_pii} span(s) ignored per IGNORED_FLAGS"
    ok_pii += "."
    print(
        f"OK: {len(df)} models × {len(task_cols)} tasks; scores in [1,10]; {ok_pii}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
