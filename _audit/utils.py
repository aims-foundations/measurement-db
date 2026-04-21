"""Shared utilities for ``data/*/scripts/*audit*.py`` scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def processed_dir_from_script(script_file: str) -> Path:
    """Return ``.../<dataset>/processed`` for a script under ``.../<dataset>/scripts/``."""
    return Path(script_file).resolve().parent.parent / "processed"


def raw_dir_from_script(script_file: str) -> Path:
    """Return ``.../<dataset>/raw`` for a script under ``.../<dataset>/scripts/``."""
    return Path(script_file).resolve().parent.parent / "raw"


def bad_pct_suffix(n_bad: int, n_total: int) -> str:
    """Suffix for audit errors: malformed row count and percentage of total."""
    if n_total <= 0:
        return " — malformed: 0 rows (empty file)"
    pct = 100.0 * n_bad / n_total
    return f" — malformed: {n_bad:,} / {n_total:,} rows ({pct:.2f}%)"


def path_errors_if_missing(paths: Iterable[Path]) -> list[str]:
    """Return one error string per path that is not a regular file."""
    return [f"Missing {p}" for p in paths if not p.is_file()]


def parse_task_column_id(col) -> int:
    """Parse a task column label (e.g. ``81`` or ``\"81\"``) as ``int`` (no strip; must match exactly)."""
    if isinstance(col, int) and not isinstance(col, bool):
        return col
    return int(str(col))


def norm_pii_literal(s: str) -> str:
    """Normalize Presidio span text for literal comparison (whitespace + casefold)."""
    return " ".join(s.split()).casefold()


def compile_ignored_flag_rules(
    flags: list[tuple[int, str, tuple[str, ...]]],
) -> list[tuple[int, str, frozenset[str]]]:
    """Turn ``IGNORED_FLAGS`` into lookup rows ``(item_id, ENTITY_TYPE, frozenset(norm(literals)))``."""
    return [
        (iid, et.upper(), frozenset(norm_pii_literal(x) for x in lits))
        for iid, et, lits in flags
    ]


def span_text(text: str, start: int, end: int) -> str:
    """Slice Presidio character offsets into a one-line display string."""
    s = text[start:end].replace("\n", " ").strip()
    return s if s else repr(text[start:end])


def pii_hit_ignored(
    item_id: int,
    entity_type: str,
    snippet: str,
    rules: list[tuple[int, str, frozenset[str]]],
) -> bool:
    """True if this hit matches an ignored (item_id, type, literal) rule."""
    sn = norm_pii_literal(snippet)
    u = entity_type.upper()
    return any(rid == item_id and ret == u and sn in lit for rid, ret, lit in rules)


def presidio_scan_content_column(
    df: pd.DataFrame,
    *,
    score_threshold: float,
    ignored_flags: list[tuple[int, str, tuple[str, ...]]],
    content_column: str = "content",
    item_id_column: str = "item_id",
) -> tuple[list[str], int]:
    """
    Run Presidio on each row's text column; return (error lines, ignored hit count).

    Needs ``presidio-analyzer`` and a spaCy model (e.g. ``en_core_web_sm``).
    """
    from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]

    if content_column not in df.columns:
        return ([f"DataFrame needs {content_column!r} for PII audit"], 0)
    if item_id_column not in df.columns:
        return ([f"DataFrame needs {item_id_column!r} for PII audit"], 0)

    rules = compile_ignored_flag_rules(ignored_flags)
    engine = AnalyzerEngine()
    actionable: list[str] = []
    n_ignored = 0

    for row in df.itertuples(index=False):
        iid = int(getattr(row, item_id_column))
        raw = getattr(row, content_column)
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        text = str(raw)
        hits = engine.analyze(
            text=text,
            language="en",
            score_threshold=score_threshold,
        )
        for h in hits:
            et = str(h.entity_type)
            snippet = span_text(text, h.start, h.end)
            if pii_hit_ignored(iid, et, snippet, rules):
                n_ignored += 1
                continue
            actionable.append(f"item_id {iid} {et} — {snippet!r} (score={h.score:.2f})")

    return (actionable, n_ignored)


def assert_frames_equal_label(
    a: pd.DataFrame,
    b: pd.DataFrame,
    a_label: str,
    b_label: str,
) -> list[str]:
    """Return an error string if two frames differ; else []."""
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_like=False)
    except AssertionError as exc:
        return [f"{a_label} vs {b_label}: {exc}"]
    return []
