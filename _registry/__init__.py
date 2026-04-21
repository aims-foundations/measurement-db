"""Registry of subjects, items, and benchmarks for measurement-db.

Every ``{dataset}/build.py`` should resolve its raw labels via
``resolve_subject`` and ``register_item`` so that every row in
``responses.parquet`` references stable IDs.

See ``DATA_FORMAT.md`` at the repo root for the full schema rationale.

Typical usage::

    from _registry import (
        resolve_subject, register_item, get_benchmark_id, save,
    )

    bench_id = get_benchmark_id("mtbench", name="MT-Bench", ...)
    for raw_label, raw_item, response in iter_raw():
        subj = resolve_subject(raw_label)
        item = register_item(bench_id, raw_item_id=..., content=...)
        rows.append((subj, item, response))
    save()  # flush any new registrations to parquet
"""
from __future__ import annotations

import hashlib
import threading
import unicodedata
from pathlib import Path

import pandas as pd

_REGISTRY_DIR = Path(__file__).resolve().parent
_SUBJECTS_PATH = _REGISTRY_DIR / "subjects.parquet"
_ITEMS_PATH = _REGISTRY_DIR / "items.parquet"
_BENCHMARKS_PATH = _REGISTRY_DIR / "benchmarks.parquet"

_SUBJECTS_COLS = [
    "subject_id", "display_name", "provider", "hub_repo", "revision",
    "params", "release_date", "raw_labels_seen", "notes",
]
_ITEMS_COLS = [
    "item_id", "benchmark_id", "raw_item_id", "content",
    "correct_answer", "test_condition", "content_hash",
]
_BENCHMARKS_COLS = [
    "benchmark_id", "name", "version", "license", "source_url", "description",
]

_lock = threading.Lock()
_subjects: pd.DataFrame | None = None
_items: pd.DataFrame | None = None
_benchmarks: pd.DataFrame | None = None
_dirty = {"subjects": False, "items": False, "benchmarks": False}


# --------------------------------------------------------------------------- #
# Normalization + ID derivation
# --------------------------------------------------------------------------- #

def _normalize_label(s: str) -> str:
    """Normalize a subject raw-label: NFC + lowercase + stripped."""
    return unicodedata.normalize("NFC", s).strip().lower()


def _normalize_content(s: str) -> str:
    """Normalize item content: NFC + stripped (preserves case)."""
    return unicodedata.normalize("NFC", s).strip()


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _subject_id_from_label(raw_label: str) -> str:
    return _hash16(_normalize_label(raw_label))


def _item_id_from_content(benchmark_id: str, content: str) -> str:
    return _hash16(f"{benchmark_id}::{_normalize_content(content)}")


def _content_hash(content: str) -> str:
    return _hash16(_normalize_content(content))


# --------------------------------------------------------------------------- #
# Lazy loading
# --------------------------------------------------------------------------- #

def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _load():
    global _subjects, _items, _benchmarks
    if _subjects is None:
        _subjects = (
            pd.read_parquet(_SUBJECTS_PATH) if _SUBJECTS_PATH.exists()
            else _empty(_SUBJECTS_COLS)
        )
    if _items is None:
        _items = (
            pd.read_parquet(_ITEMS_PATH) if _ITEMS_PATH.exists()
            else _empty(_ITEMS_COLS)
        )
    if _benchmarks is None:
        _benchmarks = (
            pd.read_parquet(_BENCHMARKS_PATH) if _BENCHMARKS_PATH.exists()
            else _empty(_BENCHMARKS_COLS)
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class UnknownSubject(KeyError):
    """Raised when a raw label doesn't match any registered subject."""


def resolve_subject(raw_label: str, *, auto_register: bool = True) -> str:
    """Return the ``subject_id`` for ``raw_label``.

    First searches ``raw_labels_seen`` (case-insensitive, normalized) across
    all registered subjects. If no match and ``auto_register`` is true, creates
    a new entry and returns its id; otherwise raises :class:`UnknownSubject`.
    """
    global _subjects
    with _lock:
        _load()
        assert _subjects is not None
        norm = _normalize_label(raw_label)

        for _, row in _subjects.iterrows():
            seen = row["raw_labels_seen"] or []
            if any(_normalize_label(a) == norm for a in seen):
                return row["subject_id"]

        if not auto_register:
            raise UnknownSubject(raw_label)

        new_id = _subject_id_from_label(raw_label)
        if (_subjects["subject_id"] == new_id).any():
            # ID collision on normalized label — should be rare; append alias.
            idx = _subjects.index[_subjects["subject_id"] == new_id][0]
            existing = list(_subjects.at[idx, "raw_labels_seen"] or [])
            if raw_label not in existing:
                existing.append(raw_label)
                _subjects.at[idx, "raw_labels_seen"] = existing
                _dirty["subjects"] = True
            return new_id

        new_row = {
            "subject_id": new_id,
            "display_name": raw_label,
            "provider": None,
            "hub_repo": None,
            "revision": None,
            "params": None,
            "release_date": None,
            "raw_labels_seen": [raw_label],
            "notes": None,
        }
        _subjects = pd.concat([_subjects, pd.DataFrame([new_row])], ignore_index=True)
        _dirty["subjects"] = True
        return new_id


def register_item(
    benchmark_id: str,
    raw_item_id: str,
    content: str | None,
    *,
    correct_answer: str | None = None,
    test_condition: str | None = None,
) -> str:
    """Register (or look up) an item under a benchmark and return its ``item_id``.

    ``item_id`` is derived from ``benchmark_id`` + normalized ``content``. If
    ``content`` is None (some benchmarks don't expose per-item text),
    ``raw_item_id`` is used as the content for hashing purposes so items remain
    distinguishable — the returned id is still deterministic.
    """
    global _items
    with _lock:
        _load()
        assert _items is not None

        hash_input = content if content is not None else f"raw:{raw_item_id}"
        iid = _item_id_from_content(benchmark_id, hash_input)

        if (_items["item_id"] == iid).any():
            return iid

        new_row = {
            "item_id": iid,
            "benchmark_id": benchmark_id,
            "raw_item_id": str(raw_item_id),
            "content": content,
            "correct_answer": correct_answer,
            "test_condition": test_condition,
            "content_hash": _content_hash(hash_input),
        }
        _items = pd.concat([_items, pd.DataFrame([new_row])], ignore_index=True)
        _dirty["items"] = True
        return iid


def get_benchmark_id(
    benchmark_id: str,
    *,
    name: str | None = None,
    version: str | None = None,
    license: str | None = None,
    source_url: str | None = None,
    description: str | None = None,
) -> str:
    """Register a benchmark once, or return its id if already registered.

    ``benchmark_id`` is the canonical short key (typically the folder name).
    Kwargs populate the row on first registration; they are ignored on
    subsequent calls for the same ``benchmark_id``.
    """
    global _benchmarks
    with _lock:
        _load()
        assert _benchmarks is not None

        if (_benchmarks["benchmark_id"] == benchmark_id).any():
            return benchmark_id

        new_row = {
            "benchmark_id": benchmark_id,
            "name": name or benchmark_id,
            "version": version,
            "license": license,
            "source_url": source_url,
            "description": description,
        }
        _benchmarks = pd.concat(
            [_benchmarks, pd.DataFrame([new_row])], ignore_index=True
        )
        _dirty["benchmarks"] = True
        return benchmark_id


def save() -> None:
    """Persist any in-memory changes to parquet. Safe to call multiple times."""
    with _lock:
        if _subjects is not None and _dirty["subjects"]:
            _subjects.to_parquet(_SUBJECTS_PATH, index=False)
            _dirty["subjects"] = False
        if _items is not None and _dirty["items"]:
            _items.to_parquet(_ITEMS_PATH, index=False)
            _dirty["items"] = False
        if _benchmarks is not None and _dirty["benchmarks"]:
            _benchmarks.to_parquet(_BENCHMARKS_PATH, index=False)
            _dirty["benchmarks"] = False


def reload() -> None:
    """Force reload from disk — mainly for tests."""
    global _subjects, _items, _benchmarks
    with _lock:
        _subjects = _items = _benchmarks = None
        _dirty.update({"subjects": False, "items": False, "benchmarks": False})
