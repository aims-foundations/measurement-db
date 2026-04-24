"""Registry of subjects, items, and benchmarks for measurement-db.

Every ``{dataset}/build.py`` resolves its raw labels via ``resolve_subject``
and ``register_item`` so each row in ``<dataset>.parquet`` references stable
IDs.  See ``DATA_FORMAT.md`` at the repo root for the full schema.

**Concurrency model.** Builds run in parallel processes. To avoid write races
on the shared ``_registry/*.parquet`` files, each build accumulates its
registrations **locally in memory** and flushes them to a per-dataset
``_contrib/`` directory at the end. A separate post-step
(``scripts/merge_registry.py``) unions every dataset's contrib into the
canonical ``_registry/*.parquet``.

IDs are deterministic from normalized input, so two builds that both see
the same subject produce the same ``subject_id``; the merge step dedupes
and set-unions ``raw_labels_seen``.

Typical usage::

    from _registry import (
        resolve_subject, register_item, get_benchmark_id, save,
    )

    bench_id = get_benchmark_id("mtbench", name="MT-Bench", ...)
    for raw_label, raw_item, response in iter_raw():
        subj = resolve_subject(raw_label)
        item = register_item(bench_id, raw_item_id=..., content=...)
        rows.append((subj, item, response))
    save(Path(__file__).resolve().parent / "_contrib")
"""
from __future__ import annotations

import hashlib
import threading
import unicodedata
from pathlib import Path

import pandas as pd

_SUBJECTS_COLS = [
    "subject_id", "display_name", "provider", "hub_repo", "revision",
    "params", "release_date", "raw_labels_seen", "notes",
]
_ITEMS_COLS = [
    "item_id", "benchmark_id", "raw_item_id", "content",
    "correct_answer", "content_hash",
]
_BENCHMARKS_COLS = [
    "benchmark_id", "name", "version", "license", "source_url", "description",
    "modality", "domain",
    "response_type", "response_scale", "categorical",
]

_lock = threading.Lock()
_subjects: pd.DataFrame | None = None
_items: pd.DataFrame | None = None
_benchmarks: pd.DataFrame | None = None


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
# In-process state (per build.py run)
# --------------------------------------------------------------------------- #

def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _ensure_init():
    global _subjects, _items, _benchmarks
    if _subjects is None:
        _subjects = _empty(_SUBJECTS_COLS)
    if _items is None:
        _items = _empty(_ITEMS_COLS)
    if _benchmarks is None:
        _benchmarks = _empty(_BENCHMARKS_COLS)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class UnknownSubject(KeyError):
    """Raised when a raw label doesn't match any registered subject."""


def resolve_subject(raw_label: str, *, auto_register: bool = True) -> str:
    """Return the ``subject_id`` for ``raw_label``.

    Within a single build.py run, the in-memory subjects table deduplicates
    on ``subject_id`` (derived from the normalized label).  If ``raw_label``
    normalizes to an already-seen subject, it is added to that subject's
    ``raw_labels_seen`` list.

    Cross-build deduplication happens later in ``scripts/merge_registry.py``;
    two builds that see the same model will emit contrib rows with the same
    ``subject_id`` and the merge step unions their alias lists.
    """
    global _subjects
    with _lock:
        _ensure_init()
        assert _subjects is not None
        sid = _subject_id_from_label(raw_label)

        mask = _subjects["subject_id"] == sid
        if mask.any():
            idx = _subjects.index[mask][0]
            existing = list(_subjects.at[idx, "raw_labels_seen"] or [])
            if raw_label not in existing:
                existing.append(raw_label)
                _subjects.at[idx, "raw_labels_seen"] = existing
            return sid

        if not auto_register:
            raise UnknownSubject(raw_label)

        new_row = {
            "subject_id": sid,
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
        return sid


def register_item(
    benchmark_id: str,
    raw_item_id: str,
    content: str | None,
    *,
    correct_answer: str | None = None,
) -> str:
    """Register (or look up) an item under a benchmark and return its ``item_id``.

    ``item_id`` is derived from ``benchmark_id`` + normalized ``content``.  If
    ``content`` is None (some benchmarks don't expose per-item text),
    ``raw_item_id`` is used as the content surrogate — the returned id is
    still deterministic.

    Note: ``test_condition`` is NOT an argument here.  Conditions under which
    an item is evaluated (few-shot=3, skill="coherence", variant="score", ...)
    are observation properties, not item properties, and live on the
    ``responses.parquet`` row — not on the item.  If two records describe the
    same prompt under different conditions, they share this ``item_id`` and
    are distinguished by ``test_condition`` on their responses.
    """
    global _items
    with _lock:
        _ensure_init()
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
            "content_hash": _content_hash(hash_input),
        }
        _items = pd.concat([_items, pd.DataFrame([new_row])], ignore_index=True)
        return iid


def get_benchmark_id(
    benchmark_id: str,
    *,
    name: str | None = None,
    version: str | None = None,
    license: str | None = None,
    source_url: str | None = None,
    description: str | None = None,
    modality: list[str] | None = None,
    domain: list[str] | None = None,
    response_type: str | None = None,
    response_scale: str | None = None,
    categorical: bool | None = None,
) -> str:
    """Register a benchmark once, or return its id if already registered.

    ``benchmark_id`` is the canonical short key (typically the folder name).
    Kwargs populate the row on first registration; subsequent calls in the
    same process return the id without updating fields.

    ``modality`` is the list of input modalities required to solve items in
    this benchmark: ``"text"``, ``"image"``, ``"grid"``, ``"gui_screenshot"``,
    ``"audio"``, etc.  Defaults to ``["text"]``.  Use a list so multimodal
    benchmarks (e.g. vision-language QA) can declare multiple.

    ``domain`` is the list of subject areas: ``"software_engineering"``,
    ``"mathematics"``, ``"medicine"``, ``"law"``, ``"finance"``, ``"safety"``,
    ``"preference"``, ``"tool_use"``, ``"gui_agent"``, ``"cybersecurity"``,
    ``"general"``, ``"translation"``, ``"summarization"``, ``"ner"``,
    ``"cultural"``, etc.  Defaults to ``["general"]``.

    ``response_type`` names how the grader emits the response:
    ``"binary"``, ``"likert_5"``, ``"likert_10"``, ``"win_rate"``,
    ``"ordinal"``, ``"fraction"``, ``"continuous_bounded"``,
    ``"continuous_unbounded"``, ``"error_presence"``, ``"mixed"``.  Defaults
    to ``"binary"``.  ``response_scale`` is a free-form string naming the
    value set (``"{0, 1}"``, ``"{1, 2, 3, 4, 5}"``, ``"k/N, N varies per
    item"``, ``"[-18, 18] continuous"``).  ``categorical`` flags whether the
    response set is finitely enumerable — downstream IRT code can filter on
    this for polytomous-vs-continuous model selection.
    """
    global _benchmarks
    with _lock:
        _ensure_init()
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
            "modality": list(modality) if modality else ["text"],
            "domain": list(domain) if domain else ["general"],
            "response_type": response_type or "binary",
            "response_scale": response_scale or "{0, 1}",
            "categorical": bool(categorical) if categorical is not None else True,
        }
        _benchmarks = pd.concat(
            [_benchmarks, pd.DataFrame([new_row])], ignore_index=True
        )
        return benchmark_id


def ensure_unique_trials(df):
    """Bump ``trial`` so ``(subject_id, item_id, trial, test_condition)`` is unique.

    When upstream has multiple observations of the same (subject, item,
    test_condition) cell — repeat runs, multiple annotators of the same
    question — the primary-key invariant is violated if they all carry
    ``trial=1``.  This helper reassigns ``trial`` within each
    (subject_id, item_id, test_condition) group as 1, 2, 3, ... preserving
    order, so every row gets a distinct key.

    Rows that are already unique pass through unchanged.  Call this at the
    end of every ``build.py``'s long-form construction, immediately before
    ``to_parquet``.  See ``DATA_FORMAT.md``'s primary-key invariant.
    """
    import pandas as pd  # local import so _registry has no hard pandas dep at load time

    if df.empty:
        return df
    group_cols = ["subject_id", "item_id", "test_condition"]
    df = df.copy()
    df["trial"] = df.groupby(group_cols, dropna=False).cumcount() + 1
    return df


def save(contrib_dir: Path | str) -> None:
    """Flush this process's registrations to ``contrib_dir``.

    Writes up to three parquet files: ``subjects.parquet``, ``items.parquet``,
    ``benchmarks.parquet``.  A file is only written if its table is non-empty.

    ``contrib_dir`` should be unique per build (conventionally
    ``{dataset}/_contrib/``), so parallel builds can never collide.  The
    post-step ``scripts/merge_registry.py`` reads these files and writes the
    canonical ``_registry/*.parquet``.
    """
    contrib_dir = Path(contrib_dir)
    with _lock:
        contrib_dir.mkdir(parents=True, exist_ok=True)
        if _subjects is not None and len(_subjects) > 0:
            _subjects.to_parquet(contrib_dir / "subjects.parquet", index=False)
        if _items is not None and len(_items) > 0:
            _items.to_parquet(contrib_dir / "items.parquet", index=False)
        if _benchmarks is not None and len(_benchmarks) > 0:
            _benchmarks.to_parquet(contrib_dir / "benchmarks.parquet", index=False)


def reload() -> None:
    """Reset the in-process state — mainly for tests."""
    global _subjects, _items, _benchmarks
    with _lock:
        _subjects = _items = _benchmarks = None
