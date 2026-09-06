"""Stable hash derivation for measurement-table identifiers.

All hash-derived measurement fields use SHA-256 truncated to 16 hexadecimal
characters, but each one intentionally defines a different identity payload.
Keep those payload builders separate: sharing the digest primitive must not
turn subject, item, content, and response identity into the same policy.

The exact serialization in this module is part of the published data contract.
Changing it is an identifier migration, not an internal refactor.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from numbers import Real
from pathlib import PurePosixPath
import re
from collections.abc import Mapping, Sequence
import unicodedata

import pandas as pd


__all__ = [
    "asset_id_from_bytes",
    "canonical_asset_manifest",
    "content_hash",
    "item_id_from_content",
    "response_id_from_row",
    "response_row_hashes",
    "sha256_16",
    "subject_id_from_row",
]


_ASSET_MANIFEST_FIELDS = (
    "asset_id",
    "path",
    "media_type",
    "role",
    "ordinal",
)


# The subjects.parquet metadata columns that enter the subject_id hash, in
# schema order (2026-08-10 decision: the id fingerprints the row at its
# observed values, not just label+features).
_SUBJECT_HASH_COLUMNS = (
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)


def sha256_16(payload: str | bytes) -> str:
    """Return the first 16 lowercase hex characters of a SHA-256 digest."""

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalize_label(value: str) -> str:
    """Normalize a subject raw label: NFC + lowercase + stripped."""

    return unicodedata.normalize("NFC", value).strip().lower()


def _normalize_content(value: str) -> str:
    """Normalize item content: NFC + stripped, preserving case."""

    return unicodedata.normalize("NFC", value).strip()


def subject_id_from_row(raw_label: str, row: Mapping[str, object]) -> str:
    """Derive a subject ID from canonical identity and observed metadata.

    Mapped aliases share ``normalized_name`` and therefore share an ID when
    their remaining metadata agrees.  An unmapped subject falls back to its
    normalized raw label so unrelated unknown subjects do not collapse.
    """

    base = row.get("normalized_name")
    if base is None:
        base = _normalize_label(raw_label)
    parts = [json.dumps(base)]
    parts += [
        f"{column}={json.dumps(row.get(column))}"
        for column in _SUBJECT_HASH_COLUMNS
    ]
    return sha256_16("||".join(parts))


def asset_id_from_bytes(payload: bytes) -> str:
    """Return the full SHA-256 content address for an attachment's bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("asset payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_asset_manifest(
    entries: Sequence[Mapping[str, object]] | None,
) -> str | None:
    """Validate and canonically serialize an item's ordered asset links.

    ``source_path`` is deliberately absent: only the stable logical path and
    immutable byte identity belong to the published item contract.
    """

    if entries is None:
        return None
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("asset manifest must be a sequence of mappings")
    if not entries:
        return None

    normalized: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for expected_ordinal, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise TypeError("every asset manifest entry must be a mapping")
        if set(entry) != set(_ASSET_MANIFEST_FIELDS):
            raise ValueError(
                "asset manifest entries must contain exactly "
                f"{list(_ASSET_MANIFEST_FIELDS)}"
            )

        asset_id = entry["asset_id"]
        path = entry["path"]
        media_type = entry["media_type"]
        role = entry["role"]
        ordinal = entry["ordinal"]
        if (
            not isinstance(asset_id, str)
            or len(asset_id) != 64
            or any(character not in "0123456789abcdef" for character in asset_id)
        ):
            raise ValueError(
                "asset manifest asset_id must be 64 lowercase hexadecimal characters"
            )
        if not isinstance(path, str) or not path:
            raise ValueError("asset manifest path must be a non-empty string")
        parsed_path = PurePosixPath(path)
        if (
            parsed_path.is_absolute()
            or path != parsed_path.as_posix()
            or any(part in ("", ".", "..") for part in parsed_path.parts)
            or re.match(r"^[A-Za-z]:", path)
            or "\\" in path
            or any(ord(character) < 32 for character in path)
        ):
            raise ValueError(
                "asset manifest path must be a normalized relative POSIX path"
            )
        if path in seen_paths:
            raise ValueError(f"asset manifest path {path!r} is repeated")
        if not isinstance(media_type, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*",
            media_type,
        ):
            raise ValueError("asset manifest media_type must be a lowercase MIME type")
        if not isinstance(role, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", role
        ):
            raise ValueError("asset manifest role must match [a-z][a-z0-9_]*")
        if type(ordinal) is not int or ordinal != expected_ordinal:
            raise ValueError(
                "asset manifest ordinals must be consecutive 1-based integers"
            )

        seen_paths.add(path)
        normalized.append(
            {
                "asset_id": asset_id,
                "path": path,
                "media_type": media_type,
                "role": role,
                "ordinal": expected_ordinal,
            }
        )

    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def item_id_from_content(
    benchmark_id: str,
    content: str,
    features_str: str | None = None,
    *,
    asset_manifest: str | None = None,
    verifier: str | None = None,
) -> str:
    """Derive an item ID from its benchmark, content, features, and assets.

    The historical text-only serialization is preserved exactly. Asset-bearing
    items use a domain-separated JSON payload so arbitrary text cannot absorb
    the manifest or feature boundary.
    """

    normalized_content = _normalize_content(content)
    if not asset_manifest:
        key = f"{benchmark_id}::{normalized_content}"
        if features_str:
            key += f"::{_normalize_content(features_str)}"
        return sha256_16(key)

    payload = json.dumps(
        {
            "asset_manifest": asset_manifest,
            "benchmark_id": benchmark_id,
            "content": normalized_content,
            "features": (
                _normalize_content(features_str) if features_str else None
            ),
            "verifier": verifier,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_16(f"measurement-db:item-with-assets:v1\0{payload}")


def content_hash(content: str) -> str:
    """Hash normalized content without benchmark or feature identity."""

    return sha256_16(_normalize_content(content))


def _stable_response_cell(value: object) -> bytes:
    """Serialize a response cell independently of pandas dtype inference."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if (
        isinstance(missing, bool) or getattr(missing, "ndim", None) == 0
    ) and bool(missing):
        return b"N"  # distinct from every non-null value via the tag
    if isinstance(value, Real) and not isinstance(value, bool):
        # A peer row can make pandas upcast 1 to 1.0. Both represent the same
        # accepted float-schema value, so normalize their decimal spelling.
        try:
            number = str(Decimal(str(value)).normalize())
        except (InvalidOperation, ValueError):
            number = str(value)
        return b"R" + number.encode("utf-8")
    return b"V" + str(value).encode("utf-8")


def _frame(payload: bytes) -> bytes:
    """Length-prefix a token so adjacent values cannot be ambiguous."""

    return len(payload).to_bytes(8, "big") + payload


def response_id_from_row(row: Mapping[str, object]) -> str:
    """Derive one response ID from an ordered finalized-row mapping.

    Mapping iteration order is part of the serialized payload because response
    extension columns follow the canonical schema columns. The caller must
    omit ``response_id`` itself and pass ``trace=None``: raw trace text lives
    in ``traces.parquet``, not in the hashed response-table representation.
    """

    encoded = bytearray()
    for column, value in row.items():
        encoded.extend(_frame(str(column).encode("utf-8")))
        encoded.extend(_frame(_stable_response_cell(value)))
    return sha256_16(bytes(encoded))


def response_row_hashes(responses: pd.DataFrame) -> list[str]:
    """Derive each response ID from all columns of its finalized row.

    Column names and values are length-prefixed, nulls are explicitly tagged,
    and accepted numeric values have a stable representation across pandas
    dtype inference.  The caller must pass the final response columns before
    inserting ``response_id`` itself.
    """

    columns = list(responses.columns)
    return [
        response_id_from_row(dict(zip(columns, values)))
        for values in responses.itertuples(index=False, name=None)
    ]
