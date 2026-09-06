"""Load and validate raw benchmark source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


class SourceDataError(RuntimeError):
    """A downloaded source is absent, corrupt, or structurally unreadable."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceDataError(f"cannot read {path}: {exc}") from None
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    minimum_size: int | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Validate one source file and return its path.

    ``minimum_size`` is inclusive. Use ``expected_size`` when a provider's
    pinned byte count is known; supplying both is an authoring error.
    """
    if minimum_size is not None and expected_size is not None:
        raise ValueError("minimum_size and expected_size are mutually exclusive")
    for name, value in (
        ("minimum_size", minimum_size),
        ("expected_size", expected_size),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")

    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise SourceDataError(f"cannot stat {path}: {exc}") from None

    if minimum_size is not None and actual_size < minimum_size:
        raise SourceDataError(
            f"{path}: expected at least {minimum_size} bytes, found {actual_size}"
        )
    if expected_size is not None and actual_size != expected_size:
        raise SourceDataError(
            f"{path}: expected {expected_size} bytes, found {actual_size}"
        )
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256.lower():
            raise SourceDataError(
                f"{path}: expected SHA-256 {expected_sha256.lower()}, "
                f"found {actual_sha256}"
            )
    return path


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL lines, requiring one JSON object per physical line."""
    records: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceDataError(
                        f"{path}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
                    ) from None
                if not isinstance(value, dict):
                    raise SourceDataError(
                        f"{path}:{line_number}: expected a JSON object, "
                        f"found {type(value).__name__}"
                    )
                records.append(value)
    except SourceDataError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        location = f"{path}:{line_number}" if line_number else str(path)
        raise SourceDataError(f"cannot read {location}: {exc}") from None
    return records
