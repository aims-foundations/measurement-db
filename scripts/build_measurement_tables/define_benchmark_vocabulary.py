"""Author-facing metadata contract for benchmark ``INFO`` dictionaries.

``build_base.BenchmarkBuild`` validates this contract before downloading data,
so malformed metadata fails with one actionable error instead of a late
``KeyError`` or parquet-schema failure.  The controlled values here are the
ones whose meanings are repository-wide.  ``domain`` and ``modality`` remain
open lists: their canonical values are documented here, but a genuinely new
kind may be proposed without first pretending it is an existing category.

The keys under ``CURATION_ONLY_KEYS`` document the source ``build.py`` for
reviewers.  They are deliberately not columns in ``benchmarks.parquet``;
persisted columns are defined by ``parquet_schemas.yaml`` and populated by
the shared build pipeline.
"""

from __future__ import annotations

from datetime import date
import re
from typing import TypedDict

from .response_scales import resolve_categorical, validate_scale_type


# Required by the BenchmarkBuild metadata contract. ``multi_single_turn`` is
# required for new builds (and present in the canonical template). It remains
# structurally optional here only because many historical source files predate
# the field; get_benchmark_id warns, and the strict parquet schema rejects the
# null until that historical builder is migrated.
REQUIRED_KEYS = (
    "description",
    "paper_url",
    "data_source_url",
    "license",
    "modality",
    "domain",
    "response_type",
    "response_scale",
    "release_date",
)

OPTIONAL_KEYS = (
    "categorical",
    "version",
    "one_line_description",
    "multi_single_turn",
    "granularity",
    "release",
    "benchmark_features",
)

CURATION_ONLY_KEYS = (
    "testing_condition",
    "subject_type",
    "item_type",
    "citation",
    "tags",
)

DOMAINS = (
    "software_engineering",
    "ml_engineering",
    "mathematics",
    "science",
    "medicine",
    "law",
    "finance",
    "cybersecurity",
    "cultural",
    "education",
    "knowledge",
    "reasoning",
    "safety",
    "agents_and_tool_use",
    "preference",
    "reward_modeling",
    "nlp_task",
    "multilingual",
    "general",
)

MODALITIES = (
    "text",
    "image",
    "video",
    "audio",
    "grid",
    "gui_screenshot",
)

RESPONSE_TYPES = (
    "binary",
    "likert_5",
    "likert_10",
    "win_rate",
    "ordinal",
    "fraction",
    "continuous_bounded",
    "continuous_unbounded",
    "error_presence",
    "mixed",
)

# ``continuous`` appears in a few historical builders.  It is accepted for
# compatibility, but new builds choose bounded vs. unbounded explicitly.
_LEGACY_RESPONSE_TYPES = {"continuous"}
MULTI_SINGLE_TURN_VALUES = {
    "single_turn",
    "multi_turn",
    "multi_turn, single_turn",
}
GRANULARITIES = {"item", "aggregate", "not_released"}
RELEASE_VALUES = {"public", "private"}
ONE_LINE_DESCRIPTION_STARTERS = (
    "Measures",
    "Evaluates",
    "Tests",
    "Assesses",
)
ONE_LINE_DESCRIPTION_MAX_LENGTH = 160


class BenchmarkInfoRequired(TypedDict):
    description: str
    paper_url: str | None
    data_source_url: str
    license: str
    modality: list[str]
    domain: list[str]
    response_type: str
    response_scale: dict | str
    release_date: str | None


class BenchmarkInfo(BenchmarkInfoRequired, total=False):
    """Typed shape of a ``BenchmarkBuild.INFO`` dictionary.

    The curation-only keys remain useful in ``build.py`` source even though
    they are not persisted in ``benchmarks.parquet``.
    """

    categorical: bool
    version: str | None
    one_line_description: str
    multi_single_turn: str
    granularity: str
    release: str
    benchmark_features: dict | str
    testing_condition: str
    subject_type: str
    item_type: str
    citation: str
    tags: list[str]


def _is_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and bool(part.strip()) for part in value)
    )


def validate_benchmark_release_date(value: object) -> None:
    """Accept an unknown date or a valid ISO month/day without filling in precision."""
    if value is None:
        return
    message = (
        "release_date must be null or a valid quoted YYYY-MM or YYYY-MM-DD "
        "calendar date (years 0001–9999)"
    )
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?", value):
        raise ValueError(message)
    parts = [int(part) for part in value.split("-")]
    try:
        # Day 1 is used only to validate a month; the stored value is unchanged.
        date(parts[0], parts[1], parts[2] if len(parts) == 3 else 1)
    except ValueError as exc:
        raise ValueError(message) from exc


def validate_info(info: object, *, context: str = "benchmark") -> None:
    """Validate one author-supplied ``INFO`` dictionary.

    This deliberately enforces structure rather than closing the extensible
    domain/modality vocabularies.  It does enforce repository-wide enums such
    as response type, granularity, release, and turn structure when supplied.
    All problems are reported together.
    """

    if not isinstance(info, dict):
        raise ValueError(f"{context}: INFO must be a dict, got {type(info).__name__}")

    problems: list[str] = []
    known_keys = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS) | set(CURATION_ONLY_KEYS)
    unknown = sorted((key for key in info if key not in known_keys), key=repr)
    if unknown:
        problems.append(
            f"unknown key(s) {unknown}; add persisted metadata to "
            "parquet_schemas.yaml or "
            "scripts/build_measurement_tables/define_benchmark_vocabulary.py "
            "rather than creating "
            "an unconsumed INFO field"
        )
    missing = [key for key in REQUIRED_KEYS if key not in info]
    if missing:
        problems.append(f"missing required key(s) {missing}")

    for key in (
        "description",
        "one_line_description",
        "data_source_url",
        "license",
    ):
        value = info.get(key)
        if key in info and (not isinstance(value, str) or not value.strip()):
            problems.append(f"{key!r} must be a non-empty string")

    one_line_description = info.get("one_line_description")
    if isinstance(one_line_description, str) and one_line_description.strip():
        if len(one_line_description) > ONE_LINE_DESCRIPTION_MAX_LENGTH:
            problems.append(
                "'one_line_description' must be at most "
                f"{ONE_LINE_DESCRIPTION_MAX_LENGTH} characters"
            )
        if "\n" in one_line_description or "\r" in one_line_description:
            problems.append("'one_line_description' must be one logical line")
        if not re.match(
            rf"^(?:{'|'.join(ONE_LINE_DESCRIPTION_STARTERS)})\b",
            one_line_description,
        ):
            problems.append(
                "'one_line_description' must start with one of "
                f"{list(ONE_LINE_DESCRIPTION_STARTERS)}"
            )

    for key in ("paper_url",):
        value = info.get(key)
        if key in info and value is not None and not isinstance(value, str):
            problems.append(f"{key!r} must be a string or None")

    version = info.get("version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        problems.append("'version' must be a non-empty string or None")

    if "release_date" in info:
        try:
            validate_benchmark_release_date(info["release_date"])
        except ValueError as exc:
            problems.append(str(exc))

    for key in ("modality", "domain"):
        if key in info and not _is_string_list(info[key]):
            problems.append(f"{key!r} must be a non-empty list of non-empty strings")

    if "categorical" in info and not isinstance(info["categorical"], bool):
        problems.append("'categorical' must be a bool")

    response_type = info.get("response_type")
    if isinstance(response_type, str):
        try:
            resolve_categorical(response_type, info.get("categorical"))
        except ValueError as exc:
            problems.append(str(exc))
    if "response_scale" in info and isinstance(response_type, str):
        try:
            validate_scale_type(response_type, info["response_scale"])
        except (TypeError, ValueError) as exc:
            problems.append(str(exc))
    allowed_response_types = set(RESPONSE_TYPES) | _LEGACY_RESPONSE_TYPES
    if "response_type" in info and (
        not isinstance(response_type, str)
        or response_type not in allowed_response_types
    ):
        problems.append(
            f"'response_type' must be one of {sorted(allowed_response_types)}, "
            f"got {response_type!r}"
        )

    enums = {
        "multi_single_turn": MULTI_SINGLE_TURN_VALUES,
        "granularity": GRANULARITIES,
        "release": RELEASE_VALUES,
    }
    for key, allowed in enums.items():
        value = info.get(key)
        if value is not None and (
            not isinstance(value, str) or value not in allowed
        ):
            problems.append(f"{key!r} must be one of {sorted(allowed)}, got {value!r}")

    features = info.get("benchmark_features")
    if features is not None and not isinstance(features, (dict, str)):
        problems.append("'benchmark_features' must be a dict, string, or None")

    for key in ("testing_condition", "subject_type", "item_type", "citation"):
        value = info.get(key)
        if value is not None and not isinstance(value, str):
            problems.append(f"{key!r} must be a string when supplied")
    if "tags" in info and not _is_string_list(info["tags"]):
        problems.append("'tags' must be a non-empty list of non-empty strings when supplied")

    if problems:
        raise ValueError(f"{context}: invalid INFO — " + "; ".join(problems))
