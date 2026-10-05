#!/usr/bin/env python3
"""Build the checked-in, source-backed benchmark attribution manifest.

Reads reviewed website references, falling back to local benchmark metadata
for new entries, and combines them with the lead-author institution registry.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CARDS_PATH = REPO_ROOT / "website/content/generated/benchmark-cards.json"
DEFAULT_HIDDEN_PATH = REPO_ROOT / "website/content/curated/hidden-benchmarks.json"
DEFAULT_AFFILIATIONS_PATH = (
    REPO_ROOT / "website/content/curated/benchmark-affiliations.json"
)
DEFAULT_OVERRIDES_PATH = (
    REPO_ROOT / "website/content/curated/benchmark-attribution-overrides.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "website/content/generated/benchmark-attributions.json"
)

AFFILIATION_SCOPE = "first_two_authors_first_listed_institutions"
ORGANIZATION_AUTHOR_PATTERN = re.compile(
    r"\b(?:Team|Lab|Laboratory|Foundation|Consortium|Collaboration)\b"
)


class AttributionError(ValueError):
    """One or more attribution records cannot be represented faithfully."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionError(f"could not read {path}: {exc}") from exc


def _validate_object_keys(
    payload: dict[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing:
        raise AttributionError(f"{label} is missing {', '.join(missing)}")
    if unknown:
        raise AttributionError(f"{label} has unknown fields {unknown}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttributionError(f"{label} must be a non-empty string")
    return value.strip()


def _http_url(value: Any, label: str) -> str:
    url = _nonempty_string(value, label)
    if not re.match(r"^https?://", url):
        raise AttributionError(f"{label} must be HTTP(S)")
    return url


def _validate_credit_entries(
    entries: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise AttributionError(f"{label} must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AttributionError(f"{label}[{index}] must be an object")
        item_label = f"{label}[{index}]"
        _validate_object_keys(
            entry,
            label=item_label,
            required=required,
            optional=optional,
        )
        _nonempty_string(entry["name"], f"{item_label}.name")
        if "role" in entry:
            _nonempty_string(entry["role"], f"{item_label}.role")
        if "url" in entry:
            _http_url(entry["url"], f"{item_label}.url")
        if "kind" in entry and entry["kind"] not in {"person", "organization"}:
            raise AttributionError(
                f"{item_label}.kind must be 'person' or 'organization'"
            )
    return entries


def _literal_info_from_build(path: Path) -> dict[str, Any]:
    """Read a literal module- or class-level ``INFO`` without executing a builder."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise AttributionError(f"could not parse {path}: {exc}") from exc

    # Some legacy builders keep INFO as a class attribute.  Walking the syntax
    # tree is still safe: only literal values are evaluated, and assignments
    # such as ``INFO = INFO`` are skipped rather than importing the module.
    candidates: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INFO"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "INFO"
        ):
            value = node.value
        if value is None:
            continue
        try:
            info = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if not isinstance(info, dict):
            raise AttributionError(f"{path}: INFO is not a mapping")
        candidates.append(info)
    if not candidates:
        raise AttributionError(f"{path}: no literal INFO mapping")
    first = candidates[0]
    if any(candidate != first for candidate in candidates[1:]):
        raise AttributionError(f"{path}: multiple distinct literal INFO mappings")
    return first


def _load_benchmark_info(
    benchmark_root: Path, slug: str
) -> tuple[dict[str, Any], Path]:
    references_path = (
        benchmark_root.parent / "website/content/curated/benchmark-references.json"
    )
    if references_path.exists():
        references = _read_json(references_path)
        if not isinstance(references, dict):
            raise AttributionError(f"{references_path}: expected a slug-keyed object")
        if slug in references:
            if not isinstance(references[slug], dict):
                raise AttributionError(f"{references_path}: {slug} must be an object")
            return references[slug], references_path

    folder = benchmark_root / slug
    metadata_path = folder / "metadata.yaml"
    if metadata_path.exists():
        try:
            payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise AttributionError(f"could not read {metadata_path}: {exc}") from exc
        benchmark = payload.get("benchmark") if isinstance(payload, dict) else None
        if not isinstance(benchmark, dict):
            raise AttributionError(
                f"{metadata_path}: expected a top-level benchmark mapping"
            )
        return benchmark, metadata_path

    build_path = folder / "build.py"
    if build_path.exists():
        return _literal_info_from_build(build_path), build_path
    raise AttributionError(f"{folder}: no metadata.yaml or build.py")


def _consume_group(text: str, start: int) -> tuple[str, int]:
    opening = text[start]
    if opening not in '{("':
        raise AttributionError(f"expected a delimited value at offset {start}")
    closing = {"{": "}", "(": ")", '"': '"'}[opening]
    if opening == '"':
        escaped = False
        brace_depth = 0
        for index in range(start + 1, len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    raise AttributionError("unbalanced braces in quoted BibTeX value")
            elif char == closing and brace_depth == 0:
                return text[start + 1 : index], index + 1
        raise AttributionError("unterminated quoted BibTeX value")

    if opening == "(":
        depth = 1
        brace_depth = 0
        quoted = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"' and brace_depth == 0:
                quoted = not quoted
                continue
            if quoted:
                continue
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    raise AttributionError("unbalanced braces in BibTeX entry")
            elif brace_depth == 0 and char == opening:
                depth += 1
            elif brace_depth == 0 and char == closing:
                depth -= 1
                if depth == 0:
                    return text[start + 1 : index], index + 1
        raise AttributionError("unterminated BibTeX entry opened with '('")

    depth = 1
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise AttributionError(f"unterminated BibTeX group opened with {opening!r}")


def _first_top_level_comma(text: str) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
        elif not quoted and depth == 0 and char == ",":
            return index
    raise AttributionError("BibTeX entry has no citation-key separator")


def parse_bibtex_entry(text: str) -> dict[str, Any]:
    """Parse the subset of BibTeX used by benchmark citation templates.

    The original template remains untouched for copying.  Parsing exists only
    to create validated display metadata, so unsupported constructs fail rather
    than being guessed at.
    """

    citation = text.strip()
    match = re.match(r"@([A-Za-z]+)\s*", citation)
    if not match:
        raise AttributionError("citation is not a BibTeX entry")
    entry_type = match.group(1).lower()
    position = match.end()
    if position >= len(citation) or citation[position] not in "{(":
        raise AttributionError("BibTeX entry must open with '{' or '('")
    body, end = _consume_group(citation, position)
    if citation[end:].strip():
        raise AttributionError("citation must contain exactly one BibTeX entry")

    key_end = _first_top_level_comma(body)
    key = body[:key_end].strip()
    if not key or not re.fullmatch(r"[^\s,{}()]+", key):
        raise AttributionError(f"invalid BibTeX citation key {key!r}")

    fields: dict[str, str] = {}
    position = key_end + 1
    while position < len(body):
        while position < len(body) and (
            body[position].isspace() or body[position] == ","
        ):
            position += 1
        if position >= len(body):
            break
        field_match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", body[position:])
        if not field_match:
            excerpt = body[position : position + 40].replace("\n", " ")
            raise AttributionError(f"malformed BibTeX field near {excerpt!r}")
        field = field_match.group(1).lower()
        position += field_match.end()
        if field in fields:
            raise AttributionError(f"duplicate BibTeX field {field!r}")
        if position >= len(body):
            raise AttributionError(f"BibTeX field {field!r} has no value")
        if body[position] in '{"':
            value, position = _consume_group(body, position)
        else:
            value_start = position
            while position < len(body) and body[position] != ",":
                position += 1
            value = body[value_start:position].strip()
        fields[field] = value.strip()
        while position < len(body) and body[position].isspace():
            position += 1
        if position < len(body) and body[position] != ",":
            raise AttributionError(f"BibTeX field {field!r} is not comma-terminated")

    return {
        "entryType": entry_type,
        "key": key,
        "fields": fields,
        "bibtex": citation,
    }


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    escaped = False
    index = 0
    lowered = value.lower()
    while index < len(value):
        char = value[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "{":
            depth += 1
            index += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise AttributionError("unbalanced braces in BibTeX name")
            index += 1
            continue
        if depth == 0 and lowered.startswith(delimiter, index):
            parts.append(value[start:index].strip())
            index += len(delimiter)
            start = index
            continue
        index += 1
    if depth:
        raise AttributionError("unbalanced braces in BibTeX name")
    parts.append(value[start:].strip())
    return parts


_COMBINING_ACCENTS = {
    '"': "\u0308",
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "u": "\u0306",
    "v": "\u030c",
    "H": "\u030b",
    "c": "\u0327",
    "k": "\u0328",
    "r": "\u030a",
}


def _accent_replacement(match: re.Match[str]) -> str:
    accent, letter = match.groups()
    return unicodedata.normalize("NFC", letter + _COMBINING_ACCENTS[accent])


def bibtex_display_text(value: str) -> str:
    """Convert reviewed BibTeX title/name text to Unicode display text."""

    text = " ".join(value.split())
    text = text.replace("$\\tau^2$", "τ²").replace("$\\tau$", "τ")
    text = text.replace("{\\i}", "ı").replace("\\i", "ı")
    accent_pattern = re.compile(r"\{\\([\"'`\^~=\.uvHckr])\s*\{?([A-Za-z])\}?\}")
    previous = None
    while previous != text:
        previous = text
        text = accent_pattern.sub(_accent_replacement, text)
    text = re.sub(r"\\(?:textsc|textrm|textit|emph)\{([^{}]*)\}", r"\1", text)
    text = (
        text.replace("\\&", "&")
        .replace("\\%", "%")
        .replace("\\_", "_")
        .replace("\\#", "#")
        .replace("~", " ")
    )
    text = text.replace("{", "").replace("}", "")
    if "\\" in text or "$" in text:
        raise AttributionError(
            f"unsupported LaTeX in display metadata {value!r}; add a reviewed override"
        )
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _outer_braced(value: str) -> bool:
    if not value.startswith("{") or not value.endswith("}"):
        return False
    try:
        _, end = _consume_group(value, 0)
    except AttributionError:
        return False
    return end == len(value)


def _display_author_name(bibtex_name: str) -> str:
    if _outer_braced(bibtex_name):
        return bibtex_display_text(bibtex_name[1:-1])
    comma_parts = _split_top_level(bibtex_name, ",")
    if len(comma_parts) == 1:
        display = comma_parts[0]
    elif len(comma_parts) == 2:
        display = f"{comma_parts[1]} {comma_parts[0]}"
    elif len(comma_parts) == 3:
        display = f"{comma_parts[2]} {comma_parts[0]}, {comma_parts[1]}"
    else:
        raise AttributionError(f"unsupported BibTeX author name {bibtex_name!r}")
    return bibtex_display_text(display)


def parse_authors(value: str) -> list[dict[str, str]]:
    names = _split_top_level(" ".join(value.split()), " and ")
    authors: list[dict[str, str]] = []
    for raw_name in names:
        if not raw_name:
            raise AttributionError("citation contains an empty author")
        if raw_name.casefold() == "others" or re.search(
            r"\bet\s+al\.?\b", raw_name, flags=re.IGNORECASE
        ):
            raise AttributionError("citation abbreviates its author list")
        authors.append(
            {
                "name": _display_author_name(raw_name),
                "bibtexName": raw_name,
            }
        )
    return authors


def _suspected_organization_authors(authors: list[dict[str, str]]) -> set[str]:
    return {
        author["name"]
        for author in authors
        if _outer_braced(author["bibtexName"])
        or ORGANIZATION_AUTHOR_PATTERN.search(author["name"])
    }


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normal_reference(
    *,
    slug: str,
    card: dict[str, Any],
    info: dict[str, Any],
    citation_text: str,
    citation_source: str,
    citation_evidence_url: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_bibtex_entry(citation_text)
    fields = parsed["fields"]
    missing = [field for field in ("title", "author", "year") if not fields.get(field)]
    if missing:
        raise AttributionError(
            f"citation is missing required field(s): {', '.join(missing)}"
        )
    authors = parse_authors(fields["author"])
    title = bibtex_display_text(fields["title"])
    year = bibtex_display_text(fields["year"])
    if not re.fullmatch(r"\d{4}", year):
        raise AttributionError(f"citation year is not YYYY: {year!r}")
    reference_url = fields.get("url") or card.get("paper") or info.get("paper_url")
    if not isinstance(reference_url, str) or not re.match(r"^https?://", reference_url):
        raise AttributionError("reference has no authoritative HTTP(S) URL")

    reference = {
        "kind": "paper",
        "title": title,
        "url": reference_url,
        "year": year,
        "authors": authors,
        "producers": [],
        "contributors": [],
        "measurementSources": [],
    }
    citation = {
        "status": "available",
        "entryType": parsed["entryType"],
        "key": parsed["key"],
        "bibtex": parsed["bibtex"],
        "source": citation_source,
        "evidenceUrl": citation_evidence_url or reference_url,
    }
    return reference, citation


def _overridden_reference(
    slug: str, override: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_object_keys(
        override,
        label="reference override",
        required={
            "kind",
            "title",
            "url",
            "year",
            "evidenceUrl",
            "reason",
            "citationStatus",
        },
        optional={"authors", "producers", "contributors", "measurementSources"},
    )
    if override["kind"] not in {"dataset", "paper", "project"}:
        raise AttributionError(f"invalid reference kind {override['kind']!r}")
    title = _nonempty_string(override["title"], "reference override title")
    url = _http_url(override["url"], "reference override URL")
    evidence_url = _http_url(override["evidenceUrl"], "reference override evidenceUrl")
    year = _nonempty_string(override["year"], "reference override year")
    if not re.fullmatch(r"\d{4}", year):
        raise AttributionError("reference override year must be YYYY")
    _nonempty_string(override["reason"], "reference override reason")
    authors = _validate_credit_entries(
        override.get("authors", []),
        label="reference override authors",
        required={"name"},
        optional={"bibtexName", "kind", "url"},
    )
    producers = _validate_credit_entries(
        override.get("producers", []),
        label="reference override producers",
        required={"name", "role", "url"},
    )
    contributors = _validate_credit_entries(
        override.get("contributors", []),
        label="reference override contributors",
        required={"name", "role", "url"},
    )
    measurement_sources = _validate_credit_entries(
        override.get("measurementSources", []),
        label="reference override measurementSources",
        required={"name", "role", "url"},
    )
    if not authors and not producers:
        raise AttributionError("reference override needs authors or producers")

    reference = {
        "kind": override["kind"],
        "title": title,
        "url": url,
        "year": year,
        "authors": authors,
        "producers": producers,
        "contributors": contributors,
        "measurementSources": measurement_sources,
    }
    status = override.get("citationStatus")
    if status != "not_provided":
        raise AttributionError(
            "a reference override without BibTeX must explicitly set "
            "citationStatus='not_provided'"
        )
    citation = {
        "status": "not_provided",
        "entryType": None,
        "key": None,
        "bibtex": None,
        "source": "producer_did_not_supply",
        "evidenceUrl": evidence_url,
    }
    return reference, citation


def build_manifest(
    *,
    repo_root: Path = REPO_ROOT,
    cards_path: Path = DEFAULT_CARDS_PATH,
    hidden_path: Path = DEFAULT_HIDDEN_PATH,
    affiliations_path: Path = DEFAULT_AFFILIATIONS_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
) -> dict[str, Any]:
    cards = _read_json(cards_path)
    hidden = _read_json(hidden_path)
    affiliations = _read_json(affiliations_path)
    overrides = _read_json(overrides_path)
    if not isinstance(cards, list) or not all(isinstance(card, dict) for card in cards):
        raise AttributionError(f"{cards_path}: expected a list of card objects")
    if not isinstance(hidden, list) or not all(
        isinstance(slug, str) for slug in hidden
    ):
        raise AttributionError(f"{hidden_path}: expected a list of slugs")
    if not isinstance(affiliations, dict):
        raise AttributionError(f"{affiliations_path}: expected an object")
    if not isinstance(overrides, dict):
        raise AttributionError(f"{overrides_path}: expected an object")
    unknown_override_sections = sorted(
        set(overrides)
        - {
            "aliases",
            "affiliations",
            "authorVerifications",
            "citations",
            "references",
        }
    )
    if unknown_override_sections:
        raise AttributionError(
            f"{overrides_path}: unknown sections {unknown_override_sections}"
        )

    card_slugs: list[str] = []
    for index, card in enumerate(cards):
        slug = card.get("slug")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
            raise AttributionError(f"{cards_path}: card {index} has an invalid slug")
        if not isinstance(card.get("name"), str) or not card["name"].strip():
            raise AttributionError(f"{cards_path}: card {index} has an invalid name")
        card_slugs.append(slug)
    duplicate_card_slugs = sorted(
        slug for slug in set(card_slugs) if card_slugs.count(slug) > 1
    )
    if duplicate_card_slugs:
        raise AttributionError(
            f"{cards_path}: duplicate card slugs {duplicate_card_slugs}"
        )
    if len(hidden) != len(set(hidden)):
        raise AttributionError(f"{hidden_path}: hidden slugs must be unique")

    hidden_set = set(hidden)
    visible_cards = sorted(
        (card for card in cards if card.get("slug") not in hidden_set),
        key=lambda card: str(card.get("slug")),
    )
    visible_slugs = {str(card.get("slug")) for card in visible_cards}
    references_path = repo_root / "website/content/curated/benchmark-references.json"
    curated_slugs = set(affiliations.get("benchmarks", {})) | (
        set(_read_json(references_path)) if references_path.exists() else set()
    )
    known_slugs = visible_slugs | curated_slugs
    aliases = overrides.get("aliases", {})
    reference_overrides = overrides.get("references", {})
    citation_overrides = overrides.get("citations", {})
    affiliation_overrides = overrides.get("affiliations", {})
    author_verifications = overrides.get("authorVerifications", {})
    for label, mapping in (
        ("aliases", aliases),
        ("references", reference_overrides),
        ("citations", citation_overrides),
        ("affiliations", affiliation_overrides),
        ("authorVerifications", author_verifications),
    ):
        if not isinstance(mapping, dict):
            raise AttributionError(f"{overrides_path}: {label} must be an object")
        orphaned = sorted(set(mapping) - known_slugs)
        if orphaned:
            raise AttributionError(
                f"{overrides_path}: {label} contain unknown slugs {orphaned}"
            )
        curated_slugs.update(mapping)
    overlapping_reference_citations = sorted(
        set(reference_overrides) & set(citation_overrides)
    )
    if overlapping_reference_citations:
        raise AttributionError(
            f"{overrides_path}: slugs have both reference and citation overrides "
            f"{overlapping_reference_citations}"
        )

    benchmark_root = repo_root / "benchmarks"
    affiliation_rows = affiliations.get("benchmarks", {})
    institution_rows = affiliations.get("institutions", {})
    if not isinstance(affiliation_rows, dict) or not isinstance(institution_rows, dict):
        raise AttributionError(
            f"{affiliations_path}: benchmarks and institutions must be objects"
        )
    records: dict[str, Any] = {}
    errors: list[str] = []
    for card in visible_cards:
        slug = str(card.get("slug"))
        if slug not in curated_slugs:
            records[slug] = None  # The page uses the source links from HF.
            continue
        try:
            alias = aliases.get(slug)
            metadata_slug = slug
            alias_provenance = None
            if alias is not None:
                if not isinstance(alias, dict):
                    raise AttributionError("alias must be an object")
                _validate_object_keys(
                    alias,
                    label="alias",
                    required={"metadataSlug", "evidenceUrl", "reason"},
                )
                metadata_slug = _nonempty_string(
                    alias["metadataSlug"], "alias metadataSlug"
                )
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", metadata_slug):
                    raise AttributionError("alias metadataSlug is not a safe slug")
                alias_evidence_url = _http_url(
                    alias["evidenceUrl"], "alias evidenceUrl"
                )
                alias_reason = _nonempty_string(alias["reason"], "alias reason")
                alias_provenance = {
                    "metadataSlug": metadata_slug,
                    "evidenceUrl": alias_evidence_url,
                    "reason": alias_reason,
                }

            info, metadata_path = _load_benchmark_info(benchmark_root, metadata_slug)
            reference_override = reference_overrides.get(slug)
            reference_override_provenance = None
            citation_override_provenance = None
            if reference_override is not None:
                if not isinstance(reference_override, dict):
                    raise AttributionError("reference override must be an object")
                reference, citation = _overridden_reference(slug, reference_override)
                reference_override_provenance = {
                    "evidenceUrl": str(reference_override["evidenceUrl"]),
                    "reason": str(reference_override["reason"]),
                }
            else:
                citation_override = citation_overrides.get(slug)
                if citation_override is not None:
                    if not isinstance(citation_override, dict):
                        raise AttributionError("citation override must be an object")
                    _validate_object_keys(
                        citation_override,
                        label="citation override",
                        required={"bibtex", "evidenceUrl", "reason"},
                    )
                    citation_text = _nonempty_string(
                        citation_override["bibtex"], "citation override bibtex"
                    )
                    citation_source = "curated_primary_source_override"
                    evidence_url = _http_url(
                        citation_override["evidenceUrl"],
                        "citation override evidenceUrl",
                    )
                    citation_reason = _nonempty_string(
                        citation_override["reason"], "citation override reason"
                    )
                    citation_override_provenance = {
                        "evidenceUrl": evidence_url,
                        "reason": citation_reason,
                    }
                else:
                    citation_text = info.get("citation")
                    citation_source = "local_metadata"
                    evidence_url = None
                if not isinstance(citation_text, str) or not citation_text.strip():
                    raise AttributionError("local metadata has no citation")
                reference, citation = _normal_reference(
                    slug=slug,
                    card=card,
                    info=info,
                    citation_text=citation_text,
                    citation_source=citation_source,
                    citation_evidence_url=evidence_url,
                )

            author_verification = author_verifications.get(slug)
            author_verification_provenance = None
            if reference["kind"] == "paper":
                authors = reference["authors"]
                suspected_organizations = _suspected_organization_authors(authors)
                if not isinstance(author_verification, dict):
                    raise AttributionError(
                        "paper author list lacks a primary-source count verification"
                    )
                organization_authors: set[str] = set()
                if author_verification is not None:
                    if not isinstance(author_verification, dict):
                        raise AttributionError("author verification must be an object")
                    required = {"expectedCount", "evidenceUrl", "reason"}
                    allowed = required | {"organizationAuthors"}
                    missing = sorted(required - set(author_verification))
                    unknown = sorted(set(author_verification) - allowed)
                    if missing:
                        raise AttributionError(
                            f"author verification is missing {', '.join(missing)}"
                        )
                    if unknown:
                        raise AttributionError(
                            f"author verification has unknown fields {unknown}"
                        )
                    expected_count = author_verification["expectedCount"]
                    if not isinstance(expected_count, int) or expected_count < 1:
                        raise AttributionError(
                            "author verification expectedCount must be a positive integer"
                        )
                    if len(authors) != expected_count:
                        raise AttributionError(
                            f"author verification expected {expected_count} authors, "
                            f"found {len(authors)}"
                        )
                    verification_url = str(author_verification["evidenceUrl"])
                    verification_reason = str(author_verification["reason"]).strip()
                    if not re.match(r"^https?://", verification_url):
                        raise AttributionError(
                            "author verification evidenceUrl must be HTTP(S)"
                        )
                    if not verification_reason:
                        raise AttributionError("author verification reason is empty")
                    organization_values = author_verification.get(
                        "organizationAuthors", []
                    )
                    if not isinstance(organization_values, list) or not all(
                        isinstance(name, str) and name.strip()
                        for name in organization_values
                    ):
                        raise AttributionError(
                            "author verification organizationAuthors must be names"
                        )
                    organization_authors = set(organization_values)
                    author_names = {author["name"] for author in authors}
                    unknown_organizations = sorted(organization_authors - author_names)
                    if unknown_organizations:
                        raise AttributionError(
                            "author verification names absent organizations "
                            f"{unknown_organizations}"
                        )
                    missing_organizations = sorted(
                        suspected_organizations - organization_authors
                    )
                    if missing_organizations:
                        raise AttributionError(
                            "suspected organization authors are not reviewed "
                            f"{missing_organizations}"
                        )
                    author_verification_provenance = {
                        "expectedCount": expected_count,
                        "organizationAuthors": sorted(organization_authors),
                        "evidenceUrl": verification_url,
                        "reason": verification_reason,
                    }
                for author in authors:
                    author["kind"] = (
                        "organization"
                        if author["name"] in organization_authors
                        else "person"
                    )
            elif author_verification is not None:
                raise AttributionError(
                    "author verification is only valid for a paper reference"
                )

            affiliation = affiliation_rows.get(slug)
            affiliation_override = affiliation_overrides.get(slug)
            affiliation_override_provenance = None
            if affiliation_override is not None:
                if not isinstance(affiliation_override, dict):
                    raise AttributionError("affiliation override must be an object")
                _validate_object_keys(
                    affiliation_override,
                    label="affiliation override",
                    required={"status", "evidenceUrl", "reason"},
                )
                if affiliation_override["status"] != "not_applicable":
                    raise AttributionError(
                        "affiliation override status must be 'not_applicable'"
                    )
                evidence_url = _http_url(
                    affiliation_override["evidenceUrl"],
                    "affiliation override evidenceUrl",
                )
                affiliation_reason = _nonempty_string(
                    affiliation_override["reason"], "affiliation override reason"
                )
                affiliation_status = "not_applicable"
                affiliation_scope = None
                lead_institutions = []
                affiliation_override_provenance = {
                    "evidenceUrl": evidence_url,
                    "reason": affiliation_reason,
                }
            else:
                if not isinstance(affiliation, dict):
                    raise AttributionError("no lead-author affiliation record")
                institution_names = affiliation.get("institutions")
                if not isinstance(institution_names, list) or not institution_names:
                    raise AttributionError("lead-author institutions are empty")
                affiliation_source = affiliation.get("source")
                if (
                    not isinstance(affiliation_source, str)
                    or not affiliation_source.strip()
                ):
                    raise AttributionError("lead-author affiliation source is empty")
                if "recheck" in affiliation_source.casefold():
                    raise AttributionError(
                        "lead-author affiliation source is marked for recheck"
                    )
                affiliation_status = "available"
                affiliation_scope = AFFILIATION_SCOPE
                lead_institutions = []
                for name in institution_names:
                    if not isinstance(name, str) or not name.strip():
                        raise AttributionError("lead-author institution name is empty")
                    institution = institution_rows.get(name)
                    if not isinstance(institution, dict):
                        raise AttributionError(
                            f"lead-author institution {name!r} has no registry entry"
                        )
                    lead_institutions.append(
                        {
                            "name": name,
                        }
                    )

            records[slug] = {
                "metadataSlug": metadata_slug,
                "metadataPath": _repo_relative(metadata_path, repo_root),
                "benchmarkName": card["name"].strip(),
                "reference": reference,
                "leadAuthorInstitutions": lead_institutions,
                "affiliationStatus": affiliation_status,
                "affiliationScope": affiliation_scope,
                "citation": citation,
                "provenance": {
                    "alias": alias_provenance,
                    "referenceOverride": reference_override_provenance,
                    "citationOverride": citation_override_provenance,
                    "authorVerification": author_verification_provenance,
                    "affiliationOverride": affiliation_override_provenance,
                    "affiliations": (
                        "website/content/curated/benchmark-affiliations.json"
                        if affiliation_override is None
                        else None
                    ),
                    "affiliationRegistrySource": (
                        affiliation.get("source")
                        if affiliation_override is None
                        and isinstance(affiliation, dict)
                        else None
                    ),
                    "affiliationEvidenceUrl": (
                        reference["url"] if affiliation_override is None else None
                    ),
                },
            }
        except (AttributionError, KeyError, TypeError) as exc:
            errors.append(f"{slug}: {exc}")

    missing_records = sorted(visible_slugs - set(records))
    if missing_records and not errors:
        errors.append(f"missing records: {', '.join(missing_records)}")
    if errors:
        raise AttributionError(
            "benchmark attribution validation failed:\n- " + "\n- ".join(errors)
        )

    unavailable = sorted(
        slug
        for slug, record in records.items()
        if record is not None and record["citation"]["status"] != "available"
    )
    unreviewed = sorted(slug for slug, record in records.items() if record is None)
    return {
        "schemaVersion": 1,
        "affiliationScope": AFFILIATION_SCOPE,
        "coverage": {
            "benchmarks": len(records),
            "withCitation": len(records) - len(unavailable) - len(unreviewed),
            "withoutProducerCitation": unavailable,
            **({"unreviewed": unreviewed} if unreviewed else {}),
        },
        "benchmarks": records,
    }


def render_manifest(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def write_manifest(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(render_manifest(payload))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cards", type=Path)
    parser.add_argument("--hidden", type=Path)
    parser.add_argument("--affiliations", type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in manifest differs instead of writing it",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    cards_path = (
        args.cards or repo_root / "website/content/generated/benchmark-cards.json"
    )
    hidden_path = (
        args.hidden or repo_root / "website/content/curated/hidden-benchmarks.json"
    )
    affiliations_path = (
        args.affiliations
        or repo_root / "website/content/curated/benchmark-affiliations.json"
    )
    overrides_path = (
        args.overrides
        or repo_root / "website/content/curated/benchmark-attribution-overrides.json"
    )
    output_path = (
        args.output
        or repo_root / "website/content/generated/benchmark-attributions.json"
    )

    try:
        payload = build_manifest(
            repo_root=repo_root,
            cards_path=cards_path,
            hidden_path=hidden_path,
            affiliations_path=affiliations_path,
            overrides_path=overrides_path,
        )
        rendered = render_manifest(payload)
        if args.check:
            current = (
                output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            )
            if current != rendered:
                raise AttributionError(
                    f"{output_path} is stale; run {Path(__file__).name}"
                )
        else:
            write_manifest(payload, output_path)
    except AttributionError as exc:
        raise SystemExit(str(exc)) from None

    verb = "verified" if args.check else "wrote"
    coverage = payload["coverage"]
    print(
        f"✓ {verb} {coverage['benchmarks']} benchmark attributions "
        f"({coverage['withCitation']} citations; "
        f"{len(coverage['withoutProducerCitation'])} explicitly unavailable)"
    )


if __name__ == "__main__":
    main()
