#!/usr/bin/env python3
"""Render the gallery from HF tables. See README.md for build and preview commands."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import mimetypes
import os
import re
import shutil
import struct
import sys
import time
from contextlib import ExitStack
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Callable, NamedTuple
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from fastparquet.cencoding import NumpyIO, ThriftObject
from huggingface_hub import HfApi, RepoFolder, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

# Paths and source
HF_REPO = "aims-foundations/measurement-db"
HF_REVISION = os.environ.get("HF_REVISION")
RESPONSE_FILE = "responses.parquet"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR))
from scripts.render_website.gallery_schema import ResponseScale, attach_features, parse_fields
HIDDEN_PATH = REPO_DIR / "website" / "content" / "curated" / "hidden-benchmarks.json"

# Display metadata
# Keep domain IDs and order aligned with website/content/measurement-db.ts.
DOMAIN_ORDER = [
    "safety", "cybersecurity", "medicine", "law", "finance",
    "mathematics", "software_engineering", "ml_engineering",
    "agents_and_tool_use", "science", "multilingual", "cultural",
    "education", "preference", "reward_modeling", "nlp_task", "knowledge",
    "reasoning", "general",
]
DOMAIN_SET = set(DOMAIN_ORDER)
SESSION_DIMS = ("agent", "user", "session", "turn")
# One cell represents one response at this key. load_raw also includes
# interactors when present, before projecting them into display labels.
KEY = ["subject_id", "item_id", "test_condition", "trial"]
# Value for a dimension absent from one condition.
MISSING_DIM = "n/a"

# Matrix rendering
# Pixels are presentation only; bundles retain exact grades separately.
RED = (214, 39, 40)
BLUE = (31, 119, 180)
GRAY = (158, 158, 158)  # recorded attempt with an unavailable grade
FOG = (242, 242, 242)   # graded midpoint
# The viewer encodes each graded response level as one character.
LEVEL_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
# Distinguish two zero grades from a loss without inventing a category meaning.
# These colors represent categories rather than a graded scale.
PAIR_LEVELS = [
    {"value": 0.0, "label": "second model preferred", "color": "#d62728"},
    {"value": 0.0, "label": "both grades are 0", "color": "#4a4a4a"},
    {"value": 0.5, "label": "tie", "color": "#b0b0b0"},
    {"value": 1.0, "label": "first model preferred", "color": "#1f77b4"},
]
P_LOSS, P_BOTHZERO, P_TIE, P_WIN = range(4)
CAP_WIDTH = 32000  # browser canvas width limit
MIN_WIDTH = 1000
MIN_HEIGHT = 600
MAX_ASPECT = 10
# Advisory size threshold: sparse grids can compress well despite many cells.
MAX_JOINED_CELLS = 2_000_000
MAX_DENSE_CELLS = 250_000_000
MAX_DENSE_ROWS = 10_000
# Answer display
TRACE_MAX_CHARS = 50000  # cap long execution logs
LOOKUP_BUCKETS = 256
DYNAMIC_ITEM_COUNT = 20
INLINE_IMAGE = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")

def categories_for(domain) -> list[str]:
    present = {str(d).strip().lower() for d in (domain or [])}
    return [d for d in DOMAIN_ORDER if d in present] or ["general"]


# Hugging Face inputs

@lru_cache(maxsize=None)
def source_revision() -> str:
    """Use one repository commit for all tables in a render."""
    return HfApi().dataset_info(HF_REPO, revision=HF_REVISION or "main").sha


class TableSource(NamedTuple):
    directory: str
    files: frozenset[str]


@lru_cache(maxsize=None)
def source_tables(slug: str) -> TableSource:
    api = HfApi()
    directory = slug
    entries = list(api.list_repo_tree(
        HF_REPO, repo_type="dataset", revision=source_revision(), path_in_repo=directory))
    if any(entry.path == f"{slug}/formatted_tables" for entry in entries):
        directory = f"{slug}/formatted_tables"
        entries = list(api.list_repo_tree(
            HF_REPO, repo_type="dataset", revision=source_revision(), path_in_repo=directory))
    return TableSource(directory, frozenset(Path(entry.path).name for entry in entries
                                           if entry.path.endswith(".parquet")))


def fetch(filename: str, cache_dir: Path, refresh: bool = False) -> Path:
    slug, name = filename.split("/", 1)
    source = f"{source_tables(slug).directory}/{name}"
    if source.endswith("/responses.parquet"):
        source = source.removesuffix("responses.parquet") + RESPONSE_FILE
    try:
        downloaded = hf_hub_download(
            HF_REPO, source, repo_type="dataset", revision=source_revision(),
            force_download=refresh,
        )
    except LocalEntryNotFoundError:
        raise
    except EntryNotFoundError as exc:
        if source.endswith("/responses.parquet"):
            downloaded = fetch(filename.removesuffix("responses.parquet") + "response.parquet",
                               cache_dir, refresh)
        else:
            raise SystemExit(f"'{source}' not found in {HF_REPO}.") from exc

    # Local analyses use the requested name, including after a legacy fallback.
    dest = cache_dir / filename
    target = Path(downloaded).resolve()
    if dest.resolve() != target:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dest.parent) as staging:
            link = Path(staging) / dest.name
            link.symlink_to(target)
            link.replace(dest)
    return dest


def list_slugs() -> list[str]:
    hidden = set(json.loads(HIDDEN_PATH.read_text())) if HIDDEN_PATH.exists() else set()
    roots = HfApi().list_repo_tree(HF_REPO, repo_type="dataset", revision=source_revision())
    available = []
    for entry in roots:
        if not isinstance(entry, RepoFolder) or entry.path in hidden:
            continue
        slug = entry.path
        if "benchmarks.parquet" not in source_tables(slug).files:
            continue
        metadata = hf_hub_download(HF_REPO, f"{slug}/metadata.yaml", repo_type="dataset",
                                   revision=source_revision())
        if yaml.safe_load(Path(metadata).read_text())["benchmark"].get("release") in {"withheld", "private"}:
            continue
        available.append(slug)
    return sorted(available)


def load_overrides(web_dir: Path) -> dict:
    p = web_dir / "content" / "curated" / "benchmark-overrides.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_affiliations(web_dir: Path) -> dict:
    p = web_dir / "content" / "curated" / "benchmark-affiliations.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"benchmarks": {}, "institutions": {}}


def load_saturation_analysis(web_dir: Path) -> dict[str, bool | None]:
    path = web_dir / "content" / "generated" / "benchmark-saturation.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(slug, str)
        or (value is not None and not isinstance(value, bool))
        for slug, value in payload.items()
    ):
        raise ValueError(f"invalid saturation analysis payload: {path}")
    return payload


def institutions_for(slug: str, affil: dict) -> list[dict] | None:
    """Return the first two authors' institutions and logos, when available."""
    names = (affil["benchmarks"].get(slug) or {}).get("institutions") or []
    if not names:
        return None
    insts = affil["institutions"]
    return [{"name": n, "logo": (insts.get(n) or {}).get("logo")} for n in names]


def read_meta(slug: str, cache_dir: Path, refresh: bool) -> dict:
    tbl = pq.read_table(fetch(f"{slug}/benchmarks.parquet", cache_dir, refresh))
    rows = tbl.to_pylist()
    if not rows:
        raise SystemExit(f"{slug}/benchmarks.parquet has no rows")
    return rows[0]


# Gallery cards

def card_for(
    slug: str,
    meta: dict,
    override: dict,
    affil: dict,
    saturation: bool | None = None,
) -> dict | None:
    """Build a card from the published HF metadata and website display settings."""
    one_line_description = str(
        meta.get("one_line_description") or ""
    ).strip()
    full_description = str(meta.get("description") or "").strip()
    return {
        "slug": meta.get("benchmark_id") or slug,
        "name": meta.get("name") or slug,
        "categories": override.get("categories")
        or ([override["category"]] if override.get("category") else None)
        or categories_for(meta.get("domain")),
        "description": one_line_description or full_description,
        "license": meta.get("license") or "",
        "code": meta.get("source_url") or meta.get("dataset_source") or "",
        "paper": meta.get("paper_url") or "",
        "image": override.get("image"),  # null selects a generated thumbnail
        "fit": override.get("fit") or "cover",
        "models": int(meta.get("n_subjects") or 0),
        "items": int(meta.get("n_items") or 0),
        "responses": int(meta.get("n_responses") or 0),
        "releaseDate": meta.get("release_date") or None,
        # Legacy metadata without granularity defaults to item-level responses.
        "itemResponses": (meta.get("granularity") or "item") == "item",
        # Saturation comes from downstream analysis, not collection metadata.
        "saturation": saturation,
        "institutions": institutions_for(slug, affil),
    }


def cmd_cards(slugs: list[str], cache_dir: Path, web_dir: Path, refresh: bool) -> None:
    overrides = load_overrides(web_dir)
    affiliations = load_affiliations(web_dir)
    saturation = load_saturation_analysis(web_dir)
    cards = []
    for slug in slugs:
        meta = read_meta(slug, cache_dir, refresh)
        card = card_for(
            slug,
            meta,
            overrides.get(slug, {}),
            affiliations,
            saturation.get(slug),
        )
        if card:
            image = card["image"]
            if image and not (web_dir / "public" / image.lstrip("/")).is_file():
                card["image"] = None
            for institution in card["institutions"] or []:
                logo = institution["logo"]
                if logo and not (web_dir / "public" / logo.lstrip("/")).is_file():
                    institution["logo"] = None
            cards.append(card)
    cards.sort(key=lambda c: c["slug"])
    out = web_dir / "content" / "generated" / "benchmark-cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=1) + "\n")
    sys.stderr.write(f"✓ wrote {len(cards)} cards → {out}\n")

# Condition parsing and matrix drawing

parse_to_sel = parse_fields


def combo_key(dims: list[str], sel: dict[str, str]) -> str:
    return ";".join(f"{d}={sel.get(d, MISSING_DIM)}" for d in dims)


def sanitize(key: str) -> str:
    return key.replace(";", "_").replace("=", "_").replace("/", "_")


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def graded_color(v: float, vmin: float, vmax: float):
    """Blue → fog → red diverging ramp for non-binary scales."""
    if vmax <= vmin:
        return FOG
    t = (v - vmin) / (vmax - vmin)
    return lerp(BLUE, FOG, t * 2) if t < 0.5 else lerp(FOG, RED, (t - 0.5) * 2)


def truncate_trace(s) -> str | None:
    """Limit answer length; preserve None to distinguish unpublished answers."""
    if s is None:
        return None
    s = str(s)
    if len(s) <= TRACE_MAX_CHARS:
        return s
    cut = s[:TRACE_MAX_CHARS]
    sp = cut.rfind(" ")
    if sp > TRACE_MAX_CHARS * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + " …"


def cell_keys(keys: list, positions=None, second: bool = False) -> dict:
    """Share constant key fields; item IDs already have a chart axis."""
    suffix = "2" if second else ""
    positions = range(len(keys)) if positions is None else positions
    present = [(position, key) for position, key in zip(positions, keys)
               if isinstance(key, list)]
    if not present:
        return {}
    first = present[0][1]
    if len(present) == len(keys) and all(key[:1] + key[2:] == first[:1] + first[2:]
                                       for _, key in present):
        return {f"key{suffix}": first}
    return {f"keys{suffix}": dict(present)}


# Joined matrices


def classify_dims(obs, dims: list[str]) -> dict[str, str]:
    """Read field ownership from the source tables, never from correlations."""
    owners = obs.attrs.get("dimension_kinds", {})
    return {d: owners.get(d, "condition") for d in dims}


def block_sort_key(v: str):
    """Sort numeric values first, then text alphabetically."""
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def joined_matrix(slug: str, obs, dims: list[str],
                 row_ids: list[str], band_labels: dict,
                 band_order: list, block_pref: str | None = None,
                 is_binary: bool = True, vmin: float = 0.0,
                 vmax: float = 1.0, block_order: list | None = None,
                 block_unit: str | None = None, scale=None, item_ids: list[str] | None = None) -> dict | None:
    """Build one matrix per trial, with item blocks and subject/condition rows."""
    cond_dims = [d for d in dims if d != "trial"]
    kinds = classify_dims(obs, cond_dims)

    # Column grouping is an explicit display choice. Registry metadata never
    # invents a new layout merely because its values happen to vary.
    block_dim = block_pref if kinds.get(block_pref) == "item" else None
    if block_pref and block_dim is None:
        sys.stderr.write(f"… {slug}: blockDim {block_pref!r} is not an item field; using one task axis\n")

    o = obs[["subject_id", "item_id", "test_condition", "trial", "response"]].copy()
    o["_sel"] = obs["_selection"]
    o["_key"] = obs["_key"]
    if block_dim:
        o["_block"] = o["_sel"].map(lambda values: values.get(block_dim, MISSING_DIM))
    else:
        o["_block"] = "all"

    # Hardest items first within each block; preserve ties for stable output.
    blocks = []
    col_pos: dict[str, dict[str, int]] = {}
    declared_blocks = list(block_order or [])

    def ordered_block_key(v: str):
        if v in declared_blocks:
            return (-1, declared_blocks.index(v), "")
        return block_sort_key(v)

    for key in sorted(o["_block"].unique(), key=ordered_block_key):
        sub = o[o["_block"] == key]
        means = sub.groupby("item_id")["response"].mean()
        if scale is None or scale.average:
            means = means.sort_values(ascending=not scale or scale.domain.get("direction") != "lower_is_better", kind="stable")
        ids = means.index.tolist()
        if block_dim is None and item_ids is not None:
            ids.extend(iid for iid in item_ids if iid not in means.index)
        blocks.append({"key": key, "colIds": ids})
        col_pos[key] = {iid: i for i, iid in enumerate(ids)}

    band_dims = [d for d in cond_dims if kinds.get(d) == "condition"]
    o["_band"] = o["_sel"].map(lambda values: ";".join(
        f"{d}={values.get(d, MISSING_DIM)}" for d in band_dims))
    # Item attributes belong to columns, including those not chosen as blocks.
    row_dims = [d for d in cond_dims if kinds.get(d) != "item"]
    o["_rowcond"] = o["_sel"].map(lambda values: combo_key(row_dims, values))
    o["_slice"] = obs["_slice"] if "_slice" in obs else o["_rowcond"]
    if o.duplicated(["trial", "subject_id", "item_id", "_slice"]).any():
        raise ValueError(f"{slug}: observation identity does not define unique cells")

    n_rows = o.groupby(["trial", "subject_id", "_slice"]).ngroups
    n_cells = n_rows * sum(len(b["colIds"]) for b in blocks)
    if n_rows > MAX_DENSE_ROWS or n_cells > MAX_DENSE_CELLS:
        return session_matrix(slug, obs, {}, observation_rows=True, scale=scale)
    if n_cells > MAX_JOINED_CELLS:
        sys.stderr.write(
            f"… {slug}: joined matrix is {n_cells:,} cells "
            f"({n_rows:,} rows), {100 * len(o) / n_cells:.1f}% observed — "
            f"over the {MAX_JOINED_CELLS:,} advisory threshold; rendering anyway\n")

    # Use the PNG color ramp for graded cells; binary cells remain 0/1.
    levels: list[dict] | None = None
    level_index: dict[float, int] = {}
    if not is_binary:
        levels = scale.levels(obs["response"]) if scale else [
            {"value": float(v), "label": f"{v:g}", "color": "#%02x%02x%02x" % graded_color(v, vmin, vmax)}
            for v in sorted(obs["response"].dropna().unique())]
        vals = [level["value"] for level in levels]
        if len(vals) > len(LEVEL_CHARS):
            sys.stderr.write(
                f"… {slug}: {len(vals)} distinct response values exceeds the "
                f"{len(LEVEL_CHARS)}-level cell encoding; using observation strips\n")
            return session_matrix(slug, obs, {}, observation_rows=True, scale=scale)
        level_index = {v: i for i, v in enumerate(vals)}


    def cell_char(v) -> str:
        if pd.isna(v):
            return "?"
        f = float(v)
        if is_binary:
            return "1" if f >= 0.5 else "0"
        return LEVEL_CHARS[level_index[f]]

    block_order = [b["key"] for b in blocks]
    trials: dict[str, list] = {}
    for (trial, sid, rowcond), g in o.groupby(["trial", "subject_id", "_slice"], sort=True):
        sel = g["_sel"].iloc[0]
        row = {
            "sid": sid,
            "band": ";".join(f"{d}={sel.get(d, MISSING_DIM)}" for d in band_dims),
            "rowcond": g["_rowcond"].iloc[0],
            "blocks": {},
            "pass": float(g["response"].sum()),
            "n": int(g["response"].count()),
        }
        if not row["n"]:
            row["score"] = "No grades"
        elif scale and not scale.average:
            row["score"] = f"{row['n']} graded"
        for key, grp in g.groupby("_block"):
            ids = blocks[block_order.index(key)]["colIds"]
            # A dot marks an unobserved cell.
            bits = ["."] * len(ids)
            for iid, v in zip(grp["item_id"], grp["response"]):
                bits[col_pos[key][iid]] = cell_char(v)
            # Keep each block's full condition for trace and audio lookup.
            row["blocks"][key] = {
                "bits": "".join(bits),
                "cond": grp["_key"].iloc[0][2],
            }
            row["blocks"][key].update(cell_keys(
                grp["_key"].tolist(), [col_pos[key][iid] for iid in grp["item_id"]]))
        trials.setdefault(str(trial), []).append(row)

    if o["_slice"].nunique() == 1:
        for rows in trials.values():
            present = {row["sid"] for row in rows}
            for sid in row_ids:
                if sid not in present:
                    rows.append({"sid": sid, "band": o["_band"].iloc[0], "rowcond": "",
                                 "blocks": {block["key"]: {"bits": "." * len(block["colIds"]), "cond": None}
                                            for block in blocks}, "pass": 0, "n": 0, "score": "No grades"})

    # Keep bands together and subjects in PNG order. Avoid ranking bands
    # by score: their item sets can differ.
    declared = band_order or []
    band_features = o.drop_duplicates("_band").set_index("_band")["_sel"].to_dict()
    def rank(band: str) -> tuple:
        vals = [band_features[band].get(d, MISSING_DIM) for d in band_dims]
        for i, d in enumerate(declared):
            if d in vals:
                return (0, i)
        return (1, band)
    band_rank = {b: rank(b) for b in o["_band"].unique()}
    subj_rank = {sid: i for i, sid in enumerate(row_ids)}
    for rows in trials.values():
        rows.sort(key=lambda r: (band_rank.get(r["band"], (2, "")),
                                 subj_rank.get(r["sid"], len(subj_rank))))

    payload = {
        "bandLabels": band_labels,
        "blockDim": block_dim,
        # Optional noun for the column count, such as "environments".
        "blockUnit": block_unit,
        "bandDims": band_dims,
        "dimKinds": kinds,
        "blocks": blocks,
        "trials": trials,
        "noneColor": "#ffffff",
    }
    # The client treats an absent levels field as binary.
    if levels is not None:
        payload["levels"] = levels
    return payload


# Session strips

# Opt in with render=sessions: observational developer streams need
# per-developer timelines rather than a shared item axis.


def session_matrix(slug: str, obs, id_to_name: dict, *, observation_rows: bool = False,
                   scale=None) -> dict | None:
    """Display every attempt in order, with the same grades and keys as a matrix."""
    selections = obs["_selection"]
    if observation_rows:
        selections = pd.Series([{"agent": "observations", "user": sid, "session": "records", "turn": i}
                                for i, sid in enumerate(obs.subject_id)], index=obs.index)
    else:
        selections = selections.map(lambda fields: fields | {"agent": fields["harness"]}
                                    if "agent" not in fields and "harness" in fields else fields)
        if any(not all(dim in fields for dim in SESSION_DIMS) for fields in selections):
            return None
    o = obs[["subject_id", "item_id", "response", "_key"]].copy()
    for dim in SESSION_DIMS:
        o[dim] = selections.map(lambda fields: fields[dim])
    o = o.sort_values(list(SESSION_DIMS), kind="stable")
    rows = []
    for (agent, user), group in o.groupby(["agent", "user"], sort=True):
        values = [rounded_score(value) if pd.isna(value) else float(value) for value in group.response]
        bits = "".join("?" if value is None else "1" if value == 1 else "0" for value in values)
        segments = list(group.groupby("session", sort=True))
        count = int(group.response.count())
        total = float(group.response.sum())
        average = scale is None or scale.average
        binary = scale is None or scale.binary
        score = ("No grades" if not count else f"{count} graded" if not average else
                 f"{total:g}/{count}" if binary else f"{total / count:.3f} avg")
        row = {"sid": group.subject_id.iloc[0], "name": id_to_name.get(user, user) if observation_rows else user,
               "band": f"agent={agent}", "rowcond": "",
               "blocks": {"all": {"bits": bits, "cond": None, **cell_keys(group._key.tolist())}},
               "cols": group.item_id.tolist(), "segs": [len(part) for _, part in segments],
               "segLabels": [str(key) for key, _ in segments],
               "subjects": [id_to_name.get(sid, sid) for sid in dict.fromkeys(group.subject_id)],
               "pass": total, "n": len(group), "gradedCount": count, "score": score}
        if scale:
            row["values"] = values
            row["labels"] = [scale.label(value, iid) for value, iid in zip(values, group.item_id)]
            row["colors"] = [("#9e9e9e" if value is None else "#1f77b4" if value == 1 else "#d62728")
                             if scale.binary else scale.color(value, iid) for value, iid in zip(values, group.item_id)]
        rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: (row["band"], -row["n"], row["name"]))
    return {"bandLabels": {"agent": {"observations": "Recorded observations"}} if observation_rows else {},
            "blockDim": None, "bandDims": ["agent"], "dimKinds": {"agent": "subject"},
            "blocks": [{"key": "all", "colIds": [""] * max(row["n"] for row in rows)}],
            "trials": {"1": rows}, "rowKind": "session", "noneColor": "#ffffff",
            "observationRows": observation_rows, "graded": bool(scale and not scale.binary),
            "summary": f"{sum(row['gradedCount'] for row in rows):,} graded / {len(o):,} recorded"}


# Faceted matrices

def faceted_matrix(slug: str, obs, id_to_name: dict, facet: dict,
                  band_labels: dict | None = None,
                  is_binary: bool = True) -> dict | None:
    """Build bands with separate item axes and panels declared in the facet override.

    Panel match fields are ANDed; onlyBands restricts panel membership.
    Return None if the layout would omit or overwrite any attempt."""
    if not is_binary:
        return None
    band_dim = facet.get("band")
    panels_decl = facet.get("panels") or []
    if not band_dim or not panels_decl:
        sys.stderr.write(
            f"… {slug}: render=faceted needs facet.band and facet.panels; "
            "using the basic matrix\n")
        return None

    sel = obs["_selection"]
    o = obs[["subject_id", "item_id", "test_condition", "response"]].copy()
    o["_sel"] = sel.values
    o["_key"] = obs["_key"]
    o["_band"] = o["_sel"].map(lambda s: s.get(band_dim, MISSING_DIM))
    if o["_band"].nunique() <= 1:
        sys.stderr.write(
            f"… {slug}: render=faceted band dim '{band_dim}' has one value; "
            "the joined view already describes this\n")
        return None

    only = facet.get("onlyBands") or {}

    def panel_of(s: dict, band: str) -> str | None:
        for p in panels_decl:
            allowed = only.get(p["key"])
            if allowed is not None and band not in allowed:
                continue
            if all(s.get(k) == v for k, v in (p.get("match") or {}).items()):
                return p["key"]
        return None

    o["_panel"] = [panel_of(s, b) for s, b in zip(o["_sel"], o["_band"])]
    dropped = int(o["_panel"].isna().sum())
    o = o[o["_panel"].notna()]
    if o.duplicated(["subject_id", "item_id", "_band", "_panel"]).any():
        return None
    if o.empty:
        sys.stderr.write(
            f"… {slug}: render=faceted matched no responses; check facet.panels\n")
        return None
    if dropped:
        return None

    declared_bands = list(facet.get("bandOrder") or [])

    def band_key(v: str):
        return ((-1, declared_bands.index(v), "") if v in declared_bands
                else block_sort_key(v))

    panel_pos = {p["key"]: i for i, p in enumerate(panels_decl)}
    bands, total_cells, observed = [], 0, 0
    for bkey in sorted(o["_band"].unique(), key=band_key):
        bsub = o[o["_band"] == bkey]
        # Share one hardest-first item order across all panels in a band.
        col_ids = bsub.groupby("item_id")["response"].mean().sort_values(
            kind="stable").index.tolist()
        pos = {iid: i for i, iid in enumerate(col_ids)}
        keys = sorted(bsub["_panel"].unique(), key=lambda k: panel_pos[k])
        panels = [{"key": k,
                   "label": next(p.get("label") or k
                                 for p in panels_decl if p["key"] == k),
                   "colIds": col_ids}
                  for k in keys]

        rows = []
        for sid, g in bsub.groupby("subject_id", sort=True):
            rp = {}
            for pkey, pg in g.groupby("_panel"):
                bits = ["."] * len(col_ids)
                for iid, v in zip(pg["item_id"], pg["response"]):
                    bits[pos[iid]] = "?" if pd.isna(v) else "1" if v == 1 else "0"
                rp[pkey] = {"bits": "".join(bits),
                            "cond": pg["_key"].iloc[0][2]}
                rp[pkey].update(cell_keys(pg["_key"].tolist(), [pos[iid] for iid in pg["item_id"]]))
            rows.append({
                "sid": sid,
                "name": id_to_name.get(sid, sid),
                "band": f"{band_dim}={bkey}",
                "rowcond": "",
                "panels": rp,
                "pass": float(g["response"].sum()),
                "n": int(g["response"].count()),
            })
        # Group by panel membership to keep partially observed populations together.
        order = [p["key"] for p in panels]
        rows.sort(key=lambda r: (tuple(k not in r["panels"] for k in order),
                                 -(r["pass"] / r["n"]) if r["n"] else 0.0,
                                 r["name"]))
        cells = len(rows) * len(col_ids) * len(panels)
        total_cells += cells
        observed += int(len(bsub))
        bands.append({"key": bkey, "panels": panels, "rows": rows})

    if total_cells > MAX_JOINED_CELLS:
        sys.stderr.write(
            f"… {slug}: faceted matrix is {total_cells:,} cells, over "
            f"{MAX_JOINED_CELLS:,}; rendering anyway\n")

    payload = {
        "rowKind": "faceted",
        "bandDim": band_dim,
        "bandLabels": band_labels or {},
        # Other viewers read these keys before checking rowKind.
        "bandDims": [band_dim],
        "blockDim": None,
        "dimKinds": {},
        "blocks": [],
        "trials": {},
        "bands": bands,
    }
    return payload


# Pairwise matrices


def detect_pairwise_dim(slug: str, df, id_to_name: dict) -> str | None:
    """Only an explicitly named opponent field requests a pairwise view."""
    fields = df["_interactor_features"]
    # Old releases stored opponent in test_condition; keep that adapter local.
    if not any("opponent" in value for value in fields):
        fields = df["_condition_features"]
    return "opponent" if all("opponent" in value for value in fields) else None


def pairwise_matrix(slug: str, df, row_ids: list[str],
                    id_to_name: dict, pair_dim: str = "opponent") -> dict | None:
    """Pair unique mirrored attempts; ambiguous comparisons use the ordinary view.

    A trial is a positional repetition, not a shared comparison ID. Repeated
    attempts for the same pair/item/condition cannot be paired from row order.
    """
    names = {}
    for sid, name in id_to_name.items():
        names.setdefault(name, []).append(sid)
    rows = []
    for row in df.to_dict("records"):
        condition = row["_condition_features"].copy()
        interactors = row["_interactor_features"].copy()
        opponent = interactors.pop(pair_dim, None)
        if opponent is None:
            opponent = condition.pop(pair_dim, None)
        candidates = [opponent] if opponent in id_to_name else names.get(opponent, [])
        if len(candidates) != 1 or candidates[0] == row["subject_id"]:
            return None
        # The recorded side names participants; it is not an experimental condition.
        if condition.get("side") in {"model_a", "model_b", "modelA", "modelB", "a", "b", "A", "B"}:
            condition.pop("side")
        context = json.dumps([condition, interactors], sort_keys=True, separators=(",", ":"))
        rows.append({"sid": row["subject_id"], "opp": candidates[0], "item_id": row["item_id"],
                     "context": context, "response": row["response"], "key": row["_key"]})
    w = pd.DataFrame(rows)
    identity = ["sid", "opp", "item_id", "context"]
    if w.duplicated(identity).any():
        sys.stderr.write(f"… {slug}: repeated pairwise attempts have no shared comparison ID; showing individual observations\n")
        return None
    mirror = w.rename(columns={"sid": "opp", "opp": "sid", "response": "other", "key": "key2"})
    w = w.merge(mirror, on=identity, how="left", validate="one_to_one", indicator=True)
    if not w["_merge"].eq("both").all():
        return None
    rank = {sid: i for i, sid in enumerate(row_ids)}
    w = w[w.sid.map(rank) > w.opp.map(rank)].copy()
    if w.empty:
        return None
    outcomes = {(0., 1.): P_LOSS, (0., 0.): P_BOTHZERO, (0.5, 0.5): P_TIE, (1., 0.): P_WIN}
    codes = []
    for a, b in zip(w.response, w.other):
        if pd.isna(a) or pd.isna(b):
            codes.append(-1)
        elif (a, b) in outcomes:
            codes.append(outcomes[a, b])
        else:
            return None
    w["code"] = codes
    groups = list(w.groupby(["sid", "opp"], sort=True))
    width = max(len(group) for _, group in groups)
    if len(groups) * width > MAX_DENSE_CELLS:
        return None
    result = []
    for (a, b), group in groups:
        group = group.sort_values(["code", "item_id", "context"], ascending=[False, True, True], kind="stable")
        n = len(group)
        wins = int(group.code.eq(P_WIN).sum())
        graded = int(group.code.ge(0).sum())
        block = {"bits": "".join("?" if code < 0 else LEVEL_CHARS[code] for code in group.code) + "." * (width - n),
                 "cond": group.key.iloc[0][2], "cond2": group.key2.iloc[0][2],
                 **cell_keys(group.key.tolist()), **cell_keys(group.key2.tolist(), second=True)}
        result.append({"sid": a, "sid2": b, "name": f"{id_to_name[a]} vs {id_to_name[b]}",
                       "band": "", "rowcond": "", "blocks": {"all": block},
                       "cols": group.item_id.tolist() + [""] * (width - n),
                       "pass": wins, "n": n, "score": f"{wins}/{graded} wins" if graded else "No grades"})
    result.sort(key=lambda row: (-row["n"], row["sid"], row["sid2"]))
    return {"bandLabels": {"_pair": {"all": "items compared · position, not a shared item"}},
            "blockDim": "_pair", "bandDims": [], "dimKinds": {},
            "blocks": [{"key": "all", "colIds": [""] * width}], "trials": {"1": result},
            "levels": PAIR_LEVELS, "rowKind": "pair", "noneColor": "#ffffff"}


# Detail records and input normalization

def load_raw(slug: str, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """Load exact observation keys and separately owned registry fields."""
    path = fetch(f"{slug}/responses.parquet", cache_dir, refresh)
    columns = pq.read_schema(path).names
    key = [*KEY, *(["interactors"] if "interactors" in columns else [])]
    table = pq.read_table(path, columns=[*key, "response", *(["response_id"] if "response_id" in columns else [])])
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"{slug}/responses.parquet has no rows")
    if frame.duplicated(key).any():
        raise ValueError(f"{slug}: duplicate observation keys")
    if "response_id" in frame and (frame.response_id.isna().any() or frame.response_id.duplicated().any()):
        raise ValueError(f"{slug}: missing or duplicate response_id")
    frame["_key"] = [list(row.values()) for row in table.select(key).to_pylist()]
    registries = []
    for name, fields in [("subjects", ["subject_id", "harness", "reasoning_effort", "harness_version", "subject_features_extra"]),
                         ("items", ["item_id", "item_features"])]:
        path = fetch(f"{slug}/{name}.parquet", cache_dir, refresh)
        available = pq.read_schema(path).names
        registry = pq.read_table(path, columns=[c for c in fields if c in available]).to_pandas()
        if registry[fields[0]].isna().any() or registry[fields[0]].duplicated().any():
            raise ValueError(f"{slug}: missing or duplicate {fields[0]}")
        registries.append(registry)
    return attach_features(frame, *registries)


def detail_metadata(slug: str, meta: dict) -> dict:
    return {
        "slug": slug,
        "name": meta.get("name") or slug,
        "description": meta.get("description"),
        "domain": list(meta.get("domain") or []),
        "modality": list(meta.get("modality") or []),
        "license": meta.get("license"),
        "responseType": meta.get("response_type"),
        "responseScale": meta.get("response_scale"),
        "sourceUrl": meta.get("source_url"),
        "paperUrl": meta.get("paper_url"),
        "releaseDate": meta.get("release_date"),
    }


def build_no_responses_detail(slug: str, meta: dict, cache_dir: Path,
                              web_dir: Path, refresh: bool,
                              override: dict | None = None) -> dict:
    """Render an unobserved matrix with clickable prompts for an item bank."""
    subj_tbl = pq.read_table(
        fetch(f"{slug}/subjects.parquet", cache_dir, refresh),
        columns=["subject_id", "display_name"],
    ).to_pandas()
    id_to_name = dict(zip(subj_tbl["subject_id"], subj_tbl["display_name"]))
    row_ids = subj_tbl["subject_id"].tolist()

    path = fetch(f"{slug}/items.parquet", cache_dir, refresh)
    columns = [c for c in ["item_id", "content", "grading_criterion"] if c in pq.read_schema(path).names]
    items_tbl = pq.read_table(path, columns=columns).to_pandas()
    declared_scale = ResponseScale(meta, pd.DataFrame(columns=["item_id", "response"]), items_tbl)
    col_ids = items_tbl["item_id"].tolist()
    questions_available = items_tbl["content"].notna().any()

    n_rows, n_items = len(row_ids), len(col_ids)
    width = min(n_items, CAP_WIDTH)

    size = matrix_size(width, n_rows)
    axes = {"colIds": col_ids, "colP": None, "items": {}}
    matrix = (pd.Series(dtype=float), row_ids, col_ids,
              f"/benchmarks/matrices/{slug}.png", pd.Series(dtype=object))
    scale = ("upstream released aggregate scores only — no per-item responses"
             if meta.get("granularity") == "aggregate"
             else "upstream released no model responses")

    detail = {
        **detail_metadata(slug, meta),
        "stats": {
            "items": int(n_items),
            "subjects": int(n_rows),
            "observed": 0.0,
            "graded": 0,
            "meanResponse": None,
        },
        "matrix": f"/benchmarks/matrices/{slug}.png",
        "matrixSize": size,
        "conditions": None,
        "categories": None,
        "binaryLabels": declared_scale.binary_labels(),
        "note": (override or {}).get("note"),
        "attacks": None,
        "matrixItemsTotal": n_items,
        "matrixItemsShown": n_items,
        "matrixSampled": False,
        "matrixRows": [id_to_name.get(sid, sid) for sid in row_ids],
        "matrixRowIds": row_ids,
        "matrixRowScores": [None] * n_rows,
        "matrixColIds": None,
        "matrixColP": None,
        "hasTraces": False,
        "traceChunkPrefix": 0,
        "isBinary": declared_scale.binary,
        "valueRange": list(declared_scale.bounds),
        "scaleLabel": f"{declared_scale.description()} · {scale}",
        "questionsAvailable": bool(questions_available),
        "subjects": [{"name": id_to_name.get(sid, sid), "score": None}
                     for sid in row_ids],
    }

    return {"detail": detail, "chart": None, "axes": axes, "matrices": {"": matrix}}


class SlicePlan(NamedTuple):

    dims: list[str]
    values: dict[str, list[str]]
    conditions: list[str]
    trial_multi: bool
    parse: Callable
    observations: pd.DataFrame


def plan_slices(slug: str, df: pd.DataFrame, sessions_mode: bool = False) -> SlicePlan:
    """Select actual conditions and trials without removing any observations."""
    selections = {}
    kinds = df.attrs["dimension_kinds"]
    for label, features in zip(df["_slice"], df["_features"]):
        selections[label] = {k: v for k, v in features.items() if kinds[k] == "condition"}
    conditions = sorted(selections)
    dims = [key for key, owner in kinds.items() if owner == "condition"] if len(conditions) > 1 else []
    # Distinct original strings can parse to identical display fields (or to
    # the literal missing-value label). Keep those recorded occasions separate.
    if len({combo_key(dims, fields) for fields in selections.values()}) < len(conditions):
        name = "recorded_condition"
        while name in kinds:
            name = "_" + name
        dims.append(name)
        kinds = {**kinds, name: "condition"}
        selections = {key: {**fields, name: key} for key, fields in selections.items()}
    values = {d: sorted({fields.get(d, MISSING_DIM) for fields in selections.values()}) for d in dims}
    trials = sorted(df["trial"].unique().tolist())
    trial_multi = len(trials) > 1
    if trial_multi:
        dims.append("trial")
        values["trial"] = [str(t) for t in trials]
    observations = df.assign(_selection=[{**features, **selections[label]}
                                          for features, label in zip(df["_features"], df["_slice"])])
    observations.attrs["dimension_kinds"] = kinds
    return SlicePlan(dims, values, conditions, trial_multi,
                     lambda label: selections[label].copy(), observations)


def matrix_size(width: int, rows: int) -> list[int]:
    width, rows = max(1, width), max(1, rows)
    width *= max(1, -(-MIN_WIDTH // width))
    return [width, rows * max(1, round(max(MIN_HEIGHT, width // MAX_ASPECT) / rows))]


def rounded_score(value) -> float | None:
    return round(float(value), 4) if pd.notna(value) else None


def prepare_matrix_slices(slug: str, df: pd.DataFrame, plan: SlicePlan,
                          row_ids: list[str], col_ids: list[str],
                          *, summary_only: bool = False, scale=None) -> tuple:
    """Build only existing condition/trial slices, retaining ungraded attempts."""
    row_pos = {sid: i for i, sid in enumerate(row_ids)}
    col_pos = {iid: i for i, iid in enumerate(col_ids)}
    groups = list(df.groupby(["_slice", "trial"], sort=True))
    if summary_only:
        groups = [max(groups, key=lambda pair: len(pair[1]))]
    matrices, metadata, columns = {}, {}, {}
    best = None
    ascending = not scale or scale.domain.get("direction") != "lower_is_better"
    for (condition, trial), observed in groups:
        cell = observed.set_index(["subject_id", "item_id"])["response"]
        if not cell.index.is_unique:
            raise ValueError(f"{slug}: duplicate cells in an observation slice")
        row_mean = observed.groupby("subject_id")["response"].mean().sort_values(ascending=ascending, kind="stable")
        col_mean = observed.groupby("item_id")["response"].mean().sort_values(ascending=ascending, kind="stable")
        rows = row_mean.index.tolist() if plan.dims else row_ids
        cols = col_mean.index.tolist() if plan.dims else col_ids
        sel = plan.parse(condition)
        if plan.trial_multi:
            sel["trial"] = str(trial)
        key = combo_key(plan.dims, sel)
        if key in matrices:
            raise ValueError(f"{slug}: condition selectors do not identify a unique slice")
        path = f"/benchmarks/matrices/{slug}/{sanitize(key)}.png" if plan.dims else f"/benchmarks/matrices/{slug}.png"
        size = matrix_size(min(len(cols), CAP_WIDTH), len(rows))
        matrices[key] = (cell, rows, cols, path, observed.set_index(["subject_id", "item_id"])["_key"])
        score = lambda value: rounded_score(value) if scale is None or scale.average else None
        metadata[key] = {"sel": sel, "matrix": path, "matrixSize": size,
                         "rowIdx": [row_pos[sid] for sid in rows],
                         "rowScores": [score(row_mean.get(sid)) for sid in rows],
                         "nItems": len(cols), "colP": None,
                         "observed": round(cell.notna().sum() / max(1, len(rows) * len(cols)), 4)}
        columns[key] = {"colIdx": [col_pos[iid] for iid in cols],
                        "colP": [score(col_mean.get(iid)) for iid in cols]}
        if best is None or len(cell) > best[0]:
            best = (len(cell), sel, path, size)
    conditions = {"dims": plan.dims, "values": plan.values,
                  "default": {d: best[1].get(d, MISSING_DIM) for d in plan.dims},
                  "collapsed": None, "matrices": metadata} if plan.dims else None
    return best[2], best[3], conditions, None, columns if plan.dims else {}, matrices


def build_detail(
    slug: str, cache_dir: Path, web_dir: Path, refresh: bool, overrides: dict,
    *, compact: bool = False,
) -> dict:
    override = overrides.get(slug, {})
    meta = read_meta(slug, cache_dir, refresh)
    if (meta.get("granularity") or "item") != "item":
        return build_no_responses_detail(slug, meta, cache_dir, web_dir, refresh, override)
    df = load_raw(slug, cache_dir, refresh)
    subjects = pq.read_table(fetch(f"{slug}/subjects.parquet", cache_dir, refresh),
                             columns=["subject_id", "display_name"]).to_pandas()
    path = fetch(f"{slug}/items.parquet", cache_dir, refresh)
    columns = [c for c in ["item_id", "content", "grading_criterion"] if c in pq.read_schema(path).names]
    items = pq.read_table(path, columns=columns).to_pandas()
    scale = ResponseScale(meta, df, items)
    id_to_name = dict(zip(subjects.subject_id, subjects.display_name))
    plan = plan_slices(slug, df)
    obs = plan.observations
    ascending = scale.domain.get("direction") != "lower_is_better"
    subject_mean = obs.groupby("subject_id").response.mean()
    item_mean = obs.groupby("item_id").response.mean()
    if scale.average:
        subject_mean = subject_mean.sort_values(ascending=ascending, kind="stable")
        item_mean = item_mean.sort_values(ascending=ascending, kind="stable")
    row_ids = subject_mean.index.tolist() + sorted(set(subjects.subject_id) - set(subject_mean.index))
    col_ids = item_mean.index.tolist() + sorted(set(items.item_id) - set(item_mean.index))
    score = lambda value: rounded_score(value) if scale.average else None
    dims = list(obs.attrs["dimension_kinds"])
    mode = override.get("render")
    joined = None
    if scale.mixed or mode == "observations":
        joined = session_matrix(slug, obs, id_to_name, observation_rows=True, scale=scale)
    elif mode == "sessions":
        joined = session_matrix(slug, obs, id_to_name, scale=scale)
    elif mode == "faceted":
        joined = faceted_matrix(slug, obs, id_to_name, override.get("facet") or {},
                                override.get("bandLabels") or {}, scale.binary)
    elif scale.response_type == "win_rate" and scale.domain.get("direction") not in {"lower_is_better", "unordered"}:
        pair_dim = override.get("pairwise") or detect_pairwise_dim(slug, df, id_to_name)
        if pair_dim:
            joined = pairwise_matrix(slug, df, row_ids, id_to_name, pair_dim)
    # Every layout uses the same attempts. A declined specialization returns to
    # the ordinary view, which preserves ambiguous/unmatched pairwise records.
    if joined is None and override.get("joined", True) and df.response.notna().any():
        joined = joined_matrix(slug, obs, dims, row_ids, override.get("bandLabels") or {},
                               override.get("bandOrder") or [], override.get("blockDim"),
                               scale.binary, *scale.bounds, block_order=override.get("blockOrder"),
                               block_unit=override.get("blockUnit"), scale=scale, item_ids=col_ids)
    if joined and joined.get("observationRows"):
        for row in joined["trials"]["1"]:
            row["name"] = id_to_name.get(row["sid"], row["sid"])
    matrix_path, size, conditions, attacks, slice_cols, matrices = prepare_matrix_slices(
        slug, df, plan, row_ids, col_ids, summary_only=bool(joined), scale=scale)
    axes = {"colIds": col_ids, "colP": [score(item_mean.get(iid)) for iid in col_ids], "items": {}}
    if slice_cols:
        axes["slices"] = slice_cols
    graded = df.dropna(subset=["response"])
    coverage = len(graded.drop_duplicates(["subject_id", "item_id"])) / max(1, len(subjects) * len(items))
    binary_labels = scale.binary_labels()
    if binary_labels:
        meanings = scale.domain.get("meanings", {})
        for name, value in [("zero", "0"), ("one", "1")]:
            binary_labels[name] = meanings.get(value, (override.get("binaryLabels") or {}).get(name, value))
    detail = {
        **detail_metadata(slug, meta),
        "stats": {"items": len(items), "subjects": len(subjects), "observed": round(coverage, 4), "graded": len(graded),
                  "meanResponse": score(df.response.mean())},
        "matrix": matrix_path, "matrixSize": size, "conditions": conditions,
        "categories": scale.levels(df.response) if scale.domain.get("direction") == "unordered" else None,
        "binaryLabels": binary_labels, "note": override.get("note"), "attacks": attacks,
        "matrixItemsTotal": len(items), "matrixItemsShown": len(col_ids), "matrixSampled": False,
        "matrixRows": [id_to_name[sid] for sid in row_ids], "matrixRowIds": row_ids,
        "matrixRowScores": [score(subject_mean.get(sid)) for sid in row_ids],
        "matrixColIds": None, "matrixColP": None,
        "hasTraces": "traces.parquet" in source_tables(slug).files,
        "hasAudio": bool(override.get("audio")),
        **({"audio": override["audio"]} if override.get("audio") else {}),
        "hasJoined": bool(joined), "traceChunkPrefix": 0,
        "isBinary": scale.binary, "valueRange": list(scale.bounds),
        "scaleLabel": scale.description(),
        "questionsAvailable": bool(items.content.notna().any()),
        "subjects": [{"name": id_to_name[sid], "score": score(subject_mean.get(sid))} for sid in reversed(row_ids)],
    }
    return {"detail": detail, "chart": joined, "axes": axes, "matrices": matrices}


def render_benchmarks(slugs: list[str], cache_dir: Path, web_dir: Path, refresh: bool) -> None:
    overrides = load_overrides(web_dir)
    details_path = web_dir / "content" / "generated" / "benchmark-details.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    built = 0
    failed: list[str] = []
    for slug in slugs:
        # Leave failed benchmarks unchanged so a later run can retry them.
        try:
            entry = build_detail(slug, cache_dir, web_dir, refresh, overrides)["detail"]
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            failed.append(slug)
            sys.stderr.write(f"✗ {slug}: skipped — {type(exc).__name__}: {exc}\n")
            continue
        details[slug] = entry
        built += 1
        drawn = (f"{len(entry['conditions']['matrices'])} condition matrices"
                 if entry.get("conditions") else "1 matrix")
        traced = "traces" if entry["hasTraces"] else "no traces"
        sys.stderr.write(f"✓ {slug}: {entry['stats']['subjects']}×{entry['stats']['items']}, "
                         f"{drawn}, {traced}\n")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=1))
    msg = f"✓ details: {built} rendered → {details_path}\n"
    if failed:
        msg += f"✗ {len(failed)} failed and skipped: {', '.join(failed)}\n"
    sys.stderr.write(msg)


def read_item(slug: str, cache_dir: Path, item_id: str) -> dict:
    path = fetch(f"{slug}/items.parquet", cache_dir)
    columns = [c for c in ["content", "asset_manifest", "grading_criterion", "reference_answer", "correct_answer"]
               if c in pq.read_schema(path).names]
    rows = pq.read_table(path, columns=columns,
                         filters=[("item_id", "=", item_id)]).to_pylist()
    if len(rows) != 1:
        raise ValueError("Item does not identify exactly one source row")
    assets = item_assets(slug, cache_dir, [rows[0].get("asset_manifest")])
    return item_content(rows[0], assets)


def input_images(manifest) -> list[dict]:
    return [link for link in json.loads(manifest or "[]")
            if link["role"] in {"input", "input_image"} and link["media_type"].startswith("image/")]


def item_assets(slug: str, cache_dir: Path, manifests, output: Path | None = None) -> dict:
    media = {link["asset_id"]: link["media_type"] for value in manifests for link in input_images(value)}
    ids = set(media)
    if not ids:
        return {}
    rows = pq.read_table(fetch(f"{slug}/assets.parquet", cache_dir),
                         columns=["asset_id", "data"], filters=[("asset_id", "in", sorted(ids))]).to_pylist()
    assets = {row["asset_id"]: row["data"] for row in rows}
    if set(assets) != ids or len(assets) != len(rows):
        raise ValueError(f"{slug}: missing or duplicate image assets")
    if output is not None:
        (output / "asset").mkdir(parents=True)
        for index, (asset_id, data) in enumerate(assets.items()):
            name = f"{index}{mimetypes.guess_extension(media[asset_id]) or '.bin'}"
            (output / "asset" / name).write_bytes(data)
            assets[asset_id] = f"/benchmark-data/{slug}/asset/{name}"
    return assets


def item_content(row: dict, assets: dict | None = None) -> dict:
    answer = row.get("reference_answer", row.get("correct_answer"))
    if row.get("grading_criterion") is not None:
        answer = json.loads(row["grading_criterion"])["reference_answer"]
    content = row.get("content")
    for link in input_images(row.get("asset_manifest")):
        source = (assets or {})[link["asset_id"]]
        if isinstance(source, bytes):
            source = f"data:{link['media_type']};base64,{base64.b64encode(source).decode('ascii')}"
        content = (content or "") + f"\n\n![Image {link['ordinal']}]({source})"
    return {"content": content,
            "answer": str(answer) if answer is not None else None}


def read_answer(slug: str, cache_dir: Path, key: list) -> dict:
    """Look up the exact source key supplied by the clicked cell."""
    if (not isinstance(key, list) or len(key) not in (4, 5)
            or not all(isinstance(value, str) for value in key[:2])
            or not isinstance(key[3], int) or isinstance(key[3], bool)
            or any(value is not None and not isinstance(value, str)
                   for value in [key[2], *key[4:]])):
        raise ValueError("Invalid observation key")
    try:
        path = fetch(f"{slug}/traces.parquet", cache_dir)
    except SystemExit:
        return {"trace": None}
    columns = [*KEY, *(["interactors"] if "interactors" in pq.read_schema(path).names else [])]
    if len(columns) != len(key):
        raise ValueError("Observation key does not match the source schema")
    predicate = ds.scalar(True)
    for column, value in zip(columns, key):
        predicate = predicate & (ds.field(column).is_null() if value is None
                                 else ds.field(column) == value)
    rows = pq.read_table(path, columns=["trace"], filters=predicate).to_pylist()
    return {"trace": truncate_trace(rows[0]["trace"]) if rows else None}


def chart_bundle(view: dict) -> dict:
    """Bundle the selected layout and its axes without embedding prompt or answer text."""
    detail = view["detail"]
    axes = {**view["axes"], "slices": view["axes"].get("slices")}
    if view["chart"]:
        axes["slices"] = None
        if detail.get("conditions"):
            detail = {**detail, "conditions": {**detail["conditions"], "matrices": {}}}
    matrices = {}
    if not view["chart"]:
        for name, (values, rows, cols, _, keys) in view["matrices"].items():
            row_pos = {sid: i for i, sid in enumerate(rows)}
            col_pos = {iid: i for i, iid in enumerate(cols)}
            width = min(len(cols), CAP_WIDTH)
            pixels, observations, grades = [], {}, {}
            categories = {entry["value"]: entry["color"] for entry in detail.get("categories") or []}
            for (sid, iid), value in values.items():
                r, c = row_pos[sid], col_pos[iid]
                color = GRAY if pd.isna(value) else (BLUE if value == 1 else RED) if detail["isBinary"] else \
                    graded_color(float(value), *detail["valueRange"])
                if value in categories:
                    color = tuple(int(categories[value][i:i+2], 16) for i in (1, 3, 5))
                pixels.append([r * width + c * width // len(cols), *color])
                observations[r * len(cols) + c] = keys.loc[(sid, iid)]
                grades[r * len(cols) + c] = None if pd.isna(value) else float(value)
            matrices[name] = {"width": width, "height": len(rows), "pixels": pixels,
                              "keys": observations, "grades": grades, "displayColors": True}
    return {"detail": detail, "chart": view["chart"] or None,
            "axes": axes,
            "matrices": matrices}


def write_lookup(directory: Path, rows) -> int:
    """Partition exact keys into compressed files that the browser can fetch."""
    directory.mkdir(parents=True)
    seen = set()
    with ExitStack() as stack:
        files = [stack.enter_context(gzip.open(directory / f"{i:02x}.json.gz", "wt",
                                               encoding="utf-8", compresslevel=6))
                 for i in range(LOOKUP_BUCKETS)]
        counts = [0] * LOOKUP_BUCKETS
        for output in files:
            output.write("{")
        for key, value in rows:
            encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if encoded in seen:
                raise ValueError(f"Duplicate {directory.name} key: {encoded}")
            seen.add(encoded)
            bucket = hashlib.sha256(encoded.encode("utf-8")).digest()[0]
            output = files[bucket]
            if counts[bucket]:
                output.write(",")
            output.write(json.dumps(encoded, ensure_ascii=False) + ":")
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            counts[bucket] += 1
        for output in files:
            output.write("}")
    for index, count in enumerate(counts):
        if not count:
            (directory / f"{index:02x}.json.gz").unlink()
    return len(seen)


def source_row_keys(path: Path, columns: list[str]):
    table = pq.ParquetFile(path)
    for group in range(table.num_row_groups):
        offset = 0
        for batch in table.iter_batches(row_groups=[group], columns=columns):
            for row in batch.to_pylist():
                key = row[columns[0]] if len(columns) == 1 else [row[c] for c in columns]
                yield key, [group, offset]
                offset += 1


def write_source_pages(path: Path, columns: list[str], output: Path, name: str) -> dict:
    """Index original Parquet pages so a click need not download a whole column chunk."""
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        length = struct.unpack("<I", stream.read(4))[0]
        stream.seek(-length - 8, 2)
        metadata = ThriftObject.from_buffer(stream.read(length), "FileMetaData")

        def groups():
            for number, group in enumerate(metadata.row_groups):
                pages = {}
                for chunk in group.columns:
                    column = chunk.meta_data
                    field = ".".join(p.decode() if isinstance(p, bytes) else p
                                     for p in column.path_in_schema)
                    if field not in columns:
                        continue
                    offset = column.dictionary_page_offset or column.data_page_offset
                    end = offset + column.total_compressed_size
                    row, locations = 0, []
                    while offset < end:
                        stream.seek(offset)
                        reader = NumpyIO(stream.read(min(65536, end - offset)))
                        header = ThriftObject.from_buffer(reader, "PageHeader")
                        size = reader.tell() + header.compressed_page_size
                        if header.type in (0, 3):
                            data = header.data_page_header if header.type == 0 else header.data_page_header_v2
                            locations.append([offset, size, row])
                            row += data.num_values
                        offset += size
                    if row != group.num_rows or offset != end:
                        raise ValueError(f"{path}: unsupported page layout for {field}")
                    pages[field] = locations
                footer = ThriftObject.from_fields(
                    "FileMetaData", version=metadata.version, schema=metadata.schema,
                    num_rows=group.num_rows, row_groups=[group], key_value_metadata=[]).to_bytes()
                footer = bytes(footer) + struct.pack("<I", len(footer)) + b"PAR1"
                yield str(number), {"footer": base64.b64encode(footer).decode(), "pages": pages}

        write_lookup(output / f"{name}-pages", groups())
    return {"size": path.stat().st_size, "columns": columns}


def static_item(slug: str, row: dict) -> dict:
    assets = {link["asset_id"]: f"/benchmark-data/{slug}/image/asset/"
              f"{quote(link['asset_id'], safe='')}?type={quote(link['media_type'], safe='')}"
              for link in input_images(row.get("asset_manifest"))}
    item = item_content(row, assets)
    for field, value in item.items():
        if value is not None:
            index = iter(range(len(INLINE_IMAGE.findall(value))))
            item[field] = INLINE_IMAGE.sub(
                lambda _: f"/benchmark-data/{slug}/image/item/{quote(row['item_id'], safe='')}/"
                          f"{field}/{next(index)}", value)
    return item


def write_website_lookups(slug: str, cache_dir: Path, output: Path, has_traces: bool) -> int:
    source = source_tables(slug)
    descriptor = {"revision": source_revision(), "directory": source.directory,
                  "dynamicItems": False, "tables": {}}
    items_path = fetch(f"{slug}/items.parquet", cache_dir)
    items = pq.ParquetFile(items_path)
    columns = [c for c in ["content", "asset_manifest", "grading_criterion",
                          "reference_answer", "correct_answer"] if c in items.schema_arrow.names]
    descriptor["tables"]["items"] = write_source_pages(items_path, columns, output, "items")
    image_ids = set()

    def item_rows():
        for batch in items.iter_batches(batch_size=128, columns=["item_id", *columns]):
            for row in batch.to_pylist():
                image_ids.update(link["asset_id"] for link in input_images(row.get("asset_manifest")))
                yield row["item_id"], static_item(slug, row)

    write_lookup(output / "item", item_rows())
    # Inline images need an item row even when its text is shipped statically.
    write_lookup(output / "item-rows", source_row_keys(items_path, ["item_id"]))
    if has_traces:
        path = fetch(f"{slug}/traces.parquet", cache_dir)
        descriptor["tables"]["traces"] = write_source_pages(path, ["trace"], output, "traces")
        key = [*KEY, *(["interactors"] if "interactors" in pq.read_schema(path).names else [])]
        write_lookup(output / "answer", source_row_keys(path, key))
    if image_ids:
        path = fetch(f"{slug}/assets.parquet", cache_dir)
        descriptor["tables"]["assets"] = write_source_pages(path, ["data"], output, "assets")
        assets = [(key, position) for key, position in source_row_keys(path, ["asset_id"])
                  if key in image_ids]
        if {key for key, _ in assets} != image_ids:
            raise ValueError(f"{slug}: missing image assets")
        write_lookup(output / "asset-rows", assets)
    descriptor["itemBytes"] = sum(p.stat().st_size for p in (output / "item").iterdir())
    with gzip.open(output / "source.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(descriptor, stream, separators=(",", ":"))
    return descriptor["itemBytes"]


def select_dynamic_items(root: Path, sizes: dict[str, int]) -> list[str]:
    selected = sorted(sizes, key=lambda slug: (-sizes[slug], slug))[:DYNAMIC_ITEM_COUNT]
    for slug in selected:
        output = root / slug
        with gzip.open(output / "source.json.gz", "rt", encoding="utf-8") as stream:
            source = json.load(stream)
        source["dynamicItems"] = True
        with gzip.open(output / "source.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(source, stream, separators=(",", ":"))
        shutil.rmtree(output / "item")
    print(f"Dynamic item content ({len(selected)}): {', '.join(selected)}", flush=True)
    catalog = {}
    for slug in sizes:
        with gzip.open(root / slug / "source.json.gz", "rt", encoding="utf-8") as stream:
            catalog[slug] = json.load(stream)
    target = root.parent.parent / "content/generated/gallery-sources.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, indent=1) + "\n")
    return selected


def separate_cell_keys(bundle: dict) -> list:
    lookups = []

    def visit(node):
        if isinstance(node, dict):
            if "bits" in node or "pixels" in node:
                keys = {name: node.pop(name) for name in ("key", "key2", "keys", "keys2")
                        if name in node}
                if keys:
                    reference = str(len(lookups))
                    node["keyRef"] = reference
                    lookups.append((reference, keys))
            for name, child in node.items():
                if name != "pixels":
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(bundle["chart"])
    visit(bundle["matrices"])
    return lookups


def build_website_data(slugs: list[str], cache_dir: Path, web_dir: Path,
                       refresh: bool = False) -> dict:
    """Build deployable viewer files; no source tables are copied into the website."""
    root = web_dir / "public" / "benchmark-data"
    root.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(web_dir)
    details = {}
    item_sizes = {}
    for slug in slugs:
        start = time.monotonic()
        bundle = chart_bundle(build_detail(slug, cache_dir, web_dir, refresh, overrides,
                                          compact=True))
        # The catalog needs summaries; selectors stay in the per-benchmark bundle.
        details[slug] = {key: value for key, value in bundle["detail"].items()
                         if key != "conditions"}
        with TemporaryDirectory(dir=root, prefix=".build-") as staging:
            output = Path(staging)
            write_lookup(output / "keys", separate_cell_keys(bundle))
            with gzip.open(output / "view.json.gz", "wt", encoding="utf-8", compresslevel=6) as stream:
                json.dump(bundle, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            item_sizes[slug] = write_website_lookups(
                slug, cache_dir, output, bundle["detail"]["hasTraces"])
            destination = root / slug
            if destination.exists():
                shutil.rmtree(destination)
            output.rename(destination)
        size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        print(f"{slug}: {size / 1_000_000:.1f} MB, {time.monotonic() - start:.1f}s", flush=True)
    select_dynamic_items(root, item_sizes)
    return details



# CLI

def resolve_slugs(arg_slug: str | None, want_all: bool) -> list[str]:
    if want_all or arg_slug is None:
        return list_slugs()
    return [arg_slug]


def main() -> None:
    global HF_REVISION, RESPONSE_FILE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode",
                    choices=["cards", "details", "all", "list", "build"])
    ap.add_argument("slug", nargs="?", help="benchmark slug (omit + --all for every benchmark)")
    ap.add_argument("--all", action="store_true", help="apply to every published benchmark")
    ap.add_argument("--web-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "website",
                    help="path to the Next app (default: repo website/)")
    ap.add_argument("--cache-dir", type=Path,
                    default=SCRIPT_DIR / ".hf-cache")
    ap.add_argument("--refresh", action="store_true", help="force re-download")
    ap.add_argument("--revision", help="HF branch, tag, or commit (default: main)")
    ap.add_argument("--response-file", choices=["responses.parquet", "response.parquet"],
                    default=RESPONSE_FILE, help="preferred filename (default: responses.parquet)")
    args = ap.parse_args()
    HF_REVISION, RESPONSE_FILE = args.revision or HF_REVISION, args.response_file
    source_revision.cache_clear()
    source_tables.cache_clear()
    published = list_slugs()
    if args.mode == "build":
        hidden = set(json.loads((args.web_dir / "content/curated/hidden-benchmarks.json").read_text()))
        expected = set(published) - hidden
        selected = set(args.slug.split(",")) if args.slug else expected
        if selected - set(published) or selected & hidden:
            ap.error(f"Unpublished or unavailable benchmarks: {sorted(selected - set(published) | selected & hidden)}; "
                     f"only {HF_REPO} is used")
        print(f"Source: {HF_REPO}@{source_revision()}", flush=True)
        selected = sorted(selected)
        details = build_website_data(selected, args.cache_dir, args.web_dir, args.refresh)
        cmd_cards(selected, args.cache_dir, args.web_dir, args.refresh)
        (args.web_dir / "content/generated/benchmark-details.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
        root = args.web_dir / "public/benchmark-data"
        for folder in root.iterdir():
            if folder.is_dir() and folder.name not in selected:
                shutil.rmtree(folder)
        return
    if args.mode == "list":
        for s in list_slugs():
            print(s)
        return

    if args.mode == "details" and not args.slug and not args.all:
        ap.error("details needs a slug or --all")

    slugs = resolve_slugs(args.slug if args.mode != "cards" else None,
                          args.all or args.mode in ("cards", "all"))

    if args.mode in ("cards", "all"):
        cmd_cards(list_slugs(), args.cache_dir, args.web_dir, args.refresh)
        # Model timeline and other analyses are generated separately.
    if args.mode in ("details", "all"):
        render_benchmarks(slugs, args.cache_dir, args.web_dir, args.refresh)


if __name__ == "__main__":
    main()
