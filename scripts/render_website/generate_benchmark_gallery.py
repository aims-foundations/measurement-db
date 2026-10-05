#!/usr/bin/env python3
"""Render the gallery from HF tables. See README.md for the local server."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Callable, NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from huggingface_hub.hf_api import RepoFolder

# Paths and source
HF_REPO = "aims-foundations/measurement-db"
HF_REVISION = None
HF_REPOS: dict[str, str | None] = {}
BENCHMARK_REPOS: dict[str, str] = {}
RESPONSE_FILE = "responses.parquet"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
GENERATED_CONTENT_DIR = REPO_DIR / "website" / "content" / "generated"
FILTERED_PATH = GENERATED_CONTENT_DIR / "published-benchmarks.json"

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
SUBJECT_FEATURE_COLS = ["harness", "reasoning_effort", "harness_version"]
SESSION_DIMS = ("agent", "user", "session", "turn")
# One cell represents one response at this key. load_raw also includes
# interactors when present, before projecting them into display labels.
KEY = ["subject_id", "item_id", "test_condition", "trial"]
# Value for a dimension absent from one condition.
MISSING_DIM = "n/a"
# Use a string sentinel so null conditions remain selectable and groupable.
NO_COND = "(none)"

# Matrix rendering
PLACEHOLDER_IMAGE = "/benchmarks/_placeholder.svg"
# Binary PNG colors also encode cell values for matrix-viewer.tsx.
# The viewer maps red/blue/gray to blue/red/white; keep these source colors.
RED = (214, 39, 40)  # correct / higher; displayed blue for binary matrices
BLUE = (31, 119, 180)  # incorrect / lower; displayed red for binary matrices
GRAY = (158, 158, 158)  # unobserved; displayed white for binary matrices
FOG = (242, 242, 242)   # graded midpoint
# The viewer encodes each graded response level as one character.
LEVEL_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
# Separate "both bad" from a loss: both are scored zero in the source.
# These colors represent categories rather than a graded scale.
PAIR_LEVELS = [
    {"value": 0.0, "label": "second model preferred", "color": "#d62728"},
    {"value": 0.0, "label": "both bad", "color": "#4a4a4a"},
    {"value": 0.5, "label": "tie", "color": "#b0b0b0"},
    {"value": 1.0, "label": "first model preferred", "color": "#1f77b4"},
]
P_LOSS, P_BOTHBAD, P_TIE, P_WIN = range(4)
CAP_WIDTH = 32000  # browser canvas width limit
MIN_WIDTH = 1000
MIN_HEIGHT = 600
MAX_ASPECT = 10
# Limit generated PNGs; omitted trials remain in the source tables.
MAX_TRIALS_RENDERED = 16
MAX_SLICES = 600  # condition combinations × trials
# Advisory size threshold: sparse grids can compress well despite many cells.
MAX_JOINED_CELLS = 2_000_000
# Advisory block count before horizontal scrolling is needed.
MAX_BLOCKS = 17
# Allow a few multi-valued IDs when identifying item/subject attributes.
DIM_ATTR_TOLERANCE = 0.05
# Minimum share of items answered by at most two subjects.
PAIRWISE_DEGENERATE_SHARE = 0.90
# Allow a few unmatched opponent names after roster changes.
PAIRWISE_NAME_MATCH = 0.95

# Answer display
TRACE_MAX_CHARS = 50000  # cap long execution logs

def categories_for(domain) -> list[str]:
    present = {str(d).strip().lower() for d in (domain or [])}
    return [d for d in DOMAIN_ORDER if d in present] or ["general"]


# Hugging Face inputs

@lru_cache(maxsize=None)
def source_revision(repo: str | None = None) -> str:
    """Use one repository commit for all tables in a render."""
    repo = repo or HF_REPO
    return HF_REPOS.get(repo) or (HF_REVISION if repo == HF_REPO else None) or HfApi().dataset_info(repo).sha


def fetch(filename: str, cache_dir: Path, refresh: bool = False) -> Path:
    repo = BENCHMARK_REPOS.get(filename.split("/")[0], HF_REPO)
    source = filename
    if source.endswith("/responses.parquet"):
        source = source.removesuffix("responses.parquet") + RESPONSE_FILE
    try:
        downloaded = hf_hub_download(
            repo, source, repo_type="dataset", revision=source_revision(repo),
            cache_dir=cache_dir / ".hub", force_download=refresh,
        )
    except LocalEntryNotFoundError:
        raise
    except EntryNotFoundError as exc:
        if source.endswith("/responses.parquet"):
            downloaded = fetch(source.removesuffix("responses.parquet") + "response.parquet",
                               cache_dir, refresh)
        else:
            raise SystemExit(f"'{source}' not found in {repo}.") from exc

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
    BENCHMARK_REPOS.clear()
    for repo in HF_REPOS or {HF_REPO: HF_REVISION}:
        entries = HfApi().list_repo_tree(
            repo, repo_type="dataset", revision=source_revision(repo))
        for entry in entries:
            if isinstance(entry, RepoFolder):
                BENCHMARK_REPOS.setdefault(entry.path, repo)
    return sorted(BENCHMARK_REPOS.keys() & set(json.loads(FILTERED_PATH.read_text())))


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
    """Use the curated publication list for eligibility; ignore metadata release flags."""
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
            cards.append(card)
    cards.sort(key=lambda c: c["slug"])
    out = web_dir / "content" / "generated" / "benchmark-cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=1) + "\n")
    sys.stderr.write(f"✓ wrote {len(cards)} cards → {out}\n")

# Condition parsing and matrix drawing

def parse_to_sel(cond: str) -> dict[str, str]:
    """Parse dim=value pairs; treat a bare label as the condition dimension."""
    if "=" not in cond:
        return {"condition": cond}
    sel: dict[str, str] = {}
    for part in cond.split(";"):
        k, _, v = part.partition("=")
        sel[k.strip()] = v.strip()
    return sel


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
    """Classify constant, item, subject, then condition dimensions, allowing tolerance."""
    if not dims:
        return {}
    sel = obs["_selection"]
    kinds: dict[str, str] = {}
    for d in dims:
        v = sel.map(lambda s: s.get(d))
        if v.nunique(dropna=True) <= 1:
            kinds[d] = "constant"
        elif (v.groupby(obs["item_id"]).nunique(dropna=True) > 1).mean() \
                <= DIM_ATTR_TOLERANCE:
            kinds[d] = "item"
        elif (v.groupby(obs["subject_id"]).nunique(dropna=True) > 1).mean() \
                <= DIM_ATTR_TOLERANCE:
            kinds[d] = "subject"
        else:
            kinds[d] = "condition"
    return kinds


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
                 block_unit: str | None = None) -> dict | None:
    """Build one matrix per trial, with item blocks and subject/condition rows."""
    cond_dims = [d for d in dims if d != "trial"]
    kinds = classify_dims(obs, cond_dims)

    # Use one item dimension for column blocks, preferring the fewest levels.
    item_dims = [d for d in cond_dims if kinds.get(d) == "item"]
    block_dim = None
    if item_dims:
        sel_all = obs["_selection"]
        levels = {d: int(sel_all.map(lambda s: s.get(d, MISSING_DIM)).nunique())
                  for d in item_dims}
        if block_pref in item_dims:
            cand = block_pref
        else:
            if block_pref:
                sys.stderr.write(
                    f"… {slug}: blockDim '{block_pref}' is not an item dim "
                    f"({', '.join(item_dims)}); ignoring the override\n")
            cand = min(item_dims, key=lambda d: (levels[d], cond_dims.index(d)))
        block_dim = cand
        if levels[cand] > MAX_BLOCKS and block_pref != cand:
            sys.stderr.write(
                f"… {slug}: {levels[cand]} blocks on `{cand}` exceeds the "
                f"{MAX_BLOCKS} that fit without scrolling "
                f"({', '.join(f'{d}={levels[d]}' for d in item_dims)}); "
                f"the figure will scroll horizontally\n")

    o = obs[["subject_id", "item_id", "test_condition", "trial", "response"]].copy()
    o["_cond"] = o["test_condition"].fillna(NO_COND)
    o["_sel"] = obs["_selection"]
    o["_key"] = obs["_key"]
    if block_dim:
        o["_block"] = o["_sel"].map(lambda values: values.get(block_dim) or MISSING_DIM)
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
        ids = sub.groupby("item_id")["response"].mean().sort_values(
            kind="stable").index.tolist()
        blocks.append({"key": key, "colIds": ids})
        col_pos[key] = {iid: i for i, iid in enumerate(ids)}

    band_dims = [d for d in cond_dims if kinds.get(d) == "condition"]
    o["_band"] = o["_sel"].map(lambda values: ";".join(
        f"{d}={values.get(d, MISSING_DIM)}" for d in band_dims))
    # Exclude the block dimension from row identity so a row spans every block.
    row_dims = [d for d in cond_dims if d != block_dim]
    o["_rowcond"] = o["_sel"].map(lambda values: ";".join(
        f"{d}={values.get(d, MISSING_DIM)}" for d in row_dims))

    n_rows = o.groupby(["trial", "subject_id", "_rowcond"]).ngroups
    n_cells = n_rows * sum(len(b["colIds"]) for b in blocks)
    if n_cells > MAX_JOINED_CELLS:
        sys.stderr.write(
            f"… {slug}: joined matrix is {n_cells:,} cells "
            f"({n_rows:,} rows), {100 * len(o) / n_cells:.1f}% observed — "
            f"over the {MAX_JOINED_CELLS:,} advisory threshold; rendering anyway\n")

    # Use the PNG color ramp for graded cells; binary cells remain 0/1.
    levels: list[dict] | None = None
    level_index: dict[float, int] = {}
    if not is_binary:
        vals = sorted(float(v) for v in obs["response"].dropna().unique())
        if len(vals) > len(LEVEL_CHARS):
            sys.stderr.write(
                f"… {slug}: {len(vals)} distinct response values exceeds the "
                f"{len(LEVEL_CHARS)}-level cell encoding; skipping joined view\n")
            return None
        level_index = {v: i for i, v in enumerate(vals)}
        levels = [{"value": v,
                   "label": f"{round(v, 4):g}",
                   "color": "#%02x%02x%02x" % graded_color(v, vmin, vmax)}
                  for v in vals]

    def cell_char(v) -> str:
        f = float(v)
        if is_binary:
            return "1" if f >= 0.5 else "0"
        return LEVEL_CHARS[level_index[f]]

    block_order = [b["key"] for b in blocks]
    trials: dict[str, list] = {}
    for (trial, sid, rowcond), g in o.groupby(["trial", "subject_id", "_rowcond"], sort=True):
        sel = g["_sel"].iloc[0]
        row = {
            "sid": sid,
            "band": ";".join(f"{d}={sel.get(d, MISSING_DIM)}" for d in band_dims),
            "rowcond": rowcond,
            "blocks": {},
            "pass": float(g["response"].sum()),
            "n": int(len(g)),
        }
        for key, grp in g.groupby("_block"):
            ids = blocks[block_order.index(key)]["colIds"]
            # A dot marks an unobserved cell.
            bits = ["."] * len(ids)
            for iid, v in zip(grp["item_id"], grp["response"]):
                bits[col_pos[key][iid]] = cell_char(v)
            # Keep each block's full condition for trace and audio lookup.
            row["blocks"][key] = {
                "bits": "".join(bits),
                "cond": None if grp["_cond"].iloc[0] == NO_COND else grp["_cond"].iloc[0],
            }
            row["blocks"][key].update(cell_keys(
                grp["_key"].tolist(), [col_pos[key][iid] for iid in grp["item_id"]]))
        trials.setdefault(str(trial), []).append(row)

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
    }
    # The client treats an absent levels field as binary.
    if levels is not None:
        payload["levels"] = levels
    return payload


# Session strips

# Opt in with render=sessions: observational developer streams need
# per-developer timelines rather than a shared item axis.


def session_matrix(slug: str, obs, id_to_name: dict) -> dict | None:
    """Build developer rows with aligned item IDs, answer cells, and session lengths."""
    sel = obs["_selection"]
    # Normalized subject registries call the agent scaffold "harness".
    sel = sel.map(lambda s: s | {"agent": s["harness"]}
                  if "agent" not in s and "harness" in s else s)
    missing = [d for d in SESSION_DIMS if not any(d in s for s in sel)]
    if missing:
        sys.stderr.write(
            f"… {slug}: render=sessions needs test_condition dims "
            f"{', '.join(SESSION_DIMS)}; missing {', '.join(missing)} — "
            "using the basic matrix\n")
        return None

    o = obs[["subject_id", "item_id", "response"]].copy()
    o["_key"] = obs["_key"]
    for d in SESSION_DIMS:
        o[d] = sel.map(lambda s, d=d: s.get(d, MISSING_DIM)).values
    # Zero-padded session/turn labels sort chronologically.
    o = o.sort_values(["agent", "user", "session", "turn"], kind="stable")

    rows = []
    for (agent, user), g in o.groupby(["agent", "user"], sort=True):
        bits, cols, segs, seg_labels, keys = [], [], [], [], []
        for sess, sg in g.groupby("session", sort=True):
            bits.append("".join("1" if v == 1.0 else "0" for v in sg["response"]))
            cols.extend(str(i) for i in sg["item_id"])
            segs.append(int(len(sg)))
            seg_labels.append(str(sess))
            keys.extend(sg["_key"])
        joined_bits = "".join(bits)
        n = len(joined_bits)
        sids = list(dict.fromkeys(str(s) for s in g["subject_id"]))
        rows.append({
            "sid": sids[0],
            "band": f"agent={agent}",
            "rowcond": "",
            "name": user,
            "blocks": {"all": {"bits": joined_bits, "cond": None}},
            "cols": cols,
            "segs": segs,
            "segLabels": seg_labels,
            "subjects": [id_to_name.get(s, s) for s in sids],
            "pass": int(joined_bits.count("1")),
            "n": n,
        })

        rows[-1]["blocks"]["all"].update(cell_keys(keys))

    if not rows:
        sys.stderr.write(f"… {slug}: render=sessions produced no rows\n")
        return None

    band_order = {f"agent={a}": i for i, a in enumerate(
        o.groupby("agent")["item_id"].count().sort_values(ascending=False).index)}
    rows.sort(key=lambda r: (band_order.get(r["band"], 99), -r["n"], r["name"]))

    width = max(r["n"] for r in rows)
    payload = {
        "bandLabels": {},
        "blockDim": None,
        "bandDims": ["agent"],
        "dimKinds": {"agent": "item"},
        "blocks": [{"key": "all", "colIds": [""] * width}],
        "trials": {"1": rows},
        "rowKind": "session",
        # Pad shorter streams with white to distinguish missing turns from failures.
        "noneColor": "#ffffff",
    }
    return payload


# Faceted matrices

def faceted_matrix(slug: str, obs, id_to_name: dict, facet: dict,
                  band_labels: dict | None = None,
                  is_binary: bool = True) -> dict | None:
    """Build bands with separate item axes and panels declared in the facet override.

    Panel match fields are ANDed; onlyBands restricts panel membership.
    Unmatched responses are omitted. Return None if the configuration does not fit."""
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
    if o.empty:
        sys.stderr.write(
            f"… {slug}: render=faceted matched no responses; check facet.panels\n")
        return None
    if dropped:
        sys.stderr.write(
            f"… {slug}: faceted drops {dropped:,} response(s) matching no "
            "declared panel — they stay in the parquets\n")

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
                    bits[pos[iid]] = ("1" if float(v) >= 0.5 else "0") if is_binary \
                        else LEVEL_CHARS[0]
                rp[pkey] = {"bits": "".join(bits),
                            "cond": pg["test_condition"].iloc[0]}
                rp[pkey].update(cell_keys(pg["_key"].tolist(), [pos[iid] for iid in pg["item_id"]]))
            rows.append({
                "sid": sid,
                "name": id_to_name.get(sid, sid),
                "band": f"{band_dim}={bkey}",
                "rowcond": "",
                "panels": rp,
                "pass": float(g["response"].sum()),
                "n": int(len(g)),
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
    """Find an opponent-name dimension when most items have at most two subjects."""
    names = set(id_to_name.values())
    if not names:
        return None
    # Check names before the more expensive item-axis grouping.
    hits: dict[str, list[int]] = {}
    distinct = df.drop_duplicates("test_condition")
    for label, features in zip(distinct["test_condition"], distinct["_features"]):
        if "=" not in label:
            continue
        for key, value in features.items():
            tally = hits.setdefault(key, [0, 0])
            tally[0] += 1
            tally[1] += value in names
    opp = [k for k, (tot, hit) in hits.items()
           if tot and hit / tot >= PAIRWISE_NAME_MATCH]
    if len(opp) != 1:
        return None

    share = float((df.groupby("item_id")["subject_id"].nunique() <= 2).mean())
    if share < PAIRWISE_DEGENERATE_SHARE:
        return None
    sys.stderr.write(
        f"… {slug}: pairwise detected — `{opp[0]}` names subjects and "
        f"{share:.1%} of items are answered by ≤2 subjects\n")
    return opp[0]


def pairwise_matrix(slug: str, df, row_ids: list[str],
                   id_to_name: dict, pair_dim: str = "opponent") -> dict | None:
    """Build one row per subject pair, with cells ordered by outcome.

    The subject with the higher raw mean is named first; this is not a fitted rank.
    Return None if the data cannot be represented as pairs."""
    prefix = f"{pair_dim}="
    features = df["_features"]
    if not features.map(lambda values: list(values) == [pair_dim]).all():
        sys.stderr.write(
            f"… {slug}: pairwise declined — not every test_condition is "
            f"\"{prefix}…\"\n")
        return None

    # Decline unmatched opponent labels to avoid dropping battles silently.
    name_to_sid: dict[str, str] = {}
    for sid, nm in id_to_name.items():
        name_to_sid.setdefault(nm, sid)
    w = df[["subject_id", "item_id", "response"]].copy()
    opponents = features.map(lambda values: values[pair_dim])
    w["opp"] = opponents.map(name_to_sid)
    w["_key"] = df["_key"]
    if w["opp"].isna().any():
        missing = sorted(set(opponents[w["opp"].isna()]))[:3]
        sys.stderr.write(
            f"… {slug}: pairwise declined — {int(w['opp'].isna().sum()):,} rows name "
            f"an opponent absent from subjects.parquet (e.g. {missing})\n")
        return None

    n_self = int((w["subject_id"] == w["opp"]).sum())
    if n_self:
        sys.stderr.write(
            f"… {slug}: pairwise dropping {n_self:,} self-battle row(s) "
            f"({n_self // 2:,} battles)\n")
        w = w[w["subject_id"] != w["opp"]]

    # Pair mirrored rows by occurrence within each item to recover "both bad".
    w["occ"] = w.groupby(["subject_id", "opp", "item_id"], sort=False).cumcount()
    mirror = w.rename(columns={"subject_id": "opp", "opp": "subject_id",
                               "response": "resp_opp", "_key": "_key2"})
    mirror_cols = ["subject_id", "opp", "item_id", "occ", "resp_opp"]
    mirror_cols.append("_key2")
    w = w.merge(mirror[mirror_cols],
                on=["subject_id", "opp", "item_id", "occ"], how="left")

    code = np.where(w["response"] == 1.0, P_WIN,
                    np.where(w["response"] == 0.5, P_TIE,
                             np.where(w["resp_opp"] == 0.0, P_BOTHBAD, P_LOSS)))
    w["code"] = code.astype(np.int8)

    # Keep each pair once, naming the higher-mean subject first.
    # row_ids follows the PNG's ascending score order.
    rank = {sid: i for i, sid in enumerate(row_ids)}
    last = -1
    keep = w["subject_id"].map(lambda s: rank.get(s, last)).to_numpy() > \
        w["opp"].map(lambda s: rank.get(s, last)).to_numpy()
    w = w[keep]
    if w.empty:
        sys.stderr.write(f"… {slug}: pairwise declined — no ordered pairs survive\n")
        return None

    groups = list(w.groupby(["subject_id", "opp"], sort=False))
    width = max(len(g) for _, g in groups)
    if len(groups) * width > MAX_JOINED_CELLS:
        sys.stderr.write(
            f"… {slug}: pairwise declined — {len(groups):,} pairs × {width:,} "
            f"items = {len(groups) * width:,} cells, over {MAX_JOINED_CELLS:,}\n")
        return None

    rows = []
    for (a, b), g in groups:
        codes = g["code"].to_numpy()
        items = g["item_id"].to_numpy()
        # Order wins, ties, both-bad outcomes, then losses; preserve ties.
        seq = np.argsort(-codes, kind="stable")
        codes, items = codes[seq], items[seq]
        n = len(codes)
        n_win = int((codes == P_WIN).sum())
        rows.append({
            "sid": a,
            "sid2": b,
            "name": f"{id_to_name.get(a, a)} vs {id_to_name.get(b, b)}",
            "band": "",
            "rowcond": "",
            # Pad to a shared width so short pairs do not appear fully sampled.
            "blocks": {"all": {
                "bits": "".join(LEVEL_CHARS[c] for c in codes) + "." * (width - n),
                "cond": f"{pair_dim}={id_to_name.get(b, b)}",
                "cond2": f"{pair_dim}={id_to_name.get(a, a)}",
            }},
            # Item positions belong to this row, not a shared column axis.
            "cols": list(items) + [""] * (width - n),
            "pass": n_win,
            "n": n,
            "score": f"{n_win}/{n}",
        })

        block = rows[-1]["blocks"]["all"]
        block.update(cell_keys(g["_key"].iloc[seq].tolist()))
        block.update(cell_keys(g["_key2"].iloc[seq].tolist(), second=True))

    rows.sort(key=lambda r: (-r["n"], -rank.get(r["sid"], last),
                             -rank.get(r["sid2"], last)))

    payload = {
        "bandLabels": {"_pair": {
            "all": "items compared · position, not a shared item",
        }},
        "blockDim": "_pair",
        "bandDims": [],
        "dimKinds": {},
        "blocks": [{"key": "all", "colIds": [""] * width}],
        "trials": {"1": rows},
        "levels": PAIR_LEVELS,
        "rowKind": "pair",
        # White padding distinguishes unobserved cells from gray ties.
        "noneColor": "#ffffff",
    }
    return payload


# Detail records and input normalization

def scale_label(is_binary: bool, override: dict, response_scale) -> str:
    bl = override.get("binaryLabels")
    if is_binary and bl:
        return f"1 = {bl['one']} · 0 = {bl['zero']}"
    if is_binary:
        return "1 = correct · 0 = incorrect"
    domain = response_scale
    if isinstance(domain, str):
        try:
            domain = json.loads(domain)
        except (TypeError, ValueError):
            pass  # Published legacy files still describe their scales in prose.
    if isinstance(domain, dict):
        if domain.get("kind") == "mixed":
            return "Item-specific grading scales"
        if domain.get("kind") == "interval":
            lower, upper = domain.get("min"), domain.get("max")
            if lower is None and upper is None:
                return "Unbounded numeric score"
            if lower is None:
                return f"Score ≤ {upper:g}"
            if upper is None:
                return f"Score ≥ {lower:g}"
            return f"Score in [{lower:g}, {upper:g}]"
        if domain.get("kind") == "discrete":
            return "Grades: " + ", ".join(f"{value:g}" for value in domain["values"])
    return str(response_scale) if response_scale else "graded score"


def registry_feature_dims(slug: str, cache_dir: Path, refresh: bool,
                          ) -> tuple[dict, dict]:
    """Read subject/item display dimensions from normalized registry fields.

    Expand verifier fields only when item_features exists; older tables
    already carry those dimensions in test_condition."""
    subj_path = fetch(f"{slug}/subjects.parquet", cache_dir, refresh)
    names = set(pq.read_schema(subj_path).names)
    cols = [c for c in [*SUBJECT_FEATURE_COLS, "subject_features_extra"]
            if c in names]
    subj_feats: dict = {}
    if cols:
        tbl = pq.read_table(subj_path, columns=["subject_id", *cols]).to_pandas()
        for row in tbl.itertuples(index=False):
            sel = {c: getattr(row, c) for c in SUBJECT_FEATURE_COLS
                   if c in names and isinstance(getattr(row, c), str)}
            extra = getattr(row, "subject_features_extra", None)
            if isinstance(extra, str) and extra:
                sel.update(parse_to_sel(extra))
            if sel:
                subj_feats[row.subject_id] = sel

    item_path = fetch(f"{slug}/items.parquet", cache_dir, refresh)
    item_feats: dict = {}
    item_names = set(pq.read_schema(item_path).names)
    if "item_features" in item_names:
        item_cols = ["item_id", "item_features"]
        if "verifier" in item_names:
            item_cols.append("verifier")
        tbl = pq.read_table(item_path, columns=item_cols).to_pandas()
        for row in tbl.itertuples(index=False):
            sel = (parse_to_sel(row.item_features)
                   if isinstance(row.item_features, str) and row.item_features
                   else {})
            verifier = getattr(row, "verifier", None)
            if isinstance(verifier, str) and verifier:
                try:
                    payload = json.loads(verifier)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if key in {"class", "spec", "judged_by"} or value is None:
                            continue
                        value = str(value)
                        if key in sel and sel[key] != value:
                            raise SystemExit(
                                f"{slug}: item {row.item_id!r} carries conflicting "
                                f"{key!r} values in item_features and verifier")
                        sel[key] = value
            if sel:
                item_feats[row.item_id] = sel
    return subj_feats, item_feats


def reassemble_display_dims(slug: str, frame: pd.DataFrame, cache_dir: Path,
                            refresh: bool) -> pd.DataFrame:
    """Parse features once; retain raw keys separately from display labels."""
    subj_feats, item_feats = registry_feature_dims(slug, cache_dir, refresh)
    inter = frame.pop("interactors") if "interactors" in frame.columns else None
    parsed = {NO_COND: {}}
    strings = list(frame["test_condition"].unique())
    if inter is not None:
        strings.extend(inter.unique())
    for value in strings:
        if value not in parsed:
            parsed[value] = parse_to_sel(value)
    labels, features = [], []
    for i, (sid, iid, cond) in enumerate(zip(
            frame["subject_id"], frame["item_id"], frame["test_condition"])):
        sel = parsed[cond].copy()
        sources = [subj_feats.get(sid), item_feats.get(iid)]
        if inter is not None:
            sources.append(parsed[inter.iat[i]])
        for source in sources:
            for key, value in (source or {}).items():
                if sel.get(key, value) != value:
                    raise SystemExit(f"{slug}: conflicting {key!r} in condition and registry features")
                sel[key] = value
        if subj_feats or item_feats or inter is not None:
            sel = dict(sorted(sel.items()))
            cond = ";".join(f"{k}={v}" for k, v in sel.items()) or NO_COND
        labels.append(cond)
        features.append(sel)
    frame["test_condition"] = labels
    frame["_features"] = features
    return frame


def load_raw(slug: str, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """Preserve observation keys and join the display features to responses."""
    resp_path = fetch(f"{slug}/responses.parquet", cache_dir, refresh)
    key = list(KEY)
    if "interactors" in pq.read_schema(resp_path).names:
        key.append("interactors")
    resp = pq.read_table(
        resp_path,
        columns=[*key, "response"],
    ).to_pandas()
    if resp.empty:
        raise SystemExit(f"{slug}/responses.parquet has no rows")
    resp["_key"] = [list(row) for row in resp[key].itertuples(index=False, name=None)]
    for col in key[2:]:
        resp[col] = resp[col].fillna(NO_COND)

    dup = int(resp.duplicated(key).sum())
    if dup:
        raise SystemExit(
            f"{slug}: {dup} rows are not unique on {key} — responses.parquet is not "
            f"cleanly keyed (bad crawl); refusing to aggregate silently.")

    return reassemble_display_dims(slug, resp, cache_dir, refresh)


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

    items_tbl = pq.read_table(
        fetch(f"{slug}/items.parquet", cache_dir, refresh),
        columns=["item_id", "content"],
    ).to_pandas()
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
            "meanResponse": None,
        },
        "matrix": f"/benchmarks/matrices/{slug}.png",
        "matrixSize": size,
        "conditions": None,
        "categories": None,
        "binaryLabels": None,
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
        # The binary filter displays unobserved gray cells as white.
        "isBinary": True,
        "valueRange": [0.0, 1.0],
        "scaleLabel": scale,
        "questionsAvailable": bool(questions_available),
        "subjects": [{"name": id_to_name.get(sid, sid), "score": None}
                     for sid in row_ids],
    }

    return {"detail": detail, "chart": None, "axes": axes, "matrices": {"": matrix}}


class SlicePlan(NamedTuple):

    dims: list[str]
    values: dict[str, list[str]]
    conditions: list[str | None]
    trials: list[int | None]
    base_condition: str | None
    trial_multi: bool
    collapsed: str | None
    parse: Callable
    observations: pd.DataFrame


def plan_slices(slug: str, df: pd.DataFrame, sessions_mode: bool = False) -> SlicePlan:
    """Choose selectors and retain their observations."""
    all_conds = df["test_condition"].unique().tolist()
    real_conds = sorted(c for c in all_conds if c != NO_COND)
    if real_conds and len(real_conds) < len(all_conds):
        n_null = int((df["test_condition"] == NO_COND).sum())
        sys.stderr.write(
            f'… {slug}: {n_null:,} of {len(df):,} rows have no test_condition while {len(real_conds)} condition value(s) exist; rendering "{NO_COND}" as a selectable value rather than dropping those rows\n'
        )
        real_conds = sorted(all_conds)
    cond_multi = len(real_conds) > 1
    base_cond = real_conds[0] if not cond_multi and real_conds else None
    if sessions_mode:
        cond_multi, base_cond = (False, None)
    all_trials = sorted(df["trial"].dropna().unique().tolist())
    render_trials = all_trials[:MAX_TRIALS_RENDERED]
    n_combos = len(real_conds) if cond_multi else 1
    if n_combos * len(render_trials) > MAX_SLICES:
        render_trials = render_trials[: max(1, MAX_SLICES // n_combos)]
    if len(render_trials) < len(all_trials):
        sys.stderr.write(
            f"… {slug}: {len(all_trials)} trials → rendering {len(render_trials)} as separate matrices (cap); omitted trials' raw data stays in the parquets\n"
        )
    trial_multi = len(render_trials) > 1
    trial_filter = trial_multi or len(render_trials) < len(all_trials)
    cond_iter = real_conds if cond_multi else [base_cond]
    trial_iter = render_trials if trial_filter else [None]
    feature_by_label = dict(zip(df["test_condition"], df["_features"]))
    parsed_conds = [feature_by_label[c] for c in real_conds] if cond_multi else []
    cond_dims = list(dict.fromkeys(d for selection in parsed_conds for d in selection))
    collapse = len(cond_dims) > 1 and (
        not any(all(d in selection for selection in parsed_conds) for d in cond_dims)
    )
    if collapse:
        merged = "condition" if "condition" not in cond_dims else "condition_set"
        sys.stderr.write(
            f"… {slug}: condition dims {cond_dims} share no universal dim — collapsing to a single '{merged}' dimension\n"
        )
        parsed_conds = [{merged: c} for c in real_conds]

        def parse_cond(c):
            return {merged: c}
    else:
        def parse_cond(c):
            return feature_by_label.get(c, {}).copy()
    dims: list[str] = []
    values: dict[str, list[str]] = {}
    if cond_multi:
        for sel_c in parsed_conds:
            for d, v in sel_c.items():
                if d not in values:
                    dims.append(d)
                    values[d] = []
                if v not in values[d]:
                    values[d].append(v)
        for d in dims:
            if any(d not in selection for selection in parsed_conds):
                values[d].append(MISSING_DIM)
    if trial_multi:
        dims.append("trial")
        values["trial"] = [str(t) for t in render_trials]
    if sessions_mode:
        dims, values = [], {}
        def parse_cond(c):
            return feature_by_label.get(c, {}).copy()
    trace_src = df
    if cond_multi:
        trace_src = trace_src[trace_src["test_condition"].isin(real_conds)]
    elif base_cond is not None:
        trace_src = trace_src[trace_src["test_condition"] == base_cond]
    if trial_filter:
        trace_src = trace_src[trace_src["trial"].isin(render_trials)]
    trace_src = trace_src.assign(_selection=trace_src["test_condition"].map(
        parse_cond if dims or sessions_mode else lambda c: {}))
    return SlicePlan(
        dims,
        values,
        cond_iter,
        trial_iter,
        base_cond,
        trial_multi,
        merged if collapse else None,
        parse_cond,
        trace_src,
    )


def matrix_size(width: int, rows: int) -> list[int]:
    width *= max(1, -(-MIN_WIDTH // width))
    return [width, rows * max(1, round(max(MIN_HEIGHT, width // MAX_ASPECT) / rows))]


def prepare_matrix_slices(slug: str, df: pd.DataFrame, plan: SlicePlan,
                          row_ids: list[str], col_ids: list[str]) -> tuple:
    row_pos = {sid: i for i, sid in enumerate(row_ids)}
    col_pos = {iid: i for i, iid in enumerate(col_ids)}
    matrices, metadata, columns = {}, {}, {}
    best = None
    observed = df.dropna(subset=["response"])
    group_columns = []
    for column, values in [("test_condition", plan.conditions), ("trial", plan.trials)]:
        if values != [None] and (column != "trial" or plan.dims):
            observed = observed[observed[column].isin(values)]
            group_columns.append(column)
    groups = {key if isinstance(key, tuple) else (key,): frame
              for key, frame in observed.groupby(group_columns, sort=False)} \
        if group_columns else {(): observed}
    for condition in plan.conditions:
        for trial in plan.trials:
            key = tuple(value for column, value in [("test_condition", condition), ("trial", trial)]
                        if column in group_columns)
            observed = groups.get(key)
            if observed is None or observed.empty:
                continue
            cell = observed.set_index(["subject_id", "item_id"], verify_integrity=True)["response"]
            row_mean = observed.groupby("subject_id")["response"].mean().sort_values(kind="stable")
            col_mean = observed.groupby("item_id")["response"].mean().sort_values(kind="stable")
            rows = row_mean.index.tolist() if plan.dims else row_ids
            cols = col_mean.index.tolist() if plan.dims else col_ids
            sel = plan.parse(condition) if condition is not None else {}
            if trial is not None and plan.trial_multi:
                sel["trial"] = str(trial)
            key = combo_key(plan.dims, sel)
            path = (f"/benchmarks/matrices/{slug}/{sanitize(key)}.png" if plan.dims
                    else f"/benchmarks/matrices/{slug}.png")
            size = matrix_size(min(len(cols), CAP_WIDTH), len(rows))
            matrices[key] = (cell, rows, cols, path,
                             observed.set_index(["subject_id", "item_id"])["_key"])
            metadata[key] = {
                "sel": sel, "matrix": path, "matrixSize": size,
                "rowIdx": [row_pos[sid] for sid in rows],
                "rowScores": [round(float(v), 4) for v in row_mean],
                "nItems": len(cols), "colP": None,
                "observed": round(len(cell) / (len(rows) * len(cols)), 4),
            }
            columns[key] = {
                "colIdx": [col_pos[iid] for iid in cols],
                "colP": [round(float(v), 4) for v in col_mean],
            }
            if best is None or len(cell) > best[0]:
                best = (len(cell), sel, path, size)
            if not plan.dims:
                break
        if not plan.dims:
            break
    if best is None:
        raise SystemExit(f"{slug}: no rendered slice has any observed response")
    conditions = {
        "dims": plan.dims, "values": plan.values,
        "default": {d: best[1].get(d, MISSING_DIM) for d in plan.dims},
        "collapsed": plan.collapsed, "matrices": metadata,
    } if plan.dims else None
    attacks = (len([v for v in plan.values["attack"] if v != MISSING_DIM])
               if plan.dims and "attack" in plan.values else None)
    return best[2], best[3], conditions, attacks, columns if plan.dims else {}, matrices


def build_detail(
    slug: str, cache_dir: Path, web_dir: Path, refresh: bool, overrides: dict,
) -> dict:
    override = overrides.get(slug, {})
    meta = read_meta(slug, cache_dir, refresh)
    if (meta.get("granularity") or "item") != "item":
        return build_no_responses_detail(
            slug, meta, cache_dir, web_dir, refresh, override
        )
    df = load_raw(slug, cache_dir, refresh)
    if df.empty:
        raise SystemExit(f"{slug}/responses.parquet has no rows")
    resp = df["response"].dropna()
    uniq = set(resp.unique().tolist())
    is_binary = uniq.issubset({0.0, 1.0})
    vmin, vmax = (float(resp.min()), float(resp.max()))
    subj_tbl = pq.read_table(
        fetch(f"{slug}/subjects.parquet", cache_dir, refresh),
        columns=["subject_id", "display_name"],
    ).to_pandas()
    id_to_name = dict(zip(subj_tbl["subject_id"], subj_tbl["display_name"]))
    items_tbl = pq.read_table(
        fetch(f"{slug}/items.parquet", cache_dir, refresh),
        columns=["item_id", "content"],
    ).to_pandas()
    questions_available = items_tbl["content"].notna().any()
    sessions_mode = override.get("render") == "sessions"
    plan = plan_slices(slug, df, sessions_mode)
    dims = plan.dims
    trace_src = plan.observations
    obs = trace_src.dropna(subset=["response"])
    subj_mean = obs.groupby("subject_id")["response"].mean().sort_values(kind="stable")
    item_solve = obs.groupby("item_id")["response"].mean().sort_values(kind="stable")
    hidden_subj = df["subject_id"].nunique() - len(subj_mean)
    hidden_item = df["item_id"].nunique() - len(item_solve)
    if hidden_subj or hidden_item:
        sys.stderr.write(
            f"… {slug}: hiding {hidden_subj} all-unobserved subject row(s) and {hidden_item} all-unobserved item column(s); raw data stays in the parquets\n"
        )
    row_ids = subj_mean.index.tolist()
    col_ids = item_solve.index.tolist()
    n_rows, n_items = (len(row_ids), len(col_ids))
    repo = BENCHMARK_REPOS.get(slug, HF_REPO)
    has_traces = HfApi().file_exists(
        repo, f"{slug}/traces.parquet", repo_type="dataset", revision=source_revision(repo))
    col_p = [round(float(item_solve[iid]), 4) for iid in col_ids]
    matrix_path, size, conditions, attacks, slice_cols, matrices = prepare_matrix_slices(
        slug, df, plan, row_ids, col_ids)
    lazy_payload = {"colIds": col_ids, "colP": col_p, "items": {}}
    if slice_cols:
        lazy_payload["slices"] = slice_cols
    faceted_mode = override.get("render") == "faceted"
    declared = override.get("pairwise")
    pair_dim = None
    if not sessions_mode and not faceted_mode:
        pair_dim = declared
        if pair_dim is None:
            pair_dim = detect_pairwise_dim(slug, df, id_to_name)
    if faceted_mode:
        joined = faceted_matrix(
            slug,
            obs,
            id_to_name,
            override.get("facet") or {},
            override.get("bandLabels") or {},
            is_binary,
        )
    elif sessions_mode:
        joined = session_matrix(slug, obs, id_to_name)
    elif pair_dim:
        joined = pairwise_matrix(slug, df, row_ids, id_to_name, pair_dim)
    else:
        joined = override.get("joined", True) and joined_matrix(
            slug,
            obs,
            dims,
            row_ids,
            override.get("bandLabels") or {},
            override.get("bandOrder") or [],
            override.get("blockDim"),
            is_binary,
            vmin,
            vmax,
            block_order=override.get("blockOrder") or [],
            block_unit=override.get("blockUnit"),
        )
    grid_cells = n_rows * n_items
    observed_overall = (
        round(obs.drop_duplicates(["subject_id", "item_id"]).shape[0] / grid_cells, 4)
        if grid_cells
        else 0.0
    )
    detail = {
        **detail_metadata(slug, meta),
        "stats": {
            "items": int(n_items),
            "subjects": int(n_rows),
            "observed": observed_overall,
            "meanResponse": round(float(resp.mean()), 4),
        },
        "matrix": matrix_path,
        "matrixSize": size,
        "conditions": conditions,
        "categories": None,
        "binaryLabels": override.get("binaryLabels"),
        "note": override.get("note"),
        "attacks": attacks,
        "matrixItemsTotal": n_items,
        "matrixItemsShown": n_items,
        "matrixSampled": False,
        "matrixRows": [id_to_name.get(sid, sid) for sid in row_ids],
        "matrixRowIds": row_ids,
        "matrixRowScores": [round(float(subj_mean[sid]), 4) for sid in row_ids],
        "matrixColIds": None,
        "matrixColP": None,
        "hasTraces": bool(has_traces),
        "hasAudio": bool(override.get("audio")),
        **({"audio": override["audio"]} if override.get("audio") else {}),
        "hasJoined": bool(joined),
        "traceChunkPrefix": 0,
        "isBinary": is_binary,
        "valueRange": [vmin, vmax],
        "scaleLabel": scale_label(is_binary, override, meta.get("response_scale")),
        "questionsAvailable": bool(questions_available),
        "subjects": [
            {"name": id_to_name.get(sid, sid), "score": round(float(subj_mean[sid]), 4)}
            for sid in reversed(row_ids)
        ],
    }
    return {"detail": detail, "chart": joined, "axes": lazy_payload, "matrices": matrices}


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
    columns = [c for c in ["content", "grading_criterion", "reference_answer", "correct_answer"]
               if c in pq.read_schema(path).names]
    rows = pq.read_table(path, columns=columns,
                         filters=[("item_id", "=", item_id)]).to_pylist()
    if len(rows) != 1:
        raise ValueError("Item does not identify exactly one source row")
    answer = rows[0].get("reference_answer", rows[0].get("correct_answer"))
    if "grading_criterion" in rows[0]:
        answer = json.loads(rows[0]["grading_criterion"])["reference_answer"]
    return {"content": rows[0].get("content"),
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
    matrices = {}
    if not view["chart"]:
        for name, (values, rows, cols, _, keys) in view["matrices"].items():
            row_pos = {sid: i for i, sid in enumerate(rows)}
            col_pos = {iid: i for i, iid in enumerate(cols)}
            width = min(len(cols), CAP_WIDTH)
            pixels, observations = [], {}
            for (sid, iid), value in values.items():
                r, c = row_pos[sid], col_pos[iid]
                color = (RED if value >= 0.5 else BLUE) if detail["isBinary"] else \
                    graded_color(float(value), *detail["valueRange"])
                pixels.append([r * width + c * width // len(cols), *color])
                observations[r * len(cols) + c] = keys.loc[(sid, iid)]
            matrices[name] = {"width": width, "height": len(rows), "pixels": pixels,
                              "keys": observations}
    return {"detail": detail, "chart": view["chart"] or None,
            "axes": {**view["axes"], "slices": view["axes"].get("slices")},
            "matrices": matrices}


def serve_benchmarks(slugs: list[str], cache_dir: Path, web_dir: Path, port: int) -> None:
    allowed = set(slugs)
    overrides = load_overrides(web_dir)
    source_revision()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            if len(parts) != 2 or parts[0] not in allowed:
                self.send_error(404)
                return
            slug, kind = parts
            query = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
            try:
                if kind == "item":
                    payload = read_item(slug, cache_dir, query["item_id"])
                elif kind == "answer":
                    payload = read_answer(slug, cache_dir, json.loads(query["key"]))
                elif kind == "view":
                    payload = chart_bundle(build_detail(
                        slug, cache_dir, web_dir, False, overrides))
                else:
                    self.send_error(404)
                    return
                body = gzip.compress(json.dumps(payload, ensure_ascii=False,
                                                allow_nan=False).encode("utf-8"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, TypeError):
                self.send_error(400, "Invalid or ambiguous data selection")
            except (Exception, SystemExit) as exc:
                sys.stderr.write(f"{slug}/{kind}: {type(exc).__name__}\n")
                self.send_error(502, "Source data could not be loaded")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving {len(slugs)} benchmarks at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# CLI

def resolve_slugs(arg_slug: str | None, want_all: bool) -> list[str]:
    if want_all or arg_slug is None:
        return list_slugs()
    return [arg_slug]


def main() -> None:
    global HF_REPO, HF_REVISION, RESPONSE_FILE, HF_REPOS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode",
                    choices=["cards", "details", "all", "list", "serve"])
    ap.add_argument("slug", nargs="?", help="benchmark slug (omit + --all for every benchmark)")
    ap.add_argument("--all", action="store_true", help="apply to every published benchmark")
    ap.add_argument("--web-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "website",
                    help="path to the Next app (default: repo website/)")
    ap.add_argument("--cache-dir", type=Path,
                    default=SCRIPT_DIR / ".hf-cache")
    ap.add_argument("--refresh", action="store_true", help="force re-download")
    ap.add_argument("--hf-repo", action="append", metavar="REPO[@COMMIT]",
                    help="repeat for multiple banks; first bank wins duplicate slugs")
    ap.add_argument("--revision", help="HF source commit (default: current commit)")
    ap.add_argument("--response-file", choices=["responses.parquet", "response.parquet"],
                    default=RESPONSE_FILE, help="preferred filename (default: responses.parquet)")
    ap.add_argument("--port", type=int, default=3050)
    args = ap.parse_args()
    HF_REPOS = {repo: revision or None for source in args.hf_repo or [HF_REPO]
                for repo, _, revision in [source.partition("@")]}
    HF_REPO, HF_REVISION, RESPONSE_FILE = next(iter(HF_REPOS)), args.revision, args.response_file
    source_revision.cache_clear()
    published = list_slugs()
    if args.mode == "serve":
        serve_benchmarks(args.slug.split(",") if args.slug else published,
                         args.cache_dir, args.web_dir, args.port)
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
