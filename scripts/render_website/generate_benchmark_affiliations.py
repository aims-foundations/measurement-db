#!/usr/bin/env python3
"""generate_benchmark_affiliations.py — author-affiliation lookup for the benchmark gallery.

Maintains website/content/curated/benchmark-affiliations.json:

    {
      "benchmarks": {
        "<slug>": { "institutions": ["MIT", "Google DeepMind"], "source": "arxiv-pdf" }
      },
      "institutions": {
        "MIT": { "logo": "/institutions/mit.png", "type": "education",
                 "canonical": "Massachusetts Institute of Technology" }
      }
    }

Per-benchmark `institutions` are the affiliations of the paper's FIRST TWO
authors (deduped — one entry if both share an institution), as printed on
page 1 of the paper. `logo` is null when no usable mark was found; the site
then renders an initials chip.

Everything is incremental — safe to re-run after new benchmarks are added;
only slugs/institutions missing from the JSON are processed.

Workflow:
  1. python3 scripts/render_website/generate_benchmark_affiliations.py fetch
       For every card in benchmark-cards.json with no entry yet, download
       page 1 of its paper (arXiv PDF, or the local ICML2025 PDF found via
       the OpenReview forum-id join against "0. papers/icml2025/metadata.jsonl")
       and write its text to scripts/render_website/.affiliation-pages/<slug>.txt.
       Slugs whose paper can't be fetched automatically are listed on stderr —
       these need step 2 to locate the paper first (arXiv mirror / web search).
       The page cache is transient: files are pruned automatically once their
       slug has an entry, so the directory holds only papers awaiting step 2.
  2. python3 scripts/render_website/generate_benchmark_affiliations.py prompt
       Prints a ready-to-paste Claude prompt covering exactly the missing
       slugs; paste it into a Claude Code session in this repo and Claude
       fills the "benchmarks" section (first-two-author affiliations).
  3. python3 scripts/render_website/generate_benchmark_affiliations.py logos
       For every institution named in "benchmarks" but absent from
       "institutions", resolve it via the OpenAlex institutions API and
       download a logo (Wikidata mark, else homepage favicon) to
       website/public/institutions/.

generate_benchmark_gallery.py cards merges the result into benchmark-cards.json,
so finish with `python3 scripts/render_website/generate_benchmark_gallery.py cards`.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
WEB_DIR = REPO_DIR / "website"
CARDS_PATH = WEB_DIR / "content" / "generated" / "benchmark-cards.json"
AFFIL_PATH = WEB_DIR / "content" / "curated" / "benchmark-affiliations.json"
# Under public/benchmarks/ so publish_gallery_assets.py ships logos with the
# rest of the gallery; the recorded path stays "/institutions/<file>", which
# assetUrl() resolves against the gallery repo root.
LOGO_DIR = WEB_DIR / "public" / "benchmarks" / "institutions"
PAGES_DIR = SCRIPT_DIR / ".affiliation-pages"
ICML_META = REPO_DIR / "0. papers" / "icml2025" / "metadata.jsonl"
ICML_PDF_DIR = REPO_DIR / "0. papers" / "icml2025" / "accepted"

UA = "measurement-db-gallery/1.0 (mailto:sttruong@cs.stanford.edu)"


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_affiliations() -> dict:
    if AFFIL_PATH.exists():
        return json.loads(AFFIL_PATH.read_text())
    return {"benchmarks": {}, "institutions": {}}


def save_affiliations(data: dict) -> None:
    data["benchmarks"] = dict(sorted(data["benchmarks"].items()))
    data["institutions"] = dict(sorted(data["institutions"].items()))
    AFFIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    AFFIL_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
    )


def inst_slug(name: str) -> str:
    """Institution display name → filesystem-safe logo basename."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "unknown"


# ---------------------------------------------------------------------------
#  fetch — page-1 text for every card that has no affiliations entry yet
# ---------------------------------------------------------------------------

def prune_pages(affil: dict) -> None:
    """The page-text cache only exists to hand papers to the extraction step;
    once a slug has its entry, its text is dead weight — drop it."""
    if not PAGES_DIR.exists():
        return
    for txt in PAGES_DIR.glob("*.txt"):
        if txt.stem in affil["benchmarks"]:
            txt.unlink()
    if not any(PAGES_DIR.iterdir()):
        PAGES_DIR.rmdir()


def arxiv_id(paper_url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", paper_url)
    return m.group(1) if m else None


def openreview_forum(paper_url: str) -> str | None:
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", paper_url) or re.search(
        r"forum/([A-Za-z0-9_-]+)", paper_url
    )
    return m.group(1) if m and "openreview" in paper_url else None


def icml_pdf_index() -> dict[str, Path]:
    """OpenReview forum id → local ICML2025 PDF path."""
    idx: dict[str, Path] = {}
    if not ICML_META.exists():
        return idx
    for line in ICML_META.read_text().splitlines():
        d = json.loads(line)
        pdf = d.get("pdf_path")
        if d.get("id") and pdf:
            p = ICML_PDF_DIR / Path(pdf).name
            if p.exists():
                idx[d["id"]] = p
    return idx


def page1_text(pdf_bytes: bytes) -> str:
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc[0].get_text()


def cmd_fetch() -> None:
    cards = json.loads(CARDS_PATH.read_text())
    affil = load_affiliations()
    prune_pages(affil)
    icml = icml_pdf_index()
    PAGES_DIR.mkdir(exist_ok=True)

    todo = [c for c in cards if c["slug"] not in affil["benchmarks"]]
    unresolved: list[str] = []
    for i, card in enumerate(todo):
        slug, paper = card["slug"], card.get("paper") or ""
        out = PAGES_DIR / f"{slug}.txt"
        if out.exists():
            continue
        pdf: bytes | None = None
        aid = arxiv_id(paper)
        forum = openreview_forum(paper)
        try:
            if aid:
                pdf = http_get(f"https://arxiv.org/pdf/{aid}")
                time.sleep(1)  # arXiv politeness
            elif forum and forum in icml:
                pdf = icml[forum].read_bytes()
        except (urllib.error.URLError, OSError) as e:
            sys.stderr.write(f"  ! {slug}: fetch failed ({e})\n")
        if pdf and pdf[:4] == b"%PDF":
            out.write_text(page1_text(pdf))
            sys.stderr.write(f"  ✓ {slug} ({i + 1}/{len(todo)})\n")
        else:
            unresolved.append(slug)

    sys.stderr.write(f"\n{len(todo) - len(unresolved)} page texts in {PAGES_DIR}\n")
    if unresolved:
        sys.stderr.write(
            f"{len(unresolved)} unresolved (locate paper manually / via Claude):\n"
        )
        for s in unresolved:
            sys.stderr.write(f"  {s}\n")


# ---------------------------------------------------------------------------
#  logos — resolve every institution mentioned by a benchmark
# ---------------------------------------------------------------------------

# Names we key benchmarks by that registry relevance-search gets wrong
# ("MIT" → MIT World Peace University / International Tourism Institute):
# query the registries with the unambiguous long form instead.
SEARCH_ALIASES = {
    "MIT": "Massachusetts Institute of Technology",
    "Meta": "Meta Platforms",
    "UCLA": "University of California, Los Angeles",
    "UT Austin": "University of Texas at Austin",
    "Caltech": "California Institute of Technology",
    "GSK": "GlaxoSmithKline",
    "Sony": "Sony (Japan)",
}

# Not real institutions — never look up, never badge with a wrong logo.
SKIP_LOOKUP = {"Independent"}

# Known registry misses with an obvious official domain (favicon source).
DOMAIN_OVERRIDES = {
    "METR": "metr.org",
    "Scale AI": "scale.com",
    "ByteDance": "bytedance.com",
    "Chinese University of Hong Kong": "www.cuhk.edu.hk",
    "Hebrew University": "huji.ac.il",
    "Huazhong University of Science and Technology": "hust.edu.cn",
    "Indian Institute of Science": "iisc.ac.in",
    "Isaacus": "isaacus.com",
    "JetBrains Research": "jetbrains.com",
    "LatchBio": "latch.bio",
    "Mercor": "mercor.com",
    "Nebius": "nebius.com",
    "Patronus AI": "patronus.ai",
    "Roboflow": "roboflow.com",
}

# Fixed placeholder images the favicon services return for unknown domains —
# an exact byte match means "no icon", so the UI falls back to initials.
PLACEHOLDER_MD5 = {
    "b8a0bf372c762e966cc99ede8682bc71",  # google s2 generic globe
    "ab1fb25b83d4b333ea661a84bd298b2e",  # duckduckgo letterbox placeholder
}


# After 3 consecutive throttled lookups, stop asking OpenAlex for the rest of
# the run (the block lasts far longer than the run; ROR covers the fallback).
_openalex_throttled = 0


def openalex_institution(query: str) -> dict | None:
    global _openalex_throttled
    if _openalex_throttled >= 3:
        return None
    q = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/institutions?search={q}&per-page=5"
        f"&mailto=sttruong@cs.stanford.edu"
    )
    for attempt in range(2):
        try:
            results = json.loads(http_get(url)).get("results", [])
            _openalex_throttled = 0
            if not results:
                return None
            # Relevance alone mismatches short names; the intended institution
            # is virtually always the most prolific match.
            return max(results, key=lambda r: r.get("works_count") or 0)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:  # throttled — one quick retry
                time.sleep(3)
                continue
            if e.code == 429:
                _openalex_throttled += 1
            return None
        except (urllib.error.URLError, OSError):
            return None
    return None


def ror_institution(query: str) -> dict | None:
    """ROR fallback (no logo images, but canonical name/type/homepage).
    Shaped like an OpenAlex record so download_logo can consume either."""
    q = urllib.parse.quote(query)
    try:
        items = json.loads(http_get(f"https://api.ror.org/v2/organizations?query={q}")).get(
            "items", []
        )
    except (urllib.error.URLError, OSError):
        return None
    if not items:
        return None
    exact = [
        r for r in items
        if any(n["value"].lower() == query.lower() for n in r.get("names", []))
    ]
    rec = (exact or items)[0]
    display = next(
        (n["value"] for n in rec.get("names", []) if "ror_display" in n["types"]),
        query,
    )
    site = next(
        (l["value"] for l in rec.get("links", []) if l.get("type") == "website"),
        None,
    )
    types = [t for t in rec.get("types", []) if t != "funder"]
    return {
        "display_name": display,
        "type": types[0] if types else None,
        "homepage_url": site,
        "image_thumbnail_url": None,
    }


def wikidata_logo(query: str) -> str | None:
    """Sharp official mark via Wikidata: entity search → P154 (logo image) or
    P158 (seal) claim → Commons thumbnail. Independent of the OpenAlex API."""
    api = "https://www.wikidata.org/w/api.php"
    try:
        s = json.loads(http_get(
            f"{api}?action=wbsearchentities&format=json&language=en&limit=1"
            f"&search={urllib.parse.quote(query)}"
        ))
        hits = s.get("search", [])
        if not hits:
            return None
        qid = hits[0]["id"]
        for prop in ("P154", "P158"):
            c = json.loads(http_get(
                f"{api}?action=wbgetclaims&format=json&entity={qid}&property={prop}"
            ))
            claims = c.get("claims", {}).get(prop, [])
            if claims:
                filename = claims[0]["mainsnak"]["datavalue"]["value"]
                return (
                    "https://commons.wikimedia.org/w/index.php?title="
                    f"Special:Redirect/file/{urllib.parse.quote(filename)}&width=256"
                )
    except (urllib.error.URLError, OSError, KeyError, TypeError):
        return None
    return None


def sniff_ext(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"\x00\x00\x01\x00":
        return ".ico"
    if data.lstrip()[:5] in (b"<svg ", b"<?xml"):
        return ".svg"
    return None


def download_logo(name: str, rec: dict | None, query: str | None = None) -> str | None:
    """Try the Wikidata mark, then the homepage favicon. Returns the site path."""
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    base = inst_slug(name)
    for existing in LOGO_DIR.glob(f"{base}.*"):
        return f"/institutions/{existing.name}"

    candidates: list[str] = []
    if rec and rec.get("image_thumbnail_url"):
        candidates.append(rec["image_thumbnail_url"])
    else:
        wd = wikidata_logo(query or name)
        if wd:
            candidates.append(wd)
    host = DOMAIN_OVERRIDES.get(name)
    homepage = (rec or {}).get("homepage_url")
    if not host and homepage:
        host = urllib.parse.urlparse(homepage).netloc
    if host:
        candidates.append(
            f"https://www.google.com/s2/favicons?domain={host}&sz=128"
        )
        candidates.append(f"https://icons.duckduckgo.com/ip3/{host}.ico")
    for url in candidates:
        try:
            data = http_get(url)
        except (urllib.error.URLError, OSError):
            continue
        ext = sniff_ext(data)
        if not ext or hashlib.md5(data).hexdigest() in PLACEHOLDER_MD5:
            continue
        path = LOGO_DIR / f"{base}{ext}"
        path.write_bytes(data)
        return f"/institutions/{path.name}"
    return None


def cmd_logos() -> None:
    affil = load_affiliations()
    prune_pages(affil)
    wanted: list[str] = []
    for entry in affil["benchmarks"].values():
        for name in entry.get("institutions") or []:
            if name not in affil["institutions"] and name not in wanted:
                wanted.append(name)
    for i, name in enumerate(wanted):
        if name in SKIP_LOOKUP:
            affil["institutions"][name] = {"logo": None, "type": None, "canonical": name}
            continue
        query = SEARCH_ALIASES.get(name, name)
        rec = openalex_institution(query) or ror_institution(query)
        logo = download_logo(name, rec, query)
        affil["institutions"][name] = {
            "logo": logo,
            "type": (rec or {}).get("type"),
            "canonical": (rec or {}).get("display_name"),
        }
        sys.stderr.write(
            f"  {'✓' if logo else '·'} {name} ({i + 1}/{len(wanted)})\n"
        )
        time.sleep(0.2)
    save_affiliations(affil)
    n_logo = sum(1 for v in affil["institutions"].values() if v["logo"])
    sys.stderr.write(
        f"\n{len(affil['institutions'])} institutions ({n_logo} with logos) → {AFFIL_PATH}\n"
    )


def cmd_prompt() -> None:
    """Print a ready-to-paste Claude prompt for the extraction step (step 2)."""
    cards = json.loads(CARDS_PATH.read_text())
    affil = load_affiliations()
    missing = [c["slug"] for c in cards if c["slug"] not in affil["benchmarks"]]
    if not missing:
        print("All cards have affiliation entries — nothing to extract.")
        return
    print(f"""\
Fill in author affiliations for {len(missing)} new benchmark(s) in the gallery.

For each slug below, read
scripts/render_website/.affiliation-pages/<slug>.txt — the extracted page-1
text of the benchmark's paper. If a file is missing, find the paper yourself
(the card's `paper` URL in website/content/generated/benchmark-cards.json, or arXiv
search by benchmark name) and read its first page.

For each paper, determine the institutional affiliations of the FIRST TWO
authors (in listed author order): resolve the superscript/footnote mapping,
take each author's first-listed institution, dedupe if both share one
(single-author papers get one institution).

Naming rules — these strings are grouping keys, so reuse an existing key from
the "institutions" section of website/content/curated/benchmark-affiliations.json
whenever it is the same place:
- short common English names ("Stanford University", "MIT", "UC Berkeley",
  "Carnegie Mellon University", "Tsinghua University"); no departments/labs
- company labs by brand ("Google DeepMind", "Microsoft Research", "Meta",
  "OpenAI", "NVIDIA", "Alibaba"); Chinese Academy of Sciences institutes
  → "Chinese Academy of Sciences"

Then add one entry per slug to the "benchmarks" section of
website/content/curated/benchmark-affiliations.json (keep it alphabetically sorted):
  "<slug>": {{"institutions": ["<inst1>", "<inst2>"], "source": "page1-llm"}}
Use "institutions": null plus a "note" only if the paper truly lists none.

Finally run:
  python3 scripts/render_website/generate_benchmark_affiliations.py logos
  python3 scripts/render_website/generate_benchmark_gallery.py cards

Slugs: {", ".join(missing)}""")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        cmd_fetch()
    elif cmd == "logos":
        cmd_logos()
    elif cmd == "prompt":
        cmd_prompt()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
