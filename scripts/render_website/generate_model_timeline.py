"""Build website/content/generated/model-timeline.json — one row per model in the bank.

Reads every published benchmark's ``subjects.parquet`` from the local
``.hf-cache`` and aggregates the registry columns filled by the subjects
auto-fill pass:

  normalized_name  canonical model name (dedup key across benchmarks)
  provider         company behind the model
  release_date     "YYYY-MM-DD" or null

Each output row is ``{name, provider, releaseDate, benchmarks}`` where
``benchmarks`` lists the SLUGS the model appears in — a list, not a count, so
the website can scope the release timeline to a section's shelf by
intersecting with that shelf's slug list (model-timeline.ts). Models whose
subjects rows carry no normalized_name are skipped (they cannot be
deduplicated across benchmarks); undated models are kept with a null
releaseDate and excluded by the chart at render time, matching how the
benchmark timeline treats undated cards.

This used to list subjects.parquet files on the HuggingFace repo, which made
the chart cover whatever the REMOTE happened to hold — including slugs the
website doesn't publish — and needed the network. The bank is defined by
benchmark-cards.json; a partial local cache aborts loudly instead of shipping
a chart that covers a subset and looks complete.

Usage:  python scripts/render_website/generate_model_timeline.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / ".hf-cache"
CARDS = REPO / "website" / "content" / "generated" / "benchmark-cards.json"
OUT = REPO / "website" / "content" / "generated" / "model-timeline.json"


def mode(values: pd.Series) -> str | None:
    """Most common non-null value; ties broken by the smallest (oldest date,
    alphabetically first provider) so reruns are deterministic."""
    counts = Counter(v for v in values if isinstance(v, str) and v.strip())
    if not counts:
        return None
    return min(counts, key=lambda v: (-counts[v], v))


def main(out: Path = OUT, cache: Path = CACHE, cards_path: Path = CARDS) -> None:
    cards = json.loads(cards_path.read_text())
    slugs = [c["slug"] for c in cards]
    missing = [s for s in slugs if not (cache / s / "subjects.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(slugs)} published benchmarks have no cached "
            f"subjects.parquet:\n  " + "\n  ".join(sorted(missing))
            + "\n\nFetch them before regenerating — the timeline would otherwise "
            "cover a subset of the bank and still look complete."
        )

    frames = []
    for slug in slugs:
        df = pd.read_parquet(
            cache / slug / "subjects.parquet",
            columns=["normalized_name", "provider", "release_date"],
        )
        df["slug"] = slug
        frames.append(df)
    subjects = pd.concat(frames, ignore_index=True)

    named = subjects[subjects["normalized_name"].fillna("").str.strip() != ""]
    skipped = len(subjects) - len(named)

    rows = []
    for name, grp in named.groupby("normalized_name"):
        rows.append(
            {
                "name": name,
                "provider": mode(grp["provider"]),
                "releaseDate": mode(grp["release_date"]),
                "benchmarks": sorted(grp["slug"].unique().tolist()),
            }
        )
    rows.sort(key=lambda r: (r["releaseDate"] or "9999", r["name"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    dated = sum(1 for r in rows if r["releaseDate"])
    print(
        f"wrote {out}: {len(rows)} models ({dated} dated, "
        f"{len(rows) - dated} undated) across {len(slugs)} benchmarks; "
        f"{skipped} un-normalized subject rows skipped"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--web-dir", type=Path, default=REPO / "website")
    args = parser.parse_args()
    generated = args.web_dir / "content" / "generated"
    main(generated / "model-timeline.json", args.cache_dir,
         generated / "benchmark-cards.json")
