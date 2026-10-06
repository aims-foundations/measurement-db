#!/usr/bin/env python3
"""Build homepage model summaries from the gallery's downloaded HF tables."""

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1] / "website"


def build_marginals(slugs: list[str], cache: Path) -> dict:
    models, benchmarks, marginals = {}, {}, []
    for slug in slugs:
        folder = cache / slug
        response = folder / "responses.parquet"
        if not response.exists():
            response = folder / "response.parquet"
        resp = pd.read_parquet(response, columns=["subject_id", "item_id", "response"])
        subjects = pd.read_parquet(
            folder / "subjects.parquet",
            columns=["subject_id", "normalized_name", "provider"],
        ).dropna(subset=["normalized_name"])
        subjects = subjects[subjects.normalized_name.str.strip().ne("")]
        subjects = subjects.rename(columns={"normalized_name": "model"})
        resp = resp.merge(subjects, on="subject_id", validate="many_to_one")
        models.update(dict(zip(subjects.model, subjects.provider.fillna("Unknown"))))

        values = resp.response.dropna()
        binary = not values.empty and set(values.unique()).issubset({0.0, 1.0})
        benchmarks[slug] = {"binary": binary}
        items = resp.groupby("model").item_id.nunique()
        scored = resp.dropna(subset=["response"]).groupby("model").response.agg(["sum", "count"])
        for model, count in items.items():
            row = {"slug": slug, "model": model, "items": int(count)}
            if binary and model in scored.index:
                row.update(sum=round(float(scored.loc[model, "sum"]), 4),
                           n=int(scored.loc[model, "count"]))
            marginals.append(row)
    return {"slugs": benchmarks, "models": models, "marginals": marginals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".hf-cache")
    parser.add_argument("--web-dir", type=Path, default=WEB)
    args = parser.parse_args()
    generated = args.web_dir / "content" / "generated"
    cards = json.loads((generated / "benchmark-cards.json").read_text())
    payload = build_marginals([c["slug"] for c in cards], args.cache_dir)
    out = generated / "chart-marginals.json"
    out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {len(payload['marginals'])} model summaries across {len(cards)} benchmarks to {out}")


if __name__ == "__main__":
    main()
