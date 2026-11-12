"""Legacy chart and analysis artifacts, retained independently of record preparation."""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path
import time
from . import gallery_artifacts as artifacts
from .gallery_packs import pack_lookups
ROOT = Path(__file__).resolve().parents[2]
FIT_OPTIONS = {"test_size": 0.2, "seed": 1, "lr": 0.05, "max_iter": 4000,
               "tol": 1e-7, "eval_every": 4000}

def timeline_counts(path: Path) -> dict:
    import pandas as pd

    subjects = pd.read_parquet(path, columns=["normalized_name", "provider", "release_date"])
    named = subjects[subjects.normalized_name.fillna("").str.strip().ne("")]
    return {name: {field: dict(Counter(v for v in group[field]
                                     if isinstance(v, str) and v.strip()))
                   for field in ("provider", "release_date")}
            for name, group in named.groupby("normalized_name")}


def render_legacy_bundle(slug: str, source: dict, tables: Path, web: Path, output: Path,
                         *, asset_root: Path = ROOT, fit_options: dict = FIT_OPTIONS) -> dict:
    """Render only local, hash-verified tables; never mutate a shared catalog."""
    import pandas as pd
    from scripts.render_website import generate_benchmark_gallery as gallery
    from scripts.render_website.generate_chart_marginals import build_marginals
    from scripts.analyze_measurements import benchmark_saturation as saturation
    from scripts.analyze_measurements import fit_rasch_models as rasch

    gallery.LOCAL_DATA_DIR, gallery.LOCAL_SOURCES = tables, {slug: source}
    gallery.HF_REVISION = source["revision"]
    gallery.source_revision.cache_clear()
    gallery.source_tables.cache_clear()
    # The historical fitter imports the gallery by its standalone module name.
    rasch.gallery, rasch.fetch = gallery, gallery.fetch
    timings = {}

    def stage(name, function):
        started = time.monotonic()
        result = function()
        timings[name] = round(time.monotonic() - started, 3)
        print(f"{slug}: {name}: {timings[name]:.3f}s", flush=True)
        return result

    dynamic = set(json.loads((web / "content/curated/dynamic-item-benchmarks.json").read_text()))
    detail = stage("gallery", lambda: gallery.build_website_data(
        [slug], tables, web, dynamic_items=dynamic))[slug]
    data = web / "public/benchmark-data" / slug
    verdict = stage("saturation", lambda: saturation.web_payload(
        saturation.analyze_folders([tables / slug]))[slug])
    card = gallery.card_for(slug, gallery.read_meta(slug, tables, False),
                            gallery.load_overrides(web).get(slug, {}),
                            gallery.load_affiliations(web), verdict)
    if card is None:
        raise ValueError(f"{slug}: selected benchmark produced no public card")
    if card["image"] and not (asset_root / "website/public" / card["image"].lstrip("/")).is_file():
        card["image"] = None
    for institution in card["institutions"] or []:
        if institution["logo"] and not (asset_root / "website/public" / institution["logo"].lstrip("/")).is_file():
            institution["logo"] = None
    has_responses = "responses.parquet" in source["tables"]
    marginals = stage("marginals", lambda: build_marginals([slug], tables) if has_responses else
                      {"slugs": {slug: {"binary": False}}, "models": {}, "marginals": []})
    timeline = stage("timeline", lambda: timeline_counts(tables / slug / "subjects.parquet"))

    def fit():
        if not has_responses:
            return None, "responses.parquet unavailable"
        try:
            frame = rasch.binary_frame(rasch.load_responses(slug, tables, False))
        except rasch.NotBinary as exc:
            return None, str(exc)
        result = rasch.fit(frame, **fit_options, device="cpu")
        fit_dir = output / "fit"
        summary = rasch.write_outputs(slug, frame, result, rasch.subject_names(slug, tables, False),
                                     fit_dir, curves=False, write_responses=False)
        theta = pd.read_csv(fit_dir / "subjects.csv", dtype={"subject_id": str})
        summary.pop("slug")
        summary.update(source={"repo": artifacts.SOURCE_REPO, "revision": source["revision"]},
                       fit=fit_options, theta={r.subject_id: round(float(r.theta), 4)
                                               for r in theta.itertuples()})
        with gzip.open(data / "irt.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(rasch.item_difficulties(fit_dir), stream, separators=(",", ":"), allow_nan=False)
        return summary, None

    irt, skipped = stage("rasch", fit)
    source_descriptor = stage("packing", lambda: pack_lookups(data))
    summary = {"card": card, "detail": detail, "saturation": verdict, "marginals": marginals,
               "timeline": timeline, "irt": irt, "irt_skip_reason": skipped}
    summary["source"] = source_descriptor
    artifacts.write_json(data / "summary.json", summary)
    return {"data": data, "timings": timings}
