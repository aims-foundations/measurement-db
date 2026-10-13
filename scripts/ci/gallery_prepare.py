"""Prepare independent static gallery bundles after strict HF validation.

Run via ``python -m scripts.ci.gallery_prepare``. Worker processes own isolated
directories; only the parent assembles shared catalogs. No HF writes occur here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from . import gallery_artifacts as artifacts
from . import gallery_cache as cache
from .gallery_packs import pack_lookups

ROOT = Path(__file__).resolve().parents[2]
FIT_OPTIONS = {"test_size": 0.2, "seed": 1, "lr": 0.05, "max_iter": 4000,
               "tol": 1e-7, "eval_every": 4000}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def generator_identity(root: Path = ROOT) -> dict:
    paths = [p for folder in ("scripts/render_website", "scripts/analyze_measurements")
             for p in (root / folder).rglob("*.py")]
    paths += [root / "scripts/ci" / name for name in
              ("gallery_prepare.py", "gallery_artifacts.py", "gallery_packs.py", "gallery-requirements.txt")]
    libraries = {name: importlib.metadata.version(name) for name in
                 ("numpy", "pandas", "pyarrow", "fastparquet", "cramjam", "scikit-learn",
                  "scipy", "torch", "PyYAML")}
    return {"code": artifacts.digest({str(p.relative_to(root)): artifacts.file_digest(p)
                                      for p in sorted(paths)}),
            "libraries": libraries, "python": platform.python_version(),
            "fit": FIT_OPTIONS, "device": "cpu"}


def source_tree(api, revision: str, path: str = "") -> list:
    """Retry transient listing failures without interpreting them as missing data."""
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(5):
        try:
            return list(api.list_repo_tree(artifacts.SOURCE_REPO, repo_type="dataset",
                                          revision=revision, path_in_repo=path or None))
        except (HfHubHTTPError, httpx.TransportError) as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 4:
                raise
            try:
                delay = float(response.headers.get("Retry-After", 2 ** attempt)) if response is not None else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            if delay > 60:
                raise  # Respect a longer server-requested pause; let CI retry later.
            print(f"Transient HF listing error for {path or '/'}; retry {attempt + 1}/4 in {max(1, delay)}s", flush=True)
            time.sleep(max(1, delay))
    raise AssertionError("unreachable")


def describe_source(api, slug: str, revision: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    import yaml

    entries = source_tree(api, revision, slug)
    directory = f"{slug}/formatted_tables" if any(
        e.path == f"{slug}/formatted_tables" for e in entries) else slug
    if directory != slug:
        entries = source_tree(api, revision, directory)
    if not any(e.path == f"{directory}/benchmarks.parquet" for e in entries):
        return None
    metadata_path = Path(hf_hub_download(artifacts.SOURCE_REPO, f"{slug}/metadata.yaml",
                                        revision=revision, repo_type="dataset"))
    metadata = yaml.safe_load(metadata_path.read_text())
    if metadata["benchmark"].get("release") in {"withheld", "private"}:
        return None
    tables = {}
    for entry in entries:
        if not entry.path.endswith(".parquet"):
            continue
        name = Path(entry.path).name
        if name == "response.parquet":
            if any(e.path == f"{directory}/responses.parquet" for e in entries):
                continue
            name = "responses.parquet"
        sha = entry.lfs.sha256 if entry.lfs else artifacts.file_digest(Path(hf_hub_download(
            artifacts.SOURCE_REPO, entry.path, revision=revision, repo_type="dataset")))
        tables[name] = {"path": entry.path, "sha256": sha, "size": entry.size}
    return {"revision": revision, "directory": directory, "tables": tables,
            "metadata_sha256": artifacts.file_digest(metadata_path)}


def benchmark_inputs(slug: str, source: dict, generator: dict, web: Path) -> dict:
    def load(name):
        return json.loads((web / "content/curated" / name).read_text())

    affiliations = load("benchmark-affiliations.json")
    benchmark_affiliations = affiliations.get("benchmarks", {}).get(slug, {})
    institutions = {name: affiliations.get("institutions", {}).get(name)
                    for name in benchmark_affiliations.get("institutions", [])}
    override = load("benchmark-overrides.json").get(slug, {})
    asset_names = [override.get("image"), *[row.get("logo") for row in institutions.values() if row]]
    assets = {name: artifacts.file_digest(web / "public" / name.lstrip("/"))
              if (web / "public" / name.lstrip("/")).is_file() else None
              for name in asset_names if name}
    return {"slug": slug, "tables": source["tables"], "directory": source["directory"],
            "metadata_sha256": source["metadata_sha256"], "generator": generator,
            "override": override, "assets": assets,
            "affiliations": benchmark_affiliations, "institutions": institutions,
            "dynamic_items": slug in load("dynamic-item-benchmarks.json")}


def timeline_counts(path: Path) -> dict:
    import pandas as pd

    subjects = pd.read_parquet(path, columns=["normalized_name", "provider", "release_date"])
    named = subjects[subjects.normalized_name.fillna("").str.strip().ne("")]
    return {name: {field: dict(Counter(v for v in group[field]
                                     if isinstance(v, str) and v.strip()))
                   for field in ("provider", "release_date")}
            for name, group in named.groupby("normalized_name")}


def combine_timelines(summaries: dict) -> list[dict]:
    models = {}
    for slug, summary in sorted(summaries.items()):
        for name, fields in summary["timeline"].items():
            model = models.setdefault(name, {"provider": Counter(), "release_date": Counter(), "slugs": []})
            model["slugs"].append(slug)
            for field in ("provider", "release_date"):
                model[field].update(fields[field])

    def mode(counts):
        return min(counts, key=lambda value: (-counts[value], value)) if counts else None

    rows = [{"name": name, "provider": mode(row["provider"]),
             "releaseDate": mode(row["release_date"]), "benchmarks": row["slugs"]}
            for name, row in models.items()]
    return sorted(rows, key=lambda row: (row["releaseDate"] or "9999", row["name"]))


def render_bundle(slug: str, source: dict, tables: Path, web: Path, output: Path) -> dict:
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
    if card["image"] and not (ROOT / "website/public" / card["image"].lstrip("/")).is_file():
        card["image"] = None
    for institution in card["institutions"] or []:
        if institution["logo"] and not (ROOT / "website/public" / institution["logo"].lstrip("/")).is_file():
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
        result = rasch.fit(frame, **FIT_OPTIONS, device="cpu")
        fit_dir = output / "fit"
        summary = rasch.write_outputs(slug, frame, result, rasch.subject_names(slug, tables, False),
                                     fit_dir, curves=False, write_responses=False)
        theta = pd.read_csv(fit_dir / "subjects.csv", dtype={"subject_id": str})
        summary.pop("slug")
        summary.update(source={"repo": artifacts.SOURCE_REPO, "revision": source["revision"]},
                       fit=FIT_OPTIONS, theta={r.subject_id: round(float(r.theta), 4)
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


def prepare_one(task: dict) -> tuple[str, dict, dict, dict]:
    from huggingface_hub import hf_hub_download

    slug, inputs = task["slug"], task["inputs"]
    key = artifacts.input_key(inputs)
    candidate = Path(task["candidate"])
    start = time.monotonic()
    with TemporaryDirectory(prefix=f"gallery-{slug}-") as temporary:
        root = Path(temporary)
        previous = task.get("previous")
        accepted = bool(previous and previous["key"] == key)
        if not accepted:
            previous = cache.lookup(slug, inputs)
        if previous and previous["key"] == key:
            path = root / "cached.tar"
            if not cache.restore(previous, path):
                if not accepted:
                    raise ValueError("Cached archive disappeared during preparation; retry the job")
                path = Path(hf_hub_download(artifacts.GALLERY_REPO, "v2/" + previous["path"],
                                           revision=task["gallery_revision"], repo_type="dataset"))
            data = root / "cached"
            artifacts.unpack(path, data, previous, {"slug": slug, "key": key})
            target = candidate / previous["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            cache.remember(slug, previous, target)
            summary = json.loads((data / "summary.json").read_text())
            timing = {"cache_hit": True, "total_seconds": round(time.monotonic() - start, 3)}
            print(f"{slug}: reused prepared bundle ({timing['total_seconds']}s)", flush=True)
            return slug, previous, summary, timing

        tables = root / "tables"
        for name, meta in inputs["tables"].items():
            path = Path(hf_hub_download(artifacts.SOURCE_REPO, meta["path"],
                                       revision=task["source"]["revision"], repo_type="dataset"))
            if path.stat().st_size != meta["size"] or artifacts.file_digest(path) != meta["sha256"]:
                raise ValueError(f"{slug}/{name}: downloaded table does not match pinned HF bytes")
            target = tables / slug / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path.resolve())
        web = root / "website"
        shutil.copytree(ROOT / "website/content/curated", web / "content/curated")
        # Public assets affect card rendering; their paths remain relative to the
        # final application. No generated data is read from another worker.
        result = render_bundle(slug, task["source"], tables, web, root)
        archive = root / "bundle.tar"
        meta = artifacts.pack(result["data"], archive, {"slug": slug, "key": key})
        relative = f"benchmarks/{slug}/{key}/{meta['sha256']}.tar"
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(archive, destination)
        entry = {**meta, "path": relative, "key": key, "inputs": inputs}
        cache.remember(slug, entry, destination)
        timing = {"cache_hit": False, **result["timings"],
                  "total_seconds": round(time.monotonic() - start, 3)}
        return slug, entry, json.loads((result["data"] / "summary.json").read_text()), timing


def assemble(summaries: dict, destination: Path) -> None:
    from scripts.render_website import generate_benchmark_attributions as attributions

    summaries = dict(sorted(summaries.items()))
    marginals = {"slugs": {}, "models": {}, "marginals": []}
    for summary in summaries.values():
        for key in ("slugs", "models"):
            marginals[key].update(summary["marginals"][key])
        marginals["marginals"].extend(summary["marginals"]["marginals"])
    payloads = {
        "benchmark-cards.json": [s["card"] for s in summaries.values()],
        "benchmark-details.json": {slug: s["detail"] for slug, s in summaries.items()},
        "benchmark-saturation.json": {slug: s["saturation"] for slug, s in summaries.items()},
        "benchmark-irt.json": {slug: s["irt"] for slug, s in summaries.items() if s["irt"] is not None},
        "gallery-sources.json": {slug: s["source"] for slug, s in summaries.items()},
        "chart-marginals.json": marginals, "model-timeline.json": combine_timelines(summaries),
    }
    for name, payload in payloads.items():
        artifacts.write_json(destination / name, payload)
    curated = ROOT / "website/content/curated"
    payload = attributions.build_manifest(
        repo_root=ROOT, cards_path=destination / "benchmark-cards.json",
        hidden_path=curated / "hidden-benchmarks.json",
        affiliations_path=curated / "benchmark-affiliations.json",
        overrides_path=curated / "benchmark-attribution-overrides.json")
    (destination / "benchmark-attributions.json").write_text(attributions.render_manifest(payload))


def prepare(report_path: Path, candidate: Path, workers: int) -> None:
    from huggingface_hub import HfApi, RepoFolder, hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    from .benchmark_release import load_release

    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four workers until resource measurements justify more")
    if candidate.exists() and any(candidate.iterdir()):
        raise ValueError("Candidate directory must be empty")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Preparation requires a clean tracked source tree")
    revision = os.environ["HF_REVISION"]
    if not artifacts.COMMIT.fullmatch(revision):
        raise ValueError("HF_REVISION must be pinned before preparation")
    sha, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    run_id, attempt = os.environ["GITHUB_RUN_ID"], int(os.environ["GITHUB_RUN_ATTEMPT"])
    report = json.loads(report_path.read_text())
    receipts = artifacts.validate_report(report, revision=revision, sha=sha, run_id=run_id, attempt=attempt)
    api = HfApi()
    gallery_revision = api.dataset_info(artifacts.GALLERY_REPO).sha
    try:
        previous = json.loads(Path(hf_hub_download(artifacts.GALLERY_REPO, "v2/latest.json",
                                                   revision=gallery_revision, repo_type="dataset")).read_text())
        artifacts.validate_manifest(previous)
    except RemoteEntryNotFoundError:
        previous = {"benchmarks": {}}
    generator = generator_identity()
    hidden = set(json.loads((ROOT / "website/content/curated/hidden-benchmarks.json").read_text()))
    tasks = []
    roots = source_tree(api, revision)
    for folder in roots:
        slug = folder.path
        if not isinstance(folder, RepoFolder) or slug in hidden or not artifacts.SLUG.fullmatch(slug):
            continue
        local_metadata = ROOT / "benchmarks" / slug / "metadata.yaml"
        if local_metadata.exists() and load_release(local_metadata)[0] == "withheld":
            continue
        source = describe_source(api, slug, revision)
        if source is None:
            continue
        if slug in receipts:
            expected = {name: {"size": item["size"], "sha256": item["sha256"]}
                        for name, item in source["tables"].items()}
            if receipts[slug] != expected:
                raise ValueError(f"{slug}: gallery source differs from byte-validated tables")
        inputs = benchmark_inputs(slug, source, generator, ROOT / "website")
        tasks.append({"slug": slug, "source": source, "inputs": inputs,
                      "candidate": str(candidate.resolve()), "gallery_revision": gallery_revision,
                      "previous": previous["benchmarks"].get(slug)})
    if not tasks:
        raise ValueError("No public benchmarks to prepare")
    entries, summaries, timings = {}, {}, {}
    # Spawn + one task per process releases large Arrow/PyTorch heaps between
    # benchmarks and keeps module-level renderer state private to each worker.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        for slug, entry, summary, timing in pool.map(prepare_one, sorted(tasks, key=lambda t: t["slug"])):
            entries[slug], summaries[slug], timings[slug] = entry, summary, timing
    with TemporaryDirectory(prefix="gallery-catalog-") as temporary:
        generated = Path(temporary) / "generated"
        assemble(summaries, generated)
        archive = Path(temporary) / "catalog.tar"
        catalog = artifacts.pack(generated, archive, {"git_tree": tree})
        catalog["path"] = f"catalog/{catalog['sha256']}.tar"
        target = candidate / catalog["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, target)
    manifest = {"schema_version": artifacts.SCHEMA_VERSION, "git_sha": sha, "git_tree": tree,
                "source_repo": artifacts.SOURCE_REPO, "source_revision": revision,
                "validation": {"run_id": run_id, "run_attempt": attempt,
                               "report_sha256": artifacts.file_digest(report_path)},
                "benchmarks": entries, "catalog": catalog}
    # Only archives already present in this accepted, immutable HF release may
    # be omitted from the portable PR handoff. Local cache hits alone never
    # justify dropping an archive from that fallback.
    known = {entry["path"] for entry in previous["benchmarks"].values()}
    manifest["base_gallery"] = {"revision": gallery_revision,
                                "paths": sorted(entry["path"] for entry in entries.values()
                                                if entry["path"] in known)}
    artifacts.validate_manifest(manifest)
    artifacts.write_json(candidate / "manifest.json", manifest)
    shutil.copyfile(report_path, candidate / "validation.json")
    artifacts.write_json(candidate / "timings.json", timings)
    stage_handoff(candidate, manifest)
    hits = sum(t["cache_hit"] for t in timings.values())
    print(f"Prepared {len(entries)} benchmarks: {hits} reused, {len(entries) - hits} generated.")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as output:
            output.write(f"### Gallery preparation\n\n{hits} reused; {len(entries) - hits} generated.\n\n"
                         "| Benchmark | Cache | Gallery (s) | Rasch (s) | Total (s) |\n|---|---|---:|---:|---:|\n")
            for slug, timing in sorted(timings.items()):
                output.write(f"| {slug} | {'hit' if timing['cache_hit'] else 'miss'} | "
                             f"{timing.get('gallery', 0)} | {timing.get('rasch', 0)} | {timing['total_seconds']} |\n")


def stage_handoff(candidate: Path, manifest: dict) -> None:
    """Cache full objects locally; upload only objects absent from accepted HF."""
    handoff = candidate / "handoff"
    handoff.mkdir()
    base = set(manifest.get("base_gallery", {}).get("paths", []))
    changed_bytes = changed = 0
    for entry in [manifest["catalog"], *manifest["benchmarks"].values()]:
        source = candidate / entry["path"]
        cache.store(source, entry)
        if entry["path"] not in base:
            target = handoff / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are immutable, verified candidate files. Hard links avoid a
            # second local copy; GitHub receives regular file bytes.
            target.hardlink_to(source)
            changed_bytes += entry["size"]
            changed += 1
    for name in ("manifest.json", "validation.json", "timings.json"):
        shutil.copyfile(candidate / name, handoff / name)
    print(f"Portable handoff: {changed} archives, {changed_bytes:,} bytes; "
          f"{len(base)} accepted archives reused from pinned HF.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.report, args.out, args.workers)


if __name__ == "__main__":
    main()
