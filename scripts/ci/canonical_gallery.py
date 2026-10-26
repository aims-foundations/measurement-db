"""Add canonical records to a prepared benchmark bundle during the migration.

Runs before artifact hashing, so CI validates and deploys the same data. Legacy
bundles stay available for every other catalog entry. No rendering occurs here.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.render_website.generate_benchmark_gallery_new import export_benchmark
from .check_gallery_records import check


def prepare(slug: str, source: dict, tables: Path, data: Path, web: Path) -> dict | None:
    selection = web / "content/curated/canonical-record-benchmarks.json"
    if not selection.exists() or slug not in json.loads(selection.read_text()):
        return None
    directory = data / "records"
    base = f"https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/{source['revision']}"
    export_benchmark(tables / slug, directory, 1024, parquet_base=base)
    (data / "records-validation.json").write_text(json.dumps(check(tables / slug, directory), indent=2))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if descriptor := manifest["tables"].get("traces"):
        # The existing server route signs pinned HF downloads; credentials never reach the browser.
        descriptor["parquet"]["url"] = f"/api/gallery-source/{slug}/traces"
    # Compression is transport only. The browser still receives canonical rows.
    for path in directory.rglob("*.json"):
        if path == manifest_path:
            continue
        path.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        path.unlink()
    manifest["compressed"] = True
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"{slug}: canonical records: " + json.dumps({k: v["rows"] for k, v in manifest["tables"].items()}), flush=True)
    return {"base": f"/benchmark-data/{slug}/records", "manifest": manifest}
