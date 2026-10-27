"""Prepare verified canonical records for every published benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.render_website.generate_benchmark_gallery_new import export_benchmark
from scripts.render_website.record_packs import pack_records
from .check_gallery_records import check


def prepare(slug: str, source: dict, tables: Path, data: Path, web: Path) -> dict | None:
    directory = data / "records"
    base = f"https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/{source['revision']}"
    export_benchmark(tables / slug, directory, 1024, parquet_base=base)
    (data / "records-validation.json").write_text(json.dumps(check(tables / slug, directory), indent=2))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, descriptor in manifest["tables"].items():
        if "parquet" in descriptor:
            descriptor["parquet"]["download"] = f"{base}/{source.get('directory', slug + '/formatted_tables')}/{name}.parquet"
            descriptor["parquet"]["url"] = f"/api/gallery-source/{slug}/{name}"
    manifest["pack"] = pack_records(directory)
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"{slug}: canonical records: " + json.dumps({k: v["rows"] for k, v in manifest["tables"].items()}), flush=True)
    return {"base": f"/benchmark-data/{slug}/records", "manifest": manifest}
