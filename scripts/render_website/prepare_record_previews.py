"""Prepare optional bounded reads; run when source tables change, never for UI edits.

python -m scripts.render_website.prepare_record_previews --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

from .record_previews import export_previews, source_ranges, verify_previews

REPO = "aims-foundations/measurement-db"


def prepare(job):
    slug, table, path, output = job
    groups = source_ranges(path)
    blocked = sum(end-start for group in groups for start, end in group["ranges"])
    if not blocked:
        return slug, table, None
    key = "item_id" if table == "items" else "response_id"
    if not (output / "manifest.json").exists():
        export_previews(path, output, key, groups)
    saved = json.loads((output / "manifest.json").read_text())
    if not saved.get("source_sha256"):
        raise ValueError("Cannot resume an archive without its original source hash")
    verification = verify_previews(path, output, key)
    manifest = json.loads((output / "manifest.json").read_text())
    return slug, table, {"manifest": manifest, "verification": verification}


def publish(output, results, catalog, branch):
    api = HfApi()
    revisions = set()
    for slug, tables in results.items():
        for table, result in tables.items():
            url = catalog[slug]["manifest"]["tables"][table]["parquet"]["download"]
            revision, path = url.split("/resolve/", 1)[1].split("/", 1)
            info = api.get_paths_info(REPO, [path], revision=revision, repo_type="dataset")[0]
            if not info.lfs or info.lfs.sha256 != result["verification"]["source_sha256"]:
                raise ValueError(f"Local table does not match pinned source: {slug}/{table}")
            revisions.add(revision)
    if len(revisions) != 1:
        raise ValueError("Publish from one verified canonical snapshot at a time")
    api.create_branch(REPO, branch=branch, revision=revisions.pop(), repo_type="dataset", exist_ok=True)
    staging = output / "upload"
    for slug, tables in results.items():
        for table in tables:
            target = staging / "website-record-previews" / slug / table
            target.mkdir(parents=True, exist_ok=True)
            for path in (output / slug / table).iterdir():
                if path.name == "manifest.json" or path.name.startswith("pack-"):
                    dest = target / path.name
                    if not dest.exists():
                        os.link(path, dest)
    api.upload_large_folder(repo_id=REPO, repo_type="dataset", revision=branch,
                            folder_path=staging, num_workers=8)
    revision = api.repo_info(REPO, revision=branch, repo_type="dataset").sha
    # Verify published bytes by the server's SHA256, not just upload completion.
    paths = [p for p in staging.rglob("*") if p.is_file() and ".cache" not in p.parts]
    for start in range(0, len(paths), 50):
        batch = paths[start:start+50]
        names = [str(p.relative_to(staging)) for p in batch]
        remote = {info.path: info for info in api.get_paths_info(REPO, names, revision=revision, repo_type="dataset")}
        for path, name in zip(batch, names):
            info = remote[name]
            if info.lfs:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != info.lfs.sha256:
                    raise ValueError(f"Published file checksum mismatch: {name}")
            else:
                raw = path.read_bytes()
                digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                if digest != info.blob_id:
                    raise ValueError(f"Published file checksum mismatch: {name}")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True, help="Local <slug>/<table>.parquet directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; completed tables can be resumed")
    parser.add_argument("--catalog", type=Path, default=Path("website/content/generated/gallery-records.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--publish-branch", help="Explicit HF preview branch; never main")
    parser.add_argument("--website-manifest", type=Path, default=Path("website/content/generated/record-previews.json"))
    args = parser.parse_args()
    if args.publish_branch and not args.publish_branch.startswith("preview/"):
        parser.error("--publish-branch must start with preview/")
    catalog = json.loads(args.catalog.read_text())
    jobs = [(slug, table, args.tables / slug / f"{table}.parquet", args.output / slug / table)
            for slug, entry in catalog.items() for table in ("items", "traces")
            if entry["manifest"]["tables"].get(table, {}).get("parquet")]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(prepare, job) for job in jobs]):
            slug, table, result = future.result()
            if result:
                results.setdefault(slug, {})[table] = result
                print(f"Verified {slug}/{table}: {result['manifest']['rows']} records", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verified.json").write_text(json.dumps(results, indent=2))
    if args.publish_branch:
        revision = publish(args.output, results, catalog, args.publish_branch)
        manifest = {slug: {table: {
            "source": catalog[slug]["manifest"]["tables"][table]["parquet"]["download"],
            "revision": revision, "directory": f"website-record-previews/{slug}/{table}",
            **{key: result["manifest"][key] for key in ("rows", "packs", "indexes")},
        } for table, result in tables.items()} for slug, tables in results.items()}
        args.website_manifest.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        print(f"Pinned website record manifest to {revision}")


if __name__ == "__main__":
    main()
