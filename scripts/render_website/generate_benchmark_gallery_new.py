#!/usr/bin/env python3
"""Export canonical Parquet records for the website's TypeScript gallery.

    python scripts/render_website/generate_benchmark_gallery_new.py \
        benchmarks/aegis benchmarks/wildvision --output /tmp/gallery-new
    pnpm --dir website gallery:prepare /tmp/gallery-new
    pnpm --dir website dev

Input is a benchmark directory (containing formatted_tables/) or a table directory.
The table builder owns schema validation. This exporter preserves rows and IDs;
the viewer owns presentation. No network access or benchmark-specific overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.build_measurement_tables.response_scales import (
    canonical_response_scale,
    validate_scale_type,
)
from scripts.render_website.record_parquet import export_parquet

PRIMARY_KEYS = {
    "subjects": "subject_id",
    "items": "item_id",
    "responses": "response_id",
    "traces": "response_id",
    "assets": "asset_id",
}


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def read_metadata(directory: Path) -> dict:
    """Use the exported metadata; catch stale tables when source YAML is present."""
    rows = pq.read_table(directory / "benchmarks.parquet").to_pylist()
    if len(rows) != 1:
        raise ValueError(f"{directory}: expected exactly one benchmark row")
    metadata = rows[0]
    scale = canonical_response_scale(metadata["response_scale"])
    validate_scale_type(metadata["response_type"], scale)
    yaml_path = (
        directory.parent if directory.name == "formatted_tables" else directory
    ) / "metadata.yaml"
    if yaml_path.exists():
        source = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["benchmark"]
        if (
            source["response_type"] != metadata["response_type"]
            or canonical_response_scale(source["response_scale"]) != scale
        ):
            raise ValueError(
                f"{yaml_path}: grade metadata differs from the tables; rebuild them first"
            )
    return metadata


def export_table(path: Path, output: Path, key: str, page_size: int) -> dict:
    """Write unchanged rows in pages, with an ID-to-page index for lazy lookups."""
    output.mkdir()
    index = {}
    subjects: dict[str, set[int]] = {}
    pages = 0
    for pages, batch in enumerate(
        pq.ParquetFile(path).iter_batches(batch_size=page_size), 1
    ):
        records = {}
        for row in batch.to_pylist():
            identity = row[key]
            if not isinstance(identity, str) or identity in index:
                raise ValueError(f"{path}: missing or duplicate {key}: {identity!r}")
            # Bytes need a file URL; every other field stays exactly as recorded.
            if path.stem == "assets":
                filename = f"{len(index)}.bin"
                (output / filename).write_bytes(row.pop("data"))
                row["data_url"] = filename
            records[identity] = row
            index[identity] = pages - 1
            if path.stem == "responses":
                subjects.setdefault(row["subject_id"], set()).add(pages - 1)
        write_json(output / f"{pages - 1}.json", records)
    write_json(output / "index.json", index)
    if path.stem == "responses":
        write_json(output / "subjects.json", {k: sorted(v) for k, v in subjects.items()})
    return {"key": key, "rows": len(index), "pages": pages,
            **({"bySubject": "subjects.json"} if path.stem == "responses" else {})}


def export_benchmark(source: Path, output: Path, page_size: int, parquet_base: str | None = None) -> dict:
    directory = (
        source / "formatted_tables"
        if (source / "formatted_tables").is_dir()
        else source
    )
    metadata = read_metadata(directory)
    required = {"subjects", "items"}
    if metadata["granularity"] == "item":
        required.add("responses")
    tables = {}
    output.mkdir()
    for name, key in PRIMARY_KEYS.items():
        path = directory / f"{name}.parquet"
        if path.exists():
            if name == "traces" and parquet_base:
                relative = f"{metadata['benchmark_id']}/"
                if directory.name == "formatted_tables":
                    relative += "formatted_tables/"
                tables[name] = export_parquet(path, output / name, key,
                                             f"{parquet_base.rstrip('/')}/{relative}{path.name}")
            else:
                tables[name] = export_table(path, output / name, key, page_size)
        elif name in required:
            raise FileNotFoundError(path)
    write_json(output / "manifest.json", {"benchmark": metadata, "tables": tables})
    return {
        "id": metadata["benchmark_id"],
        "name": metadata["name"],
        "manifest": f"{output.name}/manifest.json",
    }


def build(sources: list[Path], output: Path, *, page_size: int = 1024,
          parquet_base: str | None = None) -> None:
    """Publish a complete data export, leaving source tables and presentation alone."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gallery-", dir=output.parent) as temporary:
        stage = Path(temporary) / "site"
        stage.mkdir()
        catalog = [
            export_benchmark(source, stage / str(i), page_size, parquet_base)
            for i, source in enumerate(sources)
        ]
        if len({entry["id"] for entry in catalog}) != len(catalog):
            raise ValueError("Each benchmark may appear only once")
        write_json(stage / "index.json", catalog)
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="local canonical benchmark/table directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for the exported records",
    )
    parser.add_argument("--parquet-base", help="Pinned source URL; leave traces in Parquet and export only lookup indexes")
    args = parser.parse_args()
    build(args.sources, args.output, parquet_base=args.parquet_base)
    print(f"Gallery records written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
