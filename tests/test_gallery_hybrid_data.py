import base64
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_website import generate_benchmark_gallery as gallery
from test_gallery_output_parity import local_fetch, write_fixture


def fixture(root):
    cache = root / "tables"
    web = root / "website"
    sizes, expected = {}, {}
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
    for slug in ["binary", "normalized"]:
        write_fixture(cache, slug)
        path = cache / slug / "items.parquet"
        items = pq.read_table(path).to_pylist()
        items[0]["asset_manifest"] = json.dumps([
            {"asset_id": "image/雪", "role": "input", "media_type": "image/png", "ordinal": 1}])
        items[1]["content"] += f"\n![embedded](data:image/png;base64,{base64.b64encode(png).decode()})"
        if slug == "normalized":
            items[2]["content"] = "\n".join(f"Prompt {i}: 模型😀" for i in range(3000))
        # Keep nullable columns present even when the first row has no value.
        for row in items:
            row.setdefault("asset_manifest", None)
        pq.write_table(pa.Table.from_pylist(items), path, row_group_size=2, data_page_size=256)
        pq.write_table(pa.Table.from_pylist([{"asset_id": "image/雪", "data": png}]), cache / slug / "assets.parquet")
        path = cache / slug / "traces.parquet"
        traces = pq.read_table(path).to_pylist()
        traces[0]["trace"] = "😀 " * 30000
        traces[1]["trace"] = None
        pq.write_table(pa.Table.from_pylist(traces), path, row_group_size=3, data_page_size=256)
        output = web / "public/benchmark-data" / slug
        output.mkdir(parents=True)
        with patch.object(gallery, "fetch", side_effect=local_fetch), \
             patch.object(gallery, "source_revision", return_value="fixture-revision"), \
             patch.object(gallery, "source_tables", return_value=gallery.TableSource(slug, frozenset({"assets.parquet", "traces.parquet"}))):
            sizes[slug] = gallery.write_website_lookups(slug, cache, output, True)
        key = [*gallery.KEY, *(["interactors"] if "interactors" in traces[0] else [])]
        expected[slug] = {
            "items": {row["item_id"]: gallery.static_item(slug, row) for row in items},
            "answers": [{"key": [row[c] for c in key], "value": {"trace": gallery.truncate_trace(row["trace"])}} for row in traces],
            "image": base64.b64encode(png).decode(),
        }
    with patch.object(gallery, "DYNAMIC_ITEM_COUNT", 1):
        selected = gallery.select_dynamic_items(web / "public/benchmark-data", sizes)
    expected["dynamic"] = selected
    (root / "expected.json").write_text(json.dumps(expected, ensure_ascii=False))
    return expected


class HybridDataTests(unittest.TestCase):
    def test_largest_items_are_dynamic_and_binary_payloads_are_not_shipped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = fixture(root)
            self.assertEqual(expected["dynamic"], ["normalized"])
            for slug in ["binary", "normalized"]:
                output = root / "website/public/benchmark-data" / slug
                self.assertEqual((output / "item").exists(), slug == "binary")
                self.assertFalse((output / "asset").exists())
                for bucket in (output / "answer").glob("*.gz"):
                    values = json.loads(gzip.decompress(bucket.read_bytes())).values()
                    self.assertTrue(all(len(value) == 2 and all(isinstance(v, int) for v in value) for value in values))
                for bucket in (output / "item").glob("*.gz"):
                    self.assertNotIn(b"data:image", gzip.decompress(bucket.read_bytes()))
            catalog = json.loads((root / "website/content/generated/gallery-sources.json").read_text())
            self.assertTrue(catalog["normalized"]["dynamicItems"])
            self.assertFalse(catalog["binary"]["dynamicItems"])
            self.assertEqual(catalog["binary"]["revision"], "fixture-revision")

    def test_row_positions_preserve_groups_and_null_observation_fields(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.parquet"
            rows = [{"subject_id": "模型", "item_id": "item/😀", "test_condition": None,
                     "trial": 1, "interactors": value} for value in [None, "opponent=a", "opponent=b"]]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            keys = [*gallery.KEY, "interactors"]
            actual = list(gallery.source_row_keys(path, keys))
            self.assertEqual(actual, [([row[k] for k in keys], [i // 2, i % 2]) for i, row in enumerate(rows)])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export":
        root = Path(sys.argv[2])
        root.mkdir(parents=True, exist_ok=True)
        fixture(root)
    else:
        unittest.main()
