"""Packed lookup parity, stable content identity, and bounded browser reads."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ci import gallery_packs as packs


class GalleryPackTests(unittest.TestCase):
    def write(self, root, name, value, timestamp):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(packs.encoded(value), mtime=timestamp))

    def test_roundtrip_every_entry_and_deterministic_files_across_build_times(self):
        with TemporaryDirectory() as temporary:
            snapshots = []
            for timestamp in (1, 9999):
                root = Path(temporary) / str(timestamp)
                expected = {"item": {}, "answer": {}}
                for family in expected:
                    for bucket in range(256):
                        values = {f"{family}-{bucket}": {"text": "Unicode: 日本語", "value": bucket, "empty": None}}
                        expected[family].update(values)
                        self.write(root, f"{family}/{bucket:02x}.json.gz", values, timestamp)
                self.write(root, "source.json.gz", {"dynamicItems": False}, timestamp)
                self.write(root, "view.json.gz", {"unchanged": [1, None, 0.25]}, timestamp)
                with patch.object(packs, "CHUNK_BYTES", 400), patch.object(packs, "PACK_BYTES", 600):
                    source = packs.pack_lookups(root)
                index_path = root / source["lookupPack"]["index"]
                index = json.loads(index_path.read_bytes())
                self.assertEqual(index["format"], packs.FORMAT)
                for family, wanted in expected.items():
                    self.assertFalse((root / family).exists())
                    restored = {}
                    for pack, offset, length, sha in index["families"][family]["chunks"].values():
                        descriptor = index["packs"][pack]
                        data = (root / "lookup-pack" / descriptor["file"]).read_bytes()
                        self.assertEqual(packs.digest(data), descriptor["sha256"])
                        chunk = data[offset:offset + length]
                        self.assertEqual(packs.digest(chunk), sha)
                        restored.update(json.loads(gzip.decompress(chunk)))
                    self.assertEqual(restored, wanted)
                self.assertEqual(json.loads(gzip.decompress((root / "view.json.gz").read_bytes())),
                                 {"unchanged": [1, None, 0.25]})
                snapshots.append({str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertEqual(*snapshots)

    def test_large_single_bucket_stays_isolated_and_empty_family_remains_readable(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, "source.json.gz", {}, 1)
            self.write(root, "item/00.json.gz", {"large": "x" * 5000}, 1)
            self.write(root, "item/01.json.gz", {"small": 1}, 1)
            (root / "empty").mkdir()
            with patch.object(packs, "CHUNK_RAW", 100), patch.object(packs, "PACK_BYTES", 20):
                source = packs.pack_lookups(root)
            index = json.loads((root / source["lookupPack"]["index"]).read_bytes())
            self.assertEqual(index["families"]["item"]["bits"], 8)
            self.assertEqual(len(index["packs"]), 2)
            self.assertEqual(index["families"]["empty"]["chunks"], {})

    def test_duplicate_keys_are_rejected_before_legacy_files_are_removed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, "source.json.gz", {}, 1)
            for bucket in ("00", "01"):
                self.write(root, f"item/{bucket}.json.gz", {"duplicate": bucket}, 1)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                packs.pack_lookups(root)
            self.assertTrue((root / "item/00.json.gz").exists())
