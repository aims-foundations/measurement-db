"""An optional, disposable cache must never weaken artifact verification."""
import os
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ci import gallery_cache as cache
from test_gallery_artifacts import candidate_fixture


class GalleryCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate, manifest = candidate_fixture(self.root)
        self.entry = manifest["benchmarks"]["alpha"]
        self.archive = self.candidate / self.entry["path"]
        environment = patch.dict(os.environ, GALLERY_CACHE_DIR=str(self.root / "cache"))
        environment.start()
        self.addCleanup(environment.stop)

    def test_input_identity_and_corruption_control_hits(self):
        target = self.root / "restored.tar"
        self.assertFalse(cache.restore(self.entry, target))
        cache.remember("alpha", self.entry, self.archive)
        self.assertEqual(cache.lookup("alpha", self.entry["inputs"]), self.entry)
        self.assertIsNone(cache.lookup("alpha", {**self.entry["inputs"], "renderer": "changed"}))
        self.assertTrue(cache.restore(self.entry, target))
        target.write_bytes(b"consumer modification")
        self.assertTrue(cache.matches(cache.object_path(self.entry), self.entry))
        cache.object_path(self.entry).write_bytes(b"corrupt")
        self.assertFalse(cache.restore(self.entry, target))
        self.assertIsNone(cache.lookup("alpha", self.entry["inputs"]))
        cache.store(self.archive, self.entry)
        self.assertTrue(cache.restore(self.entry, target))

    def test_disabled_cache_and_unverified_writes(self):
        with patch.dict(os.environ, GALLERY_CACHE_DIR=""):
            cache.remember("alpha", self.entry, self.archive)
            self.assertIsNone(cache.lookup("alpha", self.entry["inputs"]))
            self.assertFalse(cache.restore(self.entry, self.root / "absent"))
        self.archive.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "unverified"):
            cache.store(self.archive, self.entry)

    def test_record_changes_reuse_charts_but_source_library_and_renderer_changes_do_not(self):
        old = {"tables": {"items": "a"}, "canonical_records": False,
               "generator": {"code": "old", "legacy_code": "charts", "libraries": {"numpy": "1"}}}
        new = copy.deepcopy(old)
        new["canonical_records"] = True
        new["generator"]["code"] = "new record exporter"
        self.assertTrue(cache.same_legacy_inputs(old, new))
        for changed in [
            {**new, "tables": {"items": "b"}},
            {**new, "generator": {**new["generator"], "legacy_code": "changed"}},
            {**new, "generator": {**new["generator"], "libraries": {"numpy": "2"}}},
        ]:
            self.assertFalse(cache.same_legacy_inputs(old, changed))
        unknown = {**old, "generator": {"code": "unknown", "libraries": {"numpy": "1"}}}
        self.assertFalse(cache.same_legacy_inputs(unknown, new))
