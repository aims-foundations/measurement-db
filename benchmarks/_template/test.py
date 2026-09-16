#!/usr/bin/env python3
"""Read-only characterization tests for the sibling benchmark builder.

The tests inspect generated Parquet tables and never download or write data.
Missing build outputs skip on a clean checkout unless
``MEASUREMENT_DB_FULL_TEST=1`` is set. Once outputs exist, a missing reviewed
``testdata/characterization.json`` is always an error.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import pandas as pd


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_measurement_tables import (  # noqa: E402
    validate_asset_relations,
    validate_table,
    validate_trace_relations,
)


CHARACTERIZATION_PATH = BENCHMARK_DIR / "testdata" / "characterization.json"
OUTPUT_NAMES = ("items", "subjects", "benchmarks", "responses")


def _digest_strings(values: Iterable[object]) -> str:
    """Return an order-independent digest of a collection of strings."""
    digest = hashlib.sha256()
    for value in sorted(str(value) for value in values):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _digest_asset_rows(assets: pd.DataFrame | None) -> str:
    """Fingerprint asset metadata and the exact bytes, independent of row order."""

    digest = hashlib.sha256()
    if assets is None:
        return digest.hexdigest()
    for row in sorted(assets.itertuples(index=False), key=lambda value: value.asset_id):
        metadata = json.dumps(
            [row.asset_id, row.benchmark_id, int(row.byte_size)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = row.data
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class ExampleBenchmarkCharacterizationTests(unittest.TestCase):
    """Logical snapshot and benchmark-specific invariants for one release."""

    @classmethod
    def setUpClass(cls) -> None:
        missing_outputs = [
            name
            for name in OUTPUT_NAMES
            if not (BENCHMARK_DIR / f"{name}.parquet").is_file()
        ]
        if missing_outputs:
            message = (
                "generated benchmark table(s) are absent: "
                f"{', '.join(missing_outputs)}; run the sibling build.py first"
            )
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)

        if not CHARACTERIZATION_PATH.is_file():
            raise RuntimeError(
                f"{CHARACTERIZATION_PATH} is absent; review the completed "
                "build and record its characterization"
            )

        cls.expected = json.loads(CHARACTERIZATION_PATH.read_text(encoding="utf-8"))[
            "release"
        ]
        cls.items = pd.read_parquet(BENCHMARK_DIR / "items.parquet")
        assets_path = BENCHMARK_DIR / "assets.parquet"
        cls.assets = pd.read_parquet(assets_path) if assets_path.is_file() else None
        cls.subjects = pd.read_parquet(BENCHMARK_DIR / "subjects.parquet")
        cls.benchmarks = pd.read_parquet(BENCHMARK_DIR / "benchmarks.parquet")
        cls.responses = pd.read_parquet(BENCHMARK_DIR / "responses.parquet")
        traces_path = BENCHMARK_DIR / "traces.parquet"
        cls.traces = pd.read_parquet(traces_path) if traces_path.is_file() else None

    def test_release_shape(self) -> None:
        self.assertEqual(len(self.items), self.expected["items"])
        observed_asset_count = 0 if self.assets is None else len(self.assets)
        self.assertEqual(observed_asset_count, self.expected["assets"])
        self.assertEqual(len(self.subjects), self.expected["subjects"])
        self.assertEqual(len(self.responses), self.expected["responses"])
        observed_trace_count = 0 if self.traces is None else len(self.traces)
        self.assertEqual(observed_trace_count, self.expected["traces"])
        self.assertEqual(len(self.benchmarks), 1)

    def test_item_and_response_integrity(self) -> None:
        for item in self.items.itertuples(index=False):
            has_text = isinstance(item.content, str) and bool(item.content.strip())
            has_manifest = isinstance(item.asset_manifest, str) and bool(
                item.asset_manifest
            )
            self.assertTrue(
                has_text or has_manifest,
                f"item {item.item_id} has neither text nor attached content",
            )
        self.assertFalse(
            self.responses.duplicated(
                [
                    "subject_id",
                    "item_id",
                    "trial",
                    "test_condition",
                    "interactors",
                ]
            ).any()
        )

    def test_asset_integrity(self) -> None:
        benchmark_ids = set(self.benchmarks["benchmark_id"])
        self.assertEqual(len(benchmark_ids), 1)
        benchmark_id = next(iter(benchmark_ids))
        self.assertEqual(set(self.items["benchmark_id"]), {benchmark_id})
        validate_asset_relations(
            self.items,
            self.assets,
            benchmark_id=benchmark_id,
            context=BENCHMARK_DIR.name,
            response_scale=self.benchmarks.iloc[0].response_scale,
        )

    def test_trace_response_links(self) -> None:
        validate_trace_relations(self.responses, self.traces, context=BENCHMARK_DIR.name)

    def test_canonical_table_schemas(self) -> None:
        validate_table("items", self.items, context=BENCHMARK_DIR.name)
        validate_table("subjects", self.subjects, context=BENCHMARK_DIR.name)
        validate_table(
            "benchmarks",
            self.benchmarks,
            include_derived=True,
            context=BENCHMARK_DIR.name,
        )
        validate_table(
            "responses",
            self.responses,
            include_derived=True,
            context=BENCHMARK_DIR.name,
        )
        if self.traces is not None:
            validate_table("traces", self.traces, context=BENCHMARK_DIR.name)

    def test_reviewed_fingerprints(self) -> None:
        item_rows = [
            json.dumps(
                [
                    row.item_id,
                    row.raw_item_id,
                    row.content,
                    row.asset_manifest,
                    row.content_hash,
                    row.item_features,
                    row.grading_criterion,
                    row.verifier,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in self.items.itertuples(index=False)
        ]
        subject_rows = [
            json.dumps(
                [
                    row.subject_id,
                    row.display_name,
                    row.normalized_name,
                    row.provider,
                    row.release_date,
                    row.access_date,
                    row.harness,
                    row.reasoning_effort,
                    row.harness_version,
                    row.subject_features_extra,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in self.subjects.itertuples(index=False)
        ]
        response_cells = [
            json.dumps(
                [
                    row.response_id,
                    row.subject_id,
                    row.item_id,
                    row.benchmark_id,
                    int(row.trial),
                    float(row.response),
                    None if pd.isna(row.test_condition) else row.test_condition,
                    None if pd.isna(row.interactors) else row.interactors,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in self.responses.itertuples(index=False)
        ]
        benchmark_rows = [
            json.dumps(
                [
                    row.benchmark_id,
                    row.name,
                    row.version,
                    row.license,
                    row.source_url,
                    row.description,
                    row.one_line_description,
                    list(row.modality),
                    list(row.domain),
                    row.multi_single_turn,
                    row.response_type,
                    row.response_scale,
                    bool(row.categorical),
                    row.paper_url,
                    row.release_date,
                    row.granularity,
                    row.release,
                    row.benchmark_features,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in self.benchmarks.itertuples(index=False)
        ]
        trace_rows = (
            []
            if self.traces is None
            else [
                json.dumps(
                    [
                        row.response_id,
                        row.subject_id,
                        row.item_id,
                        row.benchmark_id,
                        int(row.trial),
                        None if pd.isna(row.test_condition) else row.test_condition,
                        None if pd.isna(row.interactors) else row.interactors,
                        row.trace,
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for row in self.traces.itertuples(index=False)
            ]
        )
        observed = {
            "item_rows_sha256": _digest_strings(item_rows),
            "asset_rows_sha256": _digest_asset_rows(self.assets),
            "subject_rows_sha256": _digest_strings(subject_rows),
            "benchmark_rows_sha256": _digest_strings(benchmark_rows),
            "response_cells_sha256": _digest_strings(response_cells),
            "trace_rows_sha256": _digest_strings(trace_rows),
        }
        self.assertEqual(
            observed,
            {fingerprint: self.expected[fingerprint] for fingerprint in observed},
        )

    # Add short, named tests here for release-specific coverage, grading, and
    # trace-attribution invariants that generic schema validation cannot know.


if __name__ == "__main__":
    unittest.main(verbosity=2)
