#!/usr/bin/env python3
"""Source-concordance and characterization tests for FinanceBench."""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import unittest

import fitz
import pandas as pd


BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "financebench"
REPO_ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from measurement_db.scripts.build_measurement_tables.validate_benchmark_metadata import (  # noqa: E402
    load_benchmark_metadata,
)


METADATA = load_benchmark_metadata(BENCHMARK_DIR / "metadata.yaml")
from measurement_db.scripts.build_measurement_tables.validate_characterization import load_characterization, check_tables
CHARACTERIZATION = load_characterization(BENCHMARK_DIR / "characterization.yaml")
OUTPUT_NAMES = (
    "items",
    "assets",
    "subjects",
    "benchmarks",
    "responses",
    "traces",
)

EXPECTED_RESULT_LABELS = {
    ("claude-2", "inContext"): (56, 91, 3),
    ("claude-2", "inContext_reverse"): (114, 32, 4),
    ("gpt-4-1106-preview", "closedBook"): (14, 5, 131),
    ("gpt-4-1106-preview", "inContext"): (37, 54, 59),
    ("gpt-4-1106-preview", "inContext_reverse"): (118, 26, 6),
    ("gpt-4-1106-preview", "oracle"): (128, 22, 0),
    ("gpt-4-1106-preview", "oracle_reverse"): (134, 14, 2),
    ("gpt-4-1106-preview", "sharedStore"): (29, 20, 101),
    ("gpt-4-1106-preview", "singleStore"): (75, 17, 58),
    ("gpt-4", "closedBook"): (7, 1, 142),
    ("gpt-4", "oracle"): (126, 15, 9),
    ("gpt-4", "oracle_reverse"): (118, 17, 15),
    ("gpt-4", "sharedStore"): (25, 13, 112),
    ("gpt-4", "singleStore"): (63, 16, 71),
    ("llama2", "sharedStore"): (29, 104, 17),
    ("llama2", "singleStore"): (62, 81, 7),
}

# Table 2 uses the context-first/context-last choices described in the paper.
PAPER_MAIN_CORRECT_COUNTS = {
    ("gpt-4-1106-preview", "closedBook"): 14,
    ("llama2", "sharedStore"): 29,
    ("gpt-4-1106-preview", "sharedStore"): 29,
    ("llama2", "singleStore"): 62,
    ("gpt-4-1106-preview", "singleStore"): 75,
    ("claude-2", "inContext_reverse"): 114,
    ("gpt-4-1106-preview", "inContext_reverse"): 118,
    ("gpt-4-1106-preview", "oracle"): 128,
}


def _decode_jsonl(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _parse_features(serialized: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in serialized.split(";") if part)






class FinanceBenchValidationTests(unittest.TestCase):
    """Independently compare the pinned release with the generated tables."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = [
            name
            for name in OUTPUT_NAMES
            if not (BENCHMARK_DIR / "formatted_tables" / f"{name}.parquet").is_file()
        ]
        if missing:
            message = (
                "generated FinanceBench table(s) are absent: "
                f"{', '.join(missing)}; run benchmarks/financebench/build.py first"
            )
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)

        downloads = {source["name"]: source for source in METADATA["sources"]["upstream"]}
        cls.archive_path = BENCHMARK_DIR / "raw" / downloads["provider_archive"]["file"]
        cls.paper_path = BENCHMARK_DIR / "raw" / downloads["paper_v1"]["file"]
        if not cls.archive_path.is_file() or not cls.paper_path.is_file():
            raise RuntimeError("pinned FinanceBench source files are absent")
        for source_name, source_path in (
            ("provider_archive", cls.archive_path),
            ("paper_v1", cls.paper_path),
        ):
            descriptor = downloads[source_name]
            if source_path.stat().st_size != descriptor["size"]:
                raise RuntimeError(f"pinned source size mismatch: {source_name}")
            digest = hashlib.sha256()
            with source_path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != descriptor["sha256"]:
                raise RuntimeError(f"pinned source SHA-256 mismatch: {source_name}")

        cls.items = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "items.parquet")
        cls.assets = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "assets.parquet")
        cls.subjects = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "subjects.parquet")
        cls.benchmarks = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "benchmarks.parquet")
        cls.responses = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "responses.parquet")
        cls.traces = pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / "traces.parquet")

        layout = METADATA["build"]["parameters"]["paths"]
        archive_root = layout["root"]
        questions_member = layout["questions_member"]
        document_information_member = layout["document_information_member"]
        notebook_member = layout["notebook_member"]
        pdfs_prefix = layout["pdfs_prefix"]
        results_prefix = layout["results_prefix"]

        cls.questions = []
        cls.document_information = []
        cls.notebook = ""
        cls.source_results: dict[
            tuple[str, str], dict[str, dict[str, object]]
        ] = {}
        cls.source_pdf_fingerprints: dict[str, tuple[str, int]] = {}
        cls.pdf_member_count = 0
        target_documents: set[str] = set()

        # A single streaming pass keeps the test practical for the large gzip.
        with tarfile.open(cls.archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != archive_root:
                    raise AssertionError(f"unexpected archive member {member.name}")
                relative = PurePosixPath(*parts[1:])
                extracted = None
                if relative.as_posix() == questions_member:
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    cls.questions = _decode_jsonl(extracted.read())
                    target_documents = {
                        str(row["doc_name"]) for row in cls.questions
                    }
                elif relative.as_posix() == document_information_member:
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    cls.document_information = _decode_jsonl(extracted.read())
                elif relative.as_posix() == notebook_member:
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    cls.notebook = extracted.read().decode("utf-8")
                elif relative.parent.as_posix() == pdfs_prefix and relative.suffix == ".pdf":
                    cls.pdf_member_count += 1
                    if relative.stem in target_documents:
                        extracted = archive.extractfile(member)
                        assert extracted is not None
                        payload = extracted.read()
                        cls.source_pdf_fingerprints[relative.stem] = (
                            hashlib.sha256(payload).hexdigest(),
                            len(payload),
                        )
                elif (
                    relative.parent.as_posix() == results_prefix
                    and relative.suffix == ".jsonl"
                ):
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    rows = _decode_jsonl(extracted.read())
                    configs = {
                        (str(row["model_name"]), str(row["eval_mode"]))
                        for row in rows
                    }
                    if len(configs) != 1:
                        raise AssertionError(f"mixed configuration in {relative}")
                    config = configs.pop()
                    cls.source_results[config] = {
                        str(row["financebench_id"]): row for row in rows
                    }

        cls.questions_by_id = {
            str(row["financebench_id"]): row for row in cls.questions
        }
        cls.subject_configs: dict[str, tuple[str, str]] = {}
        for row in cls.subjects.itertuples(index=False):
            features = _parse_features(row.subject_features_extra)
            cls.subject_configs[row.subject_id] = (
                features["endpoint_label"],
                features["evaluation_mode"],
            )
        cls.item_qids = dict(zip(cls.items["item_id"], cls.items["raw_item_id"]))

    def test_provider_counts_and_internal_release_consistency(self) -> None:
        counts = CHARACTERIZATION["source_claims"]
        self.assertEqual(
            len(self.questions), counts["open_source_questions"]["expected"]
        )
        self.assertEqual(len(self.questions_by_id), len(self.questions))
        self.assertEqual(
            Counter(row["question_type"] for row in self.questions),
            {"metrics-generated": 50, "domain-relevant": 50, "novel-generated": 50},
        )
        self.assertEqual(len(self.document_information), 361)
        self.assertEqual(
            len({row["doc_name"] for row in self.document_information}), 360
        )
        self.assertEqual(self.pdf_member_count, 368)
        self.assertEqual(
            len(self.source_pdf_fingerprints),
            counts["referenced_source_documents"]["expected"],
        )
        self.assertEqual(
            len(self.source_results),
            counts["released_subject_configurations"]["expected"],
        )
        self.assertEqual(set(self.source_results), set(EXPECTED_RESULT_LABELS))

        total_labels: Counter[str] = Counter()
        for config, results in self.source_results.items():
            self.assertEqual(set(results), set(self.questions_by_id))
            labels = Counter(str(row["label"]) for row in results.values())
            expected = EXPECTED_RESULT_LABELS[config]
            self.assertEqual(
                (
                    labels["Correct Answer"],
                    labels["Incorrect Answer"],
                    labels["Refusal"],
                ),
                expected,
            )
            total_labels.update(labels)
            for qid, result in results.items():
                self.assertEqual(result["question"], self.questions_by_id[qid]["question"])
                self.assertEqual(result["temp"], 0.01)
                self.assertIn("model_answer", result)
                self.assertIsNotNone(result["model_answer"])

        self.assertEqual(
            total_labels,
            {"Correct Answer": 1135, "Incorrect Answer": 528, "Refusal": 737},
        )

    def test_curated_items_and_assets_match_exact_source_bytes(self) -> None:
        self.assertEqual(len(self.items), 150)
        self.assertEqual(len(self.assets), 84)
        self.assertEqual(set(self.items["raw_item_id"]), set(self.questions_by_id))

        assets_by_id = self.assets.set_index("asset_id")
        linked_ids: set[str] = set()
        for item in self.items.itertuples(index=False):
            source = self.questions_by_id[item.raw_item_id]
            self.assertEqual(item.content, source["question"])
            self.assertEqual(json.loads(item.grading_criterion)["reference_answer"], source["answer"])
            manifest = json.loads(item.asset_manifest)
            self.assertEqual(len(manifest), 1)
            link = manifest[0]
            self.assertEqual(link["path"], f"pdfs/{source['doc_name']}.pdf")
            self.assertEqual(link["media_type"], "application/pdf")
            self.assertEqual(link["role"], "source_document")
            self.assertEqual(link["ordinal"], 1)
            linked_ids.add(link["asset_id"])

            asset = assets_by_id.loc[link["asset_id"]]
            payload = asset["data"]
            source_digest, source_size = self.source_pdf_fingerprints[
                str(source["doc_name"])
            ]
            self.assertEqual(asset["benchmark_id"], "financebench")
            self.assertEqual(asset["byte_size"], len(payload))
            self.assertEqual(asset["byte_size"], source_size)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), link["asset_id"])
            self.assertEqual(link["asset_id"], source_digest)

        self.assertEqual(linked_ids, set(self.assets["asset_id"]))
        for asset in self.assets.itertuples(index=False):
            with self.subTest(asset_id=asset.asset_id):
                with fitz.open(
                    stream=io.BytesIO(asset.data),
                    filetype="pdf",
                ) as document:
                    self.assertGreater(len(document), 0)

        # Demonstrate that a consumer can reopen the exact PDF directly from
        # the Parquet byte cell without reconstructing a filesystem archive.
        first_asset_id = self.assets.iloc[0]["asset_id"]
        selected_asset = pd.read_parquet(
            BENCHMARK_DIR / "formatted_tables" / "assets.parquet",
            filters=[("asset_id", "==", first_asset_id)],
        )
        self.assertEqual(len(selected_asset), 1)
        with fitz.open(
            stream=io.BytesIO(selected_asset.iloc[0]["data"]),
            filetype="pdf",
        ) as first_pdf:
            self.assertGreater(len(first_pdf), 0)

    def test_all_curated_responses_and_traces_match_source_rows(self) -> None:
        self.assertEqual(len(self.subjects), 16)
        self.assertEqual(len(self.responses), 2400)
        self.assertEqual(len(self.traces), 2400)
        self.assertEqual(set(self.subject_configs.values()), set(self.source_results))
        self.assertEqual(set(self.responses["test_condition"]), {"temperature=0.01"})
        self.assertEqual(set(self.responses["trial"]), {1})

        trace_by_key = {
            (
                row.subject_id,
                row.item_id,
                row.trial,
                row.test_condition,
                row.interactors,
            ): row.trace
            for row in self.traces.itertuples(index=False)
        }
        observed: set[tuple[tuple[str, str], str]] = set()
        for row in self.responses.itertuples(index=False):
            config = self.subject_configs[row.subject_id]
            qid = str(self.item_qids[row.item_id])
            source = self.source_results[config][qid]
            observed.add((config, qid))
            self.assertEqual(row.response, float(source["label"] == "Correct Answer"))
            key = (
                row.subject_id,
                row.item_id,
                row.trial,
                row.test_condition,
                row.interactors,
            )
            source_trace = source["model_answer"]
            expected_trace = source_trace if isinstance(source_trace, str) else str(source_trace)
            self.assertEqual(trace_by_key[key], expected_trace)

        expected_cells = {
            (config, qid)
            for config, results in self.source_results.items()
            for qid in results
        }
        self.assertEqual(observed, expected_cells)

    def test_paper_aggregate_scores_and_disclosed_reproduction_gap(self) -> None:
        aggregate_claim = {"expected_groups": 8, "absolute_tolerance": 0.005}
        self.assertEqual(
            len(PAPER_MAIN_CORRECT_COUNTS), aggregate_claim["expected_groups"]
        )
        for config, expected_correct in PAPER_MAIN_CORRECT_COUNTS.items():
            source_correct = sum(
                row["label"] == "Correct Answer"
                for row in self.source_results[config].values()
            )
            self.assertEqual(source_correct, expected_correct)

            subject_ids = [
                subject_id
                for subject_id, subject_config in self.subject_configs.items()
                if subject_config == config
            ]
            self.assertEqual(len(subject_ids), 1)
            curated = self.responses[
                self.responses["subject_id"] == subject_ids[0]
            ]["response"].mean()
            self.assertLessEqual(
                abs(curated - expected_correct / 150),
                aggregate_claim["absolute_tolerance"],
            )

        with fitz.open(self.paper_path) as paper:
            paper_text = " ".join(page.get_text() for page in paper)
        normalized_paper = " ".join(paper_text.split())
        for table_row in (
            "GPT-4-Turbo Closed Book 14 (9%) 5 (3%) 126 (88%) 150",
            "Llama2 Shared Vector Store 29 (19%) 104 (70%) 17 (11%) 150",
            "GPT-4-Turbo Shared Vector Store 29 (19%) 20 (13%) 101 (68%) 150",
            "Llama2 Single Vector Store 62 (41%) 81 (54%) 7 (5%) 150",
            "GPT-4-Turbo Single Vector Store 75 (50%) 17 (11%) 58 (39%) 150",
            "Claude2 Long Context 114 (76%) 32 (21%) 4 (3%) 150",
            "GPT-4-Turbo Long Context 118 (79%) 26 (17%) 6 (4%) 150",
            "GPT-4-Turbo Oracle 128 (85%) 22 (15%) 0 (0%) 150",
        ):
            self.assertIn(table_row, normalized_paper)
        self.assertIn("360 documents", normalized_paper)
        self.assertIn("ALL_DOCS", self.notebook)
        self.assertIn("df_questions['doc_name'].unique().tolist()", self.notebook)
        self.assertEqual(len(self.source_pdf_fingerprints), 84)

    def test_characterization_snapshot(self) -> None:
        check_tables(CHARACTERIZATION, {name: getattr(self, name) for name in OUTPUT_NAMES})


if __name__ == "__main__":
    unittest.main()
