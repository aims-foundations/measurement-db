"""Read-only MMDocRAG characterization and independent source reconciliation.

Never downloads or updates expectations. Missing outputs skip on a clean
checkout; MEASUREMENT_DB_FULL_TEST=1 requires a completed local build.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import unicodedata
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR.parents[2]))

from measurement_db.scripts.build_measurement_tables import (
    validate_table,
    validate_trace_relations,
)
from measurement_db.scripts.build_measurement_tables.hash_measurement_ids import (
    item_id_from_content,
    response_id_from_row,
    response_identity_v1,
    subject_id_from_row,
)
from measurement_db.scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
)

METADATA = load_benchmark_metadata(BENCHMARK_DIR / "metadata.yaml")
OUTPUT_NAMES = ("items", "subjects", "benchmarks", "responses", "traces")
CELL_KEY = ["subject_key", "raw_item_id", "trial", "test_condition"]
DIMENSIONS = {
    "fluency",
    "citationquality",
    "textimagecoherence",
    "reasoninglogic",
    "factuality",
}


def normalized_label(label: str) -> str:
    return unicodedata.normalize("NFC", label).strip().lower()


def normalized_cell(value: object) -> object:
    """Canonical JSON cells, retaining float values and list order."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalized_cell(part) for part in value]
    if isinstance(value, dict):
        return {key: normalized_cell(part) for key, part in value.items()}
    return None if pd.isna(value) else value


def logical_fingerprint(table: pd.DataFrame) -> str:
    """Hash ordered columns and sorted row digests; independent of Parquet bytes."""
    rows = []
    for row in table.itertuples(index=False, name=None):
        payload = json.dumps(
            [normalized_cell(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        rows.append(hashlib.sha256(payload).digest())
    digest = hashlib.sha256(json.dumps(list(table.columns)).encode("utf-8"))
    for row_digest in sorted(rows):
        digest.update(row_digest)
    return digest.hexdigest()


def characterize_table(table: pd.DataFrame) -> dict:
    return {
        "rows": len(table),
        "columns": list(table.columns),
        "null_counts": {column: int(table[column].isna().sum()) for column in table},
        "logical_sha256": logical_fingerprint(table),
    }


def semantic_cells(tables: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Compare attempt identities across the approved subject-ID migration."""
    subjects = tables["subjects"][["subject_id", "display_name"]].copy()
    subjects["subject_key"] = subjects.pop("display_name").map(normalized_label)
    table = tables[name].merge(subjects, on="subject_id", validate="many_to_one")
    table = table.merge(
        tables["items"][["item_id", "raw_item_id"]],
        on="item_id",
        validate="many_to_one",
    )
    if "interactors" not in table:
        table["interactors"] = None
    columns = CELL_KEY + ["benchmark_id", "interactors"]
    if name == "responses":
        # Reconstruct the retired all-null slots only for the historical
        # semantic fingerprint; schema-2 files do not carry these columns.
        table["reference_answer"] = None
        table["trace"] = None
        columns += ["response", "reference_answer"]
    return table[columns + ["trace"]]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def source_configuration(filename: str) -> tuple[str, str, str]:
    """Independent full-filename parser for the released naming variants."""
    match = re.fullmatch(
        r"(.+?)_(pure-text|multimodal)(?:_response)?_quotes(\d+)"
        r"(?:_llm-judge\.jsonl|\.jsonl(?:_evaluation\.jsonl)?)",
        filename,
    )
    if match is None:
        raise AssertionError(f"Unrecognized source filename: {filename}")
    return match.groups()


def source_dimensions(payload: dict) -> dict[str, float]:
    """First valid value per recognized dimension, in released JSON order."""
    grades = {}
    for key, value in payload.items():
        dimension = re.sub("[^a-z0-9]", "", str(key).lower())
        if dimension not in DIMENSIONS or dimension in grades:
            continue
        if type(value) in (int, float) and 0 <= value <= 5:
            grades[dimension] = float(value)
    return grades


class JudgeParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "mmdocrag_builder_under_test", BENCHMARK_DIR / "build.py"
        )
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_zero_is_a_grade_and_missing_dimension_is_not(self) -> None:
        payload = dict.fromkeys(self.builder.JUDGE_DIMS, 0)
        self.assertEqual(self.builder.answer_quality(payload), 0.0)
        for invalid in (None, True, "0", -1, 6, float("nan")):
            with self.subTest(invalid=invalid):
                self.assertIsNone(
                    self.builder.answer_quality({**payload, "Fluency": invalid})
                )
        self.assertIsNone(self.builder.answer_quality({}))
        payload.pop("Fluency")
        self.assertIsNone(self.builder.answer_quality(payload))

    def test_key_noise_and_first_valid_duplicate_keep_legacy_arithmetic(self) -> None:
        payload = {
            " 'Fluency' ": 1,
            "Fluency": 5,
            "CitationQuality": 2,
            "Text Image Coherence": 3,
            "Reasoning Logic": 4,
            "Factuality": 5,
        }
        self.assertEqual(self.builder.answer_quality(payload), 15.0 / 5 / 5.0)
        payload[" 'Fluency' "] = "invalid"
        self.assertEqual(self.builder.answer_quality(payload), 19.0 / 5 / 5.0)

    def test_legacy_and_current_filename_forms(self) -> None:
        for suffix in (
            "_pure-text_quotes15_llm-judge.jsonl",
            "_pure-text_response_quotes15.jsonl",
            "_pure-text_response_quotes15.jsonl_evaluation.jsonl",
        ):
            self.assertEqual(
                self.builder.parse_eval_filename("subject" + suffix),
                ("subject", "pure-text", "15"),
            )
        self.assertIsNone(self.builder.parse_eval_filename("not-an-evaluation.jsonl"))


class MMDocRAGCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(
            (BENCHMARK_DIR / "testdata" / "characterization.json").read_text(
                encoding="utf-8"
            )
        )
        missing = [
            name
            for name in OUTPUT_NAMES
            if not (BENCHMARK_DIR / f"{name}.parquet").is_file()
        ]
        if missing:
            message = (
                f"Missing MMDocRAG tables {missing}; run benchmarks/mmdocrag/build.py"
            )
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.tables = {
            name: pd.read_parquet(BENCHMARK_DIR / f"{name}.parquet")
            for name in OUTPUT_NAMES
        }
        cls.cells = {
            name: semantic_cells(cls.tables, name) for name in ("responses", "traces")
        }

    def test_full_table_shapes_null_patterns_and_logical_fingerprints(self) -> None:
        for name, table in self.tables.items():
            with self.subTest(table=name):
                validate_table(name, table, include_derived=True)
                self.assertEqual(
                    characterize_table(table), self.expected["release"]["tables"][name]
                )

    def test_attempt_values_and_trace_attribution_match_legacy_fingerprints(
        self,
    ) -> None:
        for name, table in self.cells.items():
            with self.subTest(table=name):
                self.assertEqual(
                    logical_fingerprint(table), self.expected["preserved"][name]
                )
        item_columns = self.expected["legacy"]["items"]["columns"]
        # Compare the released question/answer bank under its historical hash;
        # current item IDs also include the grading protocol and response scale.
        legacy_items = self.tables["items"].copy()
        legacy_items["reference_answer"] = legacy_items.grading_criterion.map(
            lambda value: json.loads(value)["reference_answer"]
        )
        legacy_items["verifier"] = None
        legacy_items["item_id"] = [item_id_from_content(row.benchmark_id, row.content)
                                    for row in legacy_items.itertuples()]
        self.assertEqual(
            logical_fingerprint(legacy_items[item_columns]),
            self.expected["legacy"]["items"]["logical_sha256"],
        )

    def test_single_trials_unique_keys_and_current_response_ids(self) -> None:
        for name, key in (
            ("items", "item_id"),
            ("subjects", "subject_id"),
            ("benchmarks", "benchmark_id"),
            ("responses", "response_id"),
        ):
            self.assertTrue(self.tables[name][key].is_unique, name)
        for name, table in self.cells.items():
            self.assertFalse(table.duplicated(CELL_KEY).any(), name)
            self.assertEqual(set(table["trial"]), {1})
            self.assertTrue(table["interactors"].isna().all())
        responses = self.tables["responses"]
        self.assertEqual(
            responses["response_id"].tolist(),
            [response_id_from_row(response_identity_v1(row, None))
             for row in responses.to_dict("records")],
        )
        self.assertNotIn("reference_answer", responses)
        self.assertNotIn("trace", responses)
        validate_trace_relations(responses, self.tables["traces"])
        for column in (
            "access_date",
            "harness",
            "harness_version",
            "reasoning_effort",
        ):
            self.assertTrue(self.tables["subjects"][column].isna().all(), column)

    def test_registry_coverage_and_final_subject_identity(self) -> None:
        registry_path = (
            BENCHMARK_DIR.parents[2]
            / "measurement_db/scripts/build_measurement_tables/map_model_registry.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        features_by_label = METADATA["archive_layout"]["subject_features"]
        subjects = self.tables["subjects"]
        self.assertTrue(subjects[["normalized_name", "provider"]].notna().all().all())
        for record in subjects.to_dict(orient="records"):
            label = record["display_name"]
            entry = registry[label]
            self.assertEqual(record["normalized_name"], entry["model"], label)
            self.assertEqual(record["provider"], entry["company"], label)
            self.assertEqual(
                normalized_cell(record["release_date"]),
                entry.get("release_date"),
                label,
            )
            features = features_by_label.get(label, {})
            expected_features = (
                ";".join(f"{key}={value}" for key, value in sorted(features.items()))
                or None
            )
            self.assertEqual(
                normalized_cell(record["subject_features_extra"]),
                expected_features,
                label,
            )
            self.assertEqual(
                record["subject_id"],
                subject_id_from_row(
                    label,
                    {key: normalized_cell(value) for key, value in record.items()},
                ),
                label,
            )

    def test_fine_tuned_and_no_think_variants_do_not_merge(self) -> None:
        subjects = self.tables["subjects"].set_index("display_name")
        for size in (3, 7, 14, 32, 72):
            base = subjects.loc[f"qwen2.5-{size}b"]
            tuned = subjects.loc[f"qwen2.5-{size}b-ft"]
            self.assertNotEqual(base["normalized_name"], tuned["normalized_name"])
            self.assertNotEqual(base["subject_id"], tuned["subject_id"])
        no_think_labels = {
            label for label in subjects.index if label.endswith("-no-think")
        }
        self.assertEqual(
            no_think_labels, set(METADATA["archive_layout"]["subject_features"])
        )
        for label in no_think_labels:
            variant = subjects.loc[label]
            self.assertEqual(
                variant["subject_features_extra"], "released_variant=no-think"
            )
            base_label = (
                "qwen-qvq-max"
                if label == "qvq-max-no-think"
                else label.removesuffix("-no-think")
            )
            if base_label in subjects.index:
                base = subjects.loc[base_label]
                self.assertEqual(variant["normalized_name"], base["normalized_name"])
                self.assertNotEqual(variant["subject_id"], base["subject_id"])

    def test_grade_categories_conditions_and_ungraded_attempt_traces(self) -> None:
        responses = self.cells["responses"]
        self.assertEqual(
            sorted(responses["response"].dropna().unique()),
            self.expected["release"]["score_values"],
        )
        for name, table in self.cells.items():
            self.assertEqual(
                table.groupby("test_condition").size().to_dict(),
                self.expected["release"]["conditions"][name],
            )
        ungraded = responses.loc[responses["response"].isna()]
        self.assertEqual(len(ungraded), self.expected["release"]["ungraded_attempts"])
        self.assertEqual(
            len(
                ungraded.merge(self.cells["traces"], on=CELL_KEY, validate="one_to_one")
            ),
            self.expected["release"]["ungraded_with_traces"],
        )

    def test_pinned_source_bytes_and_question_only_items(self) -> None:
        downloads = METADATA["sources"]["downloads"]
        for source in downloads.values():
            path = BENCHMARK_DIR / "raw" / source["file"]
            with self.subTest(source=source["file"]):
                self.assertEqual(path.stat().st_size, source["size"])
                with path.open("rb") as stream:
                    self.assertEqual(
                        hashlib.file_digest(stream, "sha256").hexdigest(),
                        source["sha256"],
                    )
        gold = {}
        for source_name in METADATA["archive_layout"]["gold_sources"]:
            path = BENCHMARK_DIR / "raw" / downloads[source_name]["file"]
            for record in read_jsonl(path):
                gold.setdefault(record["q_id"], record)
        ids = METADATA["expectations"]["question_ids"]
        self.assertEqual(set(gold), set(range(ids["first"], ids["last"] + 1)))
        items = self.tables["items"].set_index("raw_item_id")
        self.assertEqual(set(items.index), {f"q_id::{qid}" for qid in gold})
        for qid, record in gold.items():
            answer = record["answer_short"]
            if isinstance(answer, list):
                answer = ", ".join(str(part) for part in answer)
            self.assertEqual(items.loc[f"q_id::{qid}", "content"], record["question"])
            self.assertEqual(json.loads(items.loc[f"q_id::{qid}", "grading_criterion"])["reference_answer"], answer)
        for column in ("asset_manifest", "item_features"):
            self.assertTrue(items[column].isna().all(), column)
        self.assertTrue(items.verifier.map(lambda value: json.loads(value)["class"] == "judge").all())

    def test_released_judge_cells_and_exact_case_trace_selection(self) -> None:
        layout = METADATA["archive_layout"]
        files = {source["file"] for source in METADATA["sources"]["downloads"].values()}
        evaluations = sorted(
            path
            for path in files
            if Path(path).parent.as_posix() == layout["evaluation_directory"]
        )
        actual_scores = (
            self.cells["responses"].set_index(CELL_KEY)["response"].to_dict()
        )
        actual_traces = self.cells["traces"].set_index(CELL_KEY)["trace"].to_dict()
        seen, seen_traces = set(), set()
        aliases, reasons, judge_labels = defaultdict(set), Counter(), Counter()
        for relative_path in evaluations:
            model, mode, quotes = source_configuration(Path(relative_path).name)
            subject_key = normalized_label(model)
            aliases[subject_key].add(model)
            candidates = (
                f"{layout['trace_directory']}/{model}_{mode}_quotes{quotes}_response.jsonl",
                f"{layout['trace_directory']}/{model}_{mode}_response_quotes{quotes}.jsonl",
            )
            traces = {}
            for candidate in candidates:
                if candidate in files:
                    traces = {
                        record["q_id"]: record.get("response")
                        for record in read_jsonl(BENCHMARK_DIR / "raw" / candidate)
                        if record.get("q_id") is not None
                    }
                    break
            for record in read_jsonl(BENCHMARK_DIR / "raw" / relative_path):
                qid, payload = record.get("q_id"), record.get("response")
                if qid is None or not isinstance(payload, dict):
                    continue
                key = (subject_key, f"q_id::{qid}", 1, f"{mode}/quotes{quotes}")
                self.assertNotIn(key, seen)
                seen.add(key)
                grades = source_dimensions(payload)
                score = sum(grades.values()) / 5 / 5.0 if len(grades) == 5 else None
                self.assertEqual(normalized_cell(actual_scores[key]), score, key)
                judge_labels[str(record.get("model"))] += 1
                if score is None:
                    reasons[
                        "empty"
                        if not payload
                        else "partial_dimensions"
                        if grades
                        else "no_usable_dimensions"
                    ] += 1
                trace = traces.get(qid)
                if isinstance(trace, str) and trace:
                    self.assertEqual(actual_traces[key], trace, key)
                    seen_traces.add(key)
                else:
                    self.assertNotIn(key, actual_traces)
        self.assertEqual(seen, set(actual_scores))
        self.assertEqual(seen_traces, set(actual_traces))
        self.assertEqual(dict(reasons), self.expected["release"]["ungraded_reasons"])
        self.assertEqual(dict(judge_labels), self.expected["release"]["judge_labels"])
        self.assertEqual(
            {key: sorted(value) for key, value in aliases.items() if len(value) > 1},
            self.expected["release"]["case_alias_groups"],
        )
        self.assertEqual(
            set(aliases),
            set(self.tables["subjects"]["display_name"].map(normalized_label)),
        )

    def test_alias_resolution_preserves_internvl_registry_metadata(self) -> None:
        subjects = self.tables["subjects"].set_index("display_name")
        for raw_label, registry_label in METADATA["archive_layout"][
            "subject_aliases"
        ].items():
            self.assertNotIn(raw_label, subjects.index)
            self.assertIn(registry_label, subjects.index)
            self.assertTrue(
                subjects.loc[
                    registry_label, ["normalized_name", "provider", "release_date"]
                ]
                .notna()
                .all()
            )
        identities = {
            normalized_label(row.display_name): row.subject_id
            for row in self.tables["subjects"].itertuples()
        }
        self.assertEqual(identities, self.expected["release"]["subject_ids"])


if __name__ == "__main__":
    unittest.main()
