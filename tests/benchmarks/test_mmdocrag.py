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
from unittest.mock import patch
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "mmdocrag"
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
    declared_source_artifacts,
)

from measurement_db.scripts.build_measurement_tables.source_snapshots import verify_snapshot_file
# Independent expectations for the pinned release, not imported from the builder.
LAYOUT = {'gold_sources': ['evaluation_20.jsonl', 'evaluation_15.jsonl'],
 'evaluation_directory': 'eval',
 'trace_directory': 'resp',
 'subject_aliases': {'Internvl3-38B': 'internvl3-38b',
                     'Internvl3-78B': 'internvl3-78b'},
 'subject_features': {'qvq-max-no-think': {'released_variant': 'no-think'},
                      'qwen3-14b-no-think': {'released_variant': 'no-think'},
                      'qwen3-30b-a3b-no-think': {'released_variant': 'no-think'},
                      'qwen3-4b-no-think': {'released_variant': 'no-think'},
                      'qwen3-8b-no-think': {'released_variant': 'no-think'}}}
QUESTION_IDS = range(2000)

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


from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    load_characterization, check_tables, check_source_claims,
)

# Reviewed migration and parsing invariants; current tables use the shared checker.
REGRESSION = {'legacy': {'items': {'columns': ['item_id',
                                  'benchmark_id',
                                  'raw_item_id',
                                  'content',
                                  'reference_answer',
                                  'verifier',
                                  'content_hash'],
                      'logical_sha256': 'c689e6722c3364ac770daa78b19f410fd113a92d2a2c9d0c03179d225710cb3e'}},
 'preserved': {'responses': 'ec638eea1c2c80d059dee555e2724d7eb9686b3729dd159b608ce07841419263',
               'traces': 'f55dfe8c4f750cf2c27f3c13300daf26a6f87fff80f8a09f66ad498fbd405da6'},
 'release': {'score_values': [0,
                              0.04,
                              0.08,
                              0.12,
                              0.16,
                              0.2,
                              0.24,
                              0.27999999999999997,
                              0.32,
                              0.36,
                              0.4,
                              0.44000000000000006,
                              0.48,
                              0.52,
                              0.5599999999999999,
                              0.6,
                              0.64,
                              0.6799999999999999,
                              0.72,
                              0.76,
                              0.8,
                              0.8400000000000001,
                              0.8800000000000001,
                              0.9199999999999999,
                              0.96,
                              1],
             'conditions': {'responses': {'multimodal/quotes15': 61966,
                                          'multimodal/quotes20': 63963,
                                          'pure-text/quotes15': 108458,
                                          'pure-text/quotes20': 108308},
                            'traces': {'multimodal/quotes15': 59756,
                                       'multimodal/quotes20': 59868,
                                       'pure-text/quotes15': 104413,
                                       'pure-text/quotes20': 104249}},
             'ungraded_attempts': 129,
             'ungraded_with_traces': 120,
             'subject_ids': {'internvl3-38b': 'e57466a395ae6879',
                             'internvl3-78b': '6132a6d2f00daf1a',
                             'claude-3.5-sonnet': 'c68b0b4cebac1009',
                             'deepseek-r1-distill-llama-70b': 'a3e762a25636977b',
                             'deepseek-r1-distill-qwen-32b': '66c4ad8d059f92bb',
                             'deepseek-r1': '50cbacf8db8037c0',
                             'deepseek-v3': '7bc3eed814b23142',
                             'gemini-1.5-pro': '563f181242d45ea1',
                             'gemini-2.0-flash-tk': '9c0918c6cebe31f9',
                             'gemini-2.0-flash': '96098bb6ed439973',
                             'gemini-2.0-pro': '943a3219f5f642a9',
                             'gemini-2.5-flash': 'e4b4ec61717e2e26',
                             'gemini-2.5-pro': 'f3c1964f1cdb72be',
                             'gpt-4-turbo': '97f2da5bc3b55368',
                             'gpt-4.1-mini': '0b840a32283e38bc',
                             'gpt-4.1-nano': 'a0c325c2fb187522',
                             'gpt-4.1': '7facc3df28eab6df',
                             'gpt-4o-mini': '1ba99d7bfc77bdf7',
                             'gpt-4o': 'b8d2171afab2c063',
                             'gpt-o3-mini': '8d0f7b00fa1ee7d4',
                             'grok-3-beta': '9605092c664cd9d1',
                             'grok-3-mini-beta': '7d65eb53a8cc7d83',
                             'internvl2.5-26b': '1bd0aa11d48be47b',
                             'internvl2.5-38b': '4c1714009675ac9f',
                             'internvl2.5-78b': '03594df58e38f87e',
                             'internvl2.5-8b': 'c62d52764d3ddafe',
                             'internvl3-14b': '6f05b1b00cca0388',
                             'internvl3-8b': 'c7822b5fffaca5ed',
                             'internvl3-9b': '6d0880e12b1fcd93',
                             'janus-pro-7b': 'a54e80c202df69de',
                             'llama3.1-8b': '55fdc5e18ebe84af',
                             'llama3.2-3b': '784ad8c8c9ae5a64',
                             'llama3.3-70b': '9f1120099dc1d940',
                             'llama4-mave-17b-128e': '6842e28b19417e47',
                             'llama4-scout-17b-16e': '887aa841d605d9ed',
                             'minicpm-o-2.6-8b': 'df5d0083a1d15a83',
                             'mistral-7b': 'fb91c581a3ab32b5',
                             'mistral-small-24b': 'cff1937ef021c5b6',
                             'mixtral-8x7b': 'e38e92acaca170e8',
                             'qvq-max-no-think': 'f1c299f63abfe2a1',
                             'qwen-max': 'e414e08afa9a7f89',
                             'qwen-plus': '9f2b6da23207a778',
                             'qwen-qvq-max': '73f13888c23d3f34',
                             'qwen-qwq-plus': 'd52f31a8bb7d041c',
                             'qwen-vl-max': 'df981a41948cb0dd',
                             'qwen-vl-plus': 'e0b9d6ea70a29061',
                             'qwen2.5-14b-ft': '02563a11cc439c01',
                             'qwen2.5-14b': '8a8a59104a846579',
                             'qwen2.5-32b-ft': '5bcc7c5d5b4602b0',
                             'qwen2.5-32b': '7299112b66af415f',
                             'qwen2.5-3b-ft': '05772d61c90364a0',
                             'qwen2.5-3b': '9213330abd9929c1',
                             'qwen2.5-72b-ft': '00f6650d983cdc8b',
                             'qwen2.5-72b': 'f7fbe8b6231489cb',
                             'qwen2.5-7b-ft': 'cfd5e8cf564e5f43',
                             'qwen2.5-7b': 'cf1a5aedf5266778',
                             'qwen2.5-vl-32b': 'b06669275cee43f3',
                             'qwen2.5-vl-72b': '45f6ddbec5ea38aa',
                             'qwen2.5-vl-7b': 'fa5687bfa12d4337',
                             'qwen3-14b-no-think': 'd1ab289874373f7f',
                             'qwen3-14b': '2ec99af5c6359e30',
                             'qwen3-235b-a22b': '2e663d8f22f5a8ea',
                             'qwen3-30b-a3b-no-think': '93aaa277c4b81776',
                             'qwen3-30b-a3b': '2a42cf4c586eab0c',
                             'qwen3-32b': '3b7ff8dae0a232cd',
                             'qwen3-4b-no-think': '244b8cc284dd22aa',
                             'qwen3-8b-no-think': '78c305a8596192ef',
                             'qwen3-8b': '8ddb957632e6146b'},
             'ungraded_reasons': {'empty': 112, 'no_usable_dimensions': 8, 'partial_dimensions': 9},
             'judge_labels': {'gpt-4o-2024-08-06': 153955, 'gpt-4o-2024-08-06_vlm': 188740},
             'case_alias_groups': {'internvl3-38b': ['Internvl3-38B', 'internvl3-38b'],
                                   'internvl3-78b': ['Internvl3-78B', 'internvl3-78b'],
                                   'internvl3-14b': ['internvl3-14B', 'internvl3-14b'],
                                   'internvl3-8b': ['internvl3-8B', 'internvl3-8b'],
                                   'internvl3-9b': ['internvl3-9B', 'internvl3-9b']}}}

class JudgeParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mmdocrag_builder_under_test", BENCHMARK_DIR / "build.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def transformed(self, payload, filename="subject_pure-text_quotes15_llm-judge.jsonl"):
        builder = self.module.MMDocRAG(str(BENCHMARK_DIR / "build.py"))
        builder.source_files = ["evaluation_20.jsonl", "evaluation_15.jsonl", "eval/" + filename]
        gold = pd.DataFrame({"q_id": [0], "question": ["Question"], "answer_short": ["Answer"]})
        judgments = pd.DataFrame({"q_id": [0], "model": ["recorded-judge"], "response": [payload]})
        def native_table(path, **kwargs):
            return (judgments if Path(path).parent.name == "eval" else gold).copy(deep=True)
        with patch.object(self.module.pd, "read_json", side_effect=native_table):
            return builder.build_tables()

    def test_zero_is_a_grade_and_missing_dimension_is_not(self):
        payload = dict.fromkeys(["Fluency", "Citation Quality", "Text-Image Coherence", "Reasoning Logic", "Factuality"], 0)
        self.assertEqual(self.transformed(payload)["responses"].response.iloc[0], 0.0)
        for invalid in (None, True, "0", -1, 6, float("nan")):
            with self.subTest(invalid=invalid):
                self.assertTrue(pd.isna(self.transformed({**payload, "Fluency": invalid})["responses"].response.iloc[0]))
        self.assertTrue(pd.isna(self.transformed({})["responses"].response.iloc[0]))
        payload.pop("Fluency")
        self.assertTrue(pd.isna(self.transformed(payload)["responses"].response.iloc[0]))

    def test_key_noise_and_first_valid_duplicate_keep_legacy_arithmetic(self):
        payload = {" 'Fluency' ": 1, "Fluency": 5, "CitationQuality": 2,
                   "Text Image Coherence": 3, "Reasoning Logic": 4, "Factuality": 5}
        self.assertEqual(self.transformed(payload)["responses"].response.iloc[0], 15.0 / 5 / 5.0)
        payload[" 'Fluency' "] = "invalid"
        self.assertEqual(self.transformed(payload)["responses"].response.iloc[0], 19.0 / 5 / 5.0)

    def test_legacy_and_current_filename_forms(self):
        for suffix in ("_pure-text_quotes15_llm-judge.jsonl", "_pure-text_response_quotes15.jsonl",
                       "_pure-text_response_quotes15.jsonl_evaluation.jsonl"):
            tables = self.transformed({}, "subject" + suffix)
            self.assertEqual(tables["subjects"].subject_key.tolist(), ["subject"])
            self.assertEqual(tables["responses"].test_condition.tolist(), ["pure-text/quotes15"])
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            self.transformed({}, "not-an-evaluation.jsonl")


class MMDocRAGCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = REGRESSION
        cls.characterization = load_characterization(BENCHMARK_DIR / "characterization.yaml")
        missing = [
            name
            for name in OUTPUT_NAMES
            if not (BENCHMARK_DIR / "formatted_tables" / f"{name}.parquet").is_file()
        ]
        if missing:
            message = (
                f"Missing MMDocRAG tables {missing}; run benchmarks/mmdocrag/build.py"
            )
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.tables = {
            name: pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / f"{name}.parquet")
            for name in OUTPUT_NAMES
        }
        cls.cells = {
            name: semantic_cells(cls.tables, name) for name in ("responses", "traces")
        }


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
        features_by_label = LAYOUT["subject_features"]
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
            no_think_labels, set(LAYOUT["subject_features"])
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
        for source in declared_source_artifacts(METADATA["sources"], benchmark_dir=BENCHMARK_DIR):
            with self.subTest(source=source["file"]):
                verify_snapshot_file(BENCHMARK_DIR / "raw" / source["file"], source)
        gold = {}
        for source_name in LAYOUT["gold_sources"]:
            path = BENCHMARK_DIR / "raw" / source_name
            for record in read_jsonl(path):
                gold.setdefault(record["q_id"], record)
        self.assertEqual(set(gold), set(QUESTION_IDS))
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
        layout = LAYOUT
        files = {source["file"] for source in declared_source_artifacts(METADATA["sources"], benchmark_dir=BENCHMARK_DIR)}
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
        gold_ids = {record["q_id"] for name in LAYOUT["gold_sources"]
                    for record in read_jsonl(BENCHMARK_DIR / "raw" / name)}
        check_source_claims(self.characterization, {
            "released_questions": len(gold_ids),
            "released_judge_attempts": len(seen),
        })
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
        for raw_label, registry_label in LAYOUT[
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
