#!/usr/bin/env python3
"""Read-only MathArena characterization and independent source reconciliation.

No downloads, fixture generation, or writes. Full mode requires all six output
tables and the pinned raw release; normal discovery skips missing build data.
"""

import base64
import hashlib
import importlib.util
import json
import math
import os
import sys
import unicodedata
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

BENCHMARK_DIR = Path(__file__).resolve().parent
SHARED_ROOT = BENCHMARK_DIR.parents[2] / "measurement_db"
sys.path.insert(0, str(SHARED_ROOT))
sys.path.insert(0, str(SHARED_ROOT.parent))
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
    declared_source_artifacts,
)

from scripts.build_measurement_tables import validate_asset_relations, validate_table, validate_trace_relations

from measurement_db.scripts.build_measurement_tables.source_snapshots import verify_snapshot_file
from measurement_db.benchmarks.matharena.build import source_competitions

OUTPUT_NAMES = ("items", "subjects", "benchmarks", "responses", "traces", "assets")
PRIMARY_KEY = ["subject_id", "item_id", "trial", "test_condition", "interactors"]


def normalized(value: object) -> object:
    """Canonical logical cells; binary payloads are represented by their hash."""
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [normalized(part) for part in value]
    if isinstance(value, dict):
        return {key: normalized(part) for key, part in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    return value


def logical_digest(rows) -> str:
    """Order-independent, length-framed SHA-256 over canonical JSON rows."""
    serialized = [
        json.dumps(
            normalized(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        for row in rows
    ]
    digest = hashlib.sha256()
    for row in sorted(serialized):
        payload = row.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def characterize(table: pd.DataFrame) -> dict:
    return {
        "rows": len(table),
        "columns": list(table.columns),
        "null_counts": {
            column: int(value) for column, value in table.isna().sum().items()
        },
        "logical_sha256": logical_digest(table.itertuples(index=False, name=None)),
    }


def features(serialized: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in serialized.split(";") if part)


def source_records(metadata: dict, competition: dict):
    """Read the upstream records independently of the builder's iterator."""
    for shard in competition["shards"]:
        parquet = pq.ParquetFile(BENCHMARK_DIR / "raw" / shard)
        columns = [
            c
            for c in parquet.schema_arrow.names
            if c not in ("all_messages", "image", "history")
        ]
        row_index = 0
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for record in batch.to_pylist():
                yield shard, row_index, record
                row_index += 1


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "matharena_build", BENCHMARK_DIR / "build.py"
        )
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_json_and_native_rubrics_are_equivalent(self):
        rubric = [{"title": "Construction", "points": 1, "max_points": 3}]
        self.assertEqual(self.builder.grading_details(json.dumps(rubric)), rubric)
        self.assertEqual(self.builder.grading_details(rubric), rubric)
        with self.assertRaises(ValueError):
            self.builder.grading_details('{"not": "a criterion list"}')

    def test_legacy_fraction_clipping_and_missing_scores(self):
        for points, maximum, expected in [
            (1, 3, 1 / 3),
            (-1, 3, 0),
            (4, 3, 1),
            (1, 0, None),
            (None, 3, None),
            (1, None, None),
        ]:
            with self.subTest(points=points, maximum=maximum):
                self.assertEqual(
                    self.builder.criterion_fraction(
                        {"points": points, "max_points": maximum}
                    ),
                    expected,
                )

    def test_answer_is_never_used_as_the_question(self):
        with self.assertRaisesRegex(ValueError, "no released prompt"):
            self.builder.prompt_components({"gold_answer": "A"})

    def test_prompt_and_trace_text_are_not_truncated(self):
        text = "  " + "proof " * 2000 + "  "
        self.assertEqual(self.builder.nonempty_text(text), text)
        self.assertEqual(
            self.builder.prompt_components({"user_message": text})[0], text
        )

    def test_effort_is_labelled_not_inferred_from_model_family(self):
        self.assertEqual(
            self.builder.subject_features("GPT-5.2 (high)", "openai/gpt-52-high")[
                "reasoning_effort"
            ],
            "high",
        )
        self.assertEqual(
            self.builder.subject_features(
                "DeepSeek-v4-Pro (Max)", "deepseek/deepseek_v4_pro"
            )["reasoning_effort"],
            "max",
        )
        self.assertNotIn(
            "reasoning_effort",
            self.builder.subject_features("Gemini 2.5 Pro", "gemini/gemini-pro-2.5"),
        )


class MathArenaCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = load_benchmark_metadata(BENCHMARK_DIR / "metadata.yaml")
        required = [BENCHMARK_DIR / f"{name}.parquet" for name in OUTPUT_NAMES]
        if all(path.is_file() for path in required):
            cls.sources = declared_source_artifacts(cls.metadata["sources"])
            cls.competitions = source_competitions(s["file"] for s in cls.sources)
            required += [BENCHMARK_DIR / "raw" / s["file"] for s in cls.sources]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            message = f"run benchmarks/matharena/build.py first; missing {missing[0]}"
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.expected = json.loads(
            (BENCHMARK_DIR / "testdata" / "characterization.json").read_text()
        )
        cls.tables = {
            name: pd.read_parquet(BENCHMARK_DIR / f"{name}.parquet")
            for name in OUTPUT_NAMES
        }

    def test_reviewed_table_shapes_null_patterns_and_fingerprints(self):
        for name, table in self.tables.items():
            with self.subTest(table=name):
                validate_table(
                    name, table, include_derived=True
                )
                self.assertEqual(
                    characterize(table), self.expected["release"]["tables"][name]
                )

    def test_primary_keys_trial_indices_and_trace_attribution(self):
        responses = self.tables["responses"]
        traces = self.tables["traces"]
        self.assertFalse(responses.duplicated(PRIMARY_KEY).any())
        self.assertTrue(responses.response_id.is_unique)
        self.assertTrue(self.tables["items"].item_id.is_unique)
        self.assertTrue(self.tables["subjects"].subject_id.is_unique)
        self.assertTrue((responses.trial >= 1).all())
        self.assertTrue(
            responses[["test_condition", "interactors"]].isna().all().all()
        )
        self.assertFalse(traces.duplicated(PRIMARY_KEY).any())
        validate_trace_relations(responses, traces)
        linked = traces.merge(
            responses, on=PRIMARY_KEY, how="left", validate="one_to_one", indicator=True
        )
        self.assertTrue(linked["_merge"].eq("both").all())
        item_features = self.tables["items"].set_index("item_id").item_features.map(features)
        competitions = responses.item_id.map(lambda item_id: item_features[item_id]["competition"])
        self.assertEqual(
            competitions.value_counts().to_dict(),
            self.expected["release"]["responses_by_competition"],
        )

    def test_image_questions_retain_assets_not_answer_letters(self):
        items = self.tables["items"]
        images = items[items.raw_item_id.str.startswith("kangaroo_")]
        self.assertTrue(images.asset_manifest.notna().all())
        self.assertFalse(images.content.isin(list("ABCDE")).any())
        self.assertEqual(images.raw_item_id.nunique(), len(self.tables["assets"]))
        validate_asset_relations(
            items, self.tables["assets"], benchmark_id="matharena",
            response_scale=self.tables["benchmarks"].iloc[0].response_scale,
        )
        for row in self.tables["assets"].itertuples():
            self.assertEqual(hashlib.sha256(row.data).hexdigest(), row.asset_id)
            self.assertEqual(len(row.data), row.byte_size)

    def test_judges_and_criteria_are_item_instruments(self):
        items = self.tables["items"].set_index("item_id")
        judged_items = items.verifier.map(lambda value: json.loads(value)["class"] == "judge")
        proof = self.tables["responses"].loc[lambda rows: rows.item_id.isin(items.index[judged_items])]
        self.assertTrue(proof.response.between(0, 1).all())
        for item_id, rows in proof.groupby("item_id"):
            verifier = json.loads(items.loc[item_id, "verifier"])
            self.assertEqual(verifier["class"], "judge")
            self.assertEqual(verifier["judged_by"], "human")
            self.assertGreaterEqual(int(verifier["judge_slot"]), 0)
            self.assertGreaterEqual(int(verifier["criterion_index"]), 0)
            self.assertEqual(
                verifier["rubric_sha256"],
                hashlib.sha256(json.loads(items.loc[item_id, "grading_criterion"])["rule"].encode()).hexdigest(),
            )

    def test_all_pinned_sources_match_hashes_and_provider_card_counts(self):
        for source in self.sources:
            with self.subTest(source=source["file"]):
                verify_snapshot_file(BENCHMARK_DIR / "raw" / source["file"], source)
        for name, competition in self.competitions.items():
            frontmatter = (
                (BENCHMARK_DIR / "raw" / competition["card"]).read_text().split("---", 2)[1]
            )
            claimed = yaml.safe_load(frontmatter)["dataset_info"]["splits"][0][
                "num_examples"
            ]
            observed = sum(
                pq.ParquetFile(
                    BENCHMARK_DIR
                    / "raw"
                    / shard
                ).metadata.num_rows
                for shard in competition["shards"]
            )
            self.assertEqual(observed, claimed, name)

    def test_every_released_score_prompt_configuration_and_trace_reconciles(self):
        """Independent row-level audit, including every formerly skipped rubric."""
        responses = self.tables["responses"]
        items = self.tables["items"].set_index("item_id").to_dict("index")
        subjects = self.tables["subjects"].set_index("subject_id").to_dict("index")
        groups = defaultdict(list)
        for response in responses.to_dict("records"):
            item_features = features(items[response["item_id"]]["item_features"])
            subject = subjects[response["subject_id"]]
            key = (item_features["competition"], item_features["problem_idx"],
                   subject["display_name"], features(subject["subject_features_extra"])["model_config"],
                   response["trial"])
            groups[key].append(response)
        traces = {
            tuple(row[:5]): row[5]
            for row in self.tables["traces"][[*PRIMARY_KEY, "trace"]].itertuples(
                index=False, name=None
            )
        }
        assets = self.tables["assets"].set_index("asset_id")["data"].to_dict()
        seen = 0
        legacy_rows = []
        for competition_name, competition in self.competitions.items():
            legacy_items = {}
            for shard, row_index, record in source_records(self.metadata, competition):
                released = []
                if competition["kind"] == "final_answer":
                    if record["correct"] is not None:
                        released.append((None, None, None, float(record["correct"])))
                else:
                    for key in sorted(
                        k for k in record if k.startswith("grading_details_judge_")
                    ):
                        details = record[key]
                        if isinstance(details, str):
                            details = json.loads(details)
                        for index, criterion in enumerate(details or []):
                            try:
                                points, maximum = (
                                    float(criterion["points"]),
                                    float(criterion["max_points"]),
                                )
                            except (TypeError, ValueError):
                                continue
                            if (
                                maximum
                                and math.isfinite(points)
                                and math.isfinite(maximum)
                            ):
                                released.append(
                                    (
                                        int(key.rsplit("_", 1)[1]),
                                        index,
                                        criterion,
                                        max(0.0, min(1.0, points / maximum)),
                                    )
                                )
                key = (competition_name, str(record["problem_idx"]), record["model_name"],
                       record["model_config"], int(record["idx_answer"]) + 1)
                actual = groups.pop(key, [])
                self.assertEqual(len(actual), len(released), (shard, row_index))
                actual.sort(
                    key=lambda row: (
                        int(json.loads(items[row["item_id"]]["verifier"]).get("judge_slot", 0)),
                        int(json.loads(items[row["item_id"]]["verifier"]).get("criterion_index", 0)),
                    )
                )
                prompt = record.get("user_message") or record.get("problem")
                reference = record.get("gold_answer")
                trace = next(
                    (
                        record[k]
                        for k in ("answer", "parsed_answer")
                        if isinstance(record.get(k), str) and record[k].strip()
                    ),
                    None,
                )
                image_payloads = []
                if competition_name.startswith("kangaroo_"):
                    parts = json.loads(prompt)
                    prompt = next(
                        part["text"]
                        for part in parts
                        if part["type"] in ("text", "input_text")
                    )
                    for part in parts:
                        if part["type"] in ("input_image", "image_url"):
                            url = part["image_url"]
                            if isinstance(url, dict):
                                url = url["url"]
                            image_payloads.append(
                                base64.b64decode(url.split(",", 1)[1], validate=True)
                            )
                        elif part["type"] == "image":
                            image_payloads.append(
                                base64.b64decode(part["source"]["data"], validate=True)
                            )
                for grade_index, (
                    response,
                    (judge, index, criterion, score),
                ) in enumerate(zip(actual, released, strict=True)):
                    item = items[response["item_id"]]
                    subject = subjects[response["subject_id"]]
                    self.assertEqual(response["response"], score)
                    self.assertEqual(
                        features(item["item_features"])["problem_idx"], str(record["problem_idx"])
                    )
                    self.assertEqual(
                        response["trial"], int(record["idx_answer"]) + 1
                    )
                    self.assertEqual(subject["display_name"], record["model_name"])
                    self.assertEqual(
                        features(subject["subject_features_extra"])["model_config"],
                        record["model_config"],
                    )
                    # Shared text identity intentionally coalesces NFC/edge-
                    # whitespace-only variants, retaining the first text.
                    self.assertEqual(
                        unicodedata.normalize("NFC", item["content"]).strip(),
                        unicodedata.normalize("NFC", prompt).strip(),
                    )
                    self.assertEqual(normalized(json.loads(item["grading_criterion"])["reference_answer"]), reference)
                    if criterion is not None:
                        verifier = json.loads(item["verifier"])
                        self.assertEqual(
                            json.loads(json.loads(item["grading_criterion"])["rule"]),
                            {
                                key: criterion.get(key)
                                for key in (
                                    "title",
                                    "grading_scheme_desc",
                                    "max_points",
                                )
                            },
                        )
                        self.assertEqual(int(verifier["judge_slot"]), judge)
                        self.assertEqual(int(verifier["criterion_index"]), index)
                    else:
                        self.assertEqual(
                            json.loads(item["verifier"])["class"], "exact_matcher"
                        )
                    if image_payloads:
                        manifest = json.loads(item["asset_manifest"])
                        self.assertEqual(
                            [assets[entry["asset_id"]] for entry in manifest],
                            image_payloads,
                        )
                    key = tuple(response[column] for column in PRIMARY_KEY)
                    self.assertEqual(
                        traces.get(key), trace if grade_index == 0 else None
                    )
                    seen += 1
                # Reconstruct the old observation multiset, excluding its
                # collision-dependent trials and using first-row legacy text.
                problem_id = str(record["problem_idx"])
                if problem_id not in legacy_items:
                    old_text = next(
                        record[k]
                        for k in ("problem", "problem_statement", "gold_answer")
                        if record.get(k) is not None
                    )
                    legacy_items[problem_id] = (str(old_text).strip()[:4000], reference)
                old_content, old_reference = legacy_items[problem_id]
                for judge, _, criterion, score in released:
                    if criterion is None or competition_name in (
                        "putnam_2025",
                        "miklos_2025",
                    ):
                        condition = (
                            None
                            if criterion is None
                            else f"judge={judge};criterion={criterion['title']}"
                        )
                        legacy_rows.append(
                            [
                                record["model_name"],
                                old_content,
                                score,
                                old_reference,
                                condition,
                            ]
                        )
        self.assertFalse(groups, "output rows have no pinned-source origin")
        self.assertEqual(seen, len(responses))
        self.assertEqual(
            logical_digest(legacy_rows),
            self.expected["legacy_baseline"]["observation_multiset_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
