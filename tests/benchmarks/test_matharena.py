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
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "matharena"
SHARED_ROOT = BENCHMARK_DIR.parents[2] / "measurement_db"
sys.path.insert(0, str(SHARED_ROOT))
sys.path.insert(0, str(SHARED_ROOT.parent))
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
    declared_source_artifacts,
)

from scripts.build_measurement_tables import validate_asset_relations, validate_table, validate_trace_relations

from measurement_db.scripts.build_measurement_tables.source_snapshots import verify_snapshot_file
def source_competitions(files):
    # Independently frozen release layout; do not import the builder's grouping.
    proof = {"imc_2025", "imo_2025", "miklos_2025", "putnam_2025", "usamo_2025"}
    competitions = {}
    for name in sorted(files):
        path = Path(name)
        if len(path.parts) != 4 or path.parts[0] != "sources" or path.suffix != ".parquet":
            continue
        competition = path.parts[1]
        group = competitions.setdefault(competition, {
            "kind": "proof" if competition in proof else "final_answer",
            "shards": [], "card": f"sources/{competition}/README.md",
        })
        group["shards"].append(name)
    return competitions

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


from measurement_db.scripts.build_measurement_tables.validate_characterization import (
    load_characterization, check_tables, check_source_claims,
)

# Reviewed migration invariants; current table snapshots live in characterization.yaml.
REGRESSION = {'legacy_baseline': {'observation_multiset_sha256': '5aa565e8d2a6287ecad65370056d516e498aab499859720464a955bd95c196de'},
 'release': {'responses_by_competition': {'aime_2025': 7915,
                                          'aime_2025_I': 1860,
                                          'aime_2025_II': 1860,
                                          'aime_2026': 2976,
                                          'aime_2026_I': 2376,
                                          'apex_2025': 7245,
                                          'apex_shortlist': 9237,
                                          'arxivmath_0126': 3279,
                                          'arxivmath_0226': 2135,
                                          'arxivmath_1225': 2832,
                                          'brumo_2025': 5280,
                                          'cmimc_2025': 5600,
                                          'hmmt_feb_2025': 7680,
                                          'hmmt_feb_2026': 3255,
                                          'hmmt_nov_2025': 2640,
                                          'imc_2025': 141,
                                          'imo_2025': 1556,
                                          'kangaroo_2025_11_12': 2280,
                                          'kangaroo_2025_1_2': 1824,
                                          'kangaroo_2025_3_4': 1824,
                                          'kangaroo_2025_5_6': 2189,
                                          'kangaroo_2025_7_8': 2160,
                                          'kangaroo_2025_9_10': 2160,
                                          'miklos_2025': 13,
                                          'putnam_2025': 72,
                                          'smt_2025': 10875,
                                          'usamo_2025': 1680}}}

class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("matharena_build", BENCHMARK_DIR / "build.py")
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def tables(self, changes=None, competition="aime_2025"):
        row = {"problem_idx": "1", "problem": "Compute the answer.", "model_name": "Test Model",
               "model_config": "test/model", "idx_answer": 0, "gold_answer": "42", "answer": "42", "correct": True}
        row.update(changes or {})
        source = pd.DataFrame([row], dtype=object)
        builder = self.builder.MathArenaBuild(BENCHMARK_DIR / "build.py")
        builder.source_files = (f"sources/{competition}/data/test.parquet",)
        with patch.object(self.builder.pq, "read_schema", return_value=SimpleNamespace(names=list(source.columns))), \
             patch.object(self.builder.pd, "read_parquet", return_value=source):
            return builder.build_tables()

    def test_json_and_native_rubrics_are_equivalent(self):
        rubric = [{"title": "Construction", "points": 1, "max_points": 3}]
        a = self.tables({"grading_details_judge_1": json.dumps(rubric)}, "putnam_2025")
        b = self.tables({"grading_details_judge_1": rubric}, "putnam_2025")
        for name in a:
            pd.testing.assert_frame_equal(a[name], b[name])
        with self.assertRaisesRegex(ValueError, "grading_details"):
            self.tables({"grading_details_judge_1": '{"not": "a criterion list"}'}, "putnam_2025")

    def test_legacy_fraction_clipping_and_missing_scores(self):
        cases = [(1, 3, 1 / 3), (-1, 3, 0), (4, 3, 1), (1, 0, None),
                 (None, 3, None), (1, None, None), (float("inf"), 3, None)]
        rubric = [{"title": str(i), "points": points, "max_points": maximum}
                  for i, (points, maximum, _) in enumerate(cases)]
        tables = self.tables({"grading_details_judge_1": rubric}, "putnam_2025")
        self.assertEqual(tables["responses"].response.tolist(), [expected for _, _, expected in cases if expected is not None])
        self.assertEqual(len(tables["traces"]), 1)
        self.assertEqual(tables["items"].verifier_features.map(lambda x: x["criterion_index"]).tolist(), [0, 1, 2])

    def test_answer_is_never_used_as_the_question(self):
        with self.assertRaisesRegex(ValueError, "no released prompt"):
            self.tables({"problem": None, "gold_answer": "A"})

    def test_proof_without_a_reference_preserves_null(self):
        tables = self.tables({"gold_answer": None, "grading_details_judge_1": [
            {"title": "Proof", "points": 1, "max_points": 3}]}, "putnam_2025")
        self.assertIsNone(tables["items"].grading_criterion.iloc[0]["reference_answer"])

    def test_prompt_and_trace_text_are_not_truncated(self):
        text = "  " + "proof " * 2000 + "  "
        tables = self.tables({"user_message": text, "answer": text})
        self.assertEqual(tables["items"].content.iloc[0], text)
        self.assertEqual(tables["traces"].trace.iloc[0], text)

    def test_effort_is_labelled_not_inferred_from_model_family(self):
        for label, effort in [("GPT-5.2 (high)", "high"), ("DeepSeek-v4-Pro (Max)", "max"), ("GPT-5", None)]:
            features = self.tables({"model_name": label})["subjects"].features.iloc[0]
            self.assertEqual(features.get("reasoning_effort"), effort)

    def test_inline_images_are_returned_as_bytes(self):
        payload = b"released image bytes"
        prompt = json.dumps([{"type": "text", "text": "Which figure?"},
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(payload).decode(), "detail": "high"}}])
        decoded = self.builder.decode_prompt(prompt, True)
        self.assertEqual(decoded["content"], "Which figure?")
        self.assertEqual(decoded["attachments"][0]["data"], payload)
        self.assertNotIn("source_path", decoded["attachments"][0])
        with self.assertRaisesRegex(ValueError, "not recovered"):
            self.builder.decode_prompt("Which figure?", True)


class MathArenaCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = load_benchmark_metadata(BENCHMARK_DIR / "metadata.yaml")
        required = [BENCHMARK_DIR / "formatted_tables" / f"{name}.parquet" for name in OUTPUT_NAMES]
        if all(path.is_file() for path in required):
            cls.sources = declared_source_artifacts(cls.metadata["sources"], benchmark_dir=BENCHMARK_DIR)
            cls.competitions = source_competitions(s["file"] for s in cls.sources)
            required += [BENCHMARK_DIR / "raw" / s["file"] for s in cls.sources]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            message = f"run benchmarks/matharena/build.py first; missing {missing[0]}"
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.expected = REGRESSION
        cls.characterization = load_characterization(BENCHMARK_DIR / "characterization.yaml")
        cls.tables = {
            name: pd.read_parquet(BENCHMARK_DIR / "formatted_tables" / f"{name}.parquet")
            for name in OUTPUT_NAMES
        }


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
        raw_rows = sum(pq.ParquetFile(BENCHMARK_DIR / "raw" / shard).metadata.num_rows
                       for competition in self.competitions.values()
                       for shard in competition["shards"])
        check_source_claims(self.characterization, {
            "provider_raw_rows": raw_rows,
            "released_response_grades": seen,
        })
        self.assertEqual(
            logical_digest(legacy_rows),
            self.expected["legacy_baseline"]["observation_multiset_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
