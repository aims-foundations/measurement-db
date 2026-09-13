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
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
)

from scripts.build_measurement_tables import validate_asset_relations, validate_table

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
        source = metadata["sources"]["downloads"][shard]
        parquet = pq.ParquetFile(BENCHMARK_DIR / "raw" / source["file"])
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
        required += [
            BENCHMARK_DIR / "raw" / s["file"]
            for s in cls.metadata["sources"]["downloads"].values()
        ]
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
                    name, table, include_derived=True, allow_extra=name == "responses"
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
        self.assertTrue((responses.trial == responses.source_idx_answer + 1).all())
        self.assertTrue(
            responses[["test_condition", "interactors", "trace"]].isna().all().all()
        )
        self.assertFalse(traces.duplicated(PRIMARY_KEY).any())
        linked = traces.merge(
            responses, on=PRIMARY_KEY, how="left", validate="one_to_one", indicator=True
        )
        self.assertTrue(linked["_merge"].eq("both").all())
        self.assertFalse(linked.duplicated(["source_shard", "source_row"]).any())
        self.assertEqual(
            responses.groupby("source_competition").size().to_dict(),
            self.expected["release"]["responses_by_competition"],
        )

    def test_image_questions_retain_assets_not_answer_letters(self):
        items = self.tables["items"]
        images = items[items.raw_item_id.str.startswith("kangaroo_")]
        self.assertTrue(images.asset_manifest.notna().all())
        self.assertFalse(images.content.isin(list("ABCDE")).any())
        self.assertEqual(images.raw_item_id.nunique(), len(self.tables["assets"]))
        validate_asset_relations(items, self.tables["assets"], benchmark_id="matharena")
        for row in self.tables["assets"].itertuples():
            self.assertEqual(hashlib.sha256(row.data).hexdigest(), row.asset_id)
            self.assertEqual(len(row.data), row.byte_size)

    def test_judges_and_criteria_are_item_instruments(self):
        proof = self.tables["responses"].dropna(subset=["source_judge_slot"])
        self.assertTrue(proof.response.between(0, 1).all())
        items = self.tables["items"].set_index("item_id")
        for item_id, rows in proof.groupby("item_id"):
            verifier = json.loads(items.loc[item_id, "verifier"])
            self.assertEqual(verifier["class"], "judge")
            self.assertEqual(verifier["judged_by"], "human")
            self.assertEqual(
                int(verifier["judge_slot"]), rows.source_judge_slot.iloc[0]
            )
            self.assertEqual(
                int(verifier["criterion_index"]), rows.source_criterion_index.iloc[0]
            )
            self.assertEqual(
                verifier["rubric_sha256"],
                hashlib.sha256(verifier["spec"].encode()).hexdigest(),
            )

    def test_all_pinned_sources_match_hashes_and_provider_card_counts(self):
        for name, source in self.metadata["sources"]["downloads"].items():
            with self.subTest(source=name):
                path = BENCHMARK_DIR / "raw" / source["file"]
                self.assertEqual(path.stat().st_size, source["size"])
                with path.open("rb") as stream:
                    self.assertEqual(
                        hashlib.file_digest(stream, "sha256").hexdigest(),
                        source["sha256"],
                    )
        for name, competition in self.metadata["sources"]["competitions"].items():
            card = self.metadata["sources"]["downloads"][competition["card"]]
            frontmatter = (
                (BENCHMARK_DIR / "raw" / card["file"]).read_text().split("---", 2)[1]
            )
            claimed = yaml.safe_load(frontmatter)["dataset_info"]["splits"][0][
                "num_examples"
            ]
            observed = sum(
                pq.ParquetFile(
                    BENCHMARK_DIR
                    / "raw"
                    / self.metadata["sources"]["downloads"][shard]["file"]
                ).metadata.num_rows
                for shard in competition["shards"]
            )
            self.assertEqual(observed, claimed, name)

    def test_every_released_score_prompt_configuration_and_trace_reconciles(self):
        """Independent row-level audit, including every formerly skipped rubric."""
        responses = self.tables["responses"]
        groups = defaultdict(list)
        for response in responses.to_dict("records"):
            groups[(response["source_shard"], response["source_row"])].append(response)
        items = self.tables["items"].set_index("item_id").to_dict("index")
        subjects = self.tables["subjects"].set_index("subject_id").to_dict("index")
        traces = {
            tuple(row[:5]): row[5]
            for row in self.tables["traces"][[*PRIMARY_KEY, "trace"]].itertuples(
                index=False, name=None
            )
        }
        assets = self.tables["assets"].set_index("asset_id")["data"].to_dict()
        seen = 0
        legacy_rows = []
        for competition_name, competition in self.metadata["sources"][
            "competitions"
        ].items():
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
                actual = groups.pop((shard, row_index), [])
                self.assertEqual(len(actual), len(released), (shard, row_index))
                actual.sort(
                    key=lambda row: (
                        row["source_judge_slot"] or 0,
                        row["source_criterion_index"] or 0,
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
                        response["source_problem_id"], str(record["problem_idx"])
                    )
                    self.assertEqual(
                        response["source_idx_answer"], record["idx_answer"]
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
                    self.assertEqual(normalized(response["reference_answer"]), reference)
                    self.assertEqual(normalized(item["reference_answer"]), reference)
                    if criterion is not None:
                        verifier = json.loads(item["verifier"])
                        self.assertEqual(
                            json.loads(verifier["spec"]),
                            {
                                key: criterion.get(key)
                                for key in (
                                    "title",
                                    "grading_scheme_desc",
                                    "max_points",
                                )
                            },
                        )
                        self.assertEqual(response["source_judge_slot"], judge)
                        self.assertEqual(response["source_criterion_index"], index)
                        self.assertEqual(
                            response["grader_feedback"], criterion.get("desc")
                        )
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
