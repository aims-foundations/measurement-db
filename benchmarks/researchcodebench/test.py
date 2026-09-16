"""Read-only source reconciliation and characterization for ResearchCodeBench."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tarfile
import unittest
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
IMPORT_ROOT = BENCHMARK_DIR.parents[2]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from measurement_db.scripts.build_measurement_tables import (
    validate_asset_relations,
    validate_table,
)
from measurement_db.scripts.build_measurement_tables.validate_benchmark_metadata import (
    load_benchmark_metadata,
    declared_source_artifacts,
)

from measurement_db.scripts.build_measurement_tables.source_snapshots import verify_snapshot_file
from measurement_db.benchmarks.researchcodebench.build import LAYOUT, EXPECTED_RELEASE

METADATA = load_benchmark_metadata(BENCHMARK_DIR / "metadata.yaml")
EXPECTED = json.loads(
    (BENCHMARK_DIR / "testdata" / "characterization.json").read_text(
        encoding="utf-8"
    )
)["release"]
SOURCE_CLAIMS = METADATA["validation"]["source_claims"]
OUTPUT_NAMES = ("items", "subjects", "benchmarks", "responses")

_PAPER_HEADER = "You are an expert in reproducing research code from a paper."
_NO_PAPER_HEADER = "You are an expert in completing research code."
_ALIAS_DISPLAY = {
    "GEMINI_2_5_PRO_PREVIEW_05_06": "GEMINI_2_5_PRO_PREVIEW_03_25"
}


def _digest_strings(values: Iterable[object]) -> str:
    """Return an order-independent, length-framed SHA-256 digest."""

    digest = hashlib.sha256()
    for value in sorted(str(value) for value in values):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalized_cell(value: object) -> object:
    """Convert one Parquet cell to a deterministic JSON-compatible value."""

    if isinstance(value, (list, tuple)):
        return [_normalized_cell(member) for member in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return [_normalized_cell(member) for member in converted]
        value = converted
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _serialized_rows(table: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Serialize selected rows for an order-independent logical fingerprint."""

    selected_columns = list(columns)
    return [
        json.dumps(
            [_normalized_cell(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in table.loc[:, selected_columns].itertuples(index=False, name=None)
    ]


def _parse_features(serialized: str) -> dict[str, str]:
    """Parse the registry's semicolon-delimited feature representation."""

    return dict(part.split("=", 1) for part in serialized.split(";") if part)


def _release_rows(
    stats: dict[str, object],
) -> list[dict[str, object]]:
    """Expand the provider result matrix independently of the builder."""

    rows: list[dict[str, object]] = []
    trial_by_cell: Counter[tuple[str, str]] = Counter()
    results = stats["results"]
    assert isinstance(results, dict)
    for paper, paper_data in results.items():
        assert isinstance(paper_data, dict)
        model_results = paper_data["results"]
        assert isinstance(model_results, dict)
        for raw_model, model_data in model_results.items():
            assert isinstance(model_data, dict)
            snippet_results = model_data["results"]
            assert isinstance(snippet_results, dict)
            display_name = _ALIAS_DISPLAY.get(raw_model, raw_model)
            for snippet, completions in snippet_results.items():
                raw_item_id = f"{paper}::{snippet}"
                assert isinstance(completions, list)
                for completion in completions:
                    assert isinstance(completion, dict)
                    cell = (display_name, raw_item_id)
                    trial_by_cell[cell] += 1
                    rows.append(
                        {
                            "display_name": display_name,
                            "raw_model": raw_model,
                            "raw_item_id": raw_item_id,
                            "trial": trial_by_cell[cell],
                            "response": 1.0 if completion.get("passed") else 0.0,
                            "completion_idx": completion.get("completion_idx"),
                            "exit_code": completion.get("exit_code"),
                            "snippet_code_line_count": completion.get(
                                "snippet_code_line_count"
                            ),
                        }
                    )
    return rows


class ResearchCodeBenchCharacterizationTests(unittest.TestCase):
    """Reconcile the pinned release and protect accepted benchmark semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        missing_outputs = [
            name
            for name in OUTPUT_NAMES
            if not (BENCHMARK_DIR / f"{name}.parquet").is_file()
        ]
        if missing_outputs:
            message = (
                "generated ResearchCodeBench table(s) are absent: "
                f"{', '.join(missing_outputs)}; run the benchmark build first"
            )
            if os.environ.get("MEASUREMENT_DB_FULL_TEST") == "1":
                raise RuntimeError(message)
            raise unittest.SkipTest(message)

        archive_descriptor = declared_source_artifacts(METADATA["sources"], benchmark_dir=BENCHMARK_DIR)[0]
        cls.archive_path = BENCHMARK_DIR / "raw" / archive_descriptor["file"]
        if not cls.archive_path.is_file():
            raise RuntimeError(
                "pinned ResearchCodeBench archive is absent; rerun build.py"
            )
        verify_snapshot_file(cls.archive_path, archive_descriptor)

        layout = LAYOUT
        archive_root = str(layout["root"])
        results_member = str(layout["results_member"])
        full_results_member = f"{archive_root}/{results_member}"
        selected_source_members: list[str] = []
        released_run_members: list[str] = []
        released_run_prefix = str(PurePosixPath(results_member).parent).rstrip("/")
        with tarfile.open(cls.archive_path, "r:gz") as archive:
            extracted = archive.extractfile(full_results_member)
            if extracted is None:
                raise RuntimeError("released overall_stats.json is absent from archive")
            cls.stats = json.load(extracted)
            released_papers = set(cls.stats["results"])
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != archive_root:
                    raise RuntimeError(f"unexpected archive member: {member.name}")
                relative = PurePosixPath(*parts[1:])
                relative_text = relative.as_posix()
                relative_parts = relative.parts
                if relative_text.startswith(f"{released_run_prefix}/"):
                    released_run_members.append(relative_text)
                if (
                    len(relative_parts) < 3
                    or relative_parts[0] != layout["pset_prefix"]
                    or relative_parts[1] not in released_papers
                ):
                    continue
                if (
                    relative_text.endswith(".py")
                    and not relative_text.endswith("paper2code_test.py")
                ) or relative_text.endswith(
                    ("paper2code_paper.tex", "paper2code.yaml")
                ):
                    selected_source_members.append(relative_text)
        cls.selected_source_members = selected_source_members
        cls.released_run_members = released_run_members
        cls.source_rows = _release_rows(cls.stats)

        cls.items = pd.read_parquet(BENCHMARK_DIR / "items.parquet")
        cls.subjects = pd.read_parquet(BENCHMARK_DIR / "subjects.parquet")
        cls.benchmarks = pd.read_parquet(BENCHMARK_DIR / "benchmarks.parquet")
        cls.responses = pd.read_parquet(BENCHMARK_DIR / "responses.parquet")
        cls.assets = None
        cls.traces = None

    def test_provider_release_counts_and_aggregates(self) -> None:
        count_claims = SOURCE_CLAIMS["counts"]
        results = self.stats["results"]
        raw_models: set[str] = set()
        raw_items: set[str] = set()
        per_model_passes: Counter[str] = Counter()
        per_model_rows: Counter[str] = Counter()
        for row in self.source_rows:
            raw_models.add(str(row["raw_model"]))
            raw_items.add(str(row["raw_item_id"]))
            per_model_passes[str(row["raw_model"])] += int(row["response"])
            per_model_rows[str(row["raw_model"])] += 1

        self.assertEqual(len(results), EXPECTED["papers"])
        self.assertEqual(len(raw_items), count_claims["released_items"]["expected"])
        self.assertEqual(
            len(raw_models),
            count_claims["released_subject_configurations"]["expected"],
        )
        self.assertEqual(
            len(self.source_rows),
            count_claims["released_response_cells"]["expected"],
        )
        self.assertEqual(count_claims["released_traces"]["expected"], 0)
        self.assertEqual(self.released_run_members, [LAYOUT["results_member"]])
        self.assertEqual(
            len(self.selected_source_members),
            EXPECTED_RELEASE["selected_pset_files"],
        )
        self.assertEqual(
            Counter(per_model_rows.values()),
            {EXPECTED["items"]: len(raw_models)},
        )

        aggregate_claim = SOURCE_CLAIMS["aggregate_tables"][
            "released_task_rates"
        ]
        overall_scores = self.stats["overall_scores"]
        self.assertEqual(len(overall_scores), aggregate_claim["expected_groups"])
        for raw_model in sorted(raw_models):
            released_rate = float(overall_scores[raw_model]["task_rates"]["mean"])
            observed_rate = 100.0 * per_model_passes[raw_model] / per_model_rows[raw_model]
            self.assertTrue(
                math.isclose(
                    observed_rate,
                    released_rate,
                    abs_tol=float(aggregate_claim["absolute_tolerance"]),
                ),
                raw_model,
            )

        self.assertEqual(
            Counter(int(row["exit_code"]) for row in self.source_rows),
            {int(key): value for key, value in EXPECTED["exit_code_counts"].items()},
        )
        self.assertEqual({row["completion_idx"] for row in self.source_rows}, {0})
        self.assertTrue(
            all(
                isinstance(row["snippet_code_line_count"], int)
                and int(row["snippet_code_line_count"]) > 0
                for row in self.source_rows
            )
        )

    def test_every_released_response_cell_is_preserved(self) -> None:
        source = pd.DataFrame(self.source_rows)[
            ["display_name", "raw_item_id", "trial", "response"]
        ].sort_values(["display_name", "raw_item_id", "trial"]).reset_index(drop=True)
        curated = (
            self.responses.merge(
                self.subjects[["subject_id", "display_name"]],
                on="subject_id",
                validate="many_to_one",
            )
            .merge(
                self.items[["item_id", "raw_item_id"]],
                on="item_id",
                validate="many_to_one",
            )[["display_name", "raw_item_id", "trial", "response"]]
            .sort_values(["display_name", "raw_item_id", "trial"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(source, curated, check_dtype=False, check_exact=True)

    def test_release_shape_and_table_schemas(self) -> None:
        self.assertEqual(len(self.items), EXPECTED["items"])
        self.assertEqual(len(self.subjects), EXPECTED["subjects"])
        self.assertEqual(len(self.responses), EXPECTED["responses"])
        self.assertEqual(len(self.benchmarks), 1)
        self.assertTrue(self.items["item_id"].is_unique)
        self.assertTrue(self.subjects["subject_id"].is_unique)
        self.assertTrue(self.responses["response_id"].is_unique)
        self.assertTrue(self.benchmarks["benchmark_id"].is_unique)
        self.assertEqual(self.subjects["normalized_name"].nunique(), EXPECTED["subjects"])

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
        validate_asset_relations(
            self.items,
            self.assets,
            benchmark_id="researchcodebench",
            context=BENCHMARK_DIR.name,
            response_scale=self.benchmarks.iloc[0].response_scale,
        )

    def test_prompt_and_reference_contract(self) -> None:
        self.assertEqual(self.items["item_id"].nunique(), EXPECTED["items"])
        self.assertEqual(self.items["content"].nunique(), EXPECTED["items"])
        self.assertEqual(
            self.items["grading_criterion"].map(lambda value: json.loads(value)["reference_answer"]).nunique(), EXPECTED["items"]
        )
        self.assertTrue(self.items["content"].str.strip().ne("").all())
        self.assertTrue(self.items["grading_criterion"].map(lambda value: json.loads(value)["reference_answer"]).str.strip().ne("").all())
        self.assertTrue(self.items["verifier"].map(lambda value: json.loads(value)["class"] == "exact_matcher").all())
        self.assertTrue(
            self.items["content"].str.contains(
                "Here is the code that you need to complete:", regex=False
            ).all()
        )
        self.assertTrue(
            self.items["content"].str.contains(
                "# TODO: Implement block", regex=False
            ).all()
        )
        self.assertFalse(
            self.items["content"].str.contains("<paper2code", regex=False).any()
        )

        with_paper = self.items["content"].str.contains(_PAPER_HEADER, regex=False)
        without_paper = self.items["content"].str.contains(
            _NO_PAPER_HEADER, regex=False
        )
        self.assertEqual(int(with_paper.sum()), EXPECTED["with_paper_prompts"])
        self.assertEqual(int(without_paper.sum()), EXPECTED["without_paper_prompts"])
        self.assertTrue(
            self.items.loc[without_paper, "raw_item_id"].str.startswith("GPS::").all()
        )

        paper_counts = Counter(
            _parse_features(value)["paper"] for value in self.items["item_features"]
        )
        self.assertEqual(dict(sorted(paper_counts.items())), EXPECTED["paper_item_counts"])
        response_references = self.responses.merge(
            self.items.assign(reference_answer=self.items.grading_criterion.map(lambda value: json.loads(value)["reference_answer"]))[["item_id", "reference_answer"]],
            on="item_id",
            validate="many_to_one",
        )
        self.assertTrue(response_references["reference_answer"].notna().all())
        self.assertNotIn("reference_answer", self.responses)

    def test_subject_identity_and_trials(self) -> None:
        raw_models = {str(row["raw_model"]) for row in self.source_rows}
        expected_displays = raw_models - set(_ALIAS_DISPLAY)
        self.assertEqual(set(self.subjects["display_name"]), expected_displays)
        for column in (
            "access_date",
            "harness",
            "reasoning_effort",
            "harness_version",
            "subject_features_extra",
        ):
            self.assertTrue(self.subjects[column].isna().all(), column)

        rows_per_subject = {
            str(row_count): int(subject_count)
            for row_count, subject_count in self.responses.groupby("subject_id")
            .size()
            .value_counts()
            .items()
        }
        trial_counts = {
            str(trial): int(count)
            for trial, count in self.responses["trial"].value_counts().items()
        }
        self.assertEqual(rows_per_subject, EXPECTED["rows_per_subject"])
        self.assertEqual(trial_counts, EXPECTED["trial_counts"])

        alias_subject_id = self.subjects.loc[
            self.subjects["display_name"].eq("GEMINI_2_5_PRO_PREVIEW_03_25"),
            "subject_id",
        ].item()
        alias_rows = self.responses[self.responses["subject_id"].eq(alias_subject_id)]
        alias_passes = {
            str(int(trial)): int(group["response"].sum())
            for trial, group in alias_rows.groupby("trial")
        }
        alias_sizes = {
            str(int(trial)): len(group) for trial, group in alias_rows.groupby("trial")
        }
        self.assertEqual(alias_passes, EXPECTED["gemini_preview_passes_by_trial"])
        self.assertEqual(alias_sizes, {"1": EXPECTED["items"], "2": EXPECTED["items"]})

    def test_binary_response_matrix_and_null_patterns(self) -> None:
        response_counts = {
            str(int(response)): int(count)
            for response, count in self.responses["response"].value_counts().items()
        }
        self.assertEqual(response_counts, EXPECTED["response_counts"])
        self.assertEqual(
            self.responses.groupby("item_id").size().value_counts().to_dict(),
            {32: EXPECTED["items"]},
        )
        for column in ("test_condition", "interactors"):
            self.assertTrue(self.responses[column].isna().all(), column)
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

    def test_no_trace_or_asset_sidecars(self) -> None:
        self.assertEqual(EXPECTED["traces"], 0)
        self.assertEqual(EXPECTED["assets"], 0)
        self.assertFalse((BENCHMARK_DIR / "traces.parquet").exists())
        self.assertFalse((BENCHMARK_DIR / "assets.parquet").exists())

    def test_logical_fingerprints(self) -> None:
        benchmark_columns = (
            "benchmark_id",
            "name",
            "version",
            "license",
            "source_url",
            "description",
            "one_line_description",
            "modality",
            "domain",
            "multi_single_turn",
            "response_type",
            "response_scale",
            "categorical",
            "paper_url",
            "release_date",
            "granularity",
            "release",
            "benchmark_features",
        )
        observed = {
            "item_ids_sha256": _digest_strings(self.items["item_id"]),
            "content_hashes_sha256": _digest_strings(self.items["content_hash"]),
            "item_rows_sha256": _digest_strings(
                _serialized_rows(self.items, self.items.columns)
            ),
            "reference_answers_sha256": _digest_strings(
                self.items["grading_criterion"].map(lambda value: json.loads(value)["reference_answer"])
            ),
            "verifiers_sha256": _digest_strings(self.items["verifier"]),
            "subject_ids_sha256": _digest_strings(self.subjects["subject_id"]),
            "subject_rows_sha256": _digest_strings(
                _serialized_rows(self.subjects, self.subjects.columns)
            ),
            "benchmark_row_sha256": _digest_strings(
                _serialized_rows(self.benchmarks, benchmark_columns)
            ),
            "response_ids_sha256": _digest_strings(self.responses["response_id"]),
            "response_cells_sha256": _digest_strings(
                _serialized_rows(self.responses, self.responses.columns)
            ),
            "asset_rows_sha256": _digest_strings([]),
            "trace_rows_sha256": _digest_strings([]),
        }
        self.assertEqual(observed, {key: EXPECTED[key] for key in observed})


if __name__ == "__main__":
    unittest.main(verbosity=2)
