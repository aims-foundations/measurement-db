"""Offline gallery regressions using mocked HF downloads and Parquet fixtures."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import pandas as pd
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_PATH = REPO_ROOT / "scripts" / "render_website" / "generate_benchmark_gallery.py"
WEBSITE_TYPES_PATH = REPO_ROOT / "website" / "content" / "measurement-db.ts"
ERROR_RESPONSE = httpx.Response(403, request=httpx.Request("GET", "https://example.test"))


def _load_gallery_module():
    spec = importlib.util.spec_from_file_location(
        "measurement_db_generate_benchmark_gallery_tests", GALLERY_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError(f"cannot load gallery builder from {GALLERY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gallery = _load_gallery_module()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _local_fetch(filename: str, cache_dir: Path, refresh: bool = False) -> Path:
    """A fetch replacement that cannot fall through to the network."""
    del refresh
    path = Path(cache_dir) / filename
    if not path.exists():
        raise SystemExit(f"offline fixture is missing {filename}")
    return path


def _write_schema_pair(root: Path) -> dict:
    """Write semantically equivalent legacy and normalized-schema fixtures."""
    slug = "schema_fixture"
    legacy_cache = root / "legacy"
    normalized_cache = root / "normalized"

    subjects = {
        "subject-a": {
            "harness": "harness-a",
            "reasoning_effort": "low",
            "quantization": "int8",
        },
        "subject-b": {
            "harness": "harness-b",
            "reasoning_effort": "high",
            "quantization": "fp16",
        },
    }
    items = {
        "item-easy": {
            "difficulty": "easy",
            "judge": "judge-a",
        },
        "item-hard": {
            "difficulty": "hard",
            "judge": "judge-b",
        },
    }

    _write_parquet(
        legacy_cache / slug / "subjects.parquet",
        [{"subject_id": subject_id} for subject_id in subjects],
    )
    _write_parquet(
        legacy_cache / slug / "items.parquet",
        [{"item_id": item_id} for item_id in items],
    )

    _write_parquet(
        normalized_cache / slug / "subjects.parquet",
        [
            {
                "subject_id": subject_id,
                "harness": features["harness"],
                "reasoning_effort": features["reasoning_effort"],
                "harness_version": None,
                "subject_features_extra": (
                    f"quantization={features['quantization']}"
                ),
            }
            for subject_id, features in subjects.items()
        ],
    )
    _write_parquet(
        normalized_cache / slug / "items.parquet",
        [
            {
                "item_id": item_id,
                "item_features": f"difficulty={features['difficulty']}",
                "verifier": json.dumps(
                    {
                        "class": "judge",
                        "judge": features["judge"],
                        "judged_by": "llm",
                        "spec": "fixture rubric",
                    },
                    sort_keys=True,
                ),
            }
            for item_id, features in items.items()
        ],
    )

    legacy_responses: list[dict] = []
    normalized_responses: list[dict] = []
    legacy_traces: list[dict] = []
    normalized_traces: list[dict] = []
    semantic_rows: list[dict] = []
    row_number = 0
    for subject_id, subject_features in subjects.items():
        for item_id, item_features in items.items():
            for temperature in ("0", "1"):
                for opponent in ("opponent-a", "opponent-b"):
                    dimensions = {
                        "difficulty": item_features["difficulty"],
                        "harness": subject_features["harness"],
                        "judge": item_features["judge"],
                        "opponent": opponent,
                        "quantization": subject_features["quantization"],
                        "reasoning_effort": subject_features["reasoning_effort"],
                        "temperature": temperature,
                    }
                    display_condition = ";".join(
                        f"{key}={value}" for key, value in sorted(dimensions.items())
                    )
                    trace = (
                        f"trace:{subject_id}:{item_id}:"
                        f"temperature={temperature}:opponent={opponent}"
                    )
                    response = float(row_number % 2)
                    row_number += 1

                    legacy_key = {
                        "subject_id": subject_id,
                        "item_id": item_id,
                        "test_condition": display_condition,
                        "trial": 1,
                    }
                    normalized_key = {
                        "subject_id": subject_id,
                        "item_id": item_id,
                        "test_condition": f"temperature={temperature}",
                        "trial": 1,
                        "interactors": f"opponent={opponent}",
                    }
                    legacy_responses.append({**legacy_key, "response": response})
                    normalized_responses.append(
                        {**normalized_key, "response": response}
                    )
                    legacy_traces.append({**legacy_key, "trace": trace})
                    normalized_traces.append({**normalized_key, "trace": trace})
                    semantic_rows.append(
                        {
                            **legacy_key,
                            "response": response,
                            "trace": trace,
                            "dimensions": dimensions,
                        }
                    )

    _write_parquet(
        legacy_cache / slug / "responses.parquet", legacy_responses
    )
    _write_parquet(legacy_cache / slug / "traces.parquet", legacy_traces)
    _write_parquet(
        normalized_cache / slug / "responses.parquet", normalized_responses
    )
    _write_parquet(
        normalized_cache / slug / "traces.parquet", normalized_traces
    )

    return {
        "slug": slug,
        "legacy_cache": legacy_cache,
        "normalized_cache": normalized_cache,
        "semantic_rows": semantic_rows,
        "dims": sorted(semantic_rows[0]["dimensions"]),
    }


class GallerySaturationAnalysisTests(unittest.TestCase):
    def test_cards_use_analysis_output_not_legacy_collection_column(self) -> None:
        with TemporaryDirectory() as tmp:
            web_dir = Path(tmp)
            path = (
                web_dir
                / "content"
                / "generated"
                / "benchmark-saturation.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text('{"fixture": true}\n', encoding="utf-8")

            analysis = gallery.load_saturation_analysis(web_dir)
            card = gallery.card_for(
                "fixture",
                {
                    "benchmark_id": "fixture",
                    # Old published tables can retain this legacy column. It
                    # must not override the standalone analysis result.
                    "saturation": False,
                },
                {},
                {"benchmarks": {}, "institutions": {}},
                analysis.get("fixture"),
            )

        self.assertIsNotNone(card)
        assert card is not None
        self.assertIs(card["saturation"], True)

    def test_missing_analysis_output_means_no_verdict(self) -> None:
        with TemporaryDirectory() as tmp:
            analysis = gallery.load_saturation_analysis(Path(tmp))

        self.assertEqual(analysis, {})


class GalleryCardDescriptionTests(unittest.TestCase):
    def test_card_prefers_trimmed_one_line_description(self) -> None:
        card = gallery.card_for(
            "fixture",
            {
                "description": "Full benchmark overview.",
                "one_line_description": "  Evaluates the core task.  ",
            },
            {},
            {"benchmarks": {}, "institutions": {}},
        )

        self.assertIsNotNone(card)
        assert card is not None
        self.assertEqual(card["description"], "Evaluates the core task.")

    def test_card_falls_back_to_trimmed_full_description(self) -> None:
        for one_line_description in (None, "", "   "):
            with self.subTest(one_line_description=one_line_description):
                card = gallery.card_for(
                    "fixture",
                    {
                        "description": "  Full benchmark overview.  ",
                        "one_line_description": one_line_description,
                    },
                    {},
                    {"benchmarks": {}, "institutions": {}},
                )

                self.assertIsNotNone(card)
                assert card is not None
                self.assertEqual(card["description"], "Full benchmark overview.")


class GalleryDownloadTests(unittest.TestCase):
    def setUp(self):
        source = patch.object(gallery, "source_tables", side_effect=lambda slug:
                              gallery.TableSource(slug, frozenset()))
        source.start()
        self.addCleanup(source.stop)

    def test_legacy_fallback_keeps_canonical_path_and_yields_to_current_file(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            legacy = cache / "sdk-legacy"
            current = cache / "sdk-current"
            legacy.write_bytes(b"old table")
            current.write_bytes(b"current table")
            with (
                patch.object(gallery, "source_revision", return_value="pinned") as revision,
                patch.object(gallery, "hf_hub_download", side_effect=[
                    gallery.EntryNotFoundError("missing"), str(legacy),
                ]) as download,
            ):
                path = gallery.fetch("fixture/responses.parquet", cache, refresh=True)
                self.assertEqual(path, cache / "fixture/responses.parquet")
                self.assertEqual(path.read_bytes(), b"old table")
                self.assertEqual([c.args[1] for c in download.call_args_list],
                                 ["fixture/responses.parquet", "fixture/response.parquet"])
                for call in download.call_args_list:
                    self.assertEqual(call.args[0], "aims-foundations/measurement-db")
                    self.assertEqual(call.kwargs["revision"], "pinned")
                    self.assertTrue(call.kwargs["force_download"])

                revision.return_value = "next-pinned"
                download.reset_mock()
                download.side_effect = None
                download.return_value = str(current)
                gallery.fetch("fixture/responses.parquet", cache)
                download.assert_called_once()
                self.assertEqual(download.call_args.args[1], "fixture/responses.parquet")
                self.assertEqual(path.read_bytes(), b"current table")

    def test_response_fallback_never_masks_access_revision_or_network_errors(self):
        errors = [gallery.LocalEntryNotFoundError("offline"),
                  GatedRepoError("gated", response=ERROR_RESPONSE),
                  RevisionNotFoundError("revision", response=ERROR_RESPONSE), OSError("network")]
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            old = cache / "fixture/response.parquet"
            old.parent.mkdir()
            old.write_bytes(b"stale local table")
            for error in errors:
                with (
                    self.subTest(error=type(error).__name__),
                    patch.object(gallery, "source_revision", return_value="pinned"),
                    patch.object(gallery, "hf_hub_download", side_effect=error) as download,
                ):
                    with self.assertRaises(type(error)):
                        gallery.fetch("fixture/responses.parquet", cache)
                    download.assert_called_once()
                    self.assertFalse((cache / "fixture/responses.parquet").exists())

    def test_both_response_files_missing_fail_without_using_stale_local_data(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "fixture/responses.parquet"
            path.parent.mkdir()
            path.write_bytes(b"stale table")
            with (
                patch.object(gallery, "source_revision", return_value="pinned"),
                patch.object(gallery, "hf_hub_download",
                             side_effect=gallery.EntryNotFoundError("missing")) as download,
            ):
                with self.assertRaises(SystemExit):
                    gallery.fetch("fixture/responses.parquet", cache)
                self.assertEqual(download.call_count, 2)
                self.assertEqual(path.read_bytes(), b"stale table")

    def test_fetch_replaces_legacy_path_with_pinned_hf_download(self) -> None:
        filename = "fixture/responses.parquet"
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            dest = cache_dir / filename
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"legacy-schema-cache")
            sdk_file = cache_dir / "sdk-immutable-file"
            sdk_file.write_bytes(b"current-schema")

            with (
                patch.object(gallery, "source_revision", return_value="pinned-commit"),
                patch.object(gallery, "hf_hub_download", return_value=str(sdk_file))
                as download,
            ):
                self.assertEqual(gallery.fetch(filename, cache_dir), dest)
                download.assert_called_with(
                    gallery.HF_REPO, filename, repo_type="dataset",
                    revision="pinned-commit",
                    force_download=False,
                )
                self.assertTrue(dest.is_symlink())
                self.assertEqual(dest.read_bytes(), b"current-schema")

                next_file = cache_dir / "sdk-next-immutable-file"
                next_file.write_bytes(b"next-schema")
                download.return_value = str(next_file)
                self.assertEqual(gallery.fetch(filename, cache_dir, refresh=True), dest)
                self.assertTrue(download.call_args.kwargs["force_download"])
                self.assertEqual(dest.read_bytes(), b"next-schema")
                self.assertEqual(sdk_file.read_bytes(), b"current-schema")

    def test_only_remote_missing_files_are_treated_as_optional(self) -> None:
        filename = "fixture/traces.parquet"
        errors = [
            (gallery.EntryNotFoundError("missing"), SystemExit),
            (gallery.LocalEntryNotFoundError("offline"), gallery.LocalEntryNotFoundError),
            (GatedRepoError("no access", response=ERROR_RESPONSE), GatedRepoError),
            (OSError("connection failed"), OSError),
        ]
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            dest = cache_dir / filename
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"previous-download")
            for error, expected in errors:
                with (
                    self.subTest(error=type(error).__name__),
                    patch.object(gallery, "source_revision", return_value="new-commit"),
                    patch.object(gallery, "hf_hub_download", side_effect=error),
                ):
                    with self.assertRaises(expected):
                        gallery.fetch(filename, cache_dir)
                    self.assertEqual(dest.read_bytes(), b"previous-download")



class GallerySchemaCompatibilityTests(unittest.TestCase):
    def setUp(self):
        source = patch.object(gallery, "source_tables", side_effect=lambda slug:
                              gallery.TableSource(slug, frozenset()))
        source.start()
        self.addCleanup(source.stop)

    def test_current_criteria_and_legacy_answers_render_consistently(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "items.parquet"
            for field, value, expected in (
                ("reference_answer", "A", "A"),
                ("correct_answer", "A", "A"),
                ("grading_criterion", json.dumps({"reference_answer": "A", "rule": None}), "A"),
                ("grading_criterion", json.dumps({"reference_answer": None, "rule": "Tests pass"}), None),
            ):
                with self.subTest(field=field, value=value):
                    _write_parquet(path, [{"item_id": "item", "content": "Question", field: value}])
                    with patch.object(gallery, "fetch", return_value=path):
                        row = gallery.read_item("fixture", Path(temporary), "item")
                    self.assertEqual(row["answer"], expected)
                    self.assertEqual(row["content"], "Question")


    def test_normalized_schema_recreates_legacy_matrix_dimensions_and_traces(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _write_schema_pair(Path(tmp))
            with patch.object(gallery, "fetch", side_effect=_local_fetch):
                legacy = gallery.load_raw(
                    fixture["slug"], fixture["legacy_cache"], refresh=False
                )
                normalized = gallery.load_raw(
                    fixture["slug"], fixture["normalized_cache"], refresh=False
                )

            legacy_plan = gallery.plan_slices(fixture["slug"], legacy).observations
            normalized_plan = gallery.plan_slices(fixture["slug"], normalized).observations
            columns = [
                "subject_id",
                "item_id",
                "test_condition",
                "trial",
                "response",
            ]
            sort_by = columns[:4]
            legacy = legacy[columns].sort_values(sort_by).reset_index(drop=True)
            normalized = (
                normalized[columns].sort_values(sort_by).reset_index(drop=True)
            )

            self.assertEqual(len(normalized), 16)
            pd.testing.assert_frame_equal(normalized, legacy)
            self.assertTrue(normalized["test_condition"].str.contains("judge=").all())
            self.assertTrue(
                normalized["test_condition"].str.contains("opponent=").all()
            )
            self.assertFalse(
                normalized["test_condition"].str.contains("judged_by=").any()
            )

            expected_kinds = {
                "difficulty": "item",
                "harness": "subject",
                "judge": "item",
                "opponent": "condition",
                "quantization": "subject",
                "reasoning_effort": "subject",
                "temperature": "condition",
            }
            self.assertEqual(
                gallery.classify_dims(
                    legacy_plan, fixture["dims"]
                ),
                expected_kinds,
            )
            self.assertEqual(
                gallery.classify_dims(
                    normalized_plan, fixture["dims"]
                ),
                expected_kinds,
            )

    def test_answers_still_match_after_feature_projection(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _write_schema_pair(Path(tmp))
            with patch.object(gallery, "fetch", side_effect=_local_fetch):
                frame = gallery.load_raw(fixture["slug"], fixture["normalized_cache"], False)
                for (_, row), expected in zip(frame.iterrows(), fixture["semantic_rows"]):
                    self.assertEqual(row["test_condition"], expected["test_condition"])
                    answer = gallery.read_answer(fixture["slug"], fixture["normalized_cache"], row["_key"])
                    self.assertEqual(answer["trace"], expected["trace"])



class GalleryRenderCommandTests(unittest.TestCase):
    def test_existing_benchmark_is_rendered_again_without_refresh(self) -> None:
        slug = "fixture"
        entry = {
            "slug": slug,
            "conditions": None,
            "hasTraces": False,
            "stats": {"subjects": 2, "items": 3},
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            web_dir, cache_dir = root / "website", root / "cache"
            details_path = web_dir / "content/generated/benchmark-details.json"
            details_path.parent.mkdir(parents=True)
            unrelated = {"slug": "other"}
            details_path.write_text(json.dumps({"other": unrelated}))
            with (
                patch.object(gallery, "load_overrides", return_value={}),
                patch.object(gallery, "build_detail", return_value={"detail": entry}) as build,
                patch.object(gallery, "HfApi", side_effect=AssertionError(
                    "The render command should not query remote file versions."
                )),
                redirect_stderr(io.StringIO()),
            ):
                for _ in range(2):
                    gallery.render_benchmarks([slug], cache_dir, web_dir, refresh=False)
                    build.assert_called_with(slug, cache_dir, web_dir, False, {})
                    self.assertEqual(json.loads(details_path.read_text()), {
                        "other": unrelated, slug: entry,
                    })
                self.assertEqual(build.call_count, 2)


class GalleryDomainVocabularyTests(unittest.TestCase):
    def test_python_and_typescript_domains_stay_in_lockstep(self) -> None:
        source = WEBSITE_TYPES_PATH.read_text(encoding="utf-8")
        union = re.search(
            r"export type BenchmarkCategoryId\s*=\s*(.*?);", source, re.DOTALL
        )
        self.assertIsNotNone(union)
        assert union is not None
        union_ids = re.findall(r'\|\s*"([^"]+)"', union.group(1))

        array_start = source.index("export const benchmarkCategories")
        array_end = source.index("] as const;", array_start)
        category_ids = re.findall(
            r'\bid:\s*"([^"]+)"', source[array_start:array_end]
        )

        self.assertEqual(len(gallery.DOMAIN_ORDER), len(gallery.DOMAIN_SET))
        self.assertEqual(union_ids, gallery.DOMAIN_ORDER)
        self.assertEqual(category_ids, gallery.DOMAIN_ORDER)
        self.assertEqual(gallery.categories_for(["education"]), ["education"])
        self.assertEqual(
            gallery.categories_for(["education", "mathematics"]),
            ["mathematics", "education"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
