"""Offline regression tests for the benchmark attribution manifest."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts/render_website/generate_benchmark_attributions.py"
MANIFEST_PATH = REPO_ROOT / "website/content/generated/benchmark-attributions.json"
CARDS_PATH = REPO_ROOT / "website/content/generated/benchmark-cards.json"
HIDDEN_PATH = REPO_ROOT / "website/content/curated/hidden-benchmarks.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "measurement_db_generate_benchmark_attributions_tests", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError(f"cannot load attribution generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


class BibtexParserTests(unittest.TestCase):
    def test_nested_braces_quoted_fields_and_corporate_author(self) -> None:
        citation = """@article{fixture2024,
          title={{A {Nested} Title}: An Evaluation},
          author={Doe, Jane and {Research and Development Lab}},
          year="2024",
          url={https://example.test/paper}
        }"""
        parsed = generator.parse_bibtex_entry(citation)
        self.assertEqual(parsed["entryType"], "article")
        self.assertEqual(parsed["key"], "fixture2024")
        self.assertEqual(parsed["fields"]["year"], "2024")
        self.assertEqual(
            generator.bibtex_display_text(parsed["fields"]["title"]),
            "A Nested Title: An Evaluation",
        )
        self.assertEqual(
            generator.parse_authors(parsed["fields"]["author"]),
            [
                {"name": "Jane Doe", "bibtexName": "Doe, Jane"},
                {
                    "name": "Research and Development Lab",
                    "bibtexName": "{Research and Development Lab}",
                },
            ],
        )

    def test_latex_accents_and_symbols_are_decoded_for_display(self) -> None:
        self.assertEqual(generator.bibtex_display_text(r"$\tau^2$-Bench"), "τ²-Bench")
        self.assertEqual(generator.bibtex_display_text(r"R{\"o}ttger"), "Röttger")
        self.assertEqual(generator.bibtex_display_text(r"R{\'e}"), "Ré")
        self.assertEqual(generator.bibtex_display_text(r"Kabakc{\i}"), "Kabakcı")

    def test_incomplete_author_sentinel_is_rejected(self) -> None:
        with self.assertRaisesRegex(generator.AttributionError, "abbreviates"):
            generator.parse_authors("Doe, Jane and others")

    def test_multiple_or_unbalanced_entries_are_rejected(self) -> None:
        with self.assertRaisesRegex(generator.AttributionError, "exactly one"):
            generator.parse_bibtex_entry(
                "@misc{one,title={One}} @misc{two,title={Two}}"
            )
        with self.assertRaisesRegex(generator.AttributionError, "unterminated"):
            generator.parse_bibtex_entry("@misc{one,title={One}")

    def test_quotes_protected_by_braces_and_parenthesized_entries(self) -> None:
        quoted = generator.parse_bibtex_entry(
            '@misc{fixture, title="A {"quoted"} title", '
            'author="Doe, Jane", year="2024", url="https://example.test"}'
        )
        self.assertEqual(quoted["fields"]["title"], 'A {"quoted"} title')

        parenthesized = generator.parse_bibtex_entry(
            "@misc(fixture, title={A title (with context)}, "
            "author={Doe, Jane}, year={2024}, url={https://example.test})"
        )
        self.assertEqual(parenthesized["fields"]["title"], "A title (with context)")

    def test_empty_optional_field_is_preserved(self) -> None:
        parsed = generator.parse_bibtex_entry(
            "@misc{fixture, title={A title}, author={Doe, Jane}, year={2024}, pages={}}"
        )
        self.assertEqual(parsed["fields"]["pages"], "")


class StaticMetadataTests(unittest.TestCase):
    def test_reviewed_website_reference_preserves_credit_without_a_builder(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "website/content/curated/benchmark-references.json"
            path.parent.mkdir(parents=True)
            reference = {"citation": "@misc{x, title={X}}", "paper_url": "https://example.test"}
            path.write_text(json.dumps({"fixture": reference}), encoding="utf-8")
            info, source = generator._load_benchmark_info(root / "benchmarks", "fixture")
            self.assertEqual(info, reference)
            self.assertEqual(source, path)

    def test_new_entry_uses_local_metadata_when_absent_from_website_references(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            references = root / "website/content/curated/benchmark-references.json"
            references.parent.mkdir(parents=True)
            references.write_text("{}", encoding="utf-8")
            metadata = root / "benchmarks/fixture/metadata.yaml"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("benchmark:\n  citation: '@misc{x, title={X}}'\n", encoding="utf-8")
            info, source = generator._load_benchmark_info(root / "benchmarks", "fixture")
            self.assertEqual(info["citation"], "@misc{x, title={X}}")
            self.assertEqual(source, metadata)

    def test_malformed_website_references_fail_instead_of_changing_credit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "website/content/curated/benchmark-references.json"
            path.parent.mkdir(parents=True)
            for payload in ([], {"fixture": "invalid"}):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(generator.AttributionError):
                        generator._load_benchmark_info(root / "benchmarks", "fixture")

    def test_build_info_is_read_without_executing_builder(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "build.py"
            marker = Path(temporary_directory) / "executed"
            path.write_text(
                "INFO = {'citation': '@misc{x, title={X}}'}\n"
                f"open({str(marker)!r}, 'w').write('bad')\n",
                encoding="utf-8",
            )
            info = generator._literal_info_from_build(path)
            self.assertIn("citation", info)
            self.assertFalse(marker.exists())

    def test_class_attribute_info_is_supported_without_importing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "build.py"
            path.write_text(
                "class Fixture:\n"
                "    INFO = {'citation': '@misc{x, title={X}}'}\n"
                "    raise RuntimeError('must not execute')\n",
                encoding="utf-8",
            )
            info = generator._literal_info_from_build(path)
            self.assertEqual(info["citation"], "@misc{x, title={X}}")

    def test_distinct_literal_info_mappings_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "build.py"
            path.write_text(
                "INFO = {'citation': 'first'}\n"
                "class Fixture:\n"
                "    INFO = {'citation': 'second'}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                generator.AttributionError, "multiple distinct"
            ):
                generator._literal_info_from_build(path)


class RepositoryManifestTests(unittest.TestCase):
    def test_new_public_benchmark_can_use_its_hf_source_credit(self):
        with TemporaryDirectory() as tmp:
            cards = Path(tmp) / "cards.json"
            rows = json.loads(CARDS_PATH.read_text())
            cards.write_text(json.dumps([*rows, {"slug": "new_public_benchmark", "name": "New benchmark"}]))
            manifest = generator.build_manifest(cards_path=cards)
        self.assertIsNone(manifest["benchmarks"]["new_public_benchmark"])
        self.assertEqual(manifest["coverage"]["unreviewed"], ["new_public_benchmark"])
        self.assertNotIn("new_public_benchmark", manifest["coverage"]["withoutProducerCitation"])
        self.assertEqual(manifest["coverage"]["withCitation"], self.generated["coverage"]["withCitation"])

    def test_incomplete_existing_curated_credit_still_fails(self):
        with TemporaryDirectory() as tmp:
            overrides = json.loads(generator.DEFAULT_OVERRIDES_PATH.read_text())
            overrides["authorVerifications"].pop("researchcodebench")
            path = Path(tmp) / "overrides.json"
            path.write_text(json.dumps(overrides))
            with self.assertRaisesRegex(generator.AttributionError, "primary-source count verification"):
                generator.build_manifest(overrides_path=path)

    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = generator.build_manifest()
        cls.checked_in = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        references = json.loads((generator.REPO_ROOT / "website/content/curated/benchmark-references.json").read_text())
        reviewed_slugs = (set(references) - {"osworld_v1"}) | {"osworld"}
        with TemporaryDirectory() as tmp:
            cards = Path(tmp) / "cards.json"
            cards.write_text(json.dumps([{"slug": slug, "name": slug} for slug in sorted(reviewed_slugs)]))
            cls.reviewed = generator.build_manifest(cards_path=cards)["benchmarks"]

    def test_manifest_is_fresh_and_deterministic(self) -> None:
        self.assertEqual(cls_render(self.generated), cls_render(self.checked_in))
        self.assertEqual(
            generator.render_manifest(self.generated),
            MANIFEST_PATH.read_text(encoding="utf-8"),
        )

    def test_manifest_covers_exactly_the_visible_detail_pages(self) -> None:
        cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        hidden = set(json.loads(HIDDEN_PATH.read_text(encoding="utf-8")))
        visible = {card["slug"] for card in cards if card["slug"] not in hidden}
        self.assertEqual(set(self.generated["benchmarks"]), visible)
        self.assertEqual(self.generated["coverage"]["benchmarks"], len(visible))

    def test_complete_author_lists_and_explicit_tengu_exception(self) -> None:
        records = self.generated["benchmarks"]
        self.assertEqual(
            self.generated["coverage"]["withoutProducerCitation"],
            ["tengu"] if "tengu" in records else []
        )
        for slug, record in records.items():
            citation = record["citation"]
            authors = record["reference"]["authors"]
            if slug == "tengu":
                self.assertEqual(citation["status"], "not_provided")
                self.assertFalse(authors)
                self.assertTrue(record["reference"]["producers"])
                continue
            self.assertEqual(citation["status"], "available")
            self.assertTrue(authors, slug)
            self.assertTrue(
                all(author["kind"] in {"person", "organization"} for author in authors),
                slug,
            )
            self.assertFalse(
                any(author["bibtexName"].casefold() == "others" for author in authors),
                slug,
            )

    def test_reviewed_author_counts_guard_high_risk_records(self) -> None:
        expected = {
            "alignment_faking": 20,
            "arc_agi_3": 1,
            "arcagi": 1,
            "bfcl": 7,
            "frontieror": 27,
            "helm_airbench": 12,
            "helm_anthropic_redteam": 36,
            "helm_bbq": 8,
            "helm_bold": 7,
            "helm_harmbench": 12,
            "helm_xstest": 6,
            "ikp": 1,
            "legalbench": 40,
            "osworld_v2": 36,
            "preference_dissection": 6,
            "supergpqa": 96,
            "tau_voice": 4,
            "terminal_bench": 84,
            "terminal_bench_2_1": 85,
            "truthfulqa_mc": 3,
        }
        for slug, count in expected.items():
            record = self.reviewed[slug]
            self.assertEqual(len(record["reference"]["authors"]), count, slug)
            self.assertEqual(
                record["provenance"]["authorVerification"]["expectedCount"],
                count,
                slug,
            )

    def test_every_paper_author_list_has_primary_source_count_lock(self) -> None:
        papers = [
            (slug, record)
            for slug, record in self.generated["benchmarks"].items()
            if record["reference"]["kind"] == "paper"
        ]
        self.assertTrue(papers)
        for slug, record in papers:
            verification = record["provenance"]["authorVerification"]
            self.assertIsNotNone(verification, slug)
            self.assertEqual(
                verification["expectedCount"],
                len(record["reference"]["authors"]),
                slug,
            )
            self.assertRegex(verification["evidenceUrl"], r"^https?://")

    def test_short_author_lists_and_corporate_author_are_explicit(self) -> None:
        for slug, record in self.generated["benchmarks"].items():
            if (
                record["reference"]["kind"] == "paper"
                and len(record["reference"]["authors"]) <= 3
            ):
                self.assertIsNotNone(record["provenance"]["authorVerification"], slug)
        arc_author = self.reviewed["arc_agi_3"]["reference"]["authors"]
        self.assertEqual(
            arc_author,
            [
                {
                    "name": "ARC Prize Foundation",
                    "bibtexName": "{ARC Prize Foundation}",
                    "kind": "organization",
                }
            ],
        )

    def test_osworld_alias_is_explicit_and_attribution_only(self) -> None:
        osworld = self.reviewed["osworld"]
        self.assertEqual(osworld["metadataSlug"], "osworld_v1")
        self.assertEqual(osworld["reference"]["title"].split(":", 1)[0], "OSWorld")
        self.assertIn("attribution-only", osworld["provenance"]["alias"]["reason"])

    def test_affiliation_scope_is_never_overstated(self) -> None:
        self.assertEqual(
            self.generated["affiliationScope"], generator.AFFILIATION_SCOPE
        )
        for slug, record in self.generated["benchmarks"].items():
            if slug in {"arc_agi_3", "tengu"}:
                self.assertEqual(record["affiliationStatus"], "not_applicable")
                self.assertIsNone(record["affiliationScope"])
                self.assertFalse(record["leadAuthorInstitutions"])
                self.assertIsNotNone(record["provenance"]["affiliationOverride"])
                continue
            self.assertEqual(record["affiliationStatus"], "available")
            self.assertEqual(record["affiliationScope"], generator.AFFILIATION_SCOPE)
            self.assertTrue(record["leadAuthorInstitutions"])
            self.assertRegex(
                record["provenance"]["affiliationEvidenceUrl"], r"^https?://"
            )
            self.assertTrue(
                all(
                    set(institution) == {"name"}
                    for institution in record["leadAuthorInstitutions"]
                ),
                slug,
            )


def cls_render(value: object) -> str:
    """Stable compact comparison helper independent of dictionary insertion order."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
