"""Tests for the AI-subject name analysis heuristic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_measurements.ai_subjects import (
    AI_SUBJECT_KEYWORDS,
    benchmark_has_ai_subjects,
    display_name_is_ai,
    subjects_include_ai,
)


class DisplayNameClassificationTests(unittest.TestCase):
    def test_every_keyword_matches_case_insensitively(self) -> None:
        for keyword in AI_SUBJECT_KEYWORDS:
            with self.subTest(keyword=keyword):
                self.assertTrue(
                    display_name_is_ai(f"provider/{keyword.upper()}-variant")
                )

    def test_non_strings_do_not_match(self) -> None:
        for value in (None, float("nan"), 7, b"gpt"):
            with self.subTest(value=value):
                self.assertFalse(display_name_is_ai(value))

    def test_short_tokens_keep_their_disambiguating_hyphen(self) -> None:
        self.assertFalse(display_name_is_ai("o1 model"))
        self.assertFalse(display_name_is_ai("phi model"))
        self.assertFalse(display_name_is_ai("yi model"))
        self.assertTrue(display_name_is_ai("o1-preview"))
        self.assertTrue(display_name_is_ai("Phi-3 Mini"))
        self.assertTrue(display_name_is_ai("Yi-34B"))


class SubjectTableClassificationTests(unittest.TestCase):
    def test_table_is_true_when_any_display_name_matches(self) -> None:
        subjects = pd.DataFrame(
            {"display_name": ["Human participant", None, "ResNet-18"]}
        )
        self.assertTrue(subjects_include_ai(subjects))

    def test_empty_or_nonmatching_table_is_false(self) -> None:
        self.assertFalse(
            subjects_include_ai(pd.DataFrame({"display_name": []}))
        )
        self.assertFalse(
            subjects_include_ai(
                pd.DataFrame({"display_name": ["Human A", "Human B"]})
            )
        )

    def test_missing_display_name_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "display_name"):
            subjects_include_ai(pd.DataFrame({"subject_id": ["a"]}))

    def test_benchmark_path_reads_subjects_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_dir = Path(tmp) / "example"
            benchmark_dir.mkdir()
            pd.DataFrame(
                {
                    "subject_id": ["human", "model"],
                    "display_name": ["Human participant", "Claude 3.5 Sonnet"],
                }
            ).to_parquet(benchmark_dir / "subjects.parquet", index=False)

            self.assertTrue(benchmark_has_ai_subjects(benchmark_dir))


if __name__ == "__main__":
    unittest.main()
