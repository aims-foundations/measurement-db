import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/render_website/generate_chart_marginals.py"
SPEC = importlib.util.spec_from_file_location("chart_marginals", PATH)
marginals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(marginals)


def test_schema_model_names_combine_configurations_without_counting_items_twice(tmp_path):
    folder = tmp_path / "binary"
    folder.mkdir()
    pd.DataFrame({
        "subject_id": ["first", "second", "human"],
        "normalized_name": ["Example Model", "Example Model", None],
        "provider": ["Example Lab", "Example Lab", None],
    }).to_parquet(folder / "subjects.parquet")
    pd.DataFrame({
        "subject_id": ["first", "second", "second", "human"],
        "item_id": ["one", "one", "two", "three"],
        "response": [1.0, 0.0, 1.0, 1.0],
    }).to_parquet(folder / "responses.parquet")
    assert marginals.build_marginals(["binary"], tmp_path) == {
        "slugs": {"binary": {"binary": True}},
        "models": {"Example Model": "Example Lab"},
        "marginals": [{"slug": "binary", "model": "Example Model", "items": 2,
                       "sum": 2.0, "n": 3}],
    }


def test_graded_values_have_coverage_without_accuracy_and_missing_tables_fail(tmp_path):
    folder = tmp_path / "graded"
    folder.mkdir()
    pd.DataFrame({"subject_id": ["s"], "normalized_name": ["Model"],
                  "provider": [None]}).to_parquet(folder / "subjects.parquet")
    pd.DataFrame({"subject_id": ["s"], "item_id": ["i"],
                  "response": [0.5]}).to_parquet(folder / "response.parquet")
    result = marginals.build_marginals(["graded"], tmp_path)
    assert result["slugs"] == {"graded": {"binary": False}}
    assert result["marginals"] == [{"slug": "graded", "model": "Model", "items": 1}]
    with pytest.raises(FileNotFoundError):
        marginals.build_marginals(["graded", "missing"], tmp_path)
