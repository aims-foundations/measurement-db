"""Response filename compatibility in the gallery's batch scripts."""

import json
import runpy
import shutil
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/render_website"


def script_copy(root, name):
    path = root / "scripts/render_website" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS / name, path)
    return path


@pytest.mark.parametrize("filenames", [
    ["responses.parquet"], ["response.parquet"], ["responses.parquet", "response.parquet"],
])
def test_model_statistics_accept_both_names_and_prefer_canonical(tmp_path, filenames):
    script = script_copy(tmp_path, "generate_model_statistics.py")
    folder = script.parent / ".hf-cache/fixture"
    folder.mkdir(parents=True)
    for name in filenames:
        # A stale singular table must not override the canonical table's two rows.
        n = 3 if name == "response.parquet" and len(filenames) > 1 else 2
        pd.DataFrame({"subject_id": ["s"] * n,
                      "item_id": [f"i{i}" for i in range(n)]}).to_parquet(folder / name)
    pd.DataFrame({"subject_id": ["s"], "display_name": ["Model"]}).to_parquet(
        folder / "subjects.parquet")
    registry = tmp_path / "scripts/build_measurement_tables/map_model_registry.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"Model": {"model": "Model", "company": "Lab"}}))
    cards = tmp_path / "website/content/generated/benchmark-cards.json"
    cards.parent.mkdir(parents=True)
    cards.write_text(json.dumps([{"slug": "fixture"}]))

    runpy.run_path(str(script), run_name="__main__")

    result = pd.read_csv(tmp_path / "artifacts/render_website/model_statistics.csv")
    assert result.total_items_asked.tolist() == [2]
    assert result.total_responses.tolist() == [2]
