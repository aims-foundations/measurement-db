"""Native score lists and model variants survive BiGGen's table conversion."""

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
spec = importlib.util.spec_from_file_location("biggen_build", ROOT / "benchmarks/biggen/build.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def source(tmp_path):
    benchmark = tmp_path / "biggen"
    benchmark.mkdir()
    metadata = yaml.safe_load((ROOT / "benchmarks/biggen/metadata.yaml").read_text())
    metadata["build"]["parameters"]["layout"]["data"] = "data"
    (benchmark / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    raw = tmp_path / "source"
    (raw / "data").mkdir(parents=True)
    text = "完整模型输出\n" * 5000
    rows = []
    for index, model in enumerate(["Llama-2-13b-hf", "Llama-2-13b-chat-hf"]):
        rows.append({
            "id": "task-1", "uuid": "generation-" + str(index), "model_name": model,
            "capability": "reasoning", "task": "reasoning", "language": "en", "instance_idx": 0,
            "system_prompt": "Solve exactly.", "input": "Compute.", "reference_answer": "42",
            "score_rubric": {"criteria": "Correctness", **{f"score{n}_description": str(n) for n in range(1, 6)}},
            "response": text, "used_for_training": False,
            "human_score": 3 if index == 0 else -1,
            "gpt4_score": 4 if index == 0 else None, "gpt4_feedback": "Published feedback.",
            "gpt4_04_turbo_score": 4, "gpt4_04_turbo_feedback": "Published feedback.",
            "claude_score": 5, "claude_feedback": "Published feedback.",
            "prometheus_8x7b_score": [1, 2, None, 4, 5], "prometheus_8x7b_feedback": "One feedback for five ratings.",
            "prometheus_8x7b_bgb_score": [2, 3, 4, 5, 5], "prometheus_8x7b_bgb_feedback": "One feedback for five ratings.",
        })
    path = raw / "data/human_eval-00000-of-00001.parquet"
    pd.DataFrame(rows, index=[101, 202]).to_parquet(path)
    return benchmark, raw, path, text


def test_complete_outputs_rating_positions_and_base_chat_identity(source, tmp_path, monkeypatch):
    benchmark, raw, _, text = source
    registrations = sys.modules["scripts.build_measurement_tables.register_measurements"]
    for name in ("_subjects", "_items", "_benchmarks"):
        monkeypatch.setattr(registrations, name, None)
    output = tmp_path / "tables"
    module.BiggenBench(benchmark / "build.py").main_from_args(["--source", str(raw), "--output", str(output)])
    subjects = pd.read_parquet(output / "subjects.parquet")
    responses = pd.read_parquet(output / "responses.parquet")
    traces = pd.read_parquet(output / "traces.parquet")
    assert len(subjects) == 2
    assert set(subjects.display_name) == {"Llama-2-13b-hf", "Llama-2-13b-chat-hf"}
    assert len(responses) == 27 and responses.response.isna().sum() == 3
    decoded = [json.loads(value) for value in traces.trace]
    assert {row["model_output"] for row in decoded} == {text}
    assert {row["source_row"] for row in decoded} == {0, 1}
    assert sum(row["score_field"] == "human_score" for row in decoded) == 1
    selected = [row for row in decoded if row["score_field"] == "prometheus_8x7b_score" and row["generation_id"] == "generation-0"]
    assert sorted(row["rating_index"] for row in selected) == list(range(5))
    assert all(row["published_scores"] == [1, 2, None, 4, 5] for row in selected)


def test_new_off_scale_values_are_rejected(source):
    benchmark, raw, path, _ = source
    frame = pd.read_parquet(path)
    frame.at[101, "gpt4_score"] = 6
    frame.to_parquet(path)
    builder = module.BiggenBench(benchmark / "build.py")
    builder.raw_dir = raw
    with pytest.raises(ValueError, match="outside the item rubric"):
        builder.build_tables()


def test_unsupported_judge_tasks_are_not_invented_ratings(source):
    benchmark, raw, path, _ = source
    frame = pd.read_parquet(path)
    frame["task"] = "llm_judge_absolute"
    for field in ["prometheus_8x7b_score", "prometheus_8x7b_bgb_score"]:
        frame[field] = None
    frame.to_parquet(path)
    builder = module.BiggenBench(benchmark / "build.py")
    builder.raw_dir = raw
    tables = builder.build_tables()
    assert len(tables["responses"]) == 7
    assert all(not json.loads(trace)["score_field"].startswith("prometheus_") for trace in tables["traces"].trace)
