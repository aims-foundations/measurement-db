"""REAL's output tables preserve source associations and grading semantics."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "benchmarks/real_webagents/build.py"
spec = importlib.util.spec_from_file_location("real_webagents_build", BUILD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_tables_preserve_nested_source_associations(builder):
    first_sites = [{"websiteId": "first", "tasks": [
        {"id": "007", "prompt": "First question", "accuracy": 100, "evalsFailed": []}]}]
    second_sites = [{"name": "second", "tasks": [
        {"id": "008", "accuracy": 0, "retrievedAnswer": "Different answer"}]}]
    source = [
        {"id": "001", "name": "same model", "run_id": "same run",
         "websites": first_sites, "verified": True},
        {"id": "skip", "name": "", "websites": second_sites},
        {"id": "002", "name": "same model", "run_id": "same run", "websites": second_sites},
        {"name": "without optional fields"},
    ]
    write_inputs(builder, source)
    source_path = builder.raw_dir / "model_data.json"
    original_bytes = source_path.read_bytes()
    tables = builder.build_tables()

    assert tables["subjects"].to_dict("records") == [
        {"subject_key": 0, "raw_label": "same model"},
        {"subject_key": 2, "raw_label": "same model"},
        {"subject_key": 3, "raw_label": "without optional fields"},
    ]
    assert tables["items"][["item_key", "raw_item_id", "content", "features"]].to_dict("records") == [
        {"item_key": 0, "raw_item_id": "007", "content": "First question", "features": {"website": "first"}},
        {"item_key": 1, "raw_item_id": "008", "content": "008", "features": {"website": "second"}},
    ]
    assert tables["responses"].to_dict("records") == [
        {"response_key": 0, "subject_key": 0, "item_key": 0, "response": 1.0},
        {"response_key": 1, "subject_key": 2, "item_key": 1, "response": 0.0},
    ]
    assert tables["traces"].to_dict("records") == [{"response_key": 1, "trace": "Different answer"}]
    assert source_path.read_bytes() == original_bytes


def test_empty_model_input_returns_empty_tables_with_required_columns(builder):
    write_inputs(builder, [])
    tables = builder.build_tables()
    expected = {
        "subjects": ["subject_key", "raw_label"],
        "items": ["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"],
        "responses": ["response_key", "subject_key", "item_key", "response"],
        "traces": ["response_key", "trace"],
    }
    assert set(tables) == set(expected)
    for name, columns in expected.items():
        assert tables[name].empty
        assert tables[name].columns.tolist() == columns


@pytest.fixture
def builder(tmp_path):
    build = module.REALWebAgents(str(BUILD))
    build.raw_dir = tmp_path
    return build


def write_inputs(builder, models, definitions=None):
    (builder.raw_dir / "model_data.json").write_text(
        json.dumps({"aiModels": models}, ensure_ascii=False))
    tasks = builder.raw_dir / "tasks"
    tasks.mkdir()
    for name, definition in (definitions or {}).items():
        (tasks / f"{name}.json").write_text(json.dumps(definition))


def model(tasks, name="fixture", website="site"):
    return {"name": name, "run_id": "repeated-upstream-id",
            "websites": [{"websiteId": website, "tasks": tasks}]}


def record_build(builder, monkeypatch):
    subjects, items, responses = [], [], []
    monkeypatch.setattr(builder, "add_subject", lambda raw_label:
                        subjects.append(raw_label) or raw_label)
    monkeypatch.setattr(builder, "add_item", lambda **kw:
                        items.append(kw) or str(len(items)))
    monkeypatch.setattr(builder, "add_response", lambda **kw: responses.append(kw))
    builder.build_subject_item_response_rows()
    return subjects, items, responses


def test_repeated_runs_and_full_answers_keep_their_context(builder, monkeypatch):
    answer = "line one\nline two\u2028line three\u2029" + "x" * 20000
    first = model([{"id": "001", "prompt": "Question", "accuracy": 0,
                    "evalsFailed": [], "retrievedAnswer": answer}])
    second = model([{"id": "001", "prompt": "Question", "accuracy": 99,
                     "retrievedAnswer": "failed answer"}])
    second["websites"][0].pop("websiteId")
    second["websites"][0]["name"] = "site"  # Preserve the website-name fallback.
    write_inputs(builder, [first, {"name": None}, second, {"name": "empty model"}])
    tables = builder.build_tables()
    assert tables["subjects"].subject_key.tolist() == [0, 2, 3]
    assert tables["responses"].subject_key.tolist() == [0, 2]
    subjects, items, responses = record_build(builder, monkeypatch)
    assert subjects == ["fixture", "fixture", "empty model"]
    assert len(items) == 1
    assert [r["response"] for r in responses] == [1.0, 0.0]
    assert [r["trial"] for r in responses] == [1, 2]
    assert [r["trace"] for r in responses] == [answer, "failed answer"]


def test_verdict_precedence_ungraded_entries_and_trace_placeholders(builder, monkeypatch):
    records = [
        {"id": "a", "accuracy": 0, "evalsFailed": [], "retrievedAnswer": " Done "},
        {"id": "b", "accuracy": 100, "evalsFailed": ["failed"], "retrievedAnswer": "No response"},
        {"id": "c", "accuracy": 100, "retrievedAnswer": " "},
        {"id": "d", "accuracy": 99},
        {"id": "ungraded", "accuracy": None, "evalsFailed": []},
        {"accuracy": 100},
    ]
    write_inputs(builder, [model(records)])
    _, _, responses = record_build(builder, monkeypatch)
    assert [r["response"] for r in responses] == [1.0, 0.0, 1.0, 0.0]
    assert all(r["trace"] is None for r in responses)


def test_checks_do_not_require_numeric_accuracy_and_nontext_answers_are_excluded(builder, monkeypatch):
    answer = "  Actual answer\n"
    write_inputs(builder, [model([
        {"id": "a", "accuracy": "not numeric", "evalsFailed": [], "retrievedAnswer": {"text": "not a trace"}},
        {"id": "b", "accuracy": "not numeric", "evalsFailed": ["failed"], "retrievedAnswer": 123},
        {"id": "c", "accuracy": "100", "retrievedAnswer": answer},
        {"id": "d", "accuracy": "99", "retrievedAnswer": ["not a trace"]},
    ])])
    _, _, responses = record_build(builder, monkeypatch)
    assert [row["response"] for row in responses] == [1.0, 0.0, 1.0, 0.0]
    assert [row["trace"] for row in responses] == [None, None, answer, None]


def test_rules_website_identity_and_missing_v2_definitions(builder, monkeypatch):
    rule = {"type": "jmespath", "query": "state.value", "expected_value": {"a.b": 1}}
    write_inputs(builder, [
        model([{"id": "known", "accuracy": 100}, {"id": "llm", "accuracy": 100}], website="one"),
        model([{"id": "known", "accuracy": 100}, {"id": "v2.known", "accuracy": 100}], website="two"),
    ], {"known": {"evals": [rule]}, "llm": {"evals": [{"type": "llm_boolean", "rubric": "Check answer"}]}})
    _, items, responses = record_build(builder, monkeypatch)
    assert len(items) == len(responses) == 4
    assert items[0]["features"] == {"website": "one"}
    assert items[2]["features"] == {"website": "two"}
    assert json.loads(items[0]["grading_criterion"]["rule"]) == [rule]
    assert isinstance(items[0]["verifier"], module.ExactMatcher)
    assert isinstance(items[1]["verifier"], module.Judge)
    assert "unavailable" in items[3]["grading_criterion"]["rule"]
    assert isinstance(items[3]["verifier"], module.ExactMatcher)
    # Simulate duplicate task discovery; the join must reject ambiguous rules.
    task_directory = builder.raw_dir / "tasks"
    task_paths = list(task_directory.glob("*.json"))
    original_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda path, pattern:
                        iter(task_paths * 2) if path == task_directory and pattern == "*.json"
                        else original_glob(path, pattern))
    with pytest.raises(pd.errors.MergeError):
        builder.build_subject_item_response_rows()


def test_verifier_descriptions_and_fallback_come_from_metadata(tmp_path, monkeypatch):
    metadata = yaml.safe_load((BUILD.parent / "metadata.yaml").read_text())
    descriptions = {
        "task_checks": {"implementation": "fixture checker", "notes": "Line one\nα"},
        "provider_result": {"implementation": "fixture provider mapping"},
    }
    metadata["grading"] = {"verifiers": descriptions, "fallback_rule": "Fixture fallback criterion"}
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    builder = module.REALWebAgents(str(tmp_path / "build.py"))
    write_inputs(builder, [model([
        {"id": "known", "accuracy": 100},
        {"id": "llm", "accuracy": 100},
        {"id": "v2.unknown", "accuracy": 100},
    ])], {
        "known": {"evals": [{"type": "jmespath", "query": "state.value", "expected_value": 1}]},
        "llm": {"evals": [{"type": "llm_boolean", "rubric": "Check answer"}]},
    })
    _, items, _ = record_build(builder, monkeypatch)
    assert [item["verifier"].spec for item in items] == [
        json.dumps(descriptions["task_checks"], sort_keys=True),
        json.dumps(descriptions["task_checks"], sort_keys=True),
        json.dumps(descriptions["provider_result"], sort_keys=True),
    ]
    assert isinstance(items[0]["verifier"], module.ExactMatcher)
    assert isinstance(items[1]["verifier"], module.Judge)
    assert items[2]["grading_criterion"] == {"rule": "Fixture fallback criterion"}


def test_numeric_ids_do_not_gain_decimal_suffixes(builder, monkeypatch):
    write_inputs(builder, [model([{"id": 7, "accuracy": 100},
                                  {"id": None, "accuracy": 100},
                                  {"id": "007", "accuracy": 100}])])
    _, items, _ = record_build(builder, monkeypatch)
    assert [item["raw_item_id"] for item in items] == ["7", "007"]


def test_empty_optional_arrays_preserve_named_models(builder, monkeypatch):
    write_inputs(builder, [{"name": "without websites"}, {"name": "empty", "websites": []},
                          {"name": "without tasks", "websites": [{"name": "site"}]},
                          {"name": None}])
    subjects, items, responses = record_build(builder, monkeypatch)
    assert subjects == ["without websites", "empty", "without tasks"]
    assert not items and not responses
