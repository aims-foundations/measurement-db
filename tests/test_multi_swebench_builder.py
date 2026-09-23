"""Check source interpretation and joins without downloading benchmark data."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "benchmarks/multi_swebench/build.py"
spec = importlib.util.spec_from_file_location("multi_swebench_build", BUILD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n")


def task(iid="00123", **overrides):
    return {"instance_id": iid, "title": "Fix the issue", "body": "Task body",
            "fix_patch": "reference patch", "test_patch": "test patch",
            "f2p_tests": {"test.with.dots": "passed"},
            "p2p_tests": {"regression": "passed"}, **overrides}


@pytest.fixture
def builder(tmp_path):
    build = module.MultiSWEBench(str(BUILD))
    build.raw_dir = tmp_path
    return build


def select_sources(builder):
    builder.source_files = tuple(str(path.relative_to(builder.raw_dir))
                                for path in builder.raw_dir.rglob("*") if path.is_file())


def test_run_scoped_joins_preserve_grades_full_patches_and_trials(builder, monkeypatch):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task()])
    patch = "line one\nline two\u2028line three\u2029" + "x" * 20000
    runs = ["go__20250101_Agent_fixture-model", "go__20250102_Agent_fixture-model"]
    (root / "results").mkdir()
    for run, grade in zip(runs, [True, False]):
        (root / "results" / f"{run}.json").write_text(json.dumps({
            "resolved": ["00123"] if grade else [],
            "unresolved_ids": [] if grade else ["00123"],
        }))
    write_jsonl(root / "preds" / f"{runs[0]}.jsonl", [
        {"instance_id": "00123", "model_patch": patch}])
    # Missing prediction data must not remove the recorded failed attempt.
    select_sources(builder)
    tables = builder.build_tables()
    assert tables["items"].raw_item_id.tolist() == ["00123"]
    assert json.loads(tables["items"].iloc[0].verifier.spec)["fail_to_pass"] == ["test.with.dots"]
    assert tables["responses"].response.tolist() == [1.0, 0.0]
    assert tables["responses"].subject_key.tolist() == runs
    assert tables["subjects"].access_date.tolist() == ["2025-01-01"] * 2
    assert tables["traces"].to_dict("records") == [{"response_key": 0, "trace": patch}]

    subjects, items, responses = [], [], []
    monkeypatch.setattr(builder, "add_subject", lambda raw_label, **kw:
                        subjects.append((raw_label, kw)) or "subject")
    monkeypatch.setattr(builder, "add_item", lambda **kw: items.append(kw) or "item")
    monkeypatch.setattr(builder, "add_response", lambda **kw: responses.append(kw))
    builder.build_subject_item_response_rows()
    assert len(items) == 1
    assert [r["trial"] for r in responses] == [1, 2]
    assert [r["response"] for r in responses] == [1.0, 0.0]
    assert [r["trace"] for r in responses] == [patch, None]


@pytest.mark.parametrize("second", [task(), task(title="Different", fix_patch="different patch")])
def test_duplicate_task_definitions_are_rejected(builder, second):
    write_jsonl(builder.raw_dir / "a.jsonl", [task()])
    write_jsonl(builder.raw_dir / "b.jsonl", [second])
    select_sources(builder)
    with pytest.raises(ValueError, match="Duplicate task definitions.*00123"):
        builder.build_tables()


@pytest.mark.parametrize("second_patch", ["first patch", "different patch"])
def test_duplicate_nonempty_patches_are_rejected(builder, second_patch):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task()])
    run = "go__model"
    (root / "results").mkdir()
    (root / "results" / f"{run}.json").write_text('{"resolved": ["00123"]}')
    write_jsonl(root / "preds" / f"{run}.jsonl", [
        {"instance_id": "00123", "model_patch": "first patch"},
        {"instance_id": "00123", "model_patch": second_patch},
    ])
    select_sources(builder)
    with pytest.raises(ValueError, match="Duplicate nonempty prediction patches.*00123"):
        builder.build_tables()


def test_empty_patches_do_not_override_the_released_patch(builder):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task()])
    run = "go__undated-model"
    (root / "results").mkdir()
    (root / "results" / f"{run}.json").write_text('{"resolved_ids": ["00123"]}')
    write_jsonl(root / "preds" / f"{run}.jsonl", [
        {"instance_id": "00123", "model_patch": "released patch"},
        {"instance_id": "00123", "model_patch": " "},
    ])
    select_sources(builder)
    tables = builder.build_tables()
    assert tables["traces"].trace.tolist() == ["released patch"]
    assert tables["subjects"].iloc[0].features is None
    assert tables["subjects"].iloc[0].access_date is None
    builder.source_files += (f"results/{run}.json",)
    with pytest.raises(ValueError, match="Duplicate result runs.*undated-model"):
        builder.build_tables()


def test_pr_number_fallback_and_supplementary_prompt(builder):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [
        task(None, org="org", repo="repo", number=7),
        task("python-task", title=None, body=None),
    ])
    pd.DataFrame({"instance_id": ["python-task", "org__repo-7"],
                  "problem_statement": [" Supplementary prompt ", "Do not overwrite"]}
                 ).to_parquet(root / "swebench_verified.parquet")
    (root / "results").mkdir()
    (root / "results/python__model.json").write_text('{"resolved": ["python-task", "org__repo-7"]}')
    select_sources(builder)
    items = builder.build_tables()["items"].set_index("raw_item_id")
    assert items.loc["org__repo-7", "content"] == "Fix the issue\nTask body"
    assert items.loc["python-task", "content"] == "Supplementary prompt"


def test_lfs_pointer_has_no_patch_but_invalid_jsonl_raises(builder):
    root = builder.raw_dir
    path = root / "pointer.jsonl"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
    assert module._read_jsonl(path).empty
    path = root / "broken.jsonl"
    path.write_text('{"instance_id":"valid"}\nnot json\n')
    with pytest.raises(ValueError, match="broken.jsonl"):
        module._read_jsonl(path)


def test_missing_task_definition_is_reported(builder):
    root = builder.raw_dir
    (root / "results").mkdir()
    (root / "results/go__model.json").write_text('{"resolved": ["missing-task"]}')
    select_sources(builder)
    with pytest.raises(ValueError, match="missing-task"):
        builder.build_tables()


def test_native_ids_and_escaped_run_names(builder):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task("org__repo-7"), task("other-task")])
    run = "go__20250101_Agent_Model+Name"
    escaped_run = "go__20250101_Agent_Model_x2b_Name"
    (root / "results").mkdir()
    (root / "results" / f"{escaped_run}.json").write_text(json.dumps({
        "resolved": [], "resolved_ids": ["org/repo:pr-7"],
        "unresolved_ids": [], "unresolved": ["other-task"],
    }))
    write_jsonl(root / "preds" / f"{escaped_run}.jsonl", [
        {"instance_id": "org__repo-7", "model_patch": "recorded patch"}])
    select_sources(builder)
    tables = builder.build_tables()
    assert tables["items"].raw_item_id.tolist() == ["org__repo-7", "other-task"]
    assert tables["responses"].response.tolist() == [1.0, 0.0]
    assert tables["subjects"].raw_label.tolist() == ["Model+Name"]
    assert tables["subjects"].subject_key.tolist() == [run]
    assert tables["subjects"].features.tolist() == [{"harness": "Agent"}]
    assert tables["traces"].trace.tolist() == ["recorded patch"]


def test_run_names_allow_dates_without_agents_and_undated_agents(builder):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task()])
    (root / "results").mkdir()
    runs = ["go__20251027_iSWE-OpenModels", "go__Agent_model_snapshot", "go__20250101_Agent_model_snapshot"]
    for run in runs:
        (root / "results" / f"{run}.json").write_text('{"resolved": ["00123"]}')
    select_sources(builder)
    subjects = builder.build_tables()["subjects"].set_index("subject_key")
    assert subjects.loc[runs[0], "raw_label"] == "iSWE-OpenModels"
    assert subjects.loc[runs[0], "features"] is None
    assert subjects.loc[runs[0], "access_date"] == "2025-10-27"
    for run in runs[1:]:
        assert subjects.loc[run, "raw_label"] == "model_snapshot"
        assert subjects.loc[run, "features"] == {"harness": "Agent"}
        assert subjects.loc[run, "access_date"] == "2025-01-01"


def test_full_item_text_patches_and_test_lists_are_preserved(builder):
    root = builder.raw_dir
    records = [task(f"task-{index:03}") for index in range(65)]
    long_body = "Body " * 3000 + "END OF ISSUE"
    reference_patch = "reference " * 600 + "END OF REFERENCE"
    test_patch = "test " * 1500 + "END OF TEST PATCH"
    records[-1].update(
        resolved_issues=[{"title": " First ", "body": long_body}, None, {"body": " Second "}],
        fix_patch=reference_patch, test_patch=test_patch,
        FAIL_TO_PASS='["literal.test", "second"]',
        PASS_TO_PASS="not JSON", p2p_tests={f"test-{index:03}": "passed" for index in range(60)},
    )
    write_jsonl(root / "tasks.jsonl", records)
    (root / "results").mkdir()
    (root / "results/go__model.json").write_text('{"resolved": ["task-000", "task-064"]}')
    select_sources(builder)
    items = builder.build_tables()["items"].set_index("raw_item_id")
    assert items.loc["task-000", "content"] == "Fix the issue\nTask body"
    last = items.loc["task-064"]
    assert last.content == "First\n" + long_body + "\n\nSecond"
    assert last.grading_criterion["reference_answer"] == reference_patch
    verifier = json.loads(last.verifier.spec)
    assert verifier["test_patch"] == test_patch
    assert verifier["fail_to_pass"] == ["literal.test", "second"]
    assert verifier["pass_to_pass"] == [f"test-{index:03}" for index in range(60)]
    assert verifier["n_pass_to_pass"] == 60


def test_empty_sources_return_empty_tables(builder):
    tables = builder.build_tables()
    assert set(tables) == {"subjects", "items", "responses", "traces"}
    assert all(frame.empty for frame in tables.values())


@pytest.mark.parametrize("resolved,unresolved", [
    (["org/repo:pr-7"], ["org__repo-7"]),
    (["org/repo:pr-7", "org__repo-7"], []),
])
def test_duplicate_or_conflicting_outcomes_are_rejected(builder, resolved, unresolved):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task("org__repo-7")])
    (root / "results").mkdir()
    (root / "results/go__model.json").write_text(json.dumps({
        "resolved": resolved, "unresolved": unresolved,
    }))
    select_sources(builder)
    with pytest.raises(ValueError, match="Duplicate or conflicting outcomes") as error:
        builder.build_tables()
    assert "org__repo-7" in str(error.value)
    assert "results/go__model.json" in str(error.value)


def test_supplementary_prompts_are_complete_and_unique(builder):
    root = builder.raw_dir
    write_jsonl(root / "tasks.jsonl", [task(title=None, body=None)])
    prompt = "Supplementary prompt " * 1000
    bank = pd.DataFrame({"instance_id": ["00123"], "problem_statement": [prompt]})
    bank.to_parquet(root / "swebench_verified.parquet")
    (root / "results").mkdir()
    (root / "results/go__model.json").write_text('{"resolved": ["00123"]}')
    select_sources(builder)
    assert builder.build_tables()["items"].iloc[0].content == prompt.strip()
    pd.concat([bank, bank]).to_parquet(root / "swebench_verified.parquet")
    with pytest.raises(ValueError, match="Duplicate supplementary prompts.*00123"):
        builder.build_tables()
