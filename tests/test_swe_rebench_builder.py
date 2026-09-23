"""Check SWE-rebench's source joins, grading definitions, and repeated attempts."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "benchmarks/swe_rebench/build.py"
spec = importlib.util.spec_from_file_location("swe_rebench_build", BUILD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def task(iid="task-b", **overrides):
    return {"instance_id": iid, "problem_statement": "Fix the issue.",
            "patch": " reference patch\n", "test_patch": "test patch",
            "FAIL_TO_PASS": ["test_z", "test_a"], "PASS_TO_PASS": ["regression"],
            "docker_image": "upstream/task:version", **overrides}


def attempt(iid="task-b", grade=1, patch="generated patch"):
    return {"instance_id": iid, "resolved": grade, "model_patch": patch}


@pytest.fixture
def builder(tmp_path):
    build = module.SWERebench(str(BUILD))
    build.raw_dir = tmp_path / "native"
    build.raw_dir.mkdir()
    return build


def write_source(builder, tasks, attempts):
    pd.DataFrame(tasks).to_parquet(builder.raw_dir / "instances.parquet", index=False)
    pd.DataFrame(attempts).to_parquet(builder.raw_dir / "openhands_trajectories.parquet", index=False)


def test_joins_preserve_first_attempt_order_grades_and_complete_artifacts(builder):
    long_patch = "  line one\nline two\u2028" + "α" * 20000 + "\n "
    regressions = [f"test_{number}" for number in range(80)]
    write_source(builder, [task("task-a", problem_statement="Second task"),
                           task(patch=long_patch, test_patch=long_patch, PASS_TO_PASS=regressions)],
                 [attempt(patch=long_patch), attempt("task-a", 0, " \n"),
                  attempt(grade=None, patch="ungraded"), attempt(grade=0, patch=None)])
    tables = builder.build_tables()
    assert tables["items"].raw_item_id.tolist() == ["task-b", "task-a"]
    assert tables["responses"].item_key.tolist() == ["task-b", "task-a", "task-b"]
    assert tables["responses"].response.tolist() == [1.0, 0.0, 0.0]
    assert tables["traces"].to_dict("records") == [{"response_key": 0, "trace": long_patch}]
    item = tables["items"].iloc[0]
    assert item.grading_criterion == {
        "reference_answer": long_patch,
        "rule": "apply test_patch; FAIL_TO_PASS (2) must pass and PASS_TO_PASS (80) must still pass: "
                f"FAIL_TO_PASS={['test_z', 'test_a']}, PASS_TO_PASS={regressions}",
    }
    assert json.loads(item.verifier.spec) == {
        "kind": "swebench_harness", "patch": long_patch, "test_patch": long_patch,
        "FAIL_TO_PASS": ["test_z", "test_a"], "PASS_TO_PASS": regressions,
        "docker_image": "upstream/task:version",
    }


def test_trial_numbering_follows_canonical_items_and_recorded_subject_settings(builder, tmp_path):
    # These different upstream IDs describe the same instrument.
    write_source(builder, [task("alias"), task()],
                 [attempt(), attempt("alias", 0, None), attempt(grade=0)])
    (builder.raw_dir / "subject_settings.json").write_text(json.dumps({
        "fixture-model": {"harness": "recorded-harness", "harness_version": "v2"}
    }))
    output = tmp_path / "tables"
    from scripts.build_measurement_tables import reload, validate_dataset
    reload()
    try:
        builder.main_from_args(["--source", str(builder.raw_dir), "--output", str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}
        validate_dataset(tables, expected_benchmark_id="swe_rebench")
        assert len(tables["items"]) == len(tables["subjects"]) == 1
        assert tables["items"].raw_item_id.tolist() == ["task-b"]
        assert tables["responses"].trial.tolist() == [1, 2, 3]
        assert tables["responses"].response.tolist() == [1.0, 0.0, 0.0]
        traces = tables["traces"].set_index("response_id").trace
        assert tables["responses"].response_id.map(traces).fillna("missing").tolist() == [
            "generated patch", "missing", "generated patch"
        ]
        subject = tables["subjects"].iloc[0]
        assert subject.harness == "recorded-harness"
        assert subject.harness_version == "v2"
    finally:
        reload()


@pytest.mark.parametrize("duplicate", [task(), task(problem_statement="Conflicting task")])
def test_duplicate_task_definitions_are_rejected(builder, duplicate):
    write_source(builder, [task(), duplicate], [attempt()])
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        builder.build_tables()


def test_missing_attempted_task_is_rejected(builder):
    write_source(builder, [task()], [attempt("absent")])
    with pytest.raises(ValueError, match="Attempted tasks lack definitions.*absent"):
        builder.build_tables()


@pytest.mark.parametrize("missing", [None, []])
def test_missing_test_list_and_blank_artifacts(builder, missing):
    write_source(builder, [task(FAIL_TO_PASS=missing, patch=" \n", test_patch=None, docker_image="")],
                 [attempt()])
    item = builder.build_tables()["items"].iloc[0]
    assert item.grading_criterion["reference_answer"] is None
    assert json.loads(item.verifier.spec) == {
        "kind": "swebench_harness", "patch": None, "test_patch": None,
        "FAIL_TO_PASS": [], "PASS_TO_PASS": ["regression"], "docker_image": None,
    }


def test_missing_all_grading_tests_is_rejected(builder):
    write_source(builder, [task(FAIL_TO_PASS=[], PASS_TO_PASS=[])], [attempt()])
    with pytest.raises(ValueError, match="Attempted tasks lack grading tests.*task-b"):
        builder.build_tables()


def test_multiple_declared_subjects_are_rejected(builder):
    write_source(builder, [task()], [attempt()])
    (builder.raw_dir / "subject_settings.json").write_text('{"one": {}, "two": {}}')
    with pytest.raises(ValueError, match="exactly one declared subject"):
        builder.build_tables()
