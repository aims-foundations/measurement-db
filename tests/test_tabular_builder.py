"""Table inputs retain the row builder's identities, traces, and validation."""

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_base import BenchmarkBuild, BuildContractError, ExactMatcher
from scripts.build_measurement_tables import reload, validate_dataset


ANSWER = "Original answer\n" + "α" * 20001
ATTACHMENT = {"source_path": "context.txt", "path": "context.txt",
              "media_type": "text/plain", "role": "context"}


class TableFixture(BenchmarkBuild):
    def build_tables(self):
        return self.inputs


class RowFixture(BenchmarkBuild):
    def build_subject_item_response_rows(self):
        subject = self.add_subject("fixture-model")
        # Two upstream aliases refer to the same canonical subject and item.
        assert self.add_subject("fixture-model") == subject
        item = self.add_item(raw_item_id="first", content="Return one.",
                             grading_criterion={"reference_answer": "1"},
                             verifier=ExactMatcher(spec="Exact binary match"),
                             attachments=[ATTACHMENT])
        assert self.add_item(raw_item_id="alias", content="Return one.",
                             grading_criterion={"reference_answer": "1"},
                             verifier=ExactMatcher(spec="Exact binary match"),
                             attachments=[ATTACHMENT]) == item
        self.add_response(subject_id=subject, item_id=item, response=0, trial=1, trace=ANSWER)
        self.add_response(subject_id=subject, item_id=item, response=1, trial=2, trace=None)
        self.add_response(subject_id=subject, item_id=item, response=None, trial=3, trace="ungraded")
        self.add_response(subject_id=subject, item_id=item, response=0, trial=1,
                          test_condition="with tools", interactors="tool", trace=None)
        self.add_response(subject_id=subject, item_id=item, response=1, trial=2,
                          test_condition="with tools", interactors="tool", trace=None)


@pytest.fixture
def fixture(tmp_path):
    reload()
    bench = tmp_path / "fixture"
    bench.mkdir()
    (bench / "metadata.yaml").write_bytes((ROOT / "benchmarks/_template/metadata.yaml").read_bytes())
    source = tmp_path / "native"
    source.mkdir()
    (source / "context.txt").write_text("Captured context.\n")
    builder = TableFixture(str(bench / "build.py"))
    builder.inputs = {
        "subjects": pd.DataFrame({"subject_key": ["s1", "s2"],
                                  "raw_label": ["fixture-model"] * 2,
                                  "features": [None, None]}),
        "items": pd.DataFrame({"item_key": ["i1", "i2"], "raw_item_id": ["first", "alias"],
                               "content": ["Return one."] * 2,
                               "grading_criterion": [{"reference_answer": "1"}] * 2,
                               "verifier": [ExactMatcher(spec="Exact binary match")] * 2,
                               "attachments": [[ATTACHMENT]] * 2}),
        "responses": pd.DataFrame({"response_key": [0, 1, 2, 3, 4],
                                   "subject_key": ["s1", "s2", "s1", "s1", "s1"],
                                   "item_key": ["i1", "i2", "i1", "i1", "i1"],
                                   "response": [0, 1, None, 0, 1],
                                   "test_condition": [None] * 3 + ["with tools"] * 2,
                                   "interactors": [None] * 3 + ["tool"] * 2}),
        # Reverse order and omit some responses to check the trace join.
        "traces": pd.DataFrame({"response_key": [2, 0], "trace": ["ungraded", ANSWER]}),
    }
    yield builder, source, tmp_path / "tables"
    reload()


def test_tabular_and_row_builds_match_all_six_tables(fixture):
    builder, source, output = fixture
    original = copy.deepcopy(builder.inputs)
    source_bytes = (source / "context.txt").read_bytes()
    builder.main_from_args(["--source", str(source), "--output", str(output)])
    tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}
    assert set(tables) == {"subjects", "items", "responses", "traces", "assets", "benchmarks"}
    validate_dataset(tables, expected_benchmark_id="fixture")
    assert len(tables["subjects"]) == len(tables["items"]) == len(tables["assets"]) == 1
    assert tables["responses"].trial.tolist() == [1, 2, 3, 1, 2]
    assert tables["responses"].response.isna().tolist() == [False, False, True, False, False]
    assert tables["traces"].trace.tolist() == [ANSWER, "ungraded"]

    reload()
    row_output = output.parent / "row-tables"
    RowFixture(str(builder.dir / "build.py")).main_from_args(
        ["--source", str(source), "--output", str(row_output)])
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, pd.read_parquet(row_output / f"{name}.parquet"))
    for name, frame in original.items():
        pd.testing.assert_frame_equal(frame, builder.inputs[name])
    assert (source / "context.txt").read_bytes() == source_bytes


def test_explicit_trials_and_omitted_traces(fixture):
    builder, source, output = fixture
    del builder.inputs["traces"]
    builder.inputs["responses"]["trial"] = [2, 4, 6, 8, 10]
    builder.main_from_args(["--source", str(source), "--output", str(output)])
    assert pd.read_parquet(output / "responses.parquet").trial.tolist() == [2, 4, 6, 8, 10]
    assert not (output / "traces.parquet").exists()


def test_subject_aliases_preserve_runs_settings_and_observations(fixture):
    builder, source, output = fixture
    aliases = ["export-a", "export-b"]
    builder.inputs["subjects"]["raw_label"] = aliases
    settings = {label: {"harness": "recorded-agent", "reasoning_effort": "high"}
                for label in aliases}
    (source / "subject_settings.json").write_text(json.dumps(settings))
    builder.main_from_args(["--source", str(source), "--output", str(output)])
    before = {p.stem: pd.read_parquet(p) for p in output.glob("*.parquet")}

    builder.build_parameters["subject_labels"] = dict.fromkeys(aliases, "gpt-4o")
    reload()
    after_output = output.parent / "normalized"
    builder.main_from_args(["--source", str(source), "--output", str(after_output)])
    after = {p.stem: pd.read_parquet(p) for p in after_output.glob("*.parquet")}
    subjects = after["subjects"]
    assert len(subjects) == subjects.subject_id.nunique() == 2
    assert subjects.normalized_name.tolist() == ["OpenAI GPT-4o"] * 2
    assert subjects.harness.tolist() == ["recorded-agent"] * 2
    assert subjects.reasoning_effort.tolist() == ["high"] * 2
    assert set(subjects.subject_features_extra) == {
        f"source_display_name={label}" for label in aliases}
    subject_ids = dict(zip(before["subjects"].subject_id, subjects.subject_id))
    response_ids = dict(zip(before["responses"].response_id, after["responses"].response_id))
    assert len(set(response_ids.values())) == len(response_ids)
    for name, frame in before.items():
        if name == "subjects":
            continue
        expected = frame.copy()
        if "subject_id" in expected:
            expected["subject_id"] = expected.subject_id.map(subject_ids)
        if "response_id" in expected:
            expected["response_id"] = expected.response_id.map(response_ids)
        pd.testing.assert_frame_equal(expected, after[name])
    validate_dataset(after, expected_benchmark_id="fixture")


@pytest.mark.parametrize("target,features,message", [
    (" ", None, "non-empty string"),
    ("gpt-4o", {"source_display_name": "different-run"}, "conflicts"),
])
def test_invalid_subject_label_mapping_fails_before_writing(fixture, target, features, message):
    builder, source, output = fixture
    builder.build_parameters["subject_labels"] = {"fixture-model": target}
    builder.inputs["subjects"]["features"] = [features, features]
    with pytest.raises(BuildContractError, match=message):
        builder.main_from_args(["--source", str(source), "--output", str(output)])
    assert not list(output.glob("*.parquet"))


def test_legacy_null_reference_identity_is_preserved_without_an_output_column(fixture):
    builder, source, output = fixture
    builder.inputs["responses"]["reference_answer"] = None
    builder.main_from_args(["--source", str(source), "--output", str(output)])

    class LegacyRowFixture(RowFixture):
        def add_response(self, **kwargs):
            return super().add_response(**kwargs, reference_answer=None)

    reload()
    old_output = output.parent / "legacy-row-tables"
    LegacyRowFixture(str(builder.dir / "build.py")).main_from_args(
        ["--source", str(source), "--output", str(old_output)])
    for path in output.glob("*.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(old_output / path.name))
    assert "reference_answer" not in pd.read_parquet(output / "responses.parquet").columns


@pytest.mark.parametrize("table,column,row,value,message", [
    ("subjects", "subject_key", 1, "s1", "duplicate subject_key"),
    ("items", "item_key", 1, "i1", "duplicate item_key"),
    ("responses", "response_key", 1, 0, "duplicate response_key"),
    ("traces", "response_key", 1, 2, "duplicate response_key"),
    ("subjects", "subject_key", 0, None, "non-null string or integer"),
    ("responses", "subject_key", 0, "absent", "unknown subject_key"),
    ("responses", "item_key", 0, "absent", "unknown item_key"),
    ("traces", "response_key", 0, 99, "unknown response_key"),
    ("responses", "response", 0, 2.0, "outside the declared response_scale"),
    ("responses", "response", 0, float("inf"), "finite"),
    ("responses", "trial", 0, 0, "positive"),
    ("responses", "trial", 0, None, "positive"),
    ("responses", "trial", 1, 1, "duplicate"),
])
def test_invalid_inputs_fail_before_writing(fixture, table, column, row, value, message):
    builder, source, output = fixture
    if column == "trial":
        builder.inputs["responses"]["trial"] = pd.Series([1, 2, 3, 1, 2], dtype=object)
    builder.inputs[table].loc[row, column] = value
    with pytest.raises(BuildContractError, match=message):
        builder.main_from_args(["--source", str(source), "--output", str(output)])
    assert not list(output.glob("*.parquet"))


def test_embedded_asset_bytes_match_file_assets_without_writing_raw(fixture):
    builder, source, output = fixture
    builder.main_from_args(["--source", str(source), "--output", str(output)])
    expected = {p.stem: pd.read_parquet(p) for p in output.glob("*.parquet")}
    payload = (source / "context.txt").read_bytes()
    embedded = {key: value for key, value in ATTACHMENT.items() if key != "source_path"}
    embedded["data"] = payload
    builder.inputs["items"]["attachments"] = [[embedded], [embedded]]
    (source / "context.txt").unlink()
    pd.DataFrame({"data": [payload]}).to_parquet(source / "images.parquet")
    original_source = (source / "images.parquet").read_bytes()
    second = output.parent / "embedded-tables"
    reload()
    builder.main_from_args(["--source", str(source), "--output", str(second)])
    assert sorted(p.name for p in source.iterdir()) == ["images.parquet"]
    assert (source / "images.parquet").read_bytes() == original_source
    for name, frame in expected.items():
        pd.testing.assert_frame_equal(frame, pd.read_parquet(second / f"{name}.parquet"))


@pytest.mark.parametrize("attachment", [
    {"data": "not bytes", "path": "context.txt", "media_type": "text/plain", "role": "context"},
    {**ATTACHMENT, "data": b"ambiguous"},
])
def test_invalid_embedded_assets_are_rejected(fixture, attachment):
    builder, source, output = fixture
    builder.inputs["items"]["attachments"] = [[attachment], [attachment]]
    with pytest.raises(BuildContractError, match="bytes|exactly one"):
        builder.main_from_args(["--source", str(source), "--output", str(output)])
    assert not list(output.glob("*.parquet"))


@pytest.mark.parametrize("change,message", [
    (lambda frames: frames.pop("items"), "must return"),
    (lambda frames: frames.update(items=frames["items"].drop(columns="verifier")), "missing columns"),
    (lambda frames: frames.update(responses=frames["responses"].assign(extra=1)), "unknown columns"),
    (lambda frames: frames.update(responses=[]), "must be a DataFrame"),
])
def test_input_table_contract_is_checked(fixture, change, message):
    builder, source, output = fixture
    change(builder.inputs)
    with pytest.raises(BuildContractError, match=message):
        builder.main_from_args(["--source", str(source), "--output", str(output)])
    assert not list(output.glob("*.parquet"))


def test_shared_immutable_asset_is_hashed_once(fixture, monkeypatch):
    import build_base
    builder, source, output = fixture
    payload = b'Large shared stimulus' * 1024
    attachment = dict(data=payload, path='stimulus.bin', media_type='application/octet-stream', role='context')
    builder.inputs['items']['attachments'] = [[attachment], [attachment]]
    original = build_base._measurement_ids.asset_id_from_bytes
    hashed = []
    def record(data):
        hashed.append(data)
        return original(data)
    monkeypatch.setattr(build_base._measurement_ids, 'asset_id_from_bytes', record)
    builder.main_from_args(['--source', str(source), '--output', str(output)])
    assert len(hashed) == 1
    assert pd.read_parquet(output/'assets.parquet').iloc[0].data == payload
    reload()
    second = output.parent / 'repeated-embedded-build'
    builder.main_from_args(['--source', str(source), '--output', str(second)])
    assert len(hashed) == 2
    pd.testing.assert_frame_equal(pd.read_parquet(output/'assets.parquet'),
        pd.read_parquet(second/'assets.parquet'))
