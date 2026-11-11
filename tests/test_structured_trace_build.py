"""Structured trace hooks preserve measurements across both builder interfaces."""

import copy
from dataclasses import FrozenInstanceError
import gzip
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_base import BenchmarkBuild, BuildContractError, ConvertedTrace, ExactMatcher, StructuredTraceBuild
from scripts.build_measurement_tables import reload
from scripts.build_measurement_tables import trace_format
from scripts.render_website.generate_benchmark_gallery import write_source_pages


CONTENT = "Original output\n" + "α" * 20001


class LegacyRows(BenchmarkBuild):
    def build_subject_item_response_rows(self):
        subject = self.add_subject("fixture-model")
        item = self.add_item(raw_item_id="first", content="Solve it.",
                             grading_criterion={"reference_answer": "1"},
                             verifier=ExactMatcher(spec="Exact binary match"))
        for trial, (grade, source) in enumerate(zip([1., None, 0.], self.sources), 1):
            self.add_response(subject_id=subject, item_id=item, response=grade,
                              trial=trial, trace=source, test_condition="original condition")


class StructuredRows(StructuredTraceBuild, LegacyRows):
    def convert_trace(self, source, *, context):
        self.contexts.append(context)
        content = source["content"] if isinstance(source, dict) else source
        native = trace_format.unused(source, {"content"}) if isinstance(source, dict) else None
        return ConvertedTrace(trace_format.trace("fixture@1", [{"type": "llm", "content": content}],
                                                  native=native))


class StructuredTables(StructuredRows):
    def build_subject_item_response_rows(self):
        return BenchmarkBuild.build_subject_item_response_rows(self)

    def build_tables(self):
        return {
            "subjects": pd.DataFrame({"subject_key": ["s"], "raw_label": ["fixture-model"]}),
            "items": pd.DataFrame({"item_key": ["i"], "raw_item_id": ["first"], "content": ["Solve it."],
                "grading_criterion": [{"reference_answer": "1"}],
                "verifier": [ExactMatcher(spec="Exact binary match")]}),
            "responses": pd.DataFrame({"response_key": [0, 1, 2], "subject_key": ["s"] * 3,
                "item_key": ["i"] * 3, "response": [1., None, 0.], "trial": [1, 2, 3],
                "test_condition": ["original condition"] * 3}),
            # Deliberately reverse order to exercise the observation-key join.
            "traces": pd.DataFrame({"response_key": [i for i in [2, 1, 0] if self.sources[i] is not None],
                "trace": [self.sources[i] for i in [2, 1, 0] if self.sources[i] is not None]}),
        }


@pytest.fixture
def workspace(tmp_path):
    reload()
    bench = tmp_path / "fixture"
    bench.mkdir()
    (bench / "metadata.yaml").write_bytes((ROOT / "benchmarks/_template/metadata.yaml").read_bytes())
    source = tmp_path / "native"
    source.mkdir()
    (source / "original.json").write_text('{"fixture":"immutable upstream record"}\n')
    yield bench, source, tmp_path
    reload()


def run_builder(workspace, kind, sources, name="tables"):
    bench, source, root = workspace
    reload()
    builder = kind(str(bench / "build.py"))
    builder.sources, builder.contexts = sources, []
    output = root / name
    rows = builder.main_from_args(["--source", str(source), "--output", str(output)])
    return builder, output, rows


def test_missing_converter_is_abstract_before_any_files_are_created(tmp_path):
    class MissingConverter(StructuredTraceBuild):
        def build_tables(self):
            raise AssertionError("must not run")
    with pytest.raises(TypeError, match="convert_trace"):
        MissingConverter(str(tmp_path / "nonexistent/build.py"))
    assert not (tmp_path / "nonexistent").exists()


@pytest.mark.parametrize("kind", [StructuredRows, StructuredTables])
def test_converter_runs_once_per_trace_with_fixed_context_and_no_measurement_changes(workspace, kind):
    _, legacy, _ = run_builder(workspace, LegacyRows, [CONTENT, "Ungraded output", None], "legacy")
    sources = [{"content": CONTENT, "original_extra": {"nested": [1, 2]}}, "Ungraded output", None]
    before = copy.deepcopy(sources)
    builder, output, rows = run_builder(workspace, kind, sources)
    assert sources == before
    for path in legacy.glob("*.parquet"):
        if path.name != "traces.parquet":
            assert path.read_bytes() == (output / path.name).read_bytes(), path.name
    legacy_traces = pq.read_table(legacy / "traces.parquet")
    traces = pq.read_table(output / "traces.parquet")
    assert trace_format.FORMAT_KEY not in legacy_traces.schema.metadata
    assert traces.schema.metadata[trace_format.FORMAT_KEY] == b"trace/1"
    assert traces.column("response_id").to_pylist() == legacy_traces.column("response_id").to_pylist()
    values = [json.loads(value) for value in traces.column("trace").to_pylist()]
    assert values[0]["steps"][0]["content"] == CONTENT
    assert values[0]["native"] == {"original_extra": {"nested": [1, 2]}}
    assert len(builder.contexts) == 2
    assert [c.trial for c in builder.contexts] == [1, 2]
    assert all(c.item_content == "Solve it." and c.test_condition == "original condition" for c in builder.contexts)
    assert [c.response_id for c in builder.contexts] == traces.column("response_id").to_pylist()
    with pytest.raises(FrozenInstanceError):
        builder.contexts[0].item_content = "changed"
    assert builder.trace_accounting["rows"] == 2
    assert builder.trace_accounting["steps"] == {"llm": 2}
    assert [json.loads(value) for value in rows.trace.dropna()] == values


def test_page_index_marks_format_and_reads_one_trace_per_page(workspace):
    _, output, _ = run_builder(workspace, StructuredRows, [CONTENT, "second", None])
    pages = output / "pages"
    descriptor = write_source_pages(output / "traces.parquet", ["trace"], pages, "traces")
    assert descriptor["format"] == "trace/1"
    groups = {}
    for path in (pages / "traces-pages").glob("*.json.gz"):
        groups.update(json.loads(gzip.decompress(path.read_bytes())))
    locations = next(iter(groups.values()))["pages"]["trace"]
    assert len(locations) == 2
    assert [p[2] for p in locations] == [0, 1]
    trace_column = pq.ParquetFile(output / "traces.parquet").metadata.row_group(0).column(7)
    assert trace_column.dictionary_page_offset is None


@pytest.mark.parametrize("kind", [StructuredRows, StructuredTables])
def test_absent_traces_stay_absent_and_accounting_resets_on_reuse(workspace, kind):
    builder, _, _ = run_builder(workspace, kind, ["present", None, None], "first")
    builder.sources = [None, None, None]
    reload()
    _, source, root = workspace
    builder.main_from_args(["--source", str(source), "--output", str(root / "second")])
    assert not (root / "second/traces.parquet").exists()
    assert builder.trace_accounting == {"rows": 0, "steps": {}, "counters": {}}


@pytest.mark.parametrize("failure", ["exception", "invalid_schema", "none", "nan", "bad_counter",
                                     "parse_failures", "unrecognized", "unmatched", "counter_mismatch"])
def test_invalid_conversion_preserves_previous_outputs(workspace, failure):
    _, output, _ = run_builder(workspace, StructuredRows, ["original", None, None])
    previous = {p.name: p.read_bytes() for p in output.glob("*.parquet")}

    class Invalid(StructuredRows):
        def convert_trace(self, source, *, context):
            if failure == "exception":
                raise ValueError("bad source record")
            if failure == "none":
                return None
            value = trace_format.trace("fixture@1", [{"type": "llm", "content": "output"}])
            counts = {}
            if failure == "invalid_schema":
                value["score"] = 1
            elif failure == "nan":
                value["native"] = {"invalid": float("nan")}
            elif failure == "bad_counter":
                counts["messages"] = -1
            elif failure == "parse_failures":
                counts["parse_failures"] = 1
            elif failure in {"unrecognized", "counter_mismatch"}:
                value["steps"] = [{"type": "event", "name": "unrecognized", "native": {"source": source}}]
                if failure == "counter_mismatch":
                    counts["unrecognized"] = 0
            elif failure == "unmatched":
                counts["unmatched_tool_results"] = 1
            return ConvertedTrace(value, counts)

    with pytest.raises(BuildContractError, match="trace"):
        run_builder(workspace, Invalid, ["new", None, None], "invalid")
    assert {p.name: p.read_bytes() for p in output.glob("*.parquet")} == previous
    assert not list((workspace[2] / "invalid").glob("*.parquet"))


def test_reviewed_allowances_are_explicit_and_unrecognized_events_are_counted(workspace):
    class Reviewed(StructuredRows):
        TRACE_ALLOWANCES = {"unrecognized": 1, "unmatched_tool_results": 1}
        def convert_trace(self, source, *, context):
            return ConvertedTrace(trace_format.trace("fixture@1", [
                {"type": "event", "name": "unrecognized", "native": {"source": source}}]),
                {"unmatched_tool_results": 1})
    builder, output, _ = run_builder(workspace, Reviewed, ["original event", None, None])
    assert builder.trace_accounting["counters"]["unrecognized"] == 1
    value = json.loads(pd.read_parquet(output / "traces.parquet").trace.iloc[0])
    assert value["steps"][0]["native"]["source"] == "original event"


def test_existing_mini_swe_agent_converter_works_through_the_package_builder_api(workspace, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT.parent))
    api = importlib.import_module("measurement_db.build_base")
    converter = importlib.import_module("measurement_db.scripts.build_measurement_tables.trace_converters.mini_swe_agent")

    class Pilot(api.StructuredTraceBuild):
        build_subject_item_response_rows = LegacyRows.build_subject_item_response_rows

        def convert_trace(self, source, *, context):
            value, counts = converter.convert(source)
            return api.ConvertedTrace(value, counts)

    trajectory = {"messages": [
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": "Checking.", "tool_calls": [
            {"id": "call-1", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "complete tool output\n"},
    ]}
    builder, output, _ = run_builder(workspace, Pilot, [trajectory, None, None])
    value = json.loads(pd.read_parquet(output / "traces.parquet").trace.iloc[0])
    assert value["converter"] == "mini_swe_agent@1"
    assert [step["type"] for step in value["steps"]] == ["message", "llm", "tool"]
    assert value["steps"][-1]["output"] == "complete tool output\n"
    assert builder.trace_accounting["counters"]["unmatched_tool_results"] == 0
