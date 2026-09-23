"""Check prompt reconstruction, result order, and fresh-run imports without APIs."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "benchmarks/researchcodebench/build.py"
spec = importlib.util.spec_from_file_location("researchcodebench_build", BUILD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LAYOUT = yaml.safe_load((BUILD.parent / "metadata.yaml").read_text())["build"]["parameters"]["archive"]

SOURCE = '''before = 0
# <paper2code name="outer">
    # Explanation
    value = 1
    # <paper2code name="inner">
        result = value + 1
    # </paper2code name="inner">
# </paper2code name="outer">
after = 2
'''


@pytest.fixture
def builder(tmp_path):
    build = module.ResearchCodeBenchBuild(str(BUILD))
    build.raw_dir = tmp_path / "native"
    build.raw_dir.mkdir()
    return build


def result(models=None):
    return {"results": {"Paper": {"results": models or {
        "fixture-model": {"results": {"outer": [{"passed": True}], "inner": [{"passed": False}]}}
    }}}}


def write_archive(builder, stats=None, files=None, extras=None):
    stats = result() if stats is None else stats
    files = {"code.py": SOURCE} if files is None else files
    members = {LAYOUT["results_member"]: json.dumps(stats),
               **{f"pset/Paper/{name}": text for name, text in files.items()}, **(extras or {})}
    with tarfile.open(builder.raw_dir / LAYOUT["file"], "w:gz") as archive:
        for name, value in members.items():
            payload = value.encode() if isinstance(value, str) else value
            member = tarfile.TarInfo(LAYOUT["root"] + "/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stats


def test_nested_markers_preserve_reference_indentation_and_non_target_code(builder):
    write_archive(builder)
    tables = builder.build_tables()
    items = tables["items"].set_index("raw_item_id")
    outer, inner = items.loc["Paper::outer"], items.loc["Paper::inner"]
    assert outer.grading_criterion["reference_answer"] == (
        "    # Explanation\n    value = 1\n        result = value + 1"
    )
    assert inner.grading_criterion["reference_answer"] == "        result = value + 1"
    assert 'before = 0\n    # TODO: Implement block "outer"\n\n    # Approximately 2 line(s) of code.\n\nafter = 2' in outer.content
    assert '    value = 1\n        # TODO: Implement block "inner"\n\n        # Approximately 1 line(s) of code.\n\nafter = 2' in inner.content
    assert "<paper2code" not in outer.content + inner.content
    assert outer.content.startswith("\nYou are an expert in completing research code.\n")
    assert tables["responses"].response.tolist() == [1.0, 0.0]
    assert tables["traces"].empty


@pytest.mark.parametrize("paths", ["context_file_paths: z.py", "context_file_paths:\n  - z.py\n  - a.py"])
def test_context_declaration_order_paper_text_and_newlines(builder, paths):
    write_archive(builder, files={
        "code.py": SOURCE.replace("\n", "\r\n"), "paper2code_paper.tex": "Paper\r\nbody",
        "paper2code.yaml": paths, "z.py": "z = 1\r\n", "a.py": "a = 2\n",
    })
    content = builder.build_tables()["items"].iloc[0].content
    assert "Paper\nbody" in content
    assert "## z.py\n\nz = 1\n\n\n" in content
    if "a.py" in paths:
        assert content.index("## z.py") < content.index("## a.py") < content.index("before = 0")
    assert "\r" not in content


def test_longest_snippet_then_first_file_wins_without_truncation(builder):
    longer = "    value = '" + "x" * 20000 + "'"
    def source(prefix, body):
        return f'{prefix}\n# <paper2code name="outer">\n{body}\n# </paper2code name="outer">\n'
    write_archive(builder, stats=result({"model": {"results": {"outer": [{"passed": True}]}}}), files={
        "a.py": source("selected_file = True", longer),
        "b.py": source("other_file = True", longer),
        "c.py": source("short_file = True", "x = 1"),
    })
    item = builder.build_tables()["items"].iloc[0]
    assert item.grading_criterion["reference_answer"] == longer
    assert "selected_file = True" in item.content
    assert "other_file = True" not in item.content


def test_fresh_results_keep_alias_trials_traces_and_recorded_model_order(builder, tmp_path):
    write_archive(builder)
    trace = "  generated code\n" + "α" * 20000
    first, second = "GEMINI_2_5_PRO_PREVIEW_03_25", "GEMINI_2_5_PRO_PREVIEW_05_06"
    fresh = result({
        first: {"results": {"inner": [], "outer": [{"passed": True, "completion": trace}]}},
        second: {"results": {"outer": [{"passed": False}, {"passed": True, "completion": "second"}]}},
    })
    (builder.raw_dir / "overall_stats.json").write_text(json.dumps(fresh))
    from scripts.build_measurement_tables import reload, validate_dataset
    reload()
    try:
        output = tmp_path / "tables"
        builder.main_from_args(["--source", str(builder.raw_dir), "--output", str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}
        validate_dataset(tables, expected_benchmark_id="researchcodebench")
        assert len(tables["subjects"]) == 1
        assert tables["items"].raw_item_id.tolist() == ["Paper::inner", "Paper::outer"]
        assert tables["responses"].trial.tolist() == [1, 2, 3]
        assert tables["responses"].response.tolist() == [1.0, 0.0, 1.0]
        traces = tables["traces"].set_index("response_id").trace
        assert tables["responses"].response_id.map(traces).fillna("missing").tolist() == [trace, "missing", "second"]
    finally:
        reload()


def test_missing_snippet_fails_instead_of_creating_a_placeholder_item(builder):
    write_archive(builder, stats=result({"model": {"results": {"absent": [{"passed": True}]}}}))
    with pytest.raises(module.BuildContractError, match="missing full snippet content"):
        builder.build_tables()


@pytest.mark.parametrize("completions,message", [
    (None, "completions must be lists"),
    ({"passed": True}, "completions must be lists"),
    ([None], "malformed completion"),
    ([True], "malformed completion"),
])
def test_malformed_completions_are_rejected(builder, completions, message):
    write_archive(builder, stats=result({"model": {"results": {"outer": completions}}}))
    with pytest.raises(module.BuildContractError, match=message):
        builder.build_tables()


def test_invalid_text_identifies_its_archive_member(builder):
    write_archive(builder, files={"code.py": b"\xff"})
    with pytest.raises(module.BuildContractError, match="not UTF-8: .*code.py"):
        builder.build_tables()
