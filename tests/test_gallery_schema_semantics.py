"""Interpretation and observation-retention checks, independent of snapshots."""

import json

import pandas as pd
import pytest

from scripts.render_website import generate_benchmark_gallery as gallery
from test_gallery_output_parity import local_fetch, write_fixture


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "fetch", local_fetch)
    monkeypatch.setattr(gallery, "source_tables", lambda slug: gallery.TableSource(slug, frozenset({"traces.parquet"})))

    class Source:
        def __init__(self, case="binary"):
            self.case = case
            self.overrides = write_fixture(tmp_path, case)
            self.folder = tmp_path / case

        def edit(self, name, change):
            path = self.folder / f"{name}.parquet"
            change(pd.read_parquet(path)).to_parquet(path, index=False)

        def scale(self, response_type, domain):
            self.edit("benchmarks", lambda frame: frame.assign(response_type=response_type, response_scale=json.dumps(domain)))

        def build(self):
            return gallery.build_detail(self.case, tmp_path, tmp_path / "web", False, self.overrides, compact=True)

        def raw(self):
            return gallery.load_raw(self.case, tmp_path, False)

    return Source


def chart_records(chart):
    """Expand compressed keys using the actual cell axis, including null grades."""
    if chart.get("rowKind") == "faceted":
        groups = [(band["rows"], {panel["key"]: panel["colIds"] for panel in band["panels"]})
                  for band in chart["bands"]]
    else:
        groups = [(rows, {block["key"]: block["colIds"] for block in chart["blocks"]})
                  for rows in chart["trials"].values()]
    records = []
    for rows, axes in groups:
        for row in rows:
            for name, block in row.get("blocks", row.get("panels", {})).items():
                for index, bit in enumerate(block["bits"]):
                    if bit == ".":
                        continue
                    iid = row.get("cols", axes[name])[index]
                    for suffix in ["", "2"]:
                        key = block.get(f"keys{suffix}", {}).get(index, block.get(f"key{suffix}"))
                        if key is not None:
                            records.append(tuple([key[0], iid, *key[2:]]))
    return records


def assert_all_attempts(data, view):
    assert sorted(chart_records(view["chart"]), key=repr) == sorted(map(tuple, data.raw()._key), key=repr)


def test_binary_meanings_come_from_metadata(source):
    data = source()
    data.scale("binary", {"kind": "discrete", "values": [0, 1], "direction": "lower_is_better",
                          "meanings": {"0": "safe", "1": "unsafe"}})
    data.overrides["binary"]["binaryLabels"] = {"zero": "incorrect", "one": "correct"}
    data.edit("responses", lambda frame: frame.assign(response=frame.subject_id.map({"s0": 0., "s1": 1.})))
    detail = data.build()["detail"]
    assert detail["binaryLabels"] == {"zero": "safe", "one": "unsafe"}
    assert detail["scaleLabel"] == "1 = unsafe · 0 = safe"
    assert detail["subjects"][0] == {"name": "Model 0", "score": 0.0}


def test_endpoint_observations_do_not_turn_an_interval_into_binary(source):
    data = source()
    data.scale("continuous_bounded", {"kind": "interval", "min": 0, "max": 100})
    view = data.build()
    assert view["detail"]["isBinary"] is False
    assert view["detail"]["valueRange"] == [0, 100]
    assert view["detail"]["binaryLabels"] is None
    assert {level["label"] for level in view["chart"]["levels"]} == {"0", "1"}


@pytest.mark.parametrize("domain", [None, "[0, 1]", {"kind": "discrete", "values": [0, 2]}])
def test_invalid_or_missing_metadata_is_not_guessed_from_grades(source, domain):
    data = source()
    data.edit("benchmarks", lambda frame: frame.assign(response_scale=json.dumps(domain)))
    with pytest.raises(ValueError, match="response_scale"):
        data.build()


def test_nominal_categories_keep_meanings_without_an_average(source):
    data = source()
    data.scale("nominal", {"kind": "discrete", "values": [0, 1], "direction": "unordered",
                           "meanings": {"0": "refusal", "1": "compliance"}})
    view = data.build()
    assert view["detail"]["stats"]["meanResponse"] is None
    assert view["detail"]["stats"]["graded"] == 8
    assert view["detail"]["matrixRowScores"] == [None, None]
    assert [level["label"] for level in view["chart"]["levels"]] == ["refusal (0)", "compliance (1)"]
    assert all(row["score"] == "4 graded" for row in view["chart"]["trials"]["1"])


def test_mixed_scales_keep_each_items_meaning_and_native_value(source):
    data = source()
    data.scale("mixed", {"kind": "mixed"})
    data.edit("items", lambda frame: frame.assign(grading_criterion=[json.dumps({
        "reference_answer": None, "rule": "Released grade",
        "response_scale": {"kind": "discrete", "values": [0, 1], "meanings": {"0": "unsafe", "1": "safe"}}
        if iid == "i0" else {"kind": "interval", "min": 0, "max": 100}}) for iid in frame.item_id]))
    data.edit("responses", lambda frame: frame.assign(response=frame.item_id.map({"i0": 1., "i1": 20., "i2": 50., "i3": 100.})))
    view = data.build()
    assert view["detail"]["stats"]["meanResponse"] is None
    assert view["chart"]["observationRows"]
    for row in view["chart"]["trials"]["1"]:
        assert row["values"] == [1., 20., 50., 100.]
        assert row["labels"] == ["safe (1)", "20", "50", "100"]
        assert row["score"] == "4 graded"
    assert_all_attempts(data, view)


def test_sparse_registry_features_keep_their_declared_roles(source):
    data = source()
    data.edit("subjects", lambda frame: frame.assign(harness=["ha", "hb"]))
    data.edit("items", lambda frame: frame.assign(item_features=["language=en", "language=fr"] * 2))
    data.edit("responses", lambda frame: frame.iloc[[0, 5]].assign(test_condition=["temperature=0", "temperature=1"]))
    raw = data.raw()
    assert raw.test_condition.tolist() == ["temperature=0", "temperature=1"]
    assert raw.attrs["dimension_kinds"] == {"temperature": "condition", "harness": "subject", "language": "item"}
    assert raw._subject_features.tolist() == [{"harness": "ha"}, {"harness": "hb"}]
    assert raw._item_features.tolist() == [{"language": "en"}, {"language": "fr"}]
    assert_all_attempts(data, data.build())


def test_many_item_features_do_not_remove_trials(source):
    data = source()
    data.edit("items", lambda _: pd.DataFrame([{"item_id": f"i{i}", "content": f"Task {i}", "item_features": f"source_id={i}"} for i in range(601)]))
    data.edit("responses", lambda _: pd.DataFrame([{"subject_id": "s0", "item_id": f"i{i}", "trial": trial,
        "test_condition": None, "response": float(trial % 2)} for i in range(601) for trial in range(1, 19)]))
    view = data.build()
    assert set(view["chart"]["trials"]) == {str(i) for i in range(1, 19)}
    assert len(chart_records(view["chart"])) == 601 * 18
    assert_all_attempts(data, view)


def canonical_pairs(data):
    data.edit("subjects", lambda frame: frame.assign(harness="evaluation harness"))
    data.edit("items", lambda frame: frame.assign(item_features="language=en"))
    data.edit("responses", lambda frame: frame.assign(interactors=frame.test_condition,
        test_condition=frame.subject_id.map({"s0": "side=model_a;temperature=0", "s1": "side=model_b;temperature=0"})))


def test_pairwise_registry_metadata_and_source_order_do_not_change_comparisons(source):
    data = source("pairwise")
    canonical_pairs(data)
    view = data.build()
    assert view["chart"]["rowKind"] == "pair"
    assert_all_attempts(data, view)
    data.edit("responses", lambda frame: frame.sample(frac=1, random_state=17))
    assert data.build()["chart"] == view["chart"]


def test_repeated_pairwise_attempts_remain_separate_without_guessing_matches(source):
    data = source("pairwise")
    canonical_pairs(data)
    data.edit("responses", lambda frame: pd.concat([frame, frame.assign(trial=2, response=1-frame.response)], ignore_index=True).sample(frac=1, random_state=5))
    view = data.build()
    assert view["chart"].get("rowKind") != "pair"
    assert set(view["chart"]["trials"]) == {"1", "2"}
    assert_all_attempts(data, view)


def test_ambiguous_opponent_names_are_not_resolved_arbitrarily(source):
    data = source("pairwise")
    data.edit("subjects", lambda frame: pd.concat([frame, pd.DataFrame([{"subject_id": "other", "display_name": "Model 0"}])], ignore_index=True))
    view = data.build()
    assert view["chart"].get("rowKind") != "pair"
    assert_all_attempts(data, view)


@pytest.mark.parametrize("case", ["binary", "pairwise", "sessions", "faceted"])
def test_ungraded_attempts_keep_clickable_keys_in_every_layout(source, case):
    data = source(case)
    data.edit("responses", lambda frame: frame.assign(response=frame.response.mask(frame.index == 0)))
    view = data.build()
    chart = view["chart"]
    assert "?" in json.dumps(chart)
    assert_all_attempts(data, view)
    assert view["detail"]["stats"]["graded"] == data.raw().response.count()


@pytest.mark.parametrize("repeated", [False, True])
def test_facets_fall_back_without_overwriting_repeated_or_graded_results(source, repeated):
    data = source("faceted")
    if repeated:
        data.edit("responses", lambda frame: pd.concat([frame, frame.assign(trial=2, response=1-frame.response)], ignore_index=True))
    else:
        data.scale("continuous_bounded", {"kind": "interval", "min": 0, "max": 1})
        data.edit("responses", lambda frame: frame.assign(response=frame.response * .5 + .25))
    view = data.build()
    assert view["chart"].get("rowKind") != "faceted"
    assert_all_attempts(data, view)


def test_coverage_uses_registered_banks_and_retains_ungraded_subjects(source):
    data = source()
    data.edit("subjects", lambda frame: pd.concat([frame, pd.DataFrame([{"subject_id": "s2", "display_name": "Model 2"}])], ignore_index=True))
    data.edit("items", lambda frame: pd.concat([frame, pd.DataFrame([{"item_id": "i4", "content": "Unanswered task"}])], ignore_index=True))
    detail = data.build()["detail"]
    assert detail["stats"] == {"subjects": 3, "items": 5, "observed": .5333, "meanResponse": .5, "graded": 8}
    assert "s2" in detail["matrixRowIds"]


def test_response_ids_are_retained_and_duplicates_rejected(source):
    data = source()
    data.edit("responses", lambda frame: frame.assign(response_id=[f"response-{i}" for i in range(len(frame))]))
    assert data.raw().response_id.tolist() == [f"response-{i}" for i in range(8)]
    data.edit("responses", lambda frame: frame.assign(response_id="duplicate"))
    with pytest.raises(ValueError, match="response_id"):
        data.raw()


def test_distinct_raw_conditions_with_equal_display_fields_keep_their_keys(source):
    data = source()
    data.edit("responses", lambda frame: pd.concat([
        frame.assign(test_condition="temperature=0"),
        frame.assign(test_condition="temperature = 0"),
    ], ignore_index=True))
    view = data.build()
    assert_all_attempts(data, view)


def test_basic_matrices_keep_exact_native_grades_and_missingness(source):
    data = source("continuous")
    data.overrides["continuous"]["joined"] = False
    data.edit("responses", lambda frame: frame.assign(response=frame.response.mask(frame.index == 0)))
    view = data.build()
    bundle = gallery.chart_bundle(view)
    raw = data.raw().set_index(["subject_id", "item_id"])
    for matrix in bundle["matrices"].values():
        assert len(matrix["grades"]) == 8
        for offset, key in matrix["keys"].items():
            expected = raw.loc[(key[0], key[1]), "response"]
            assert matrix["grades"][offset] == (None if pd.isna(expected) else expected)


def test_item_bank_uses_declared_scale_even_without_responses(source):
    data = source("item_bank")
    data.scale("ordinal", {"kind": "discrete", "values": [1, 2, 3, 4, 5]})
    detail = data.build()["detail"]
    assert detail["isBinary"] is False
    assert detail["valueRange"] == [1, 5]
    assert detail["stats"]["graded"] == 0


def test_fractional_categories_preserve_float64_roundoff_and_meanings(source):
    data = source()
    value = 0.10000000000000002
    data.scale("ordinal", {"kind": "discrete", "values": [0, .1, 1], "meanings": {"0.1": "partial"}})
    data.edit("responses", lambda frame: frame.assign(response=value))
    chart = data.build()["chart"]
    assert {level["value"] for level in chart["levels"]} == {0, .1, value, 1}
    assert any(level["label"] == f"partial ({value})" for level in chart["levels"])
    assert_all_attempts(data, data.build())
