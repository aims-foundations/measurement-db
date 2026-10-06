"""Bundle snapshots verified against the original renderer before consolidation."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from scripts.render_website import generate_benchmark_gallery as gallery

EXPECTED = Path(__file__).parent / "fixtures/gallery-bundle-parity.json"
CASES = (
    "binary",
    "continuous",
    "ragged",
    "collapsed",
    "trial_cap",
    "normalized",
    "pairwise",
    "sessions",
    "faceted",
    "item_bank",
)


def write_fixture(root: Path, case: str) -> dict:
    bank = root / case
    bank.mkdir(parents=True)
    subjects = [{"subject_id": f"s{s}", "display_name": f"Model {s}"} for s in range(2)]
    items = [
        {
            "item_id": f"i{i}",
            "content": f"Question {i}",
            "reference_answer": None if i == 0 else str(i),
        }
        for i in range(4)
    ]
    conditions = {
        "ragged": ["protocol=a", "protocol=b;opponent=extra"],
        "collapsed": ["protocol=a", "dataset=b"],
        "normalized": ["temperature=0", "temperature=1"],
    }.get(case, [None])
    trials = (
        range(1, 19) if case == "trial_cap" else (1, 2) if case == "ragged" else (1,)
    )
    rows = []
    override = {}
    for s in range(2):
        for i in range(4):
            for condition in conditions:
                for trial in trials:
                    value = (s + i + trial) % 2
                    if case == "continuous":
                        value = (s + i) / 5
                    if case == "ragged" and s == 1 and i == 3:
                        value = None
                    row = {
                        "subject_id": f"s{s}",
                        "item_id": f"i{i}",
                        "test_condition": condition,
                        "trial": trial,
                        "response": value,
                    }
                    if case == "normalized":
                        row["interactors"] = "opponent=reference"
                    if case == "pairwise":
                        row["test_condition"] = f"opponent=Model {1 - s}"
                        row["response"] = float(s)
                    if case == "sessions":
                        row["item_id"] = f"i{s * 2 + i % 2}"
                        if i > 1:
                            continue
                        row["test_condition"] = (
                            f"agent=agent-a;user=user-{s};session=01;turn={i:02}"
                        )
                    if case == "faceted":
                        row["test_condition"] = f"task=task-{s};stage=" + (
                            "pre" if i < 2 else "post"
                        )
                    rows.append(row)
    if case == "normalized":
        for s, subject in enumerate(subjects):
            subject.update(harness=f"h{s}", subject_features_extra="precision=full")
        for i, item in enumerate(items):
            item.update(
                item_features=f"difficulty={i % 2}",
                verifier=json.dumps({"class": "judge", "judge": "judge-a"}),
            )
    if case == "pairwise":
        override["pairwise"] = "opponent"
    elif case == "sessions":
        override["render"] = "sessions"
    elif case == "faceted":
        override.update(
            render="faceted",
            facet={
                "band": "task",
                "panels": [
                    {"key": "pre", "label": "Before", "match": {"stage": "pre"}},
                    {"key": "post", "label": "After", "match": {"stage": "post"}},
                ],
            },
        )
    meta = {
        "benchmark_id": case,
        "name": case,
        "description": "Rendering parity fixture",
        "domain": ["reasoning"],
        "modality": ["text"],
        "granularity": "not_released" if case == "item_bank" else "item",
        "n_subjects": 2,
        "n_items": 4,
        "n_responses": 0 if case == "item_bank" else len(rows),
        "response_scale": "[0, 1]",
    }
    frames = {"benchmarks": [meta], "subjects": subjects, "items": items}
    if case != "item_bank":
        frames["responses"] = rows
        frames["traces"] = [
            {
                **{k: v for k, v in row.items() if k != "response"},
                "trace": f"Answer for {index}",
            }
            for index, row in enumerate(rows)
        ]
    for name, records in frames.items():
        pd.DataFrame(records).to_parquet(bank / f"{name}.parquet", index=False)
    return {case: override}


def local_fetch(filename: str, cache_dir: Path, refresh: bool = False) -> Path:
    path = cache_dir / filename
    if not path.exists():
        raise SystemExit(f"Missing fixture file: {filename}")
    return path


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class GalleryOutputParityTests(unittest.TestCase):
    def setUp(self):
        source = patch.object(gallery, "source_tables", side_effect=lambda slug:
                              gallery.TableSource(slug, frozenset({"traces.parquet"})))
        source.start()
        self.addCleanup(source.stop)

    def test_sdk_loading_preserves_every_layout_with_both_filename_conventions(self):
        expected = json.loads(EXPECTED.read_text())["cases"]
        for case in CASES:
            for filename in ("responses.parquet", "response.parquet", "both"):
                with self.subTest(case=case, filename=filename), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    overrides = write_fixture(root / "sdk", case)
                    canonical = root / "sdk" / case / "responses.parquet"
                    if canonical.exists() and filename == "both":
                        stale = pd.read_parquet(canonical)
                        stale["response"] = 0.0
                        stale.to_parquet(canonical.with_name("response.parquet"))
                    elif canonical.exists() and filename != canonical.name:
                        canonical.rename(canonical.with_name(filename))

                    def download(repo, path, **kwargs):
                        source = root / "sdk" / path
                        if not source.exists():
                            raise gallery.EntryNotFoundError("missing fixture")
                        return str(source)

                    with (
                        patch.object(gallery, "hf_hub_download", side_effect=download),
                        patch.object(gallery, "source_revision", return_value="pinned"),
                        redirect_stderr(io.StringIO()),
                    ):
                        bundle = gallery.chart_bundle(gallery.build_detail(
                            case, root / "cache", root / "web", False, overrides))
                    self.assertEqual(digest_json(bundle), expected[case]["bundle"])

    def test_bundles_match_verified_pre_consolidation_output(self):
        expected = json.loads(EXPECTED.read_text())["cases"]
        for case in CASES:
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp)
                overrides = write_fixture(root / "data", case)
                with (
                    patch.object(gallery, "fetch", side_effect=local_fetch),
                    redirect_stderr(io.StringIO()),
                ):
                    bundle = gallery.chart_bundle(
                        gallery.build_detail(
                            case, root / "data", root / "web", False, overrides
                        )
                    )
                self.assertEqual(digest_json(bundle), expected[case]["bundle"])
                self.assertFalse((root / "web").exists())


def without_keys(value):
    if isinstance(value, dict):
        return {
            k: without_keys(v)
            for k, v in value.items()
            if k not in {"keys", "keys2"}
            and not (k in {"key", "key2"} and isinstance(v, list))
        }
    if isinstance(value, list):
        return [without_keys(v) for v in value]
    return value


def chart_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"key", "key2"} and isinstance(child, list):
                yield child
            elif key in {"keys", "keys2"}:
                yield from child.values()
            else:
                yield from chart_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from chart_keys(child)


class DirectTableTests(unittest.TestCase):
    def test_source_row_references_remain_opaque_condition_labels(self):
        for label in ["tests/results.csv:0;model=chatgpt", "prm800k/data/test.jsonl:row=0"]:
            self.assertEqual(gallery.parse_to_sel(label), {"condition": label})
        self.assertEqual(gallery.parse_to_sel("model=chatgpt;temperature=0"),
                         {"model": "chatgpt", "temperature": "0"})

    def setUp(self):
        source = patch.object(gallery, "source_tables", side_effect=lambda slug:
                              gallery.TableSource(slug, frozenset({"traces.parquet"})))
        source.start()
        self.addCleanup(source.stop)

    def test_observation_rows_preserve_repeated_items_and_source_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            records = [{"subject_id": "s0", "item_id": "i0", "trial": 1,
                        "test_condition": f"source.jsonl:{index}:0", "response": value}
                       for index, value in enumerate([1.0, 0.0, 1.0])]
            pd.DataFrame(records).to_parquet(root / "binary/responses.parquet", index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                view = gallery.build_detail("binary", root, root / "web", False,
                                            {"binary": {"render": "observations"}}, compact=True)
            self.assertTrue(view["chart"]["observationRows"])
            rows = view["chart"]["trials"]["1"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cols"], ["i0"] * 3)
            self.assertEqual(rows[0]["blocks"]["all"]["bits"], "101")
            self.assertEqual(list(chart_keys(view["chart"])),
                             [["s0", "i0", record["test_condition"], 1] for record in records])
            self.assertEqual(view["detail"]["stats"]["items"], 1)

    def test_ungraded_observations_keep_prompts_answers_conditions_and_trials(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = write_fixture(root, "trials")
            path = root / "trials/responses.parquet"
            responses = pd.read_parquet(path)
            responses["response"] = float("nan")
            responses.to_parquet(path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                view = gallery.build_detail("trials", root, root / "web", False, overrides)
                bundle = gallery.chart_bundle(view)
            detail = bundle["detail"]
            self.assertEqual(detail["stats"]["observed"], 0)
            self.assertIsNone(detail["stats"]["meanResponse"])
            self.assertTrue(detail["hasTraces"])
            keys = [key for matrix in bundle["matrices"].values() for key in matrix["keys"].values()]
            self.assertEqual(len(keys), len(responses))
            self.assertEqual({key[3] for key in keys}, set(responses.trial))
            json.dumps(bundle, allow_nan=False)

    def test_large_graded_chart_keeps_native_values_and_exact_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "continuous")
            path = root / "continuous/responses.parquet"
            source = pd.read_parquet(path)
            source["test_condition"] = [f"record-{i}" for i in range(len(source))]
            source["item_id"] = "i0"
            source.to_parquet(path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch), \
                    patch.object(gallery, "MAX_DENSE_ROWS", 0):
                view = gallery.build_detail("continuous", root, root / "web", False, {}, compact=True)
            chart = view["chart"]
            self.assertTrue(chart["observationRows"])
            self.assertTrue(chart["graded"])
            actual = {}
            for row in chart["trials"]["1"]:
                keys = list(chart_keys(row))
                self.assertEqual(len(keys), len(row["values"]))
                for key, value, color in zip(keys, row["values"], row["colors"]):
                    actual[tuple(key)] = value
                    self.assertEqual(color, "#%02x%02x%02x" % gallery.graded_color(value, *view["detail"]["valueRange"]))
            source = pd.read_parquet(root / "continuous/responses.parquet")
            expected = {tuple(record[column] for column in gallery.KEY): record["response"]
                        for record in source.to_dict("records") if pd.notna(record["response"])}
            self.assertEqual(actual, expected)

    def test_feature_name_collisions_preserve_both_sources_and_raw_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            path = root / "binary/responses.parquet"
            responses = pd.read_parquet(path)
            responses["test_condition"] = "dataset=collection;prover=Isabelle"
            responses.to_parquet(path, index=False)
            item_path = root / "binary/items.parquet"
            items = pd.read_parquet(item_path)
            items["item_features"] = "dataset=source"
            items.to_parquet(item_path, index=False)
            subject_path = root / "binary/subjects.parquet"
            subjects = pd.read_parquet(subject_path)
            subjects["subject_features_extra"] = "prover=Isabelle/PISA"
            subjects.to_parquet(subject_path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                frame = gallery.load_raw("binary", root, False)
            for features in frame["_features"]:
                self.assertEqual(features["dataset"], "collection")
                self.assertEqual(features["item.dataset"], "source")
                self.assertEqual(features["prover"], "Isabelle")
                self.assertEqual(features["subject.prover"], "Isabelle/PISA")
            self.assertTrue(all(key[2] == "dataset=collection;prover=Isabelle" for key in frame["_key"]))

    def test_nullable_source_keys_survive_pandas_string_conversion(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            path = root / "binary/responses.parquet"
            rows = pq.read_table(path).to_pylist()
            rows[0]["test_condition"] = "protocol=a"
            for index, row in enumerate(rows):
                row["interactors"] = "opponent=x" if index == 0 else None
            pd.DataFrame(rows).to_parquet(path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                actual = gallery.load_raw("binary", root, False)["_key"].tolist()
            expected = [[row[k] for k in [*gallery.KEY, "interactors"]] for row in rows]
            self.assertEqual(actual, expected)
            json.dumps(actual, allow_nan=False)

    def test_json_interactors_keep_nested_configuration_and_raw_lookup_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            path = root / "binary/responses.parquet"
            rows = pd.read_parquet(path)
            payload = {"reply_candidates": [{"model_name": "model,max_tokens=400;temperature=1"}]}
            raw = json.dumps(payload)
            rows["interactors"] = raw
            rows.to_parquet(path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                frame = gallery.load_raw("binary", root, False)
                bundle = gallery.chart_bundle(gallery.build_detail("binary", root, root / "web", False, {}))
            for features in frame["_features"]:
                self.assertEqual(set(features), {"reply_candidates"})
                self.assertEqual(json.loads(features["reply_candidates"]), payload["reply_candidates"])
            self.assertTrue(all(key[4] == raw for key in frame["_key"]))
            self.assertEqual(bundle["detail"]["stats"]["items"], 4)
            json.dumps(bundle, allow_nan=False)

    def test_item_attributes_do_not_split_subject_rows(self):
        records = []
        for sid in ["a", "b"]:
            for i in range(4):
                condition = f"task={i % 2};difficulty={i // 2}"
                records.append({"subject_id": sid, "item_id": str(i), "test_condition": condition,
                                "trial": 1, "response": i % 2,
                                "_selection": {"task": str(i % 2), "difficulty": str(i // 2)},
                                "_key": [sid, str(i), condition, 1]})
        chart = gallery.joined_matrix("fixture", pd.DataFrame(records),
                                      ["task", "difficulty"], ["a", "b"], {}, [])
        self.assertEqual(len(chart["trials"]["1"]), 2)
        self.assertEqual(sum(row["n"] for row in chart["trials"]["1"]), 8)
        self.assertEqual(sorted(chart_keys(chart)), sorted(row["_key"] for row in records))

    def test_older_answer_column_and_embedded_images_remain_readable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            path = root / "binary/items.parquet"
            items = pd.read_parquet(path).rename(
                columns={"reference_answer": "correct_answer"}
            )
            prompt = "Diagram ![image](data:image/png;base64,aGVsbG8=)"
            items.loc[1, "content"] = prompt
            items.to_parquet(path, index=False)
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                self.assertEqual(
                    gallery.read_item("binary", root, "i1"),
                    {"content": prompt, "answer": "1"},
                )

    def test_normalized_harness_preserves_session_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "sessions")
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                frame = gallery.load_raw("sessions", root, False)
            obs = gallery.plan_slices(
                "sessions", frame, sessions_mode=True
            ).observations
            before = gallery.session_matrix("sessions", obs, {})
            obs["_selection"] = obs["_selection"].map(
                lambda s: {
                    **{k: v for k, v in s.items() if k != "agent"},
                    "harness": s["agent"],
                }
            )
            self.assertEqual(gallery.session_matrix("sessions", obs, {}), before)

    def test_initial_bundle_never_loads_answers_or_writes_assets(self):
        for case in CASES:
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp)
                overrides = write_fixture(root / "data", case)

                def table_only(filename, cache_dir, refresh=False):
                    self.assertFalse(filename.endswith("/traces.parquet"))
                    return local_fetch(filename, cache_dir, refresh)

                with (
                    patch.object(gallery, "fetch", side_effect=table_only),
                    redirect_stderr(io.StringIO()),
                ):
                    gallery.chart_bundle(
                        gallery.build_detail(
                            case, root / "data", root / "web", False, overrides
                        )
                    )
                self.assertFalse((root / "web").exists())

    def test_build_preserves_bundles_items_and_exact_answer_keys(self):
        expected = json.loads(EXPECTED.read_text())["cases"]
        for case in CASES:
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp)
                overrides = write_fixture(root / "data", case)
                with (
                    patch.object(gallery, "fetch", side_effect=local_fetch),
                    patch.object(gallery, "load_overrides", return_value=overrides),
                    patch.object(gallery, "source_revision", return_value="fixture-revision"),
                    patch.object(gallery, "DYNAMIC_ITEM_COUNT", 0),
                    redirect_stderr(io.StringIO()),
                ):
                    summaries = gallery.build_website_data([case], root / "data", root / "web")
                    folder = root / "web/public/benchmark-data" / case
                    bundle = json.loads(gzip.decompress((folder / "view.json.gz").read_bytes()))
                    self.assertNotIn("conditions", summaries[case])
                    self.assertEqual(summaries[case], {key: value for key, value in bundle["detail"].items()
                                                      if key != "conditions"})

                    def lookup(kind, key):
                        encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                        bucket = hashlib.sha256(encoded.encode()).hexdigest()[:2]
                        data = json.loads(gzip.decompress((folder / kind / f"{bucket}.json.gz").read_bytes()))
                        return data.get(encoded)

                    def restore_keys(node):
                        if isinstance(node, dict):
                            if "keyRef" in node:
                                node.update(lookup("keys", node.pop("keyRef")))
                            for child in node.values():
                                restore_keys(child)
                        elif isinstance(node, list):
                            for child in node:
                                restore_keys(child)

                    restore_keys(bundle)
                    self.assertEqual(digest_json(bundle), expected[case]["bundle"])
                    for row in pd.read_parquet(root / "data" / case / "items.parquet").to_dict("records"):
                        key = row["item_id"]
                        self.assertEqual(lookup("item", key), gallery.read_item(case, root / "data", key))
                    if case != "item_bank":
                        for key in gallery.load_raw(case, root / "data", False)["_key"]:
                            group, offset = lookup("answer", key)
                            trace = pq.ParquetFile(root / "data" / case / "traces.parquet").read_row_group(
                                group, columns=["trace"])["trace"][offset].as_py()
                            self.assertEqual({"trace": gallery.truncate_trace(trace)},
                                             gallery.read_answer(case, root / "data", key))

    def test_lookup_retains_unicode_nulls_and_distinct_interactors(self):
        keys = [["模型", "item/😀", None, 1, value] for value in [None, "opponent=a", "opponent=b"]]
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "answer"
            gallery.write_lookup(folder, ((key, {"trace": str(i)}) for i, key in enumerate(keys)))
            for i, key in enumerate(keys):
                encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                bucket = hashlib.sha256(encoded.encode()).hexdigest()[:2]
                payload = json.loads(gzip.decompress((folder / f"{bucket}.json.gz").read_bytes()))
                self.assertEqual(payload[encoded], {"trace": str(i)})

    def test_lookup_rejects_duplicate_keys(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                gallery.write_lookup(Path(tmp) / "item", [("same", {}), ("same", {})])

    def test_source_image_assets_are_included_in_item_content(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            path = root / "binary/items.parquet"
            items = pd.read_parquet(path)
            items["asset_manifest"] = None
            items.loc[0, "asset_manifest"] = json.dumps([
                {"asset_id": "image", "role": "input_image", "ordinal": 1, "media_type": "image/png"},
                {"asset_id": "modern", "role": "input", "ordinal": 2, "media_type": "image/jpeg"},
                {"asset_id": "reference", "role": "reference", "ordinal": 3, "media_type": "image/png"},
                {"asset_id": "audio", "role": "input", "ordinal": 4, "media_type": "audio/wav"},
            ])
            items.to_parquet(path, index=False)
            pd.DataFrame([{"asset_id": "image", "data": b"png bytes"},
                          {"asset_id": "modern", "data": b"jpeg bytes"}]).to_parquet(root / "binary/assets.parquet")
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                item = gallery.read_item("binary", root, "i0")
                self.assertIn("data:image/png;base64,cG5nIGJ5dGVz", item["content"])
                self.assertIn("data:image/jpeg;base64,anBlZyBieXRlcw==", item["content"])
                self.assertEqual(item["content"].count("![Image"), 2)
                assets = gallery.item_assets("binary", root, items.asset_manifest)
                self.assertEqual(gallery.item_content(pq.read_table(path).to_pylist()[0], assets), item)
                output = root / "viewer"
                links = gallery.item_assets("binary", root, list(items.asset_manifest) * 2, output)
                linked = gallery.item_content(pq.read_table(path).to_pylist()[0], links)
                self.assertNotIn("data:image", linked["content"])
                for asset_id, url in links.items():
                    self.assertIn(url, linked["content"])
                    self.assertEqual((output / "asset" / Path(url).name).read_bytes(), assets[asset_id])
                self.assertEqual(len(list((output / "asset").iterdir())), 2)

    def test_answer_queries_match_every_raw_observation(self):
        for case in CASES:
            if case == "item_bank":
                continue
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_fixture(root, case)
                with patch.object(gallery, "fetch", side_effect=local_fetch):
                    displayed = gallery.load_raw(case, root, False)
                    for key, trace in zip(
                        displayed["_key"],
                        pd.read_parquet(root / case / "traces.parquet")["trace"],
                    ):
                        with (
                            patch.object(
                                gallery,
                                "load_raw",
                                side_effect=AssertionError(
                                    "Answer lookup must not rebuild observations"
                                ),
                            ),
                            patch.object(
                                gallery,
                                "registry_feature_dims",
                                side_effect=AssertionError(
                                    "Answer lookup must not parse display features"
                                ),
                            ),
                        ):
                            result = gallery.read_answer(case, root, key)
                        self.assertEqual(result["trace"], trace)
                    item = gallery.read_item(case, root, "i1")
                    self.assertEqual(item, {"content": "Question 1", "answer": "1"})

    def test_answer_requires_a_complete_typed_source_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "ragged")
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                for invalid in [
                    [],
                    ["s0", "i0"],
                    ["s0", "i0", None, "2"],
                    ["s0", "i0", None, True],
                ]:
                    with self.assertRaisesRegex(ValueError, "Invalid observation key"):
                        gallery.read_answer("ragged", root, invalid)
                result = gallery.read_answer(
                    "ragged", root, ["s0", "i0", "protocol=a", 2]
                )
                self.assertEqual(result["trace"], "Answer for 1")

    def test_answer_preserves_null_keys_and_distinct_interactors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, "binary")
            observations = [
                {
                    "subject_id": "s0",
                    "item_id": "i0",
                    "test_condition": None,
                    "trial": 1,
                    "interactors": interactor,
                    "response": 1.0,
                }
                for interactor in [None, "opponent=a", "opponent=b"]
            ]
            pd.DataFrame(observations).to_parquet(root / "binary/responses.parquet")
            pd.DataFrame(
                [{**row, "trace": f"answer-{i}"} for i, row in enumerate(observations)]
            ).to_parquet(root / "binary/traces.parquet")
            with patch.object(gallery, "fetch", side_effect=local_fetch):
                for i, interactor in enumerate([None, "opponent=a", "opponent=b"]):
                    result = gallery.read_answer(
                        "binary", root, ["s0", "i0", None, 1, interactor]
                    )
                    self.assertEqual(result["trace"], f"answer-{i}")
                with self.assertRaisesRegex(ValueError, "source schema"):
                    gallery.read_answer("binary", root, ["s0", "i0", None, 1])

    def test_bundle_keeps_keys_aligned_and_does_not_reparse_features(self):
        for case in CASES:
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp)
                overrides = write_fixture(root / "data", case)
                with (
                    patch.object(gallery, "fetch", side_effect=local_fetch),
                    redirect_stderr(io.StringIO()),
                ):
                    if case != "item_bank":
                        frame = gallery.load_raw(case, root / "data", False)
                        expected = {
                            tuple(key): trace
                            for key, trace in zip(
                                frame["_key"],
                                pd.read_parquet(
                                    root / "data" / case / "traces.parquet"
                                )["trace"],
                            )
                        }
                        with (
                            patch.object(gallery, "load_raw", return_value=frame),
                            patch.object(
                                gallery,
                                "parse_to_sel",
                                side_effect=AssertionError(
                                    "Feature strings must not be parsed by layouts"
                                ),
                            ),
                        ):
                            view = gallery.build_detail(
                                case,
                                root / "data",
                                root / "web",
                                False,
                                overrides,
                            )
                    else:
                        expected = {}
                        view = gallery.build_detail(
                            case,
                            root / "data",
                            root / "web",
                            False,
                            overrides,
                        )
                    bundle = gallery.chart_bundle(view)
                    json.dumps(bundle, allow_nan=False)
                    for key in chart_keys(bundle):
                        self.assertIn(tuple(key), expected)
                        self.assertEqual(
                            gallery.read_answer(case, root / "data", key)["trace"],
                            expected[tuple(key)],
                        )
                    self.assertFalse((root / "web").exists())
                    self.assertEqual(bundle["axes"]["items"], {})
                    if bundle["chart"]:
                        self.assertEqual(bundle["matrices"], {})
                        self.assertTrue(list(chart_keys(bundle)))

    def test_shared_keys_preserve_item_identity_and_varying_keys_stay_explicit(self):
        common = [
            ["s0", "i1", None, 1, "opponent=x"],
            ["s0", "i0", None, 1, "opponent=x"],
        ]
        self.assertEqual(gallery.cell_keys(common, [1, 0]), {"key": common[0]})
        varied = [common[0], ["s0", "i0", "temperature=1", 2, None]]
        self.assertEqual(
            gallery.cell_keys(varied, [1, 0]), {"keys": {1: varied[0], 0: varied[1]}}
        )
        self.assertEqual(
            gallery.cell_keys([common[0], None], second=True), {"keys2": {0: common[0]}}
        )

    def test_basic_bundle_pixels_match_export_downsampling_and_keys(self):
        import numpy as np

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = write_fixture(root / "data", "continuous")
            overrides["continuous"]["joined"] = False
            with (
                patch.object(gallery, "fetch", side_effect=local_fetch),
                patch.object(gallery, "CAP_WIDTH", 2),
            ):
                view = gallery.build_detail(
                    "continuous",
                    root / "data",
                    root / "web",
                    False,
                    overrides,
                )
                bundle = gallery.chart_bundle(view)
                for name, matrix in bundle["matrices"].items():
                    grid = np.full(
                        (matrix["height"], matrix["width"], 3),
                        gallery.GRAY,
                        dtype=np.uint8,
                    )
                    for offset, r, g, b in matrix["pixels"]:
                        grid.reshape(-1, 3)[offset] = [r, g, b]
                    expected = [
                        [[136, 180, 211], [228, 141, 141]],
                        [[242, 242, 242], [214, 39, 40]],
                    ]
                    np.testing.assert_array_equal(grid, expected)
                    self.assertEqual(len(matrix["keys"]), 8)


if __name__ == "__main__":
    unittest.main()
