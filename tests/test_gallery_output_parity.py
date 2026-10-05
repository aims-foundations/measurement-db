"""Bundle snapshots verified against the original renderer before consolidation."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import pandas as pd

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
                        patch.object(gallery.HfApi, "file_exists", return_value=case != "item_bank"),
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
                    patch.object(
                        gallery.HfApi, "file_exists", return_value=case != "item_bank"
                    ),
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
                    patch.object(
                        gallery.HfApi, "file_exists", return_value=case != "item_bank"
                    ),
                    redirect_stderr(io.StringIO()),
                ):
                    gallery.chart_bundle(
                        gallery.build_detail(
                            case, root / "data", root / "web", False, overrides
                        )
                    )
                self.assertFalse((root / "web").exists())

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
                    patch.object(
                        gallery.HfApi, "file_exists", return_value=case != "item_bank"
                    ),
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
                patch.object(gallery.HfApi, "file_exists", return_value=True),
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
