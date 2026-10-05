"""Path and rollup tests for benchmark-local Rasch 1PL fits."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
IRT_PATH = REPO_ROOT / "scripts" / "analyze_measurements" / "fit_rasch_models.py"


def _load_irt_module():
    spec = importlib.util.spec_from_file_location(
        "measurement_db_fit_rasch_models_tests", IRT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError(f"cannot load IRT analysis from {IRT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


irt = _load_irt_module()


class IrtOutputLayoutTests(unittest.TestCase):
    def test_default_outputs_are_colocated_with_the_benchmark(self) -> None:
        self.assertEqual(
            irt.fit_output_dir("mmlu", None),
            REPO_ROOT / "benchmarks" / "mmlu" / "model_fits" / "rasch1pl",
        )
        self.assertEqual(
            irt.summary_output_path(None),
            REPO_ROOT
            / "artifacts"
            / "analyze_measurements"
            / "model_fits"
            / "rasch1pl"
            / "summary.csv",
        )

    def test_explicit_output_preserves_the_common_root_layout(self) -> None:
        root = Path("scratch-fits")
        self.assertEqual(irt.fit_output_dir("mmlu", root), root / "mmlu")
        self.assertEqual(irt.summary_output_path(root), root / "summary.csv")

    def test_rollup_is_rebuilt_from_existing_fit_summaries(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch.object(irt, "REPO", repo):
                for slug, auc in (("beta", 0.8), ("alpha", 0.9)):
                    fit_dir = irt.fit_output_dir(slug, None)
                    fit_dir.mkdir(parents=True)
                    (fit_dir / "summary.json").write_text(
                        json.dumps({"slug": slug, "aucTest": auc}) + "\n"
                    )

                stale_rollup = irt.summary_output_path(None)
                stale_rollup.parent.mkdir(parents=True)
                stale_rollup.write_text("slug,aucTest\nghost,0.1\n")

                path, count = irt.write_summary_rollup(None)

            self.assertEqual(path, stale_rollup)
            self.assertEqual(count, 2)
            table = pd.read_csv(path)
            self.assertEqual(table["slug"].tolist(), ["alpha", "beta"])

    def test_difficulties_preserve_ids_and_do_not_depend_on_chart_order(self) -> None:
        with TemporaryDirectory() as tmp:
            fit_dir = Path(tmp)
            pd.DataFrame(
                [
                    {"item_id": "000123", "z": 1.23456},
                    {"item_id": "easy", "z": -0.5},
                ]
            ).to_csv(fit_dir / "items.csv", index=False)
            self.assertEqual(
                irt.item_difficulties(fit_dir), {"000123": 1.2346, "easy": -0.5}
            )

    def test_web_emission_reads_benchmark_local_fit_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            web_payload = repo / "website" / "benchmark-irt.json"
            with (
                patch.object(irt, "REPO", repo),
                patch.object(irt, "WEB_IRT", web_payload),
            ):
                fit_dir = irt.fit_output_dir("fixture", None)
                fit_dir.mkdir(parents=True)
                pd.DataFrame([{"subject_id": "000123", "theta": 1.23456}]).to_csv(
                    fit_dir / "subjects.csv", index=False
                )

                pd.DataFrame([{"item_id": "item-a", "z": 0.125}]).to_csv(
                    fit_dir / "items.csv", index=False
                )
                irt.emit_web(
                    {"fixture": {"slug": "fixture", "aucTest": 0.9}},
                    {},
                    None,
                )

            payload = json.loads(web_payload.read_text())
            self.assertEqual(payload["fixture"]["zByItem"], {"item-a": 0.125})
            self.assertEqual(payload["fixture"]["theta"], {"000123": 1.2346})

    def test_explicit_output_routes_rollup_and_web_emission(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "scratch-fits"
            web_payload = root / "benchmark-irt.json"
            fit_dir = irt.fit_output_dir("fixture", out_root)
            fit_dir.mkdir(parents=True)
            summary = {"slug": "fixture", "aucTest": 0.9}
            (fit_dir / "summary.json").write_text(json.dumps(summary) + "\n")
            pd.DataFrame([{"subject_id": "model-a", "theta": 0.75}]).to_csv(
                fit_dir / "subjects.csv", index=False
            )

            rollup, count = irt.write_summary_rollup(out_root)
            with patch.object(irt, "WEB_IRT", web_payload):
                pd.DataFrame([{"item_id": "item-a", "z": 0.125}]).to_csv(
                    fit_dir / "items.csv", index=False
                )
                irt.emit_web(
                    {"fixture": summary},
                    {},
                    out_root,
                )

            self.assertEqual(rollup, out_root / "summary.csv")
            self.assertEqual(count, 1)
            self.assertEqual(pd.read_csv(rollup)["slug"].tolist(), ["fixture"])
            payload = json.loads(web_payload.read_text())
            self.assertEqual(payload["fixture"]["zByItem"], {"item-a": 0.125})
            self.assertEqual(payload["fixture"]["theta"], {"model-a": 0.75})

    def test_current_schema_uses_source_ids_and_interactor_conditions(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.parquet"
            pd.DataFrame({
                "subject_id": ["0001", "0002"],
                "item_id": ["0010", "0011"],
                "test_condition": [None, "temperature=0.5"],
                "interactors": [None, "user=simulator"],
                "trial": [1, 2],
                "response": [0.0, 1.0],
            }).to_parquet(path)
            with patch.object(irt, "fetch", return_value=path) as fetch:
                df = irt.load_responses("fixture", Path(tmp), False)
            fetch.assert_called_once_with("fixture/responses.parquet", Path(tmp), False)
            self.assertEqual(df.subject_id.tolist(), ["0001", "0002"])
            self.assertEqual(df.item_id.tolist(), ["0010", "0011"])
            self.assertEqual(df.test_condition.tolist(), ["", "temperature=0.5;user=simulator"])
            self.assertEqual(len(irt.binary_frame(df)), 2)
            df.loc[0, "response"] = 0.5
            with self.assertRaises(irt.NotBinary):
                irt.binary_frame(df)


if __name__ == "__main__":
    unittest.main()
