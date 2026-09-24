"""Native-source audits must distinguish wrong grades, missing grades and clipped traces."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from measurement_db.scripts.curate_benchmarks.batch2_audits import verify_batch2


class NativeTrajectoryAuditTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "artifacts"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "theagentcompany"
        self.tables = self.directory / "formatted_tables"
        self.tables.mkdir(parents=True)
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump({"build": {"parameters": {
            "models": {"run-a": "model-a", "run-b": "model-b"},
            "harnesses": {"run-a": "Harness", "run-b": "Harness"},
        }}}))
        raw = self.directory / "raw"
        items = []
        for task in ("task-a", "task-b"):
            folder = raw / "tasks" / task
            folder.mkdir(parents=True)
            (folder / "task.md").write_text(task)
            (folder / "evaluator.py").write_text("released evaluator")
            verifier = {"kind": "provider_result_mapping", "grading_code": "released evaluator",
                        "source_revision": "98b68ef82a47690c316f42fddb05baafaab56851"}
            items.append(dict(item_id=task, raw_item_id=task, content=task,
                              verifier=json.dumps({"spec": json.dumps(verifier)})))
        traces, responses = [], []
        for index, (run, task, grade) in enumerate([
            ("run-a", "task-a", 1.), ("run-b", "task-a", 0.), ("run-a", "task-b", None),
        ]):
            folder = raw / "experiments/evaluation/1.0.0" / run
            (folder / "results").mkdir(parents=True, exist_ok=True)
            (folder / "trajectories").mkdir(exist_ok=True)
            if grade is not None:
                (folder / "results" / f"eval_{task}-image.json").write_text(json.dumps({
                    "final_score": {"result": grade, "total": 1},
                }))
            payload = json.dumps([{"content": "完整轨迹" * 5000}], ensure_ascii=False)
            trace = folder / "trajectories" / f"traj_{task}-image.json"
            trace.write_text(payload)
            if index == 0:
                trace.with_name(trace.name + ".gz").write_bytes(gzip.compress(payload.encode()))
            responses.append(dict(response_id=str(index), subject_id=run, item_id=task, response=grade))
            traces.append(dict(response_id=str(index), trace=payload))
        self.frames = {
            "subjects": pd.DataFrame([dict(subject_id=run, display_name=model, harness="Harness")
                                      for run, model in (("run-a", "model-a"), ("run-b", "model-b"))]),
            "items": pd.DataFrame(items), "responses": pd.DataFrame(responses), "traces": pd.DataFrame(traces),
        }
        for name, frame in self.frames.items():
            frame.to_parquet(self.tables / f"{name}.parquet", index=False)

    def test_native_compressed_copies_and_ungraded_attempt_are_preserved(self):
        self.assertEqual(verify_batch2(self.directory), {
            "source_responses": 3, "source_items": 2, "source_traces": 3,
            "source_trace_files": 4, "source_ungraded_observations": 1,
        })

    def test_equal_sum_swaps_missing_grade_to_zero_and_trace_clipping_are_rejected(self):
        for change in ("swap", "null_to_zero", "clip"):
            with self.subTest(change=change):
                name = "traces" if change == "clip" else "responses"
                altered = self.frames[name].copy()
                if change == "swap":
                    altered.loc[[0, 1], "response"] = [0., 1.]
                elif change == "null_to_zero":
                    altered.loc[2, "response"] = 0.
                else:
                    altered.loc[0, "trace"] = altered.loc[0, "trace"][:16000]
                altered.to_parquet(self.tables / f"{name}.parquet", index=False)
                with self.assertRaises(AssertionError):
                    verify_batch2(self.directory)
                self.frames[name].to_parquet(self.tables / f"{name}.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
