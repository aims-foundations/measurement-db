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


class EmbodiedArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        from zipfile import ZipFile
        scratch = ROOT / "artifacts"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "embodied_agent_interface"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        self.metadata = {"build": {"parameters": {"paths": {
            "archive": "answers.zip", "prompts": "prompts.json", "conditions": "conditions.json",
        }}}, "grading": {"rule": "native goal comparison", "verifiers": {"official_pipeline": {"kind": "symbolic"}}}}
        self.prompts = {"task-a": "Original full prompt A", "task-b": "Original full prompt B"}
        conditions = {key: {"initial_conditions": {}, "goal_conditions": {"goal": [[[key]]]}}
                      for key in self.prompts}
        (raw / "prompts.json").write_text(json.dumps(self.prompts))
        (raw / "conditions.json").write_text(json.dumps(conditions))
        member = "helm_output/behavior/goal_interpretation/model-a_outputs.json"
        records = [{"identifier": "task-a", "llm_output": "原始答案" * 5000},
                   {"identifier": "task-b", "llm_output": None}]
        with ZipFile(raw / "answers.zip", "w") as archive:
            archive.writestr(member, json.dumps(records, ensure_ascii=False))
            archive.writestr("__MACOSX/" + member, b"not JSON: macOS resource metadata")
        subjects = [dict(subject_id="model-a", display_name="Embodied Agent Interface / model-a",
            harness="Embodied Agent Interface",
            subject_features_extra="recorded_model_label=model-a;historical_inference_settings=not_recorded",
            normalized_name=None, release_date=None, access_date=None, harness_version=None, reasoning_effort=None)]
        items, responses, traces = [], [], []
        for index, original in enumerate(records, 1):
            key = original["identifier"]
            items.append(dict(item_id=key, raw_item_id=key, content=self.prompts[key],
                item_features="simulator=behavior;module=goal_interpretation",
                grading_criterion=json.dumps(dict(reference_answer=json.dumps(conditions[key]["goal_conditions"]),
                                                  rule="native goal comparison")),
                verifier=json.dumps(dict(spec=json.dumps({"kind": "symbolic"})))))
            responses.append(dict(response_id=str(index), subject_id="model-a", item_id=key,
                response=None, trial=1, interactors=None,
                test_condition="simulator=behavior;module=goal_interpretation;historical_judgments_unavailable"))
            traces.append(dict(response_id=str(index), trace=json.dumps(dict(source_archive="answers.zip",
                source_member=member, source_row=index, source_record=original, source_conditions=conditions[key],
                grade_status="historical_judgments_unavailable"), ensure_ascii=False)))
        self.tables = {name: pd.DataFrame(rows) for name, rows in dict(subjects=subjects, items=items,
            responses=responses, traces=traces, benchmarks=[dict(response_scale=json.dumps(
                dict(kind="interval", min=0, max=1, direction="higher_is_better")))]).items()}

    def test_archive_sidecars_are_ignored_and_missing_grades_and_outputs_preserved(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _embodied_agent_interface
        observed = _embodied_agent_interface(self.directory, self.tables, self.metadata)
        self.assertEqual(observed, dict(source_responses=2, source_items=2, source_subjects=1,
            source_traces=2, source_ungraded=2, source_missing_outputs=1))

    def test_builder_reads_only_native_members_and_preserves_null_output(self):
        import runpy
        folder = ROOT / "benchmarks/embodied_agent_interface"
        metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        metadata["build"]["parameters"]["paths"] = {
            **self.metadata["build"]["parameters"]["paths"],
            "members": "helm_output/behavior/goal_interpretation/*_outputs.json",
        }
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(metadata))
        builder = runpy.run_path(str(folder / "build.py"))["EmbodiedAgentInterface"]
        tables = builder(str(self.directory / "build.py")).build_tables()
        self.assertEqual(len(tables["responses"]), 2)
        self.assertTrue(tables["responses"].response.isna().all())
        traces = [json.loads(value)["source_record"] for value in tables["traces"].trace]
        self.assertEqual(traces, [{"identifier": "task-a", "llm_output": "原始答案" * 5000},
                                  {"identifier": "task-b", "llm_output": None}])

    def test_null_to_zero_item_swap_and_clipped_answer_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _embodied_agent_interface
        for change in ("grade", "item", "answer", "prompt", "reference", "model"):
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "grade":
                    tables["responses"].loc[0, "response"] = 0.
                elif change == "item":
                    tables["responses"].loc[0, "item_id"] = "task-b"
                elif change == "answer":
                    trace = json.loads(tables["traces"].loc[0, "trace"])
                    trace["source_record"]["llm_output"] = trace["source_record"]["llm_output"][:16000]
                    tables["traces"].loc[0, "trace"] = json.dumps(trace)
                elif change == "prompt":
                    tables["items"].loc[0, "content"] = self.prompts["task-b"]
                elif change == "reference":
                    tables["items"].loc[0, "grading_criterion"] = tables["items"].loc[1, "grading_criterion"]
                else:
                    tables["subjects"].loc[0, "display_name"] = "Embodied Agent Interface / guessed-new-label"
                with self.assertRaises(ValueError):
                    _embodied_agent_interface(self.directory, tables, self.metadata)


class SweTogetherAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from scripts.build_measurement_tables import reload

        reload()
        self.addCleanup(reload)

        scratch = ROOT / "artifacts"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "swe_together"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/swe_together"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        self.metadata["build"]["parameters"]["paths"] = {
            "metrics": "metrics.js", "tasks": "tasks.jsonl", "website_tasks": "tasks.js"}
        self.metadata["build"]["parameters"]["task_aliases"] = {"task-b-short": "task-b-shortened"}
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        tasks = []
        for identifier in ["task-a", "task-b-shortened"]:
            tasks.append(dict(task_id=identifier, instruction="Native instruction " + identifier + "\n",
                repo="author/project", repo_url="https://example.org/author/project", base_commit="abcdef",
                language=None, difficulty="hard", category="bugfix", tags=["code"], docker_image="example:commit",
                allow_internet=False, agent_timeout_sec=3600.0, reference_patch="  patch " + identifier + "\n",
                completeness_goals='[{"goal": "original goal"}]', oracle_intents='["original user intent"]',
                fail_to_pass=["test_new"], pass_to_pass=["test_existing"], test_manifest="{}", test_cmd=None,
                log_parser=None))
        (raw / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in tasks) + "\n")
        (raw / "tasks.js").write_text("window.TASKS = " + json.dumps([dict(name=row["task_id"]) for row in tasks]) + ";")
        self.metrics = {
            "task-a": {"models": {"model-a": {"trials": [0.85, None], "j": 0.85},
                                     "model-b": {"trials": [0.84, 0.], "j": 0.42}}},
            "task-b-short": {"models": {"model-a": {"trials": [None, 1.], "j": 1.}}},
        }
        (raw / "metrics.js").write_text("window.METRICS = " + json.dumps(self.metrics) + ";")
        builder = runpy.run_path(str(folder / "build.py"))["SweTogether"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_missing_slots_threshold_and_original_replicate_positions(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _swe_together
        self.assertEqual(_swe_together(self.directory, self.tables, self.metadata), dict(
            source_responses=4, source_tasks=2, source_subjects=2, source_traces=4,
            source_successes=2, missing_replicates=2, missing_model_task_cells=1))
        trace = next(json.loads(value) for value in self.tables["traces"].trace
                     if json.loads(value)["source_task"] == "task-b-short")
        self.assertEqual(trace["source_trial"], 2)
        self.assertEqual(trace["source_score"], 1.)

    def test_wrong_associations_grades_references_and_snapshot_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _swe_together
        for change in ["grade", "model", "item", "replicate", "score", "reference", "prompt", "snapshot", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "grade":
                    tables["responses"].loc[0, "response"] = 1. - tables["responses"].loc[0, "response"]
                elif change in {"model", "item"}:
                    field = "subject_id" if change == "model" else "item_id"
                    current = tables["responses"].loc[0, field]
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != current)
                elif change == "replicate":
                    tables["responses"].loc[0, "trial"] = 3
                elif change in {"score", "snapshot"}:
                    trace = json.loads(tables["traces"].loc[0, "trace"])
                    trace["source_score" if change == "score" else "snapshot"] = 0.123 if change == "score" else "old-export"
                    tables["traces"].loc[0, "trace"] = json.dumps(trace)
                elif change == "reference":
                    tables["items"].loc[0, "grading_criterion"] = tables["items"].loc[1, "grading_criterion"]
                elif change == "prompt":
                    tables["items"].loc[0, "content"] = "truncated prompt"
                else:
                    tables["responses"] = tables["responses"].iloc[1:].copy()
                with self.assertRaises(ValueError):
                    _swe_together(self.directory, tables, self.metadata)


class PublishedHTMLAuditTests(unittest.TestCase):
    def test_algotune_preserves_multiple_final_files_and_code_whitespace(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _algotune_html
        source = ('<div class="file-name">solver.py</div><pre class="best-code"><code>def solve():\n'
                  '    return 1 &lt; 2\n</code></pre>'
                  '<div class="file-name">helper.pyx</div><pre class="best-code">cdef int value = 2\n</pre>'
                  '<div class="message assistant"><div class="message-content"><pre>  original\n\n'
                  '    indentation</pre></div></div>')
        messages, files = _algotune_html(source)
        self.assertEqual(files, [dict(name='solver.py', content='def solve():\n    return 1 < 2\n'),
                                 dict(name='helper.pyx', content='cdef int value = 2\n')])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0][0], 'assistant')
        altered, _ = _algotune_html(source.replace('    indentation', '  indentation'))
        self.assertNotEqual(messages, altered)
        with self.assertRaises(ValueError):
            _algotune_html(source.removesuffix('</div>'))


if __name__ == "__main__":
    unittest.main()
