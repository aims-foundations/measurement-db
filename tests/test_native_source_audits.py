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


class FrontierORAuditTests(unittest.TestCase):
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
        self.directory = Path(self.temporary.name) / "frontieror"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/frontieror"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        paths = self.metadata["build"]["parameters"]["paths"]
        tasks = [dict(paper_id="task-a"), dict(paper_id="task-b")]
        values = [dict(model="model-a", kind="model", feasibility=.2, sol_quality=0., qte=0.),
                  dict(model="model-b", kind="model", feasibility=.6, sol_quality=.2, qte=.4),
                  dict(model="model-a", kind="model", feasibility=None, sol_quality=None, qte=None),
                  dict(model="GPT-5.3-Codex + CoRAL", kind="self_evolve", feasibility=1., sol_quality=.2, qte=.2)]
        index = raw / paths["index"]
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps(tasks))
        index = raw / paths["website_index"]
        index.parent.mkdir(parents=True)
        index.write_text("paper_id\ntask-a\ntask-b\n")
        for i, task in enumerate(tasks):
            identifier = task["paper_id"]
            path = raw / paths["details"].format(paper_id=identifier)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(paper_id=identifier, per_model=values[2*i:2*i+2])))
            for field, text in dict(description="Original task " + identifier + "\n", instance_schema='{"input": "original"}\n',
                    solution_schema='{"objective_value": "number"}\n', checker="# original checker\n" + "# " + "x" * 20000).items():
                path = raw / paths[field].format(paper_id=identifier)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        builder = runpy.run_path(str(folder / "build.py"))["FrontierOR"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_fractional_grades_unknown_grade_and_distinct_harnesses(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _frontieror
        self.assertEqual(_frontieror(self.directory, self.tables, self.metadata), dict(source_responses=4,
            source_tasks=2, source_subjects=3, source_traces=4, source_ungraded=1, source_one_shot=3, source_self_evolution=1))

    def test_aggregate_preserving_swaps_null_filling_and_source_corruption_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _frontieror
        for change in ["swap", "fill_null", "model", "item", "replicate", "score", "checker", "schema", "snapshot", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "swap":
                    tables["responses"].loc[[0, 1], "response"] = [.6, .2]
                elif change == "fill_null":
                    tables["responses"].loc[tables["responses"].response.isna(), "response"] = 0.
                elif change in {"model", "item"}:
                    field = "subject_id" if change == "model" else "item_id"
                    current = tables["responses"].loc[0, field]
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != current)
                elif change == "replicate":
                    tables["responses"].loc[0, "trial"] = 5
                elif change in {"score", "snapshot"}:
                    trace = json.loads(tables["traces"].loc[0, "trace"])
                    if change == "score": trace["source_record"]["feasibility"] = .4
                    else: trace["snapshot"] = "older-version"
                    tables["traces"].loc[0, "trace"] = json.dumps(trace)
                elif change == "checker":
                    verifier = json.loads(tables["items"].loc[0, "verifier"])
                    spec = json.loads(verifier["spec"])
                    spec["reference_checker"] = spec["reference_checker"][:16000]
                    verifier["spec"] = json.dumps(spec)
                    tables["items"].loc[0, "verifier"] = json.dumps(verifier)
                elif change == "schema":
                    tables["items"].loc[0, "content"] = "Original task task-a\n"
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _frontieror(self.directory, tables, self.metadata)


class PlanBenchAuditTests(unittest.TestCase):
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
        self.directory = Path(self.temporary.name) / "planbench"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/planbench"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        rows = [dict(instance_id=1, query="Original planning prompt one\n", ground_truth_plan=["a", "b"],
                     llm_raw_response="original output\n" + "x" * 20000, llm_correct=True),
                dict(instance_id=2, query="Original planning prompt two\n", ground_truth_plan=" \n",
                     llm_raw_response="", llm_correct=False),
                dict(instance_id=3, query="Attempt without judgment", llm_raw_response="ungraded plan"),
                dict(instance_id=4, query="Query only; no recorded attempt")]
        header = dict(task="t1", engine="model-a", domain="blocksworld", prompt_type="oneshot")
        adaptive = dict(instance_id=1, messages=[dict(role="system", content="Original system instructions"),
            dict(role="user", content="Original adaptive problem\n"), dict(role="assistant", content="wrong plan"),
            dict(role="user", content="Private later feedback, excluded from item content"),
            dict(role="assistant", content="corrected plan")], act_correct=True, verifier_states_correct=False,
            steps=2, feedback_messages=["original feedback"])
        variants = [
            ("blocksworld/model-a/task_1_plan_generation.json", dict(header, instances=rows)),
            ("blocksworld/copy/task_1_plan_generation.json", dict(header, instances=[rows[0]])),
            ("blocksworld/model-b/task_1_plan_generation_backprompting.json", dict(header, engine="model-b", instances=[adaptive])),
            ("unsolvable_blocksworld/model-a/task_1_plan_generation.json", dict(header, domain="unsolvable_blocksworld", instances=[
                dict(instance_id=5, query="Unsolvable prompt", llm_raw_response="No valid plan", correct=-2, llm_correct=False)])),
        ]
        for name, data in variants:
            path = raw / "llm_planning_analysis/results" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        builder = runpy.run_path(str(folder / "build.py"))["PlanBench"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_adaptive_and_ungraded_attempts_duplicate_exports_and_native_annotation_codes(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _planbench
        self.assertEqual(_planbench(self.directory, self.tables, self.metadata), dict(source_responses=5,
            source_items=5, source_subjects=2, source_models=2, source_traces=5, source_query_only=1,
            source_duplicate_exports=1, source_adaptive=1, source_single_prompt=4, source_ungraded=1,
            source_numeric_annotations=1, source_blank_outputs=1))

    def test_changed_associations_feedback_leakage_clipping_and_imputed_grades_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _planbench
        for change in ["swap", "fill_null", "subject", "item", "trial", "feedback", "clip", "grade_code",
                       "alias", "header", "reference", "grader", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "swap":
                    positive = tables["responses"].index[tables["responses"].response.eq(1.)][0]
                    negative = tables["responses"].index[tables["responses"].response.eq(0.)][0]
                    tables["responses"].loc[[positive, negative], "response"] = [0., 1.]
                elif change == "fill_null":
                    tables["responses"].loc[tables["responses"].response.isna(), "response"] = 0.
                elif change in {"subject", "item"}:
                    field = change + "_id"
                    current = tables["responses"].loc[0, field]
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != current)
                elif change == "trial":
                    tables["responses"].loc[0, "trial"] = 3
                elif change == "feedback":
                    tables["items"].loc[0, "content"] += "\nLater feedback must not leak here"
                elif change in {"clip", "grade_code", "alias", "header"}:
                    for index, body in tables["traces"].trace.items():
                        trace = json.loads(body)
                        record = trace["source_record"]
                        if change == "clip" and len(record.get("llm_raw_response", "")) > 16000:
                            record["llm_raw_response"] = record["llm_raw_response"][:16000]
                        elif change == "grade_code" and type(record.get("correct")) is int:
                            record["correct"] = 0
                        elif change == "alias" and len(trace["source_locations"]) > 1:
                            trace["source_locations"] = trace["source_locations"][:1]
                        elif change == "header":
                            trace["source_header"]["engine"] = "guessed-model"
                        else:
                            continue
                        tables["traces"].loc[index, "trace"] = json.dumps(trace)
                        break
                    else:
                        self.fail("Missing fixture for corruption: " + change)
                elif change == "reference":
                    tables["items"].loc[0, "grading_criterion"] = json.dumps(dict(reference_answer="guessed", rule="incorrect"))
                elif change == "grader":
                    verifier = json.loads(tables["items"].loc[0, "verifier"])
                    verifier["class"] = "judge"
                    tables["items"].loc[0, "verifier"] = json.dumps(verifier)
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _planbench(self.directory, tables, self.metadata)


class WeaveBenchAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from scripts.build_measurement_tables import reload

        reload()
        self.addCleanup(reload)
        scratch = ROOT / "artifacts"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "weavebench"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/weavebench"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        paths = self.metadata["build"]["parameters"]["paths"]
        rows = [dict(task_id="task-a", model="model-a", harness="codex", score=.8),
                dict(task_id="task-b", model="model-a", harness="openclaw", score=.7999999999999),
                dict(task_id="task-c", model="model-b", harness="codex", score=None),
                dict(task_id="task-d", model="model-b", harness="codex", score=0.)]
        for i, row in enumerate(rows):
            row.update(category="WEB", task_prompt="Original problem " + row["task_id"] + "\n" + "p" * 18000,
                checks="def grade():\n    return 1.0\n# " + "c" * 20000,
                deliverables=["original deliverable"], files=[["output.txt", 123]], is_hack=i == 3,
                hack_confidence=1. if i == 3 else 0., dimensions=dict(evidence=dict(score=.9, reason="Native evidence")),
                steps=[dict(kind="cli", thinking="Full published reasoning\n" + "t" * 22000,
                            action="original action", shot="001.webp", output="original tool output"),
                       dict(kind="note", thinking="Keep published notes too", action=None, shot=None, output=None)])
        manifest = [dict(id=row["task_id"], model=row["model"], harness=row["harness"], score=row["score"],
                         hack=row["is_hack"], steps=1) for row in rows]
        members = {paths["root"] + "/" + paths["manifest"]: json.dumps(manifest).encode()}
        for row in rows:
            members[paths["root"] + "/" + paths["records"] + row["task_id"] + ".json"] = json.dumps(row).encode()
            members[paths["root"] + "/trajectories/shots/" + row["task_id"] + "/001.webp"] = b"original screenshot bytes"
        with tarfile.open(raw / paths["archive"], "w:gz") as archive:
            for name, body in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
        builder = runpy.run_path(str(folder / "build.py"))["WeaveBench"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_threshold_precision_missing_grade_runtime_identity_and_full_trace(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _weavebench
        self.assertEqual(_weavebench(self.directory, self.tables, self.metadata), dict(source_responses=4,
            source_items=4, source_subjects=3, source_models=2, source_traces=4, source_screenshots=4,
            source_trace_entries=8, source_action_steps=4, source_passes=1, source_ungraded=1))

    def test_scores_configuration_linkage_checks_screenshots_and_selection_policy(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _weavebench
        for change in ["swap", "fill_null", "subject", "item", "trial", "prompt", "checks", "clip", "screenshot", "score", "selection", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "swap":
                    tables["responses"].loc[[0, 1], "response"] = [0., 1.]
                elif change == "fill_null":
                    tables["responses"].loc[tables["responses"].response.isna(), "response"] = 0.
                elif change in {"subject", "item"}:
                    field = change + "_id"
                    current = tables["responses"].loc[0, field]
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != current)
                elif change == "trial":
                    tables["responses"].loc[0, "trial"] = 2
                elif change == "prompt":
                    tables["items"].loc[0, "content"] = tables["items"].loc[0, "content"][:16000]
                elif change == "checks":
                    criterion = json.loads(tables["items"].loc[0, "grading_criterion"])
                    rule = json.loads(criterion["rule"])
                    rule["checks"] = rule["checks"][:16000]
                    criterion["rule"] = json.dumps(rule)
                    tables["items"].loc[0, "grading_criterion"] = json.dumps(criterion)
                elif change in {"clip", "screenshot", "score"}:
                    trace = json.loads(tables["traces"].loc[0, "trace"])
                    if change == "clip":
                        trace["source_record"]["steps"][0]["thinking"] = trace["source_record"]["steps"][0]["thinking"][:16000]
                    elif change == "screenshot":
                        trace["screenshot_members"] = []
                    else:
                        trace["source_record"]["score"] = .99
                    tables["traces"].loc[0, "trace"] = json.dumps(trace)
                elif change == "selection":
                    tables["responses"].loc[0, "test_condition"] = "unbiased_full_evaluation"
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _weavebench(self.directory, tables, self.metadata)


class MTBenchAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from scripts.build_measurement_tables import reload

        reload()
        self.addCleanup(reload)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "mtbench"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/mtbench"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        paths = self.metadata["build"]["parameters"]["paths"]

        def write(name, records):
            target = raw / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(json.dumps(row) for row in records) + "\n")

        questions = [dict(question_id=81, category="writing", turns=["First question\n" + "q" * 18000, "Follow-up question"], reference=["Task reference", ""]),
                     dict(question_id=82, category="math", turns=["Math question", "Math follow-up"], reference=["Task math reference", "Task second reference"])]
        reference = dict(question_id=82, answer_id="reference-id", model_id="gpt-4", choices=[dict(index=0, turns=["Judge reference 1", "Judge reference 2"])])
        prompts = []
        for math in [False, True]:
            for turn in [1, 2]:
                name = "single-" + ("math-v1" if math else "v1") + ("-multi-turn" if turn == 2 else "")
                text = "Q: {question}\nA: {answer}" if turn == 1 else "Q1: {question_1}\nA1: {answer_1}\nQ2: {question_2}\nA2: {answer_2}"
                if math:
                    text += "\nR1: {ref_answer_1}\nR2: {ref_answer_2}"
                prompts.append(dict(name=name, type="single", system_prompt="Rate the answer from 1 to 10", prompt_template=text, output_format="[[rating]]"))
        lookup = {row["name"]: row for row in prompts}
        judgments = []
        scores = iter([8.5, -1, 5, 6, 7, 9.5, 3, 1])
        for model in ["model-a", "model-b"]:
            answers = []
            for q in questions:
                a = dict(question_id=q["question_id"], answer_id=model + str(q["question_id"]), model_id="internal-" + model,
                         choices=[dict(index=0, turns=[model + " first answer\n" + "a" * 20000, model + " current answer"])], tstamp=123.456)
                answers.append(a)
                for turn in [1, 2]:
                    template = "single-" + ("math-v1" if q["category"] == "math" else "v1") + ("-multi-turn" if turn == 2 else "")
                    p = lookup[template]
                    at, qt = a["choices"][0]["turns"], q["turns"]
                    user_prompt = p["prompt_template"].format(question=qt[0], question_1=qt[0], question_2=qt[1],
                        answer=at[0], answer_1=at[0], answer_2=at[1], ref_answer_1="Judge reference 1", ref_answer_2="Judge reference 2")
                    judgments.append(dict(model=model, question_id=q["question_id"], turn=turn, judge=["gpt-4", template],
                        score=next(scores), judgment="Full native assessment\n" + "j" * 24000, user_prompt=user_prompt, tstamp=234.567))
            write(paths["answers"].replace("*", model), answers)
        for name, rows in [("questions", questions), ("judgments", judgments), ("prompts", prompts), ("references", [reference])]:
            write(paths[name], rows)
        builder = runpy.run_path(str(folder / "build.py"))["MTBench"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_fractional_ratings_nulls_distinct_references_and_complete_turn_history(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _mtbench
        self.assertEqual(_mtbench(self.directory, self.tables, self.metadata), dict(source_responses=8, source_items=6,
            source_subjects=2, source_questions=2, source_conversations=4, source_traces=8, source_ungraded=1, source_fractional_ratings=2))

    def test_source_associations_history_judging_and_full_traces_reject_corruption(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _mtbench
        for change in ["swap", "fill_null", "round", "subject", "item", "trial", "prompt", "history", "reference",
                       "judge_reference", "judge_prompt", "clip", "model_alias", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "swap":
                    tables["responses"].loc[[0, 2], "response"] = list(reversed(tables["responses"].loc[[0, 2], "response"].tolist()))
                elif change == "fill_null":
                    tables["responses"].loc[tables["responses"].response.isna(), "response"] = 1.
                elif change == "round":
                    tables["responses"]["response"] = tables["responses"].response.round()
                elif change in {"subject", "item"}:
                    field = change + "_id"
                    tables["responses"].loc[0, field] = next(x for x in tables["responses"][field] if x != tables["responses"].loc[0, field])
                elif change == "trial":
                    tables["responses"].loc[0, "trial"] = 2
                elif change == "prompt":
                    tables["items"].loc[0, "content"] = "Wrong question"
                elif change == "history":
                    index = next(i for i, row in tables["items"].iterrows() if "_turn2@" in row.raw_item_id)
                    tables["items"].loc[index, "content"] = json.loads(tables["items"].loc[index, "content"])[-1]["content"]
                elif change in {"reference", "judge_reference"}:
                    index = next(i for i, row in tables["items"].iterrows() if row.raw_item_id.startswith("82_"))
                    criterion = json.loads(tables["items"].loc[index, "grading_criterion"])
                    if change == "reference":
                        criterion["reference_answer"] = "Wrong task reference"
                    else:
                        rule = json.loads(criterion["rule"])
                        rule["judge_reference"] = None
                        criterion["rule"] = json.dumps(rule)
                    tables["items"].loc[index, "grading_criterion"] = json.dumps(criterion)
                elif change == "judge_prompt":
                    verifier = json.loads(tables["items"].loc[0, "verifier"])
                    spec = json.loads(verifier["spec"])
                    spec["prompt"]["system_prompt"] = "Different rubric"
                    verifier["spec"] = json.dumps(spec)
                    tables["items"].loc[0, "verifier"] = json.dumps(verifier)
                elif change == "clip":
                    trace = json.loads(tables["traces"].loc[0, "trace"])
                    trace["judgment"]["record"]["judgment"] = trace["judgment"]["record"]["judgment"][:16000]
                    tables["traces"].loc[0, "trace"] = json.dumps(trace)
                elif change == "model_alias":
                    tables["subjects"].loc[0, "subject_features_extra"] = tables["subjects"].loc[0, "subject_features_extra"].replace("internal-model", "invented-model")
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _mtbench(self.directory, tables, self.metadata)


class NYUCTFAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from scripts.build_measurement_tables import reload

        reload()
        self.addCleanup(reload)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "nyu_ctf_bench"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/nyu_ctf_bench"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        parameters = self.metadata["build"]["parameters"]
        paths = parameters["paths"]
        bank, definitions = {}, {}
        for key, category, title in [("a", "rev", "Short"), ("b", "crypto", "Beta"), ("c", "misc", "Gamma")]:
            task = f"2023q-{category}-{key}"
            path = f"test/2023/CSAW-Quals/{category}/{title}"
            bank[task] = dict(path=path, challenge=title, category=category, year=2023, event="CSAW-Quals")
            definitions[path + "/challenge.json"] = dict(name="Long alternate?" if key == "a" else title, category=category,
                description="Original challenge\n" + key * 20000, flag="flag{" + key + "}", files=[
                    "./instruction.txt" if key == "a" else "Challenge/input.txt" if key == "b" else "external.zip"])
        tasks = list(bank)
        parameters["external_inputs"] = {bank[tasks[2]]["path"] + "/external.zip": "external.zip"}
        (raw / "external.zip").write_bytes(b"original external input bytes")
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))

        def add(archive, name, value):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

        with tarfile.open(raw / paths["tasks_archive"], "w:gz") as archive:
            prefix = paths["tasks_root"] + "/"
            add(archive, prefix + paths["bank"], bank)
            for name, record in definitions.items():
                add(archive, prefix + name, record)
            add(archive, prefix + bank[tasks[0]]["path"] + "/instruction.txt", b"exact task input A")
            add(archive, prefix + bank[tasks[1]]["path"] + "/dist/input.txt", b"exact task input B")
            link = tarfile.TarInfo(prefix + bank[tasks[1]]["path"] + "/Challenge")
            link.type, link.linkname = tarfile.SYMTYPE, "dist"
            archive.addfile(link)
        with tarfile.open(raw / paths["results_archive"], "w:gz") as archive:
            for submission in ["baseline_demo", "craken_demo", "craken_graph_demo", "enigma_demo"]:
                prefix = paths["results_root"] + "/transcripts/" + submission + "/"
                baseline, enigma = submission.startswith("baseline_"), submission.startswith("enigma_")
                outcomes = [True, False] if baseline else [True, False, None] if enigma else [False, True, False]
                meta = dict(agent=submission.split("_", 1)[0] + (" graph" if "graph" in submission else ""),
                    model="same-recorded-model", date="2026-01-01", comment="pass@5" if baseline else "pass@1")
                add(archive, prefix + "summary.json", dict(metadata=meta, results=dict(zip(tasks, outcomes))))
                for task, outcome in zip(tasks, outcomes):
                    if (baseline and not outcome) or outcome is None:
                        continue
                    if baseline:
                        for attempt in [1, 2]:
                            add(archive, prefix + bank[task]["path"][5:] + f"/conversation.model.{attempt}.json",
                                dict(args=dict(model="actual-snapshot", backend="openai", max_rounds=30),
                                     solved=attempt == 2, messages=[dict(role="assistant", content="complete output\n" + "a" * 24000)]))
                    elif enigma:
                        title = "longalternate" if task == tasks[0] else bank[task]["challenge"].lower()
                        add(archive, prefix + bank[task]["category"] + "_" + title + ".traj",
                            dict(info=dict(exit_status="submitted" if outcome else "failed"),
                                 history=[dict(role="assistant", content="complete Enigma output\n" + "e" * 24000)]))
                    else:
                        add(archive, prefix + task + ".json", dict(success=outcome,
                            planner_model="actual-snapshot", executor_model="actual-snapshot", autoprompter_model="actual-snapshot",
                            planner=[dict(role="assistant", content="complete plan\n" + "p" * 24000)]))
        builder = runpy.run_path(str(folder / "build.py"))["NYUCTFBench"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _nyu_ctf_sources
        self.source = _nyu_ctf_sources(self.directory, self.metadata)

    def test_native_formats_allowances_aliases_symlinks_assets_and_missingness(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _nyu_ctf
        observed = _nyu_ctf(self.directory, self.tables, self.metadata, self.source)
        self.assertEqual(observed, dict(source_responses=11, source_items=5, source_subjects=4, source_traces=11,
            source_challenges=3, source_assets=3, source_declared_input_files=3, source_native_trace_files=10,
            source_assessments_without_released_trace=2, source_unpublished_cells=1, source_passes=4))
        self.assertEqual(self.tables["responses"].response.isna().sum(), 1)

    def test_grade_protocol_trace_and_input_corruption_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _nyu_ctf
        for change in ["grade", "fill_null", "subject", "item", "trial", "protocol", "content", "flag", "verifier",
                       "asset_bytes", "asset_link", "configuration", "clip", "drop_trace", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "grade":
                    tables["responses"].loc[0, "response"] = 1 - tables["responses"].loc[0, "response"]
                elif change == "fill_null":
                    tables["responses"].loc[tables["responses"].response.isna(), "response"] = 0.
                elif change in {"subject", "item"}:
                    field = change + "_id"
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != tables["responses"].loc[0, field])
                elif change == "trial":
                    tables["responses"].loc[0, "trial"] = 5
                elif change == "protocol":
                    tables["responses"].loc[0, "test_condition"] = json.dumps(dict(kind="individual_attempt", protocol="pass@1", released_trace_files=1))
                elif change == "content":
                    tables["items"].loc[0, "content"] = json.dumps(dict(description="wrong challenge"))
                elif change == "flag":
                    criterion = json.loads(tables["items"].loc[0, "grading_criterion"])
                    criterion["reference_answer"] = "wrong flag"
                    tables["items"].loc[0, "grading_criterion"] = json.dumps(criterion)
                elif change == "verifier":
                    verifier = json.loads(tables["items"].loc[0, "verifier"])
                    spec = json.loads(verifier["spec"])
                    spec["maximum_attempts"] = 99
                    verifier["spec"] = json.dumps(spec)
                    tables["items"].loc[0, "verifier"] = json.dumps(verifier)
                elif change == "asset_bytes":
                    original = tables["assets"].loc[0, "data"]
                    tables["assets"].at[0, "data"] = b"x" * len(original)
                elif change == "asset_link":
                    manifest = json.loads(tables["items"].loc[0, "asset_manifest"])
                    manifest[0]["role"] = "grading"
                    tables["items"].loc[0, "asset_manifest"] = json.dumps(manifest)
                elif change == "configuration":
                    tables["subjects"]["subject_features_extra"] = tables["subjects"].subject_features_extra.str.replace("actual-snapshot", "invented-snapshot")
                elif change in {"clip", "drop_trace"}:
                    index = next(i for i, row in tables["traces"].iterrows() if json.loads(row.trace)["released_traces"])
                    trace = json.loads(tables["traces"].loc[index, "trace"])
                    if change == "clip":
                        trace["released_traces"][0]["source_record"] = {"summary": "clipped"}
                    else:
                        trace["released_traces"] = []
                    tables["traces"].loc[index, "trace"] = json.dumps(trace)
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _nyu_ctf(self.directory, tables, self.metadata, self.source)


class PerfCodeBenchAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from scripts.build_measurement_tables import reload

        reload()
        self.addCleanup(reload)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "perfcodebench"
        raw = self.directory / "raw"
        raw.mkdir(parents=True)
        folder = ROOT / "benchmarks/perfcodebench"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        paths = self.metadata["build"]["parameters"]["paths"]
        with tarfile.open(raw / paths["archive"], "w:gz") as archive:
            def add(name, value):
                data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
                member = tarfile.TarInfo(paths["root"] + "/" + name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            for index, task in enumerate(["task-a", "task-b", "task-c"]):
                root = f"executable_tasks/{task}/"
                filename = "solution.py" if index == 1 else "solution.cpp"
                record = dict(task_id=task, title="Original task " + task, goal="Preserve the result while optimizing",
                    metric="elapsed nanoseconds", correctness_rule="Match the original oracle", allowed_external_includes=["original-library"],
                    build=dict(compiler="original-compiler", sources=["{variant_dir}/" + filename]))
                if index == 1:
                    record["solution_filename"] = filename
                add(root + "instance.json", record)
                add(root + "baseline/" + filename, "  original baseline\n" + "b" * 21000)
                add(root + "reference/" + filename, "  original reference\n" + "r" * 22000)
                add(root + "harness/oracle.txt", "Original task-specific oracle " + task)
                if index == 0:
                    add(root + "harness/interface.h", "First original interface\n" + "h" * 23000)
                    add(root + "harness/interface.hpp", "Unused second interface")
                if index == 2:
                    add(root + "harness/interface.txt", "Original text interface")
                for model in ["vendor/model-a", "vendor/model-b"]:
                    name = root + "candidate/" + model.replace("/", "__") + "/"
                    status = (["ok", "error", "missing_candidate"] if model.endswith("a") else ["timeout", "ok", "ok"])[index]
                    candidate = dict(task_id=task, variant="candidate", runs=3, status=status)
                    if status == "missing_candidate":
                        add(name + "eval_result.json", dict(task_id=task, model=model, status=status, candidate=dict(status=status, all_ok=False)))
                        continue
                    code = "" if status == "error" else "original generated source\n" + "c" * 24000
                    add(name + filename, code)
                    if status == "ok":
                        candidate.update(all_ok=index != 1, median_elapsed_ns=123, all_elapsed_ns=[122, 123, 124])
                    else:
                        candidate.update(error_type="CalledProcessError" if status == "error" else "TimeoutExpired", error="Original execution diagnostic\n" + "e" * 25000)
                    add(name + "eval_result.json", dict(task_id=task, model=model, dry_run=False, candidate_path=name + filename,
                        benchmark_timeout_sec=5 + index, model_output_summary="original reasoning summary", model_output_solution_source=code,
                        baseline=dict(status="ok", median_elapsed_ns=345), reference=dict(status="ok", median_elapsed_ns=100), candidate=candidate))
        builder = runpy.run_path(str(folder / "build.py"))["PerfCodeBench"]
        output = self.directory / "test_tables"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.tables = {p.stem: pd.read_parquet(p) for p in output.glob("*.parquet")}
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perfcode_sources
        self.source = _perfcode_sources(self.directory, self.metadata)

    def test_original_prompts_interfaces_statuses_and_full_execution_records(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perfcodebench
        self.assertEqual(_perfcodebench(self.directory, self.tables, self.metadata, self.source),
            dict(source_responses=5, source_items=3, source_subjects=2, source_traces=5, source_passes=2,
                source_missing_candidates=1, source_errors=1, source_timeouts=1, source_empty_solutions=1, source_tasks_with_interface=2))

    def test_grades_prompt_contracts_grading_and_native_records_reject_corruption(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perfcodebench
        for change in ["grade", "subject", "item", "trial", "timeout", "prompt", "interface", "reference", "oracle",
                       "model_alias", "trace_code", "trace_error", "trace_time", "missing"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
                if change == "grade":
                    tables["responses"].loc[0, "response"] = 1 - tables["responses"].loc[0, "response"]
                elif change in {"subject", "item"}:
                    field = change + "_id"
                    tables["responses"].loc[0, field] = next(value for value in tables["responses"][field] if value != tables["responses"].loc[0, field])
                elif change == "trial":
                    tables["responses"].loc[0, "trial"] = 3
                elif change == "timeout":
                    condition = json.loads(tables["responses"].loc[0, "test_condition"])
                    condition["timeout_sec"] = 99
                    tables["responses"].loc[0, "test_condition"] = json.dumps(condition)
                elif change == "prompt":
                    tables["items"].loc[0, "content"] = "Task name only"
                elif change == "interface":
                    index = next(i for i, row in tables["items"].iterrows() if "First original interface" in row.content)
                    tables["items"].loc[index, "content"] = tables["items"].loc[index, "content"].replace("First original interface", "Wrong interface")
                elif change in {"reference", "oracle"}:
                    criterion = json.loads(tables["items"].loc[0, "grading_criterion"])
                    if change == "reference":
                        criterion["reference_answer"] = "Wrong reference source"
                    else:
                        rule = json.loads(criterion["rule"])
                        rule["harness_sources"] = []
                        criterion["rule"] = json.dumps(rule)
                    tables["items"].loc[0, "grading_criterion"] = json.dumps(criterion)
                elif change == "model_alias":
                    tables["subjects"]["subject_features_extra"] = tables["subjects"].subject_features_extra.str.replace("vendor/model", "wrong/model")
                elif change.startswith("trace_"):
                    index = next(i for i, row in tables["traces"].iterrows() if json.loads(row.trace)["source_record"]["candidate"]["status"] == ("error" if change == "trace_error" else "ok"))
                    trace = json.loads(tables["traces"].loc[index, "trace"])
                    record = trace["source_record"]
                    if change == "trace_code":
                        record["model_output_solution_source"] = record["model_output_solution_source"][:16000]
                    elif change == "trace_error":
                        record["candidate"]["error"] = "clipped"
                    else:
                        record["candidate"]["median_elapsed_ns"] = 999
                    tables["traces"].loc[index, "trace"] = json.dumps(trace)
                else:
                    tables["responses"] = tables["responses"].iloc[1:]
                with self.assertRaises(ValueError):
                    _perfcodebench(self.directory, tables, self.metadata, self.source)


class OSWorldAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / "artifacts"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(self.temporary.name) / "osworld"
        self.directory.mkdir()
        folder = ROOT / "benchmarks/osworld"
        self.metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        (self.directory / "metadata.yaml").write_text(yaml.safe_dump(self.metadata))
        raw = self.directory / "raw"
        tasks = ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"]
        for index, task in enumerate(tasks):
            path = raw / "tasks/chrome" / (task + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(id=task, snapshot="chrome", instruction=f"Current task {index}",
                config=[dict(command="original setup; preserve=exactly")], related_apps=["chrome"],
                evaluator=dict(func="original_check", expected=index))))
        self.long_trace = '{"response":"' + "原始轨迹" * 5000 + '"}\n'
        self.records = [
            ("release.zip", "agent:desktop/turn_2/", tasks[0], "1\n", "{malformed original JSONL\n", "Historical wording"),
            ("release.zip", "agent:desktop/turn_1/", tasks[0], "0.028626288575876013\n", self.long_trace, None),
            ("release.zip", "agent:desktop/turn_1/", tasks[1], None, None, None),
            ("results_only.zip", "", tasks[0], "0\n", None, None),
        ]
        for archive, prefix, task, score, trace, instruction in self.records:
            safe_prefix = prefix.replace(":", "_x3a_")
            folder_path = raw / "native" / archive / safe_prefix / "chrome" / task
            folder_path.mkdir(parents=True, exist_ok=True)
            if score is not None:
                (folder_path / "result.txt").write_bytes(score.encode())
            if trace is not None:
                (folder_path / "traj.jsonl").write_bytes(trace.encode())
            if archive == "release.zip":
                (folder_path / "runtime.log").write_bytes(b"Original runtime log.\n\n")
                args = raw / "native" / archive / safe_prefix / "args.json"
                args.write_text(json.dumps(dict(model="recorded-model", temperature=0.4, stop_token=";=",
                    client_password="not subject metadata", model_api_key="do-not-copy-api-value")))
            if instruction is not None:
                (folder_path / "instruction.txt").write_bytes(instruction.encode())
        # A definition or empty log alone does not establish an attempted run.
        empty = raw / "native/results_only.zip/chrome" / tasks[1]
        empty.mkdir(parents=True)
        (empty / "runtime.log").write_bytes(b"")
        (empty / "instruction.txt").write_bytes(b"Unattempted task")
        output = self.directory.parent / "tables"
        builder = runpy.run_path(str(folder / "build.py"))["OSWorld"]
        with contextlib.redirect_stdout(io.StringIO()):
            builder(str(self.directory / "build.py")).main_from_args(["--source", str(raw), "--output", str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob("*.parquet")}

    def test_original_scores_missing_grades_turns_and_instruction_variants_are_preserved(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _osworld
        result = _osworld(self.directory, self.frames, self.metadata)
        self.assertEqual(result, dict(source_responses=4, source_graded=3, source_ungraded=1, source_items=3,
            source_subjects=2, source_traces=3, source_archives=2, source_partial_credit=1,
            source_nonempty_trajectories=2, source_nonempty_runtime_logs=3, source_task_ids=2,
            source_recorded_instructions=1, source_changed_instructions=1))
        grades = self.frames["responses"].response.dropna().tolist()
        self.assertIn(float("0.028626288575876013"), grades)
        original = [json.loads(value)["native_files"].get("traj.jsonl") for value in self.frames["traces"].trace]
        self.assertIn(self.long_trace, original)
        self.assertIn("{malformed original JSONL\n", original)
        self.assertNotIn("do-not-copy-api-value", " ".join(self.frames["subjects"].subject_features_extra))

    def test_native_grades_trace_associations_settings_and_historical_wording_reject_corruption(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _osworld
        responses = self.frames["responses"]
        fractional = responses.index[responses.response.between(0, 1, inclusive="neither")][0]
        missing = responses.index[responses.response.isna()][0]
        historical = responses.index[responses.trial.eq(2)][0]
        trace_row = self.frames["traces"].index[self.frames["traces"].response_id.eq(responses.loc[fractional, "response_id"])][0]
        for change in ["grade", "rounding", "null_to_zero", "clip", "trial", "run", "subject", "item", "historical_wording",
                       "setup", "verifier", "settings", "drop_ungraded", "duplicate", "missing_trace", "scale"]:
            with self.subTest(change=change):
                tables = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == "grade":
                    tables["responses"].loc[[fractional, historical], "response"] = responses.loc[[historical, fractional], "response"].to_numpy()
                elif change == "rounding":
                    tables["responses"].loc[fractional, "response"] = pd.to_numeric(pd.Series(["0.028626288575876013"]))[0]
                elif change == "null_to_zero":
                    tables["responses"].loc[missing, "response"] = 0.
                elif change == "clip":
                    record = json.loads(tables["traces"].loc[trace_row, "trace"])
                    record["native_files"]["traj.jsonl"] = record["native_files"]["traj.jsonl"][:16000]
                    tables["traces"].loc[trace_row, "trace"] = json.dumps(record)
                elif change == "trial":
                    tables["responses"].loc[historical, "trial"] = 1
                elif change == "run":
                    tables["responses"].loc[fractional, "test_condition"] = json.dumps(dict(source_archive="release.zip", run_prefix="agent:desktop/turn_2/"))
                elif change == "subject":
                    other = responses.loc[responses.response.eq(0), "subject_id"].iloc[0]
                    tables["responses"].loc[fractional, "subject_id"] = other
                elif change == "item":
                    tables["responses"].loc[fractional, "item_id"] = responses.loc[missing, "item_id"]
                elif change == "historical_wording":
                    tables["responses"].loc[historical, "item_id"] = responses.loc[fractional, "item_id"]
                elif change in {"setup", "verifier"}:
                    index = tables["items"].index[tables["items"].item_id.eq(responses.loc[fractional, "item_id"])][0]
                    column = "grading_criterion" if change == "setup" else "verifier"
                    record = json.loads(tables["items"].loc[index, column])
                    if change == "setup":
                        rule = json.loads(record["rule"])
                        rule["task_definition"]["config"] = []
                        record["rule"] = json.dumps(rule)
                    else:
                        record["spec"] = json.dumps(dict(kind="different grader"))
                    tables["items"].loc[index, column] = json.dumps(record)
                elif change == "settings":
                    index = tables["subjects"].index[tables["subjects"].subject_id.eq(responses.loc[fractional, "subject_id"])][0]
                    tables["subjects"].loc[index, "subject_features_extra"] = tables["subjects"].loc[index, "subject_features_extra"].replace("0.4", "0.9")
                elif change == "drop_ungraded":
                    tables["responses"] = tables["responses"].drop(index=missing)
                elif change == "duplicate":
                    tables["responses"] = pd.concat([tables["responses"], tables["responses"].loc[[fractional]]], ignore_index=True)
                elif change == "missing_trace":
                    tables["traces"] = tables["traces"].drop(index=trace_row)
                else:
                    tables["benchmarks"].loc[0, "response_scale"] = json.dumps(dict(kind="discrete", values=[0, 1]))
                with self.assertRaises(ValueError):
                    _osworld(self.directory, tables, self.metadata)


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


class TabArenaNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import numpy as np
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'tabarena'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/tabarena'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.metadata['build']['parameters']['methods'] = {'CatBoost': 'CatBoost', 'KNeighbors': 'KNeighbors'}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        raw = self.directory / 'raw'
        inputs = raw / 'openml'
        inputs.mkdir(parents=True)
        tasks = pd.DataFrame([dict(dataset=kind, tid=i + 11, did=i + 21)
                              for i, kind in enumerate(['binary', 'multiclass', 'regression'])])
        predictions = {'binary': [0.5, 0.9], 'multiclass': [[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]],
                       'regression': [16.994932174682617, 2.5]}
        labels = {'binary': [1, 0], 'multiclass': [1, 2], 'regression': [17.65, -8.25]}
        originals = {'binary': ['zebra', 'unused', 'ant'], 'multiclass': ['two', 'unused', 'one'],
                     'regression': [-8.25, 99., 17.65]}
        for row in tasks.itertuples():
            pd.DataFrame({'feature': [0.12345678901234567, None, 3.5],
                          'text': ['original ' * 500, 'training only', 'test'],
                          'ignored': ['not an input'] * 3, 'target': originals[row.dataset]}).to_parquet(inputs / f'data-{row.did}.parquet')
            (inputs / f'data-{row.did}.json').write_text(json.dumps({'data_set_description': dict(
                default_target_attribute='target', ignore_attribute='ignored', licence='CC0', citation='Fixture source; second citation\nOriginal line', original_data_url='https://example.org/source?version=1')}))
            (inputs / f'task-{row.tid}.json').write_text(json.dumps({'task': {'input': [dict(name='source_data',
                data_set=dict(data_set_id=str(row.did), target_feature='target'))]}}))
            (inputs / f'splits-{row.tid}.arff').write_text('@relation fixture\n@attribute type {TRAIN,TEST}\n'
                '@attribute rowid numeric\n@attribute repeat numeric\n@attribute fold numeric\n@data\n'
                'TRAIN,1,0,0\nTEST,2,0,0\nTEST,0,0,0\n')
        for method in ['CatBoost', 'KNeighbors']:
            method_root = raw / 'methods' / method
            method_root.mkdir(parents=True)
            tasks.to_parquet(method_root / 'task_metadata.parquet')
            default = method + '_c1_BAG_L1'
            (method_root / 'configs_hyperparameters.json').write_text(json.dumps({default: {'setting': 'original;value=1'}}))
            configurations = []
            for row in tasks.itertuples():
                if method == 'KNeighbors' and row.dataset == 'multiclass':
                    continue
                configurations.append(dict(dataset=row.dataset, tid=row.tid, fold=0, framework=default, problem_type=row.dataset))
                path = method_root / 'model_predictions' / row.dataset / '0'
                path.mkdir(parents=True)
                values = np.array([predictions[row.dataset], predictions[row.dataset]], dtype='float32')
                values[0] = 0
                values.tofile(path / 'pred-test.dat')
                (path / 'metadata.json').write_text(json.dumps(dict(dataset=row.dataset, fold=0, dtype='float32',
                    models=['unselected', default], pred_test_shape=list(values.shape))))
                pd.DataFrame({'target': labels[row.dataset]}, index=[2, 0]).to_csv(path / 'label-test.csv.zip',
                    compression={'method': 'zip', 'archive_name': 'label-test.csv'})
            pd.DataFrame(configurations).to_parquet(method_root / 'configs.parquet')
        output = self.directory.parent / 'tables'
        builder = runpy.run_path(str(folder / 'build.py'))['TabArena']
        with contextlib.redirect_stdout(io.StringIO()):
            builder(str(self.directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_full_features_prediction_precision_and_fitted_models_are_preserved(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _tabarena
        counts = _tabarena(self.directory, self.frames, self.metadata)
        self.assertEqual(counts, dict(source_responses=10, source_subjects=5, source_items=6, source_traces=10,
            source_datasets=3, source_methods=2, source_classification_responses=6, source_regression_responses=4))
        self.assertTrue(self.frames['items'].content.str.len().max() > 1500)

    def test_independent_audit_rejects_corrupt_grades_inputs_outputs_and_links(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _tabarena, _tabarena_sources
        original = _tabarena_sources(self.directory, self.metadata)
        for change in ['grade', 'item', 'subject', 'trace', 'clip', 'reference', 'scale', 'settings', 'trial', 'condition', 'drop', 'duplicate', 'drop_trace']:
            with self.subTest(change=change):
                tables = {key: frame.copy(deep=True) for key, frame in self.frames.items()}
                if change == 'grade':
                    tables['responses'].loc[0, 'response'] += 0.125
                elif change == 'item':
                    tables['responses'].loc[0, 'item_id'] = tables['responses'].loc[1, 'item_id']
                elif change == 'subject':
                    tables['responses'].loc[0, 'subject_id'] = tables['subjects'].subject_id.iloc[-1]
                elif change == 'trace':
                    data = json.loads(tables['traces'].loc[0, 'trace']); data['prediction'] = 0.333
                    tables['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'clip':
                    data = json.loads(tables['items'].loc[0, 'content']); data['features']['text'] = 'truncated'
                    tables['items'].loc[0, 'content'] = json.dumps(data)
                elif change in ['reference', 'scale']:
                    data = json.loads(tables['items'].loc[0, 'grading_criterion'])
                    data['reference_answer' if change == 'reference' else 'response_scale'] = 'wrong'
                    tables['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'settings':
                    tables['subjects'].loc[0, 'subject_features_extra'] = tables['subjects'].loc[0, 'subject_features_extra'].replace('original', 'changed')
                elif change == 'trial':
                    tables['responses'].loc[0, 'trial'] = 2
                elif change == 'condition':
                    tables['responses'].loc[0, 'test_condition'] = 'fold=1;repeat=0'
                elif change == 'drop':
                    tables['responses'] = tables['responses'].iloc[1:]
                elif change == 'duplicate':
                    tables['responses'] = pd.concat([tables['responses'], tables['responses'].iloc[:1]])
                else:
                    tables['traces'] = tables['traces'].iloc[1:]
                with self.assertRaises(ValueError):
                    _tabarena(self.directory, tables, self.metadata, original)


class ExploitGymNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'exploitgym'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/exploitgym'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        raw = self.directory / 'raw'
        (raw / 'archives').mkdir(parents=True)
        template_prefix = 'src/cybergym/task/workspace/templates/'
        for index, revision in enumerate(['a' * 40, 'b' * 40]):
            files = {}
            for domain, filename, description_path in [('user', 'metadata.json', 'description.txt'),
                    ('kernel', 'kernel_metadata.json', 'docs/vulnerability.md'),
                    ('v8', 'v8_metadata.json', 'pov/description.md')]:
                records = [dict(task_id=domain + ':hashed-id', entry_name='fixture/task',
                                attributes='Original; annotation=1')] if domain == 'user' else []
                files['src/cybergym/task/' + filename] = json.dumps(records).encode()
                files[template_prefix + domain + '.md.j2'] = (f'Original version {index}: {{{{ configuration }}}}').encode()
                if records:
                    prefix = 'data/tasks/user/fixture/task/'
                    files[prefix + description_path] = ('完整描述 ' * 5000 + '\n').encode()
                    files[prefix + 'original.bin'] = b'\x00\xfforiginal task bytes'
            files[template_prefix + '_includes/environment.md.j2'] = b'Original shared template\n'
            with tarfile.open(raw / 'archives' / (revision + '.tar.gz'), 'w:gz') as archive:
                for name, content in files.items():
                    member = tarfile.TarInfo('exploitgym-' + revision + '/' + name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            submission = raw / 'results/submissions' / ('agent-' + str(index))
            submission.mkdir(parents=True)
            (submission / 'metadata.yml').write_text(yaml.safe_dump(dict(name='Original harness',
                models=['model-' + str(index)], benchmark_commit=revision, date='2026-06-30')))
            complete = [dict(task_id='user:fixture/task', mitigation_enabled=mitigation,
                flag_captured=not mitigation, on_target=not mitigation, judge_models=[] if mitigation else ['judge-a'],
                models={'model-' + str(index): {'input_tokens': 123}}, time=9000.0, total={'input_tokens': 123})
                for mitigation in [False, True]]
            (submission / 'results.json').write_text(json.dumps(complete))
            if index == 1:
                shorter = [dict(row, on_target=False, flag_captured=False, time=7200.0, judge_models=['judge-b']) for row in complete]
                (submission / 'results-2h.json').write_text(json.dumps(shorter))
        builder = runpy.run_path(str(folder / 'build.py'))['ExploitGym']
        self.builder = builder
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            builder(str(self.directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_original_views_tasks_grading_and_complete_materials_are_preserved(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _exploitgym
        counts = _exploitgym(self.directory, self.frames, self.metadata)
        self.assertEqual(counts, dict(source_responses=6, source_subjects=2, source_items=2,
            source_traces=6, source_assets=2, source_unique_attempt_keys=4, source_task_ids=1,
            source_benchmark_revisions=2, source_successes=2, source_primary_results=4, source_two_hour_views=2))
        self.assertTrue(self.frames['items'].content.str.len().min() > 16000)

    def test_audit_rejects_changed_grades_links_protocols_materials_and_clipping(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _exploitgym, _exploitgym_sources
        source = _exploitgym_sources(self.directory, self.metadata)
        for change in ['grade', 'item', 'subject', 'trace', 'clip', 'criterion', 'judge', 'settings',
                       'trial', 'condition', 'drop', 'duplicate', 'drop_trace', 'asset', 'asset_link']:
            with self.subTest(change=change):
                tables = {key: frame.copy(deep=True) for key, frame in self.frames.items()}
                if change == 'grade':
                    tables['responses'].loc[0, 'response'] = 1 - tables['responses'].loc[0, 'response']
                elif change == 'item':
                    current = tables['responses'].loc[0, 'item_id']
                    tables['responses'].loc[0, 'item_id'] = tables['items'].loc[tables['items'].item_id.ne(current), 'item_id'].iloc[0]
                elif change == 'subject':
                    current = tables['responses'].loc[0, 'subject_id']
                    tables['responses'].loc[0, 'subject_id'] = tables['subjects'].loc[tables['subjects'].subject_id.ne(current), 'subject_id'].iloc[0]
                elif change == 'trace':
                    data = json.loads(tables['traces'].loc[0, 'trace']); data['record']['time'] = 1.
                    tables['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'clip':
                    data = json.loads(tables['items'].loc[0, 'content']); data['description'] = data['description'][:16000]
                    tables['items'].loc[0, 'content'] = json.dumps(data)
                elif change == 'criterion':
                    data = json.loads(tables['items'].loc[0, 'grading_criterion']); data['reference_answer'] = 'invented'
                    tables['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'judge':
                    data = json.loads(tables['items'].loc[0, 'verifier']); spec = json.loads(data['spec'])
                    spec['recorded_judge_models'] = ['outcome-dependent']; data['spec'] = json.dumps(spec)
                    tables['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change == 'settings':
                    tables['subjects'].loc[0, 'subject_features_extra'] = tables['subjects'].loc[0, 'subject_features_extra'].replace('model-', 'changed-')
                elif change == 'trial':
                    tables['responses'].loc[0, 'trial'] = 2
                elif change == 'condition':
                    data = json.loads(tables['responses'].loc[0, 'test_condition']); data['mitigation_enabled'] = not data['mitigation_enabled']
                    tables['responses'].loc[0, 'test_condition'] = json.dumps(data)
                elif change == 'drop':
                    tables['responses'] = tables['responses'].iloc[1:]
                elif change == 'duplicate':
                    tables['responses'] = pd.concat([tables['responses'], tables['responses'].iloc[:1]])
                elif change == 'drop_trace':
                    tables['traces'] = tables['traces'].iloc[1:]
                elif change == 'asset':
                    tables['assets'].loc[0, 'data'] = b'truncated'
                else:
                    data = json.loads(tables['items'].loc[0, 'asset_manifest']); data[0]['role'] = 'runtime_workspace'
                    tables['items'].loc[0, 'asset_manifest'] = json.dumps(data)
                with self.assertRaises((ValueError, KeyError)):
                    _exploitgym(self.directory, tables, self.metadata, source)

    def test_missing_verdict_is_not_converted_to_failure(self):
        import contextlib
        import io
        raw = self.directory / 'raw'
        source = raw / 'results/submissions/agent-0/results.json'
        original = json.loads(source.read_text())
        original[0]['on_target'] = None
        source.write_text(json.dumps(original))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Boolean on_target'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(raw), '--output', str(self.directory.parent / 'invalid')])


class ERBenchNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import csv
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'erbench'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/erbench'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.metadata['build']['parameters']['models'] = dict.fromkeys(['claude', 'gpt35'], 'fixture')
        self.metadata['build']['parameters']['cot_results'] = {'claude/movie_foreign_year': 'fixture'}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        long_reasoning = ('完整说明 ' * 4000).strip()
        for model in ['claude', 'gpt35']:
            domain = 'movie_foreign_year'
            directory = self.directory / 'raw/binary/results' / model
            (directory / 'crafted_df').mkdir(parents=True)
            records, blocks = [], []
            for index, (entity, question, gold, answer, rationale) in enumerate([
                    ('0', 'Is the title Alpha: Beta correct?', 'yes', 'yes', long_reasoning),
                    ('1', 'Is the title Gamma correct?', 'no', 'unsure', 'I cannot tell.'),
                    ('0', 'Is the title Alpha: Beta correct?', 'yes', 'no', 'A second recorded attempt.')]):
                blocks.append(f'{entity}-0th question\nQ: {question}\nA:{answer.capitalize()}. {rationale}\nGold Answer: {gold}\nGold Entity: native entity\n\n')
                records.append({'': str(index), 'question': ('Q: ' + question).split(':')[1],
                    'entity_idx': entity, 'question_idx': '0', 'model_answer': answer,
                    'model_reasoning': rationale.lower(), 'gold_answer': gold, 'gold_entity': 'native entity'})
            filename = domain + ('_cot' if model == 'claude' else '') + '.log'
            (directory / filename).write_text(''.join(blocks))
            if model == 'claude':
                (directory / (domain + '.log')).write_text(''.join(blocks).replace('A:Yes.', 'A:No.'))
            with (directory / 'crafted_df' / (domain + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        self.builder = runpy.run_path(str(folder / 'build.py'))['ERBench']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_colon_questions_conditions_repeated_records_and_complete_outputs(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _erbench
        counts = _erbench(self.directory, self.frames, self.metadata)
        self.assertEqual(counts, dict(source_responses=6, source_subjects=2, source_items=2, source_traces=6,
            source_files=2, source_successes=2, source_chain_of_thought=3, source_restored_question_rows=4))
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)
        self.assertEqual(self.frames['responses'].trial.max(), 2)

    def test_independent_check_rejects_changed_data_links_conditions_and_truncation(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _erbench, _erbench_sources
        source = _erbench_sources(self.directory)
        for change in ['grade', 'subject', 'item', 'criterion', 'verifier', 'question', 'trace', 'position',
                       'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade':
                    frames['responses'].loc[0, 'response'] = 1 - frames['responses'].loc[0, 'response']
                elif change in ['subject', 'item']:
                    column = change + '_id'
                    current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'criterion':
                    data = json.loads(frames['items'].loc[0, 'grading_criterion']); data['reference_answer'] = 'no'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'verifier':
                    data = json.loads(frames['items'].loc[0, 'verifier']); data['class'] = 'judge'
                    frames['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change == 'question':
                    frames['items'].loc[0, 'content'] = frames['items'].loc[0, 'content'].split(':')[0]
                elif change in ['trace', 'position']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'trace': data['log_entry'] = data['log_entry'][:16000]
                    else: data['source_row'] += 1
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition':
                    data = json.loads(frames['responses'].loc[0, 'test_condition']); data['prompting'] = 'standard'
                    frames['responses'].loc[0, 'test_condition'] = json.dumps(data)
                elif change == 'trial':
                    frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _erbench(self.directory, frames, self.metadata, source)

    def test_missing_answer_is_rejected_without_a_failure_grade(self):
        import contextlib
        import io
        path = self.directory / 'raw/binary/results/gpt35/crafted_df/movie_foreign_year.csv'
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, 'model_answer'] = ''
        frame.to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Missing or unknown answers'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])


class PeruMedQANativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'perumedqa'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/perumedqa'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        params = self.metadata['build']['parameters']
        files = [('medgemma4B_TF', '09.finetuned', 0, 4), ('OctoMed-7B', '10.OctoMed/Chunk_1', 0, 2),
                 ('OctoMed-7B', '10.OctoMed/Chunk_2', 2, 4)]
        params['model_identifiers'] = {model: params['model_identifiers'][model] for model in ['medgemma4B_TF', 'OctoMed-7B']}
        params['inference_sources'] = {model: json.dumps(['02.LLMs_x20_Results/' + directory + '/LLMs_Answers.py'
            for name, directory, _, _ in files if name == model]) for model in params['model_identifiers']}
        params['aggregate_models'] = {model: params['aggregate_models'][model] for model in params['model_identifiers']}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        bank = pd.DataFrame([dict(questions=question, option_A='Sí', option_B='No', option_C='Otra',
            option_D='Ninguna', option_E='NA', correct_answer=gold, source_file='Examen', source_folder='Year' + str(year), year=year)
            for question, gold, year in [('¿Primera?', 'A', 2024), ('¿Primera?', 'A', 2024),
                                        ('¿Primera?', 'B', 2025), ('¿Otra?\nLínea dos.', 'C', 2025)]])
        bank_path = self.directory / 'raw/01.Datasets/combined_exam_dataset.csv'
        bank_path.parent.mkdir(parents=True)
        bank.to_csv(bank_path, index=False)
        outputs = {'medgemma4B_TF': ['Respuesta final: A\nRespuesta final: B\n' + 'Explicación 完整 ' * 2000,
            'respuesta final: A', 'Respuesta final: A', ''],
            'OctoMed-7B': ['Respuesta final: (A)', 'Respuesta final: B', 'Respuesta final: B', 'Respuesta final: C']}
        for model, directory, start, stop in files:
            path = self.directory / 'raw/02.LLMs_x20_Results' / directory
            path.mkdir(parents=True)
            frame = bank.iloc[start:stop].copy()
            frame['question'] = frame.questions + '\nA) Sí\nB) No\nC) Otra\nD) Ninguna\nE) NA'
            frame['answer_llm'] = outputs[model][start:stop]
            frame['model_basename'] = model
            frame.to_parquet(path / 'results.parquet')
            (path / 'LLMs_Answers.py').write_text('model_name = ' + repr(params['model_identifiers'][model]) + '\n')
        pd.DataFrame([dict(source_file='Examen', year=year, count_of_matches=correct, total_rows=total,
            percent_correct=100 * correct / total, model_name=params['aggregate_models'][model], source_file_eng='Exam')
            for model, year, correct, total in [('medgemma4B_TF', 2024, 1, 1), ('medgemma4B_TF', 2025, 0, 1),
                ('OctoMed-7B', 2024, 1, 2), ('OctoMed-7B', 2025, 2, 2)]]).to_csv(
                    self.directory / 'raw/02.LLMs_x20_Results/All_Models_Results_2026-01-21.csv', index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['PeruMedQA']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_first_match_missing_grades_chunk_indices_and_training_pool(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perumedqa
        self.assertEqual(_perumedqa(self.directory, self.frames, self.metadata), dict(
            source_responses=8, source_subjects=2, source_items=3, source_traces=8, source_files=3,
            source_bank_rows=4, source_summary_groups=4, source_graded_observations=6, source_ungraded_observations=2,
            source_empty_outputs=1, source_successes=4, source_finetuned_development_observations=2,
            source_finetuned_held_out_year_observations=2))
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)
        self.assertEqual(self.frames['responses'].trial.max(), 2)

    def test_independent_check_rejects_changed_grades_links_indices_and_output(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perumedqa, _perumedqa_sources
        source = _perumedqa_sources(self.directory)
        for change in ['grade', 'null_to_zero', 'subject', 'item', 'criterion', 'verifier', 'question',
                       'trace', 'position', 'bank_index', 'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade': frames['responses'].loc[0, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'
                    current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'criterion':
                    data = json.loads(frames['items'].loc[0, 'grading_criterion']); data['reference_answer'] = 'D'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'verifier':
                    data = json.loads(frames['items'].loc[0, 'verifier']); data['class'] = 'judge'
                    frames['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change == 'question': frames['items'].loc[0, 'content'] = frames['items'].loc[0, 'content'].split('\n')[0]
                elif change in ['trace', 'position', 'bank_index']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'trace': data['native_record']['answer_llm'] = data['native_record']['answer_llm'][:16000]
                    elif change == 'position': data['source_row'] += 1
                    else: data['source_dataset_row'] += 1
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition':
                    data = json.loads(frames['responses'].loc[0, 'test_condition']); data['fine_tuning_partition'] = 'held_out_year'
                    frames['responses'].loc[0, 'test_condition'] = json.dumps(data)
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _perumedqa(self.directory, frames, self.metadata, source)

    def test_shifted_chunk_index_cannot_silently_join_to_another_question(self):
        import contextlib
        import io
        path = self.directory / 'raw/02.LLMs_x20_Results/10.OctoMed/Chunk_2/results.parquet'
        frame = pd.read_parquet(path).reset_index(drop=True)
        frame.to_parquet(path)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'question-bank row'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])

    def test_published_summary_is_an_independent_denominator_check(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _perumedqa, _perumedqa_sources
        source = _perumedqa_sources(self.directory)
        source['summary'][0]['total_rows'] = '2'
        with self.assertRaisesRegex(ValueError, 'denominator'):
            _perumedqa(self.directory, self.frames, self.metadata, source)


class MorphKVNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'morphkv'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/morphkv'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        models = ['Llama_1000_snapkv', 'Qwen30B_1000_snapkv']
        self.metadata['build']['parameters']['runs'] = dict.fromkeys(models, 'fixture')
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        bank = [dict(prompt='Write a diary.\n#*# Week 1:', type='Week', number=3, prefix='#*# Week 1:',
            checks_once={'1': 'rain', '3': 'sports'}, checks_range={'2': 'music'}, checks_periodic={'1': 'sleep'}),
            dict(prompt='Design a building.\n#*# Floor 1:', type='Floor', number=2, prefix='#*# Floor 1:',
                checks_once={'1': 'lobby'}, checks_range={}, checks_periodic={'2': 'lift'}),
            dict(prompt='Plan menus.\n#*# Menu Week 1:', type='Menu Week', number=2, prefix='#*# Menu Week 1:',
                checks_once={'2': 'beans'}, checks_range={}, checks_periodic={'1': 'fruit'})]
        bank_path = self.directory / 'raw/LongGenBench/Dataset/Dataset_short.json'
        bank_path.parent.mkdir(parents=True)
        bank_path.write_text(json.dumps(bank))
        blocks = [['Introduction', 'Week 1 first ' + '完整 explanation ' * 2000, 'Week 1 duplicate context', 'Week 2 music'],
                  ['Floor 1 lobby', 'Floor 2 lift'], ['Menu Week 1 fruit', 'Menu Week 2 beans']]
        for model in models:
            records = []
            indices = [0, 1, 0] if model == models[0] else [0, 2]
            for position, index in enumerate(indices):
                row = {key: value for key, value in bank[index].items() if key not in ['prompt', 'prefix']}
                row.update(input=bank[index]['prompt'], output_blocks=blocks[index], word_count=123,
                           source_note='Keep every original field.')
                if model == models[0]:
                    grades = [({'1': 'yes'}, {'2': 'no'}, {'1': 'yes'}),
                              ({'1': 'no'}, {}, {'2': 'yes'}),
                              ({'1': 'no'}, {'2': 'yes'}, {'1': 'no'})][position]
                    for kind, values in zip(['once', 'range', 'periodic'], grades):
                        row['results_' + kind] = values
                        row['count_' + kind] = len(values)
                records.append(row)
            path = self.directory / 'raw/LongGenBench/Evalution/paper_results' / (model + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records, ensure_ascii=False))
        pd.DataFrame([{'Model': models[0], 'Completion Rate': (200 / 3 + 100 + 200 / 3) / 3,
            'Accuracy Once': 1 / 3, 'Accuracy Range': .5, 'Accuracy Periodic': 2 / 3, 'Average Accuracy': .5}]).to_csv(
            self.directory / 'raw/LongGenBench/Evalution/paper_results/paper_results.csv', index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['MorphKV']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_constraint_identity_complete_outputs_and_ungraded_generations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _morphkv
        self.assertEqual(_morphkv(self.directory, self.frames, self.metadata), dict(
            source_responses=13, source_subjects=2, source_items=7, source_traces=13, source_files=2,
            source_declared_constraints=16, source_absent_block_constraints=3, source_native_attempts=5,
            source_summary_rows=1, source_bank_rows=3, source_generation_tasks=3,
            source_graded_observations=8, source_ungraded_observations=5, source_successes=4))
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)
        self.assertEqual(self.frames['responses'].trial.max(), 2)

    def test_audit_rejects_corrupt_grades_links_constraints_and_traces(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _morphkv, _morphkv_sources
        source = _morphkv_sources(self.directory)
        for change in ['grade', 'equal_sum_swap', 'null_to_zero', 'subject', 'item', 'criterion', 'verifier', 'prompt',
                       'trace', 'position', 'bank_index', 'constraint', 'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade': frames['responses'].loc[0, 'response'] = 0.
                elif change == 'equal_sum_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    frames['responses'].loc[0, 'response'] = 0.; frames['responses'].loc[zero, 'response'] = 1.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'
                    current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'criterion':
                    data = json.loads(frames['items'].loc[0, 'grading_criterion']); rule = json.loads(data['rule']); rule['requirement'] = 'changed'
                    data['rule'] = json.dumps(rule); frames['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'verifier':
                    data = json.loads(frames['items'].loc[0, 'verifier']); data['class'] = 'exact_matcher'
                    frames['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change == 'prompt': frames['items'].loc[0, 'content'] = frames['items'].loc[0, 'content'].split('\n')[0]
                elif change in ['trace', 'position', 'bank_index', 'constraint']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'trace': data['native_record']['output_blocks'][1] = data['native_record']['output_blocks'][1][:16000]
                    elif change == 'position': data['source_row'] += 1
                    elif change == 'bank_index': data['bank_row'] += 1
                    else: data['block_id'] = '3'
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'another run'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _morphkv(self.directory, frames, self.metadata, source)

    def test_judgment_for_absent_block_cannot_be_silently_discarded(self):
        import contextlib
        import io
        path = self.directory / 'raw/LongGenBench/Evalution/paper_results/Llama_1000_snapkv.json'
        records = json.loads(path.read_text()); records[0]['results_once']['3'] = 'no'; path.write_text(json.dumps(records))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'matching generated-block constraint'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])

    def test_bank_grading_changes_require_review(self):
        import contextlib
        import io
        path = self.directory / 'raw/LongGenBench/Dataset/Dataset_short.json'
        bank = json.loads(path.read_text()); bank[0]['checks_once']['1'] = 'different requirement'; path.write_text(json.dumps(bank))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'original bank definition'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])


class MORQANativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'morqa'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/morqa'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        parameters = self.metadata['build']['parameters']
        selected = ['en/iiyi/test', 'zh/woundcare/valid', 'en/liveqa/test']
        parameters['result_files'] = {key: parameters['result_files'][key] for key in selected}
        parameters['rating_files'] = {key: parameters['rating_files'][key] for key in selected}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        for name, relative in parameters['question_files'].items():
            dataset, split = name.split('/')
            question = dict(encounter_id='q1', post_id='q1', split='valid' if split == 'all' else split,
                query_title_en='Medical question', query_content_en='Full patient query.',
                query_title_zh='问题', query_content_zh='完整问题。')
            path = self.directory / 'raw' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([question], ensure_ascii=False))
        gpt, gemini, deepseek = 'gpt-4o_##__ratings_prompt', 'gemini-1.5-pro_##__ratings_prompt', 'deepseekv3_##__ratings_prompt'
        long_candidate = 'Complete native answer. ' + '完整 and untruncated. ' * 1500
        cases = {
            selected[0]: [('a', long_candidate, {gpt: .75, gemini: .5}), ('b', '', {gpt: 0., gemini: 1.})],
            selected[1]: [('a', '中文答复', {gpt: .9}), ('b', '待评分答复', {gpt: None, deepseek: 1.})],
            selected[2]: [('a', 'LiveQA answer', {gpt: 1., gemini: 0.})]}
        for collection, samples in cases.items():
            lang, dataset, split = collection.split('/')
            records, annotations = [], []
            for author, candidate, ratings in samples:
                records.append(dict(post_id='q1', author_id_candidate=author, candidate=candidate,
                    responses=[dict(author_id='expert', **{'content_' + lang: 'Native reference'})],
                    auxiliary_metric=0.12345678901234568, **ratings))
                metrics = [('overall', 'rater1', -2 if dataset == 'liveqa' else .5)]
                if collection == selected[0] and author == 'a':
                    metrics += [('overall', 'rater2', 1.), ('style', 'rater1', .5), ('style', 'rater2', 0.)]
                for metric, rater, value in metrics:
                    annotations.append(dict(dataset=dataset, encounter_id='q1', post_id='q1', lang=lang,
                        system_input=candidate, author_id=author, metric=metric, author_metric=rater, value=value))
            for key, data in [('result_files', records), ('rating_files', annotations)]:
                path = self.directory / 'raw' / parameters[key][collection]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['MORQA']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_original_ratings_empty_answers_nulls_and_complete_references(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _morqa
        self.assertEqual(_morqa(self.directory, self.frames, self.metadata), dict(
            source_responses=9, source_subjects=3, source_items=5, source_traces=9, source_result_files=3,
            source_candidates=5, source_empty_candidates=1, source_human_annotations=8,
            source_off_rubric_ratings=2, source_ungraded_ratings=1, source_empty_candidate_ratings=2))
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)

    def test_audit_rejects_wrong_ratings_inputs_and_source_associations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _morqa, _morqa_sources
        source = _morqa_sources(self.directory)
        for change in ['snap', 'equal_sum_swap', 'null_to_zero', 'subject', 'item', 'query', 'candidate', 'references',
                       'human_rating', 'verifier', 'trace', 'position', 'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'snap': frames['responses'].loc[frames['responses'].response.eq(.75), 'response'] = .5
                elif change == 'equal_sum_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change in ['query', 'candidate', 'references']:
                    data = json.loads(frames['items'].loc[0, 'content'])
                    data[{'query': 'query_content', 'candidate': 'candidate', 'references': 'references'}[change]] = ''
                    frames['items'].loc[0, 'content'] = json.dumps(data)
                elif change == 'human_rating':
                    data = json.loads(frames['items'].loc[0, 'grading_criterion'])
                    annotations = json.loads(data['reference_answer']); annotations[0]['native_record']['value'] = 999
                    data['reference_answer'] = json.dumps(annotations); frames['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'verifier':
                    data = json.loads(frames['items'].loc[0, 'verifier']); data['class'] = 'exact_matcher'
                    frames['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change in ['trace', 'position']:
                    index = frames['traces'].trace.str.len().idxmax() if change == 'trace' else 0
                    data = json.loads(frames['traces'].loc[index, 'trace'])
                    if change == 'trace': data['native_record']['candidate'] = data['native_record']['candidate'][:16000]
                    else: data['source_row'] += 1
                    frames['traces'].loc[index, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'other'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _morqa(self.directory, frames, self.metadata, source)

    def test_changed_annotation_text_cannot_attach_to_the_wrong_candidate(self):
        import contextlib
        import io
        relative = self.metadata['build']['parameters']['rating_files']['en/iiyi/test']
        path = self.directory / 'raw' / relative
        records = json.loads(path.read_text()); records[0]['system_input'] = 'different answer'; path.write_text(json.dumps(records))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'exact released candidate text'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])

    def test_duplicate_human_identity_requires_review(self):
        import contextlib
        import io
        relative = self.metadata['build']['parameters']['rating_files']['en/iiyi/test']
        path = self.directory / 'raw' / relative
        records = json.loads(path.read_text()); records.append(records[0]); path.write_text(json.dumps(records))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Repeated human annotation identity'):
            self.builder(str(self.directory / 'build.py')).main_from_args(
                ['--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'invalid')])


if __name__ == "__main__":
    unittest.main()
