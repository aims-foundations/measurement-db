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


class MTBBenchNativeAuditTests(unittest.TestCase):
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
        self.directory = Path(temporary.name) / 'mtbbench'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/mtbbench'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        raw = self.directory / 'raw'
        for cohort, case in [('hancock', '104'), ('msk', 'P-1')]:
            first, later = f'data/{cohort}/{case}/history.txt', f'data/{cohort}/{case}/later.txt'
            for relative, content in [(first, 'Original patient information: ' + cohort), (later, 'Later information')]:
                path = raw / 'tasks' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            events = [dict(context='Patient context ' + cohort), dict(file_paths=[first]),
                dict(question='Repeated question text?', answer='A) first'),
                dict(context='Later clinical event'), dict(file_paths=[later]),
                dict(question='Next question?', answer='B) second')]
            filename = self.metadata['build']['parameters']['questions'][cohort]
            (raw / 'tasks' / filename).write_text(json.dumps({case: events}))
            for model, grades in [('model-a', [True, False]), ('model-b', [None, True])]:
                conversation = [dict(role='system', content='Protocol ' + (cohort if model == 'model-b' else 'shared')),
                    dict(role='user', content='Repeated question text?'),
                    dict(role='assistant', content='Full answer 完整 ' * 2000),
                    dict(role='user', content='Earlier file was accessed by you'),
                    dict(role='user', content='Next question?'), dict(role='assistant', content='[ANSWER: B) second]')]
                records = [dict(question=events[index]['question'], answer=events[index]['answer'],
                    response='Native answer', files_accessed=[first], files_hallucinated=[],
                    question_time=0.12345678901234568, **({} if grade is None else dict(correct=grade)))
                    for index, grade in zip([2, 5], grades)]
                records.append(dict(conversation=conversation))
                path = raw / 'release' / ('agent_logs_' + cohort) / model / (case + '_chatlog_2025.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(records, ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['MTBBench']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_case_prefixes_protocols_null_grades_and_full_conversations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _mtbbench
        self.assertEqual(_mtbbench(self.directory, self.frames, self.metadata), dict(source_responses=8,
            source_subjects=3, source_items=4, source_traces=8, source_assets=3, source_asset_paths=4,
            source_case_logs=4, source_successes=4, source_failures=2, source_ungraded=2))
        self.assertGreater(self.frames['traces'].trace.str.len().min(), 16000)

    def test_audit_rejects_corrupt_grades_context_assets_and_links(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _mtbbench, _mtbbench_sources
        source = _mtbbench_sources(self.directory)
        for change in ['grade_swap', 'null_to_zero', 'subject', 'item', 'prefix', 'future', 'reference', 'verifier',
                       'asset_bytes', 'asset_link', 'asset_role', 'system', 'trace', 'native_float', 'position',
                       'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change in ['prefix', 'future']:
                    data = json.loads(frames['items'].loc[0, 'content'])
                    if change == 'prefix': data['case_events'] = data['case_events'][-1:]
                    else: data['case_events'].append(dict(context='Unrevealed future event'))
                    frames['items'].loc[0, 'content'] = json.dumps(data)
                elif change == 'reference':
                    data = json.loads(frames['items'].loc[0, 'grading_criterion']); data['reference_answer'] = 'C) other'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(data)
                elif change == 'verifier':
                    data = json.loads(frames['items'].loc[0, 'verifier']); data['class'] = 'judge'
                    frames['items'].loc[0, 'verifier'] = json.dumps(data)
                elif change == 'asset_bytes': frames['assets'].at[0, 'data'] = b'Altered patient file'
                elif change in ['asset_link', 'asset_role']:
                    links = json.loads(frames['items'].loc[0, 'asset_manifest'])
                    if change == 'asset_link': links[0]['path'] = 'future.txt'
                    else: links[0]['role'] = 'input'
                    frames['items'].loc[0, 'asset_manifest'] = json.dumps(links)
                elif change == 'system':
                    value = frames['subjects'].loc[0, 'subject_features_extra']
                    self.assertIn('Protocol shared', value)
                    frames['subjects'].loc[0, 'subject_features_extra'] = value.replace('Protocol shared', 'Changed')
                elif change in ['trace', 'native_float', 'position']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'trace': data['conversation'][2]['content'] = data['conversation'][2]['content'][:16000]
                    elif change == 'native_float': data['native_record']['question_time'] = .12
                    else: data['source_row'] += 1
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'other'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _mtbbench(self.directory, frames, self.metadata, source)

    def test_changed_task_reference_is_rejected(self):
        import contextlib
        import io
        path = self.directory / 'raw/tasks/questions_hancock_bench.json'
        bank = json.loads(path.read_text()); bank['104'][2]['answer'] = 'C) changed'; path.write_text(json.dumps(bank))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'differs from the frozen case bank'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])


class RealPOCQiNativeAuditTests(unittest.TestCase):
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
        self.directory = Path(temporary.name) / 'real_pocqi'
        raw = self.directory / 'raw/release'
        raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/real_pocqi'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        questions = [dict(question_id='q1', question_text='Full clinical question 完整', specialty='Cardiology'),
                     dict(question_id='q2', question_text='A different original question', specialty='Neurology')]
        answers = [dict(question_id=question, provider_key=provider,
            answer_markdown=('  Full untruncated answer 完整. ' * 1500 if (question, provider) == ('q1', 'model-a') else
                '  Complete answer for ' + question + ' / ' + provider + '  '))
            for question in ['q1', 'q2'] for provider in ['model-a', 'model-b']]
        ratings = [dict(question_id=question, axis=axis, choice=choice, slot_a_provider='model-a',
            slot_b_provider='model-b', render_mode='qa_text_citations' if question == 'q1' else 'qa_text_only')
            for question, axis, choice in [('q1', 'accuracy', 'strongly_a'), ('q1', 'accuracy', 'strongly_a'),
                ('q1', 'clinical_utility', 'tie'), ('q2', 'accuracy', 'slightly_b'), ('q2', 'accuracy', None)]]
        for name, records in [('questions', questions), ('answers', answers), ('ratings', ratings)]:
            pd.DataFrame(records).to_parquet(raw / (name + '.parquet'), index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['RealPOCQi']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(raw.parent), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_both_perspectives_complete_answers_nulls_and_repeated_votes(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _real_pocqi
        self.assertEqual(_real_pocqi(self.directory, self.frames, self.metadata), dict(source_responses=10,
            source_subjects=2, source_items=3, source_traces=10, source_questions=2, source_answers=4,
            source_rating_rows=5, source_identical_rating_rows=1, source_long_answers=1,
            source_wins=3, source_losses=3, source_ties=2, source_ungraded=2))

    def test_audit_rejects_changed_votes_dimensions_answers_and_associations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _real_pocqi, _real_pocqi_sources
        source = _real_pocqi_sources(self.directory)
        for change in ['grade_swap', 'null_to_tie', 'subject', 'item', 'query', 'axis', 'verifier', 'strength',
                       'answer', 'opponent_answer', 'position', 'slot', 'condition', 'opponent', 'trial',
                       'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_tie': frames['responses']['response'] = frames['responses'].response.fillna(.5)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'query': frames['items'].loc[0, 'content'] = 'Wrong question'
                elif change == 'axis':
                    criterion = json.loads(frames['items'].loc[0, 'grading_criterion']); criterion['rule'] = 'Different grading dimension'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
                elif change == 'verifier':
                    verifier = json.loads(frames['items'].loc[0, 'verifier']); verifier['judged_by'] = 'llm'
                    frames['items'].loc[0, 'verifier'] = json.dumps(verifier)
                elif change in ['strength', 'answer', 'opponent_answer', 'position', 'slot']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'strength': data['native_rating']['choice'] = 'slightly_a'
                    elif change == 'answer': data['native_answer']['answer_markdown'] = data['native_answer']['answer_markdown'][:12000]
                    elif change == 'opponent_answer': data['opponent_answer']['answer_markdown'] = 'Another answer'
                    elif change == 'position': data['source_row'] += 1
                    else: data['slot'] = 'slot_b_provider'
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = '{}'
                elif change == 'opponent': frames['responses'].loc[0, 'interactors'] = 'opponent=other'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _real_pocqi(self.directory, frames, self.metadata, source)

    def test_unrecognized_choice_is_not_silently_a_tie(self):
        import contextlib
        import io
        path = self.directory / 'raw/release/ratings.parquet'
        ratings = pd.read_parquet(path); ratings.loc[0, 'choice'] = 'unrecognized'; ratings.to_parquet(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Unknown native preference'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])

    def test_missing_opponent_answer_cannot_be_silently_dropped(self):
        import contextlib
        import io
        path = self.directory / 'raw/release/answers.parquet'
        pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'no released answer'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])


class VisualRiddlesNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from PIL import Image
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'visual_riddles'
        raw = self.directory / 'raw/release/test'
        raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/visual_riddles'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        records = []
        for image_id, color, grade in [('image-a', 'red', True), ('image-b', 'blue', None)]:
            Image.new('RGB', (2, 2), color=color).save(raw / (image_id + '.jpg'))
            answers = [dict(type='LVLM', model_name='same-model', human_rating=grade,
                model_answer='  A complete answer 完整 ' * 2000),
                dict(type='gemini-1.5-pro-caption_LLM', model_name='same-model', human_rating=False, model_answer='Different pipeline'),
                dict(type='human', model_name='human_1', human_rating=True, model_answer='Human response')]
            if image_id == 'image-a':
                answers.append(dict(type='human-caption_LLM', model_name='same-model', human_rating=True, model_answer='Caption answer'))
            records.append(dict(file_name=image_id + '.jpg', image_id=image_id, question='Same question for different images',
                ground_truth_answer='Reference for ' + image_id, category='World knowledge', difficulty_level_index='2',
                **{'human-caption': '  Original human caption 完整 ' + image_id + '  ',
                   'model_rated_answers-open_ended': repr(answers)}))
        pd.DataFrame(records).to_csv(raw / 'metadata.csv', index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['VisualRiddles']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_images_captions_pipelines_full_answers_and_missing_grades(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _visual_riddles
        self.assertEqual(_visual_riddles(self.directory, self.frames, self.metadata), dict(source_responses=5,
            source_subjects=3, source_items=3, source_traces=5, source_riddles=2, source_assets=2,
            source_caption_items=1, source_human_responses=2, source_direct_responses=2,
            source_human_caption_responses=1, source_generated_caption_responses=2,
            source_successes=2, source_failures=2, source_ungraded=1))

    def test_audit_rejects_corrupt_grades_inputs_images_and_source_associations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _visual_riddles, _visual_riddles_sources
        source = _visual_riddles_sources(self.directory)
        for change in ['grade_swap', 'null_to_zero', 'subject', 'item', 'question', 'caption', 'reference', 'verifier',
                       'image', 'image_link', 'role', 'answer', 'position', 'entry', 'condition', 'trial',
                       'drop', 'duplicate', 'drop_trace', 'settings']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'question': frames['items'].loc[0, 'content'] = 'Another question'
                elif change == 'caption':
                    index = frames['items'].index[frames['items'].raw_item_id.str.endswith(':human_caption')][0]
                    content = json.loads(frames['items'].loc[index, 'content']); content['human_caption'] = 'Wrong caption'
                    frames['items'].loc[index, 'content'] = json.dumps(content)
                elif change == 'reference':
                    criterion = json.loads(frames['items'].loc[0, 'grading_criterion']); criterion['reference_answer'] = 'Wrong reference'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
                elif change == 'verifier':
                    verifier = json.loads(frames['items'].loc[0, 'verifier']); verifier['judged_by'] = 'llm'
                    frames['items'].loc[0, 'verifier'] = json.dumps(verifier)
                elif change == 'image': frames['assets'].loc[0, 'data'] = bytes(frames['assets'].loc[0, 'data']) + b'corruption'
                elif change in ['image_link', 'role']:
                    links = json.loads(frames['items'].loc[0, 'asset_manifest'])
                    if change == 'role': links[0]['role'] = 'source'
                    else: links[0]['asset_id'] = frames['assets'].loc[frames['assets'].asset_id.ne(links[0]['asset_id']), 'asset_id'].iloc[0]
                    frames['items'].loc[0, 'asset_manifest'] = json.dumps(links)
                elif change in ['answer', 'position', 'entry']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'answer': data['native_record']['model_answer'] = data['native_record']['model_answer'][:12000]
                    elif change == 'position': data['source_row'] += 1
                    else: data['source_entry'] += 1
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'Other protocol'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _visual_riddles(self.directory, frames, self.metadata, source)

    def test_nonboolean_source_grade_is_not_coerced_to_success(self):
        import ast
        import contextlib
        import io
        path = self.directory / 'raw/release/test/metadata.csv'
        table = pd.read_csv(path, keep_default_na=False)
        answers = ast.literal_eval(table.loc[0, 'model_rated_answers-open_ended']); answers[0]['human_rating'] = 'false'
        table.loc[0, 'model_rated_answers-open_ended'] = repr(answers); table.to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'must be boolean'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])

    def test_unknown_source_pipeline_cannot_be_silently_dropped(self):
        import ast
        import contextlib
        import io
        path = self.directory / 'raw/release/test/metadata.csv'
        table = pd.read_csv(path, keep_default_na=False)
        answers = ast.literal_eval(table.loc[0, 'model_rated_answers-open_ended']); answers[0]['type'] = 'unknown'
        table.loc[0, 'model_rated_answers-open_ended'] = repr(answers); table.to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Unrecognized released answer type'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])


class SustainableFoodNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import html
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'sustainable_food'
        self.raw = self.directory / 'raw/release/recipe-rating-prediction'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/sustainable_food'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        paragraphs = self.metadata['build']['parameters']['instructions']['prefix'].split('\n\n')
        (self.directory / 'raw/paper.html').write_text(''.join('<span id="A4.F10.pic1.2.' + str(i) + '.1">' +
            html.escape(text) + '</span>' for i, text in enumerate(paragraphs, 1)))
        models = list(self.metadata['build']['parameters']['parsers'])
        output_columns = [['Answer: 1', '2', '1'], ['', '1', '2'],
            ['Recipe 2 appears before Recipe 1. ' + 'Complete answer 完整. ' * 2000, 'Recipe 2', '1'],
            ['2', '1', 'not a selection'], ['1', '1', '2'], ['2 explanation', '2', '1']]
        pairs, outputs = [], []
        for position, (key, gold) in enumerate([('9', '1.0'), ('4', '2.0'), ('2', '2.0')]):
            pairs.append({'': key, 'index': str(100 + position), 'id_1': 'first-' + key, 'id_2': 'second-' + key,
                'text_1': '  Recipe one 完整 for ' + key + '\n', 'text_2': '\nRecipe two for ' + key + '  ',
                'ground_truth': gold, 'actual_score_1': '4.8' if gold == '1.0' else '3.9', 'actual_score_2': '4.5'})
            outputs.append({'': key, **{model: output_columns[index][position] for index, model in enumerate(models)}})
        pd.DataFrame(pairs).to_csv(self.raw / 'pairs_metadata.csv', index=False)
        pd.DataFrame(outputs[::-1]).to_csv(self.raw / 'collected_outputs.csv', index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['SustainableFood']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_reordered_keys_native_parser_precedence_and_ungraded_attempts(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _sustainable_food
        self.assertEqual(_sustainable_food(self.directory, self.frames, self.metadata), dict(source_responses=18,
            source_subjects=6, source_items=3, source_traces=18, source_successes=8, source_failures=8,
            source_ungraded=2, source_missing_outputs=1, source_present_outputs=17, source_unparseable_outputs=1))

    def test_audit_rejects_corrupt_grades_prompts_outputs_and_joins(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _sustainable_food, _sustainable_food_sources
        source = _sustainable_food_sources(self.directory)
        for change in ['grade_swap', 'null_to_zero', 'subject', 'item', 'instructions', 'recipe', 'reference', 'verifier',
                       'output', 'pair', 'position', 'row_key', 'column', 'condition', 'trial',
                       'drop', 'duplicate', 'drop_trace', 'settings', 'request_options']:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'instructions': frames['items'].loc[0, 'content'] = frames['items'].loc[0, 'content'].split('Recipe 1:')[1]
                elif change == 'recipe': frames['items'].loc[0, 'content'] += ' Wrong added input'
                elif change == 'reference':
                    criterion = json.loads(frames['items'].loc[0, 'grading_criterion']); criterion['reference_answer'] = 'Wrong recipe'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
                elif change == 'verifier':
                    verifier = json.loads(frames['items'].loc[0, 'verifier']); verifier['class'] = 'judge'
                    frames['items'].loc[0, 'verifier'] = json.dumps(verifier)
                elif change in ['output', 'pair', 'position', 'row_key', 'column']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'output': data['native_output'] += ' altered'
                    elif change == 'pair': data['native_pair']['id_1'] = 'other recipe'
                    elif change == 'position': data['source_row'] += 1
                    elif change == 'row_key': data['row_key'] = 'other row'
                    else: data['source_column'] = 'other-model'
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'Other release'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                elif change == 'settings': frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                else:
                    text = frames['subjects'].loc[0, 'subject_features_extra']
                    self.assertIn('"echo_prompt": false', text)
                    frames['subjects'].loc[0, 'subject_features_extra'] = text.replace('"echo_prompt": false', '"echo_prompt": true')
                with self.assertRaises((ValueError, KeyError)):
                    _sustainable_food(self.directory, frames, self.metadata, source)

    def test_unmatched_source_rows_cannot_be_silently_truncated(self):
        import contextlib
        import io
        path = self.raw / 'collected_outputs.csv'
        pd.read_csv(path, keep_default_na=False).iloc[:-1].to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'exactly matching row keys'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])

    def test_unknown_model_column_cannot_be_silently_dropped(self):
        import contextlib
        import io
        path = self.raw / 'collected_outputs.csv'
        table = pd.read_csv(path, keep_default_na=False); table['another-model'] = '1'; table.to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'model columns differ'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])


class SugarCrepeNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from zipfile import ZipFile
        from PIL import Image
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'sugarcrepe'
        self.raw = self.directory / 'raw'
        (self.raw / 'coco').mkdir(parents=True)
        folder = ROOT / 'benchmarks/sugarcrepe'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        images = []
        with ZipFile(self.raw / 'coco/val2017.zip', 'w') as archive:
            for index, color in enumerate(['red', 'blue'], 1):
                name = f'{index:012d}.jpg'
                data = io.BytesIO()
                Image.new('RGB', (2, 2), color=color).save(data, format='JPEG')
                archive.writestr('val2017/' + name, data.getvalue())
                images.append(dict(id=index, file_name=name, license=index, flickr_url='https://example.test/' + name))
        licenses = [dict(id=1, name='Attribution', url='https://creativecommons.org/licenses/by/2.0/'),
                    dict(id=2, name='Attribution Noncommercial', url='https://creativecommons.org/licenses/by-nc/2.0/')]
        with ZipFile(self.raw / 'coco/annotations_trainval2017.zip', 'w') as archive:
            archive.writestr('annotations/captions_val2017.json', json.dumps(dict(images=images, licenses=licenses)))
        for category in ['add_obj', 'swap_obj']:
            bank = {}
            for order in ['negative-first', 'positive-first']:
                records = {}
                for key, image in zip(['0', '108'], images):
                    grade = None if (category, order, key) == ('swap_obj', 'negative-first', '108') else order == 'positive-first'
                    caption, negative = '  Original caption 完整 ' + key + '  ', 'Other caption ' + key
                    records[key] = dict(filename=image['file_name'], caption=caption, negative_caption=negative,
                        correct=grade, answer=dict(free_form_answer='  Complete model output 完整 ' * 2000,
                            multiple_choice_answer=caption if grade else negative))
                    if category != 'swap_obj' or key != '108':
                        bank[key] = dict(filename=image['file_name'], caption=caption, negative_caption=negative)
                path = self.raw / 'release/gpt-4v-results' / order / ('gpt4v-' + category + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(records, accuracy=0.5), ensure_ascii=False))
            path = self.raw / 'release/data' / (category + '.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(bank, ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['SugarCrepe']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.raw), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_ordered_inputs_images_full_answers_nulls_and_historical_records(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _sugarcrepe
        self.assertEqual(_sugarcrepe(self.directory, self.frames, self.metadata), dict(source_responses=8,
            source_subjects=1, source_items=8, source_traces=8, source_assets=2, source_categories=2,
            source_correct=4, source_incorrect=3, source_ungraded=1, source_historical_only_records=2,
            source_overlapping_stimulus_records=4, source_distinct_ordered_stimuli=4))

    def test_audit_rejects_corrupted_observations_and_inputs(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _sugarcrepe, _sugarcrepe_sources
        source = _sugarcrepe_sources(self.directory)
        changes = ['grade_swap', 'null_to_zero', 'subject', 'item', 'prompt', 'order', 'reference', 'verifier',
            'image', 'image_link', 'role', 'answer', 'source_key', 'source_file', 'attribution', 'license',
            'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change == 'subject': frames['responses'].loc[0, 'subject_id'] = 'incorrect_model'
                elif change == 'item': frames['responses'].loc[0, 'item_id'] = frames['items'].loc[1, 'item_id']
                elif change == 'prompt': frames['items'].loc[0, 'content'] += '\nThe correct caption is marked positive.'
                elif change == 'order': frames['items'].loc[0, 'item_features'] = frames['items'].loc[0, 'item_features'].replace('negative-first', 'positive-first')
                elif change == 'reference':
                    criterion = json.loads(frames['items'].loc[0, 'grading_criterion']); criterion['reference_answer'] = 'Wrong reference'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
                elif change == 'verifier':
                    verifier = json.loads(frames['items'].loc[0, 'verifier']); verifier['class'] = 'judge'
                    frames['items'].loc[0, 'verifier'] = json.dumps(verifier)
                elif change == 'image': frames['assets'].loc[0, 'data'] = bytes(frames['assets'].loc[0, 'data']) + b'corruption'
                elif change in ['image_link', 'role']:
                    links = json.loads(frames['items'].loc[0, 'asset_manifest'])
                    if change == 'role': links[0]['role'] = 'source'
                    else: links[0]['asset_id'] = frames['assets'].loc[frames['assets'].asset_id.ne(links[0]['asset_id']), 'asset_id'].iloc[0]
                    frames['items'].loc[0, 'asset_manifest'] = json.dumps(links)
                elif change in ['answer', 'source_key', 'source_file', 'attribution', 'license']:
                    trace = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'answer': trace['native_record']['answer']['free_form_answer'] = trace['native_record']['answer']['free_form_answer'][:16000]
                    elif change == 'source_key': trace['source_key'] = '108'
                    elif change == 'source_file': trace['source_file'] = trace['source_file'].replace('negative-first', 'positive-first')
                    elif change == 'attribution': trace['coco_image']['flickr_url'] = 'https://example.test/wrong'
                    else: trace['coco_license']['name'] = 'Wrong license'
                    frames['traces'].loc[0, 'trace'] = json.dumps(trace)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'wrong_order'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _sugarcrepe(self.directory, frames, self.metadata, source)

    def test_source_checker_rejects_invalid_flags_bank_drift_and_missing_images(self):
        from zipfile import ZipFile
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _sugarcrepe_sources
        path = self.raw / 'release/gpt-4v-results/positive-first/gpt4v-add_obj.json'
        original = path.read_text()
        for change in ['flag', 'bank', 'grade']:
            record = json.loads(original)
            if change == 'flag': record['0']['correct'] = 'false'
            elif change == 'bank': record['0']['filename'] = '000000000002.jpg'
            else: record['0']['correct'] = False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): _sugarcrepe_sources(self.directory)
            path.write_text(original)
        with ZipFile(self.raw / 'coco/val2017.zip', 'w'): pass
        with self.assertRaises(KeyError): _sugarcrepe_sources(self.directory)


class LiveAgentRiskNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'live_agent_risk'
        self.directory.mkdir()
        (self.directory / 'raw').mkdir()
        folder = ROOT / 'benchmarks/live_agent_risk'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        parameters = self.metadata['build']['parameters']
        self.files, self.records = {}, {}
        self.game = 'experiments/nested/game__fixture'
        self.result_file = self.game + '/end_game_results.json'
        players = [dict(name=name, model='gpt-4o', provider='OpenAI', reasoning_effort='medium',
            turn_time_limit_seconds=90, placement_reasoning_effort='low', planning_reasoning_effort='high') for name in ['Seat A', 'Seat B']]
        self.records[self.result_file] = dict(winner='Seat A', victory_condition='Recorded rule', total_rounds=3,
            players=[dict(name='Seat A', territories_controlled=1), dict(name='Seat B', territories_controlled=41)], original_field='unchanged')
        self.records[self.game + '/game_manifest.json'] = dict(players=players,
            rules=dict(max_rounds=3, territory_control_percentage=0.65), git_revision='reported_revision',
            include_initial_troop_placement=True, generated_at_utc='original timestamp')
        self.turn_file = self.game + '/turn_summary_turn_1.json'
        self.records[self.turn_file] = dict(player=dict(players[0]), turn_number=1,
            events=[dict(action='original invalid action', error='retained')], plan=dict(original='unmodified'))
        self.call_file = 'experiments/nested/llm_interactions/game__fixture/Seat_A/round_01/turn_0001/0001_attack.json'
        self.records[self.call_file] = dict(player='Seat A', model='gpt-4o', provider='OpenAI', phase='attack',
            interaction_index=1, request=dict(prompt='Original full prompt\n\u2028' * 1200, timeout_seconds=25),
            response=dict(raw_response='def incomplete(', error='Original API error', used_fallback_response=True),
            original_unknown_field='retain exactly')
        for i, (name, field) in enumerate(parameters['supplements'].items()):
            game = 'summary_game_' + str(i)
            self.records[name] = dict(game_folder='game_results/' + game,
                config=dict(max_rounds=5 + i), **{field: [dict(name='Seat C', model='gpt-4.1', provider='OpenAI', reasoning_effort='none'),
                    dict(name='Seat D', model='gpt-5', provider='OpenAI', reasoning_effort='high')]}, winner='Seat D')
            self.records[game + '/end_game_results.json'] = dict(winner='Seat D', victory_condition='Native rule', total_rounds=5,
                players=[dict(name='Seat C'), dict(name='Seat D')], original_field='unchanged')
            self.records[game + '/turn_summary_turn_1.json'] = dict(player=dict(name='Seat C', model='gpt-4.1',
                provider='OpenAI', reasoning_effort=None), original_setting=None)
        self.records['unknown_game/end_game_results.json'] = dict(winner='nano-medium-b', victory_condition='Original rule',
            players=[dict(name=name) for name in ['nano-medium-a', 'nano-medium-b', 'nano-medium-c']], original_field='unchanged')
        self.records['calibration/end_game_results.json'] = dict(winner='Alpha', players=[dict(name='Alpha'), dict(name='Bravo')])
        self.records['llm_interactions/uncompleted_game/Seat_A/0001.json'] = dict(player='Seat A', model='gpt-4o', provider='OpenAI',
            response=dict(raw_response='An ungraded probe'), request=dict(prompt='Probe only'))
        for name in list(self.records):
            if name.endswith('/end_game_results.json'):
                self.files[str(Path(name).parent) + '/game_state_turn_0.csv'] = b'Territory,Post-placement troops\nOriginal territory,17\n'
        layout = parameters['layout']
        with tarfile.open(self.directory / 'raw' / layout['source'], 'w:gz') as archive:
            for name in parameters['resources'].values():
                data=('# Original reference ' + name).encode(); member=tarfile.TarInfo(layout['source_prefix'] + name)
                member.size=len(data); archive.addfile(member, io.BytesIO(data))
        self._write_archive()
        self.builder = runpy.run_path(str(folder / 'build.py'))['LiveAgentRisk']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'tables')])
        self.frames = {p.stem: pd.read_parquet(p) for p in (self.directory.parent / 'tables').glob('*.parquet')}

    def _write_archive(self):
        import io
        import tarfile
        with tarfile.open(self.directory / 'raw' / self.metadata['build']['parameters']['layout']['results'], 'w:gz') as archive:
            data = dict(self.files, **{name: json.dumps(value, ensure_ascii=False).encode() for name, value in self.records.items()})
            for name, body in data.items():
                member = tarfile.TarInfo('./' + name); member.size = len(body)
                archive.addfile(member, io.BytesIO(body))

    def test_complete_seats_configurations_unknown_labels_and_full_failed_calls(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _live_agent_risk
        result = _live_agent_risk(self.directory, self.frames, self.metadata)
        self.assertEqual(result['source_responses'], 9)
        self.assertEqual(result['source_items'], 4)
        self.assertEqual(result['source_placeholder_games'], 1)
        self.assertEqual(result['source_manifest_games'], 1)
        self.assertEqual(result['source_supplement_games'], 2)
        self.assertEqual(result['source_full_calls'], 1)
        self.assertEqual(result['source_fallback_calls'], 1)
        self.assertEqual(result['source_unknown_model_seats'], 3)
        self.assertEqual(result['source_long_traces'], 1)
        self.assertTrue(self.frames['responses'].trial.eq(2).any())
        self.assertFalse(self.frames['items'].content.str.contains('Post-placement troops').any())
        self.assertFalse(self.frames['items'].content.str.contains('territories_controlled').any())
        self.assertTrue(self.frames['traces'].trace.str.contains('Post-placement troops').all())
        self.assertEqual(self.frames['responses'].response.sum(), 4)

    def test_corruption_of_scores_settings_game_context_and_logs_is_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _live_agent_risk, _live_agent_risk_sources
        source = _live_agent_risk_sources(self.directory, self.metadata)
        changes = ['score', 'subject', 'item', 'trial', 'condition', 'interactors', 'drop', 'duplicate',
            'extra_subject', 'extra_item', 'extra_asset', 'model', 'configuration', 'effort', 'verified_revision',
            'scale', 'task', 'rules', 'roster', 'post_state_leak', 'winner_leak', 'reference', 'verifier',
            'raw_item_id', 'asset_bytes', 'asset_path', 'asset_role', 'trace_clip', 'missing_call', 'missing_turn',
            'missing_history', 'missing_field', 'source_file', 'seat', 'fallback', 'missing_trace']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, assets, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'assets', 'traces']]
                if change == 'score': responses.loc[0, 'response'] = 1 - responses.loc[0, 'response']
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'unknown'
                elif change == 'item': responses.loc[0, 'item_id'] = next(x for x in items.item_id if x != responses.loc[0, 'item_id'])
                elif change == 'trial': responses.loc[0, 'trial'] = 99
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'wrong seat'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented'
                elif change == 'drop': frames['responses'] = responses.iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'extra_item': frames['items'] = pd.concat([items, items.iloc[:1]], ignore_index=True)
                elif change == 'extra_asset': frames['assets'] = pd.concat([assets, assets.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'Other model'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] += ';invented=yes'
                elif change == 'effort': subjects.loc[0, 'reasoning_effort'] = 'xhigh'
                elif change == 'verified_revision': subjects.loc[0, 'harness_version'] = 'claimed-executed-code'
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=1))
                elif change in ['task', 'rules', 'roster', 'post_state_leak', 'winner_leak']:
                    value = json.loads(items.loc[0, 'content'])
                    key = dict(task='task', rules='known_rules', roster='released_roster', post_state_leak='initial_board', winner_leak='winner')[change]
                    value[key] = 'wrong or leaked context'; items.loc[0, 'content'] = json.dumps(value, sort_keys=True)
                elif change == 'reference': items.loc[0, 'grading_criterion'] = json.dumps(dict(reference_answer='Seat A', rule='guess winner'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(**{'class': 'judge'}, spec='{}'))
                elif change == 'raw_item_id': items.loc[0, 'raw_item_id'] = 'unknown game'
                elif change == 'asset_bytes': assets.loc[0, 'data'] = assets.loc[0, 'data'][:-1]
                elif change in ['asset_path', 'asset_role']:
                    value = json.loads(items.loc[0, 'asset_manifest']);value[0]['path' if change == 'asset_path' else 'role']='wrong'
                    items.loc[0, 'asset_manifest'] = json.dumps(value)
                elif change == 'missing_trace': frames['traces'] = traces.iloc[1:]
                else:
                    index = next(i for i, text in enumerate(traces.trace) if len(text) > 16000)
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['interactions'][0]['record']['request']['prompt'] = value['interactions'][0]['record']['request']['prompt'][:16000]
                    elif change == 'missing_call': value['interactions'] = []
                    elif change == 'missing_turn': value['turns'] = []
                    elif change == 'missing_history': value['game_history'] = {}
                    elif change == 'missing_field': value['record'].pop('original_field')
                    elif change == 'source_file': value['source_file'] = 'wrong/end_game_results.json'
                    elif change == 'seat': value['seat'] = 'Seat B'
                    elif change == 'fallback': value['interactions'][0]['record']['response']['used_fallback_response'] = False
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _live_agent_risk(self.directory, frames, self.metadata, source)

    def test_ambiguous_rosters_and_conflicting_model_identity_are_rejected(self):
        import copy
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _live_agent_risk_sources
        baseline = copy.deepcopy(self.records)
        for change in ['winner', 'empty_roster', 'duplicate_seat', 'conflicting_model']:
            with self.subTest(change=change):
                self.records = copy.deepcopy(baseline)
                if change == 'winner': self.records[self.result_file]['winner'] = 'Unknown player'
                elif change == 'empty_roster': self.records[self.result_file]['players'] = []
                elif change == 'duplicate_seat': self.records[self.result_file]['players'][1]['name'] = 'Seat A'
                else: self.records[self.turn_file]['player']['model'] = 'Wrong model'
                self._write_archive()
                with self.assertRaises(ValueError):
                    _live_agent_risk_sources(self.directory, self.metadata)
                with self.assertRaises(ValueError):
                    self.builder(str(self.directory / 'build.py')).build_tables()


class GoodAILTMNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'goodai_ltm_benchmark'
        self.directory.mkdir()
        (self.directory / 'raw').mkdir()
        folder = ROOT / 'benchmarks/goodai_ltm_benchmark'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        parameters = self.metadata['build']['parameters']
        self.files = {name: ('# Original source reference: ' + name).encode() for name in
            [*parameters['task_programs'].values(), *parameters['shared_resources'].values()]}
        self.files['data/Restaurant/menu.json'] = b'{"menu": ["Original dish"]}'
        sessions = ['GPTChatSession - gpt-4-1106-preview - 8192', 'MemGPTChatSession - 4096',
            'GeminiProInterface', 'LTMAgentWrapper - claude-3-opus-20240229 - 16384 - QG_JSON_USER_INFO',
            'MemGPTChatSession', 'HFChatSession - huggingface-gradientai-Llama-3-70B-Instruct-Gradient-262k - 32768']
        self.results = {}
        self.definitions = {}
        self.colour_file = None
        for index, task in enumerate(parameters['task_programs']):
            group = 'Experiment ' + str(index % 2)
            example, repetition = ('story_4', 2) if task == 'ChapterBreak' else ('0', 0)
            name = f'data/tests/{group}/results/{sessions[index % len(sessions)]}/{task}/{example}_{repetition}.json'
            reference = ['Original reference ' + task]
            script = ['Complete instruction ' + task + '\n\u2028', 'Original next question']
            definition = dict(script=script, expected_responses=reference, evaluation_fn='original grader',
                is_question=[False, True], time_jumps=[0, 60], token_spacings=[0, 1000],
                can_be_interleaved=True, uses_callback=task == 'Prospective Memory',
                is_temporal=task == 'Trigger Response', extra_source_setting=dict(unchanged=True))
            messages = ['Test (2000-01-01): ' + script[0], 'Agent (2000-01-01): Original answer',
                'System: Original filler\ncontinued', 'Test (2000-01-02): ' + script[1]]
            record = dict(score=0.0123456789012345, max_score=1, task_log=messages,
                actual_responses=['Original answer'], full_log=['Unprefixed original full conversation'],
                expected_responses=reference, tokens=5, characters=42, reasoning=['Published explanation'],
                extra_original_field=dict(retain='exactly'))
            if task in ['Instruction Recall', 'Spy Meeting']:
                record['expected_responses'] = ['Older reference', 'Accepted synonym']
            if task == 'Restaurant':
                definition['script'], definition['expected_responses'] = [], []
                record.update(score=3, max_score=5, auto_score=1, auto_actual_responses=['Earlier answer'],
                    expected_responses=['Original dynamic rubric'])
                record['task_log'] = ['Test: Original initial instructions', 'Agent: Original answer',
                    'Test: Waiter: Good morning. ' + parameters['labels']['menu_marker'] + '\n1. Original dish',
                    'Agent: Original order', 'Test: Later prompt quoting Original order']
            if task == 'NameList':
                record['auto_reasoning'] = ['Original automatically generated explanation']
            if task == 'Delayed Recall':
                record['score'], record['max_score'] = 4, 10
            if task == 'Colours':
                self.colour_file = name
                record['full_log'] = ['Unprefixed original conversation\n\u2028' * 1000]
            self.results[name] = record
            self.definitions[f'data/tests/{group}/definitions/{task}/{example}.def.json'] = definition
            self.files[f'data/tests/{group}/definitions/config.yml'] = yaml.safe_dump(dict(
                config=dict(run_name='Original mismatched label', filler_tokens=1000 + index % 2),
                datasets=['original scheduling configuration'])).encode()
        # A second source alias for an identical definition must retain its observation.
        alias = self.colour_file.replace('/0_0.json', '/1_0.json')
        self.results[alias] = json.loads(json.dumps(self.results[self.colour_file]))
        self.results[alias]['score'] = 0
        self.results[alias]['full_log'] = ['Original second observation']
        definition_file = 'data/tests/Experiment 0/definitions/Colours/0.def.json'
        self.definitions[definition_file.replace('/0.def.json', '/1.def.json')] = self.definitions[definition_file]
        self.files['data/tests/Experiment 0/results/GeminiProInterface/runstats.json'] = b'{"duration": 5}'
        self._write_archive()
        self.builder = runpy.run_path(str(folder / 'build.py'))['GoodAILTM']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'tables')])
        self.frames = {p.stem: pd.read_parquet(p) for p in (self.directory.parent / 'tables').glob('*.parquet')}

    def _write_archive(self):
        import zipfile
        layout = self.metadata['build']['parameters']['layout']
        with zipfile.ZipFile(self.directory / 'raw' / layout['archive'], 'w') as archive:
            for name, body in self.files.items():
                archive.writestr(layout['prefix'] + name, body)
            for name, value in {**self.definitions, **self.results}.items():
                archive.writestr(layout['prefix'] + name, json.dumps(value))

    def test_complete_native_grades_contexts_aliases_and_revised_references(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _goodai_ltm
        observed = _goodai_ltm(self.directory, self.frames, self.metadata)
        self.assertEqual(observed, dict(source_subjects=6, source_items=13, source_responses=14,
            source_traces=14, source_assets=16, source_result_files=14, source_run_statistics_excluded=1,
            source_reference_differences=3, source_marked_revisions=2, source_revised_grades=1,
            source_dynamic_records=1, source_long_full_logs=1, source_task_definitions=14))
        self.assertTrue(self.frames['responses'].response.eq(4).any())
        self.assertTrue(self.frames['responses'].trial.eq(3).any())
        self.assertFalse(self.frames['items'].content.str.contains('Later prompt quoting').any())
        self.assertTrue(self.frames['traces'].trace.str.contains('Later prompt quoting').any())

    def test_corrupted_grades_definitions_resources_and_trace_links_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _goodai_ltm, _goodai_ltm_sources
        source = _goodai_ltm_sources(self.directory, self.metadata)
        changes = ['score', 'precision', 'normalize', 'subject', 'item', 'trial', 'condition', 'interactors',
            'drop', 'duplicate', 'extra_subject', 'extra_item', 'extra_asset', 'model', 'configuration', 'scale',
            'definition', 'schedule', 'release_group', 'menu', 'answer_leak', 'reference', 'rubric', 'verifier',
            'effective_scale', 'raw_item_id', 'asset_bytes', 'asset_path', 'asset_role', 'trace_clip',
            'missing_field', 'source_file', 'automatic_grade', 'trace_drop']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, assets, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'assets', 'traces']]
                if change == 'score': responses.loc[0, 'response'] = 0.5
                elif change == 'precision': responses.loc[responses.response.eq(0.0123456789012345), 'response'] = 0.0123456789
                elif change == 'normalize': responses.loc[responses.response.eq(4), 'response'] = 0.4
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'unknown'
                elif change == 'item': responses.loc[0, 'item_id'] = next(x for x in items.item_id if x != responses.loc[0, 'item_id'])
                elif change == 'trial': responses.loc[0, 'trial'] = 99
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'wrong source'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented'
                elif change == 'drop': frames['responses'] = responses.iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'extra_item': frames['items'] = pd.concat([items, items.iloc[:1]], ignore_index=True)
                elif change == 'extra_asset': frames['assets'] = pd.concat([assets, assets.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'Wrong model'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] += ';invented=yes'
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=1))
                elif change in ['definition', 'schedule', 'release_group', 'menu', 'answer_leak']:
                    index = next(i for i, text in enumerate(items.content) if ('initial_menu_message' in text) == (change in ['menu', 'answer_leak']))
                    value = json.loads(items.loc[index, 'content'])
                    if change == 'definition': value['definition']['script'][0] = 'wrong question'
                    elif change == 'schedule': value['configuration']['config']['filler_tokens'] = 999
                    elif change == 'release_group': value['release_group'] = 'invented'
                    elif change == 'menu': value['initial_menu_message'] = 'truncated menu'
                    else: value['answer'] = 'evaluated answer'
                    items.loc[index, 'content'] = json.dumps(value, sort_keys=True, ensure_ascii=False)
                elif change in ['reference', 'rubric', 'effective_scale']:
                    index = next(i for i, text in enumerate(items.content) if ('initial_menu_message' in text) == (change == 'rubric'))
                    value = json.loads(items.loc[index, 'grading_criterion'])
                    if change == 'reference': value['reference_answer'] = 'Wrong reference'
                    elif change == 'rubric': value['rule'] = 'Wrong rubric'
                    else: value['response_scale']['max'] = 99
                    items.loc[index, 'grading_criterion'] = json.dumps(value)
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(**{'class': 'judge'}, spec='{}'))
                elif change == 'raw_item_id': items.loc[0, 'raw_item_id'] = 'unknown definition'
                elif change == 'asset_bytes': assets.loc[0, 'data'] = assets.loc[0, 'data'][:-1]
                elif change in ['asset_path', 'asset_role']:
                    value = json.loads(items.loc[0, 'asset_manifest']); value[0]['path' if change == 'asset_path' else 'role'] = 'wrong'
                    items.loc[0, 'asset_manifest'] = json.dumps(value)
                elif change == 'trace_drop': frames['traces'] = traces.iloc[1:]
                else:
                    index = next(i for i, text in enumerate(traces.trace) if ('auto_score' in text) == (change == 'automatic_grade') and (change != 'trace_clip' or len(text) > 16000))
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['record']['full_log'][0] = value['record']['full_log'][0][:16000]
                    elif change == 'missing_field': value['record'].pop('extra_original_field')
                    elif change == 'source_file': value['source_file'] = 'wrong.json'
                    elif change == 'automatic_grade': value['record']['auto_score'] = 99
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _goodai_ltm(self.directory, frames, self.metadata, source)

    def test_invalid_native_scores_and_wrong_task_definitions_are_rejected(self):
        import contextlib
        import io
        from measurement_db.build_base import BuildContractError
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _goodai_ltm_sources
        record = self.results[self.colour_file]
        for value in [float('inf'), float('nan'), True, -0.1, 1.1]:
            with self.subTest(value=value):
                record['score'] = value
                self._write_archive()
                with self.assertRaises(ValueError):
                    _goodai_ltm_sources(self.directory, self.metadata)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, BuildContractError)):
                    self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                        '--output', str(self.directory.parent / 'invalid')])
        record['score'] = 0
        self.definitions['data/tests/Experiment 0/definitions/Colours/0.def.json']['script'][0] = 'Wrong opening instruction'
        self._write_archive()
        with self.assertRaises(ValueError):
            _goodai_ltm_sources(self.directory, self.metadata)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.builder(str(self.directory / 'build.py')).build_tables()


class InterCodeNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'intercode'
        (self.directory / 'raw').mkdir(parents=True)
        folder = ROOT / 'benchmarks/intercode'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        self.files, self.results = {}, {}
        sql = 'data/sql/spider/ic_spider_dev.json'
        python = 'data/python/mbpp/ic_mbpp.json'
        ctf = 'data/ctf/ic_ctf.json'
        self.files[sql] = json.dumps([dict(query='SQL question ' + str(i), db='database_' + str(i),
            gold='SELECT ' + str(i), db_tables=['table_' + str(i)], hardness='easy') for i in range(2)]).encode()
        self.files[python] = json.dumps([dict(query='Python task', gold='def f(): return 1', tests=['assert f() == 1'],
            task_id=987, test_setup_code='')]).encode()
        initial = next(iter(self.metadata['build']['parameters']['initial_ctf_files']))
        initial_key = initial + ':0'
        self.files[ctf] = json.dumps([dict(query=self.metadata['build']['parameters']['definition_variants'][initial_key],
            gold='CURRENT_FLAG_NOT_KNOWN_TO_BE_HISTORICAL', task_id=0, source='Released fixture')]).encode()
        self.files['data/sql/spider/ic_spider_dbs.sql'] = b'CREATE TABLE example(id INTEGER);'
        self.files['data/ctf/task_assets/0/input.txt'] = b'original task input\n'
        self.files['data/ctf/task_assets/0/solution/README.md'] = b'not an input'
        self.files['data/ctf/task_assets/0/.placeholder'] = b''
        self.files['docker/nl2bash.Dockerfile'] = b'FROM declared-image\n'
        for i in range(1, 5):
            self.files[f'data/nl2bash/nl2bash_fs_{i}.json'] = json.dumps([dict(query='Bash task', gold='ls')]).encode()
            self.files[f'docker/bash_scripts/setup_nl2b_fs_{i}.sh'] = f'echo filesystem-{i}\n'.encode()

        def episode(env, dataset, index, query, score, turns=10):
            return dict(environment=env, dataset=dataset, task_id=index, query=query,
                turn_history=dict(actions=['Original command'], rewards=[score], observations=['Full observation\u2028']),
                summary=dict(max_reward=score, max_reward_idx=0, turns_taken=1, turns_max=turns),
                original_extra='complete source field')

        self.sql_file = 'data/results/sql/gpt-3.5/ic_sql_multiturn_gpt-3.5_10_turns.json'
        self.results[self.sql_file] = {str(i): episode('ic_sql', './data/spider/dev_spider.json', i,
            'SQL question ' + str(i), score) for i, score in enumerate([-0.18, 0.0123456789012345])}
        self.results[self.sql_file]['0']['turn_history']['actions'][0] = 'Long original command\n' * 1300
        self.results['data/results/sql/gpt-3.5/ic_sql_multiturn_gpt-3.5_10_turns_handicap.json'] = {
            '0': episode('ic_sql', './data/spider/dev_spider.json', 0, 'SQL question 0', 0.8)}
        plan = episode('ic_sql', './data/spider/dev_spider.json', 0, 'SQL question 0', 0)
        plan['summary'] = dict(max_reward=0, max_reward_idx=-1)
        self.results['data/results/sql/gpt-3.5/ic_sql_plan_solve_refine_3_turns.json'] = dict(
            meta=dict(refine=True, refine_turns=3, seed=32, proportion=0.05), logs={'0': plan})
        self.results['data/results/python/gpt-3.5/ic_python_multiturn_gpt-4_7_turns.json'] = {
            '0': episode('ic_python', './' + python, 0, 'Python task', 1 / 3, 7)}
        for i in [1, 2]:
            self.results[f'data/results/bash/gpt-4/ic_bash_multiturn_gpt-4_10_turns_fs_{i}.json'] = {
                '0': episode('ic_bash', f'./data/nl2bash/nl2bash_fs_{i}.json', 0, 'Bash task', 0.7100000000000001)}
        original_query = self.metadata['build']['parameters']['question_variants'][initial_key]
        self.results[initial] = {'0': episode('ic_ctf', './data/ctf/ctf_test.json', 0, original_query, 0, 15)}
        self.results[initial]['0']['summary']['turns_taken'] = 3
        self.results['data/results/ctf/ic_ctf_multiturn_gpt-4_10_turns.json'] = {
            '0': episode('ic_ctf', './' + ctf, 0, json.loads(self.files[ctf])[0]['query'], 1)}
        self.results['data/results/sql/human/human.json'] = {'0': episode('ic_sql', './data/spider/dev_spider.json', 0, 'SQL question 0', 1)}
        self._write_archive()
        self.builder = runpy.run_path(str(folder / 'build.py'))['InterCode']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def _write_archive(self):
        import zipfile
        layout = self.metadata['build']['parameters']['layout']
        with zipfile.ZipFile(self.directory / 'raw' / layout['archive'], 'w') as archive:
            for path, body in self.files.items():
                archive.writestr(layout['prefix'] + path, body)
            for path, records in self.results.items():
                archive.writestr(layout['prefix'] + path, json.dumps(records))

    def test_complete_records_signed_scales_task_positions_and_configurations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _intercode
        observed = _intercode(self.directory, self.frames, self.metadata)
        self.assertEqual(observed['source_responses'], 9)
        self.assertEqual(observed['source_subjects'], 6)
        self.assertEqual(observed['source_items'], 8)
        self.assertEqual(observed['source_result_files'], 8)
        self.assertEqual(observed['source_wrapped_files'], 1)
        self.assertEqual(observed['source_human_records_excluded'], 1)
        self.assertEqual(observed['source_negative_scores'], 1)
        self.assertEqual(observed['source_long_records'], 1)
        self.assertEqual(observed['source_question_variants'], 1)
        self.assertEqual(observed['source_ctf_unknown_references'], 2)
        self.assertEqual(observed['source_turn_count_differences'], 1)
        self.assertEqual(observed['source_best_step_index_differences'], 1)
        self.assertTrue(self.frames['responses'].response.eq(0.0123456789012345).any())
        self.assertFalse(self.frames['items'].grading_criterion.str.contains('CURRENT_FLAG').any())
        self.assertFalse(self.frames['items'].asset_manifest.fillna('').str.contains('/solution/').any())

    def test_corruption_of_data_contexts_configurations_and_trace_links_is_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _intercode, _intercode_sources
        source = _intercode_sources(self.directory, self.metadata)
        changes = ['score', 'precision', 'clip_negative', 'subject', 'item', 'trial', 'condition', 'interactors',
            'drop', 'duplicate', 'extra_subject', 'extra_item', 'extra_asset', 'model', 'configuration', 'benchmark_scale',
            'question', 'schema', 'reference', 'verifier', 'effective_scale', 'asset_bytes', 'asset_path', 'asset_role',
            'trace_clip', 'missing_field', 'source_file', 'source_record', 'run_metadata', 'turn_count', 'trace_drop']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, assets, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'assets', 'traces']]
                if change == 'score': responses.loc[0, 'response'] = 0.5
                elif change == 'precision': responses.loc[responses.response.eq(0.0123456789012345), 'response'] = 0.0123456789
                elif change == 'clip_negative': responses.loc[responses.response.lt(0), 'response'] = 0
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'unknown'
                elif change == 'item': responses.loc[0, 'item_id'] = next(value for value in items.item_id if value != responses.loc[0, 'item_id'])
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'invented run'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented'
                elif change == 'drop': frames['responses'] = responses.iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'extra_item': frames['items'] = pd.concat([items, items.iloc[:1]], ignore_index=True)
                elif change == 'extra_asset': frames['assets'] = pd.concat([assets, assets.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'Another model'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] += ';invented_setting=yes'
                elif change == 'benchmark_scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=1))
                elif change in ['question', 'schema']:
                    index = next(i for i, text in enumerate(items.content) if 'provided_schema' in text)
                    content = json.loads(items.loc[index, 'content'])
                    content['question' if change == 'question' else 'provided_schema'] = 'wrong'
                    items.loc[index, 'content'] = json.dumps(content, ensure_ascii=False, sort_keys=True)
                elif change in ['reference', 'effective_scale']:
                    value = json.loads(items.loc[0, 'grading_criterion'])
                    value['reference_answer' if change == 'reference' else 'response_scale'] = ('invented' if change == 'reference' else dict(kind='interval', min=-100, max=100))
                    items.loc[0, 'grading_criterion'] = json.dumps(value)
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(**{'class': 'exact_matcher'}, spec='{}'))
                elif change == 'asset_bytes': assets.loc[0, 'data'] = assets.loc[0, 'data'][:-1]
                elif change in ['asset_path', 'asset_role']:
                    index = next(i for i, value in enumerate(items.asset_manifest) if isinstance(value, str) and value != '[]')
                    value = json.loads(items.loc[index, 'asset_manifest']); value[0]['path' if change == 'asset_path' else 'role'] = 'wrong'
                    items.loc[index, 'asset_manifest'] = json.dumps(value)
                elif change == 'trace_drop': frames['traces'] = traces.iloc[1:]
                else:
                    index = next(i for i, text in enumerate(traces.trace) if len(text) > 16000)
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['record']['turn_history']['actions'][0] = value['record']['turn_history']['actions'][0][:16000]
                    elif change == 'missing_field': value['record'].pop('original_extra')
                    elif change == 'source_file': value['source_file'] = 'wrong.json'
                    elif change == 'source_record': value['source_record'] = '999'
                    elif change == 'run_metadata': value['run_metadata'] = {'seed': 999}
                    elif change == 'turn_count': value['record']['summary']['turns_taken'] = 999
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _intercode(self.directory, frames, self.metadata, source)

    def test_invalid_scores_and_undocumented_task_mapping_are_rejected(self):
        import contextlib
        import io
        from measurement_db.build_base import BuildContractError
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _intercode_sources
        record = self.results[self.sql_file]['0']
        for value in [float('inf'), float('nan'), True, -1.1, 1.1]:
            with self.subTest(value=value):
                record['summary']['max_reward'] = value
                self._write_archive()
                with self.assertRaises(ValueError):
                    _intercode_sources(self.directory, self.metadata)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, BuildContractError)):
                    self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                        '--output', str(self.directory.parent / 'invalid')])
        record['summary']['max_reward'] = -0.18
        record['query'] = 'Wrong task'
        self._write_archive()
        with self.assertRaises((ValueError, KeyError)):
            _intercode_sources(self.directory, self.metadata)
        with self.assertRaisesRegex(ValueError, 'undocumented'):
            self.builder(str(self.directory / 'build.py')).build_tables()


class QatchNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import csv
        import io
        import runpy
        import sqlite3
        import zipfile
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'qatch'
        self.directory.mkdir()
        folder = ROOT / 'benchmarks/qatch'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        raw = self.directory / 'raw'; raw.mkdir()
        self.native = {}
        metrics = list(self.metadata['grading']['verifiers'])
        for task, model, values in [('question-answering', 'chatgpt', [[1, 2], [3, 4]]),
                ('question-answering', 'tapas-wtq', [[10, 20], [30, 40]]),
                ('semantic-parsing', 'resdsql', [[1, 2], [3, 4]])]:
            rows = []
            for index, (query, target, tag) in enumerate([
                    ('SELECT * FROM t', values, 'SELECT'),
                    ('SELECT a FROM t', [[row[0]] for row in values], 'SELECT'),
                    ('SELECT b FROM t', [[row[1]] for row in values], 'SELECT'),
                    ('SELECT a FROM t ORDER BY a DESC', [[row[0]] for row in reversed(values)], 'ORDERBY'),
                    ('SELECT count(*) FROM t', [[999]], 'SIMPLEAGGR')]):
                row = dict(db_id='example', tbl_name='t', sql_tags=tag, query=query,
                    question='Question ' + str(index) + ' — complete text.\u2028', query_result=repr(target), predictions='DROP TABLE t;',
                    source_extra='0.0123456789012345', **{metric: '0.0123456789012345' for metric in metrics})
                if index != 3: row['tuple_order'] = ''
                if task == 'question-answering' and model == 'chatgpt' and index == 0:
                    row['predictions'] = 'A complete answer.\n\u2028' * 1200
                if model == 'tapas-wtq' and index == 4: row['predictions'] = ''
                rows.append(row)
            name = f'generated_tests/{task}/custom-data/{model}/tests_with_results.csv'
            self.native[name] = rows
        official = [dict(db_id='example', question='Spider full table', query='SELECT * FROM t'),
            dict(db_id='example', question='Spider ordered values', query='SELECT a FROM t ORDER BY a DESC')]
        merged, single = [], []
        for index, original in enumerate(official):
            row = dict(**original, tbl_name='t', sql_tags='SELECT' if index == 0 else 'ORDERBY',
                query_result=repr([[1, 2], [3, 4]] if index == 0 else [[3], [1]]), source_extra='unchanged')
            many = dict(row)
            for model in self.metadata['build']['parameters']['merged_models']:
                many[model + '_predictions'] = 'Original output for ' + model
                for metric in metrics:
                    many[metric + '_' + model] = '' if metric == 'tuple_order' and index == 0 else '0.5'
            if index == 0: many['cell_precision_chatgpt'] = ''
            merged.append(many)
            single.append(dict(row, predictions='Original SQL output', **{metric: '' if metric == 'tuple_order' and index == 0 else '0.5' for metric in metrics}))
        self.native[self.metadata['build']['parameters']['layout']['merged']] = merged
        self.native['generated_tests/semantic-parsing/spider/resdsql-large/tests_with_results_dev.csv'] = single
        for name, rows in self.native.items():
            if '/semantic-parsing/' in name:
                self.native[name] = [dict({'': str(index)}, **row) for index, row in enumerate(rows)]
        self._write_results()
        database = self.directory / 'fixture.sqlite'
        connection = sqlite3.connect(database)
        connection.execute('CREATE TABLE t(a INTEGER, b INTEGER)')
        connection.executemany('INSERT INTO t VALUES(?,?)', [(1, 2), (3, 4)])
        connection.commit(); connection.close()
        self.database_bytes = database.read_bytes()
        schema = [dict(db_id='example', table_names_original=['t'], column_names_original=[[-1, '*'], [0, 'a'], [0, 'b']])]
        layout = self.metadata['build']['parameters']['layout']
        with zipfile.ZipFile(raw / layout['spider'], 'w') as archive:
            archive.writestr(layout['spider_prefix'] + 'tables.json', json.dumps(schema))
            archive.writestr(layout['spider_prefix'] + 'train_spider.json', json.dumps(official))
            archive.writestr(layout['spider_prefix'] + 'dev.json', '[]')
            archive.writestr(layout['spider_prefix'] + 'database/example/example.sqlite', self.database_bytes)
        self.builder = runpy.run_path(str(folder / 'build.py'))['QATCH']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(raw), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def _write_results(self):
        import csv
        import io
        import zipfile
        path = self.directory / 'raw' / self.metadata['build']['parameters']['layout']['results']
        with zipfile.ZipFile(path, 'w') as archive:
            for name, rows in self.native.items():
                stream = io.StringIO(newline='')
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
                archive.writestr(name, stream.getvalue())

    def test_complete_contexts_native_precision_missing_grades_and_separate_protocols(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _qatch
        self.assertEqual(_qatch(self.directory, self.frames, self.metadata), dict(
            source_subjects=5, source_items=81, source_responses=99, source_traces=99, source_assets=3,
            source_csv_rows=19, source_attempt_records=23, source_unavailable_grades=1,
            source_inapplicable_order_fields=16, source_long_predictions=1, source_empty_predictions=1,
            source_custom_contexts=3, source_spider_contexts=2))
        self.assertEqual(self.frames['responses'].response.isna().sum(), 1)
        self.assertTrue(self.frames['responses'].response.eq(0.0123456789012345).any())

    def test_corrupted_scores_records_contexts_and_trace_links_are_rejected(self):
        import hashlib
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _qatch, _qatch_sources
        source = _qatch_sources(self.directory, self.metadata)
        changes = ['score', 'precision', 'null_to_zero', 'subject', 'item', 'trial', 'condition', 'interactors',
            'drop', 'duplicate', 'extra_subject', 'extra_item', 'extra_asset', 'model', 'configuration', 'scale',
            'question', 'column_order', 'reference', 'verifier', 'metric', 'asset_bytes', 'asset_path', 'asset_role',
            'trace_clip', 'trace_precision', 'missing_field', 'source_file', 'source_row', 'source_model']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, assets, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'assets', 'traces']]
                if change == 'score': responses.loc[0, 'response'] = 0.75
                elif change == 'precision': responses.loc[responses.response.eq(0.0123456789012345), 'response'] = 0.0123456789
                elif change == 'null_to_zero': responses.loc[responses.response.isna(), 'response'] = 0
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'unknown'
                elif change == 'item': responses.loc[0, 'item_id'] = next(key for key in items.item_id if key != responses.loc[0, 'item_id'])
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'new trial'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented'
                elif change == 'drop': frames['responses'] = responses.iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'extra_item': frames['items'] = pd.concat([items, items.iloc[:1]], ignore_index=True)
                elif change == 'extra_asset': frames['assets'] = pd.concat([assets, assets.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'Other model'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] = 'prediction_task=wrong'
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=100))
                elif change in ['question', 'column_order']:
                    value = json.loads(items.loc[0, 'content'])
                    if change == 'question': value['question'] = 'Different question'
                    else: value['columns'] = list(reversed(value['columns']))
                    items.loc[0, 'content'] = json.dumps(value, sort_keys=True, ensure_ascii=False)
                elif change == 'reference': items.loc[0, 'grading_criterion'] = json.dumps(dict(reference_answer='Corrected reference', rule='New rule'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(**{'class': 'exact_matcher'}, spec='{}'))
                elif change == 'metric':
                    verifier = json.loads(items.loc[0, 'verifier'])
                    protocol = json.loads(verifier['spec'])
                    protocol['metric'] = 'unknown'
                    verifier['spec'] = json.dumps(protocol)
                    items.loc[0, 'verifier'] = json.dumps(verifier)
                elif change == 'asset_bytes': assets.loc[0, 'data'] = assets.loc[0, 'data'][:-1]
                elif change in ['asset_path', 'asset_role']:
                    value = json.loads(items.loc[0, 'asset_manifest'])
                    value[0]['path' if change == 'asset_path' else 'role'] = 'wrong'
                    items.loc[0, 'asset_manifest'] = json.dumps(value)
                else:
                    index = next(i for i, text in enumerate(traces.trace) if len(json.loads(text)['record'].get('predictions', '')) > 16000)
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['record']['predictions'] = value['record']['predictions'][:16000]
                    elif change == 'trace_precision': value['record']['source_extra'] = '0.0123456789'
                    elif change == 'missing_field': value['record'].pop('source_extra')
                    elif change == 'source_file': value['source_file'] = 'wrong.csv'
                    elif change == 'source_row': value['source_row'] = 999
                    elif change == 'source_model': value['source_model'] = 'resdsql-large'
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _qatch(self.directory, frames, self.metadata, source)

    def test_invalid_native_scores_are_flagged(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _qatch_sources
        first = next(iter(self.native))
        for value in ['true', 'inf', 'nan', '-0.1', '1.1']:
            with self.subTest(value=value):
                self.native[first][0]['cell_precision'] = value
                self._write_results()
                with self.assertRaises(ValueError):
                    _qatch_sources(self.directory, self.metadata)
                with self.assertRaises(ValueError):
                    self.builder(str(self.directory / 'build.py')).build_tables()

    def test_missing_column_evidence_is_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _qatch_sources
        first = next(iter(self.native))
        self.native[first][1]['query'] = 'SELECT a FROM t WHERE a > 0'
        self._write_results()
        with self.assertRaisesRegex(ValueError, 'unambiguous'):
            _qatch_sources(self.directory, self.metadata)
        with self.assertRaisesRegex(ValueError, 'unambiguous'):
            self.builder(str(self.directory / 'build.py')).build_tables()

    def test_published_wrong_reference_and_sql_output_are_preserved_without_execution(self):
        references = [json.loads(row)['reference_answer'] for row in self.frames['items'].grading_criterion]
        self.assertTrue(any(json.loads(value)['released_target'] == '[[999]]' for value in references))
        records = [json.loads(value)['record'] for value in self.frames['traces'].trace]
        self.assertTrue(any(row.get('predictions') == 'DROP TABLE t;' for row in records))
        self.assertIn(self.database_bytes, self.frames['assets'].data.tolist())


class RakudaNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'rakuda'
        folder = ROOT / 'benchmarks/rakuda'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.directory.mkdir()
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        questions = [dict(question_id='question-' + str(index), category='社会', text=text)
            for index, text in enumerate(['完全な質問。\u2028' * 2000, 'もう一つの質問。'])]
        bank = self.directory / 'raw' / self.metadata['build']['parameters']['layout']['questions']
        bank.parent.mkdir(parents=True)
        bank.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in questions))
        self.native = {}
        for group_index, group in enumerate(list(self.metadata['grading']['verifiers'])[:3]):
            for model in ['model-alpha', 'qwen__qwen3.6-plus:free']:
                rows = []
                for index, question in enumerate(questions):
                    score = None if group_index == 2 and index == 1 else 8
                    if group_index == 0 and model == 'model-alpha' and index == 0: score = 5.8
                    if group_index == 1 and model == 'qwen__qwen3.6-plus:free' and index == 0: score = 0
                    answer = (('Original complete answer 完整。\u2028' * 1000) if index == 0 else 'Answer ') + group + model
                    if group_index == 1 and model == 'qwen__qwen3.6-plus:free' and index == 1: answer = ''
                    row = dict(question_id=question['question_id'], category=question['category'], Question=question['text'],
                        ModelAnswer=answer, score=score, judge_output='Published explanation',
                        metadata=dict(probability=0.0123456789012345))
                    if group_index == 0 and model == 'model-alpha' and index == 1: row.pop('judge_output')
                    rows.append(row)
                name = f'data/judgements/{group}/yuzuai__rakuda-questions/{model}.json'
                name = name.replace(':', '_x3a_')
                path = self.directory / 'raw' / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                self.native[name] = rows
        self.builder = runpy.run_path(str(folder / 'build.py'))['Rakuda']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_original_scale_nulls_full_unicode_outputs_and_distinct_judges(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _rakuda
        self.assertEqual(_rakuda(self.directory, self.frames, self.metadata), dict(
            source_questions=2, source_subjects=2, source_items=6, source_responses=12, source_traces=12,
            source_null_scores=2, source_empty_outputs=1, source_long_outputs=6, source_fractional_scores=1,
            source_zero_scores=1, source_judge_explanations=11, source_model_question_variants=4))
        self.assertIn('qwen__qwen3.6-plus:free', set(self.frames['subjects'].display_name))
        self.assertTrue(self.frames['responses'].trial.eq(1).all())
        self.assertEqual(self.frames['responses'].response.isna().sum(), 2)

    def test_invalid_native_score_is_flagged(self):
        import copy
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _rakuda_sources
        name = next(iter(self.native))
        for value in [True, float('inf'), '9', -1, 11]:
            with self.subTest(value=value):
                rows = copy.deepcopy(self.native[name]); rows[0]['score'] = value
                (self.directory / 'raw' / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                with self.assertRaisesRegex(ValueError, 'finite numeric rating'):
                    _rakuda_sources(self.directory, self.metadata)

    def test_changed_units_missing_attempts_wrong_judges_and_truncation_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _rakuda, _rakuda_sources
        source = _rakuda_sources(self.directory, self.metadata)
        changes = ['rescale', 'null_to_zero', 'round', 'clip_zero', 'subject', 'item', 'question', 'rule', 'verifier',
            'alias', 'feature', 'trial', 'condition', 'interactors', 'drop', 'duplicate', 'model', 'configuration',
            'extra_subject', 'scale', 'trace_clip', 'unicode_separator', 'trace_grade', 'judge_explanation', 'precision',
            'source_file', 'source_row', 'share_different_answer']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'traces']]
                if change == 'rescale': responses['response'] = responses.response / 10
                elif change == 'null_to_zero': responses.loc[responses.response.isna(), 'response'] = 0
                elif change == 'round': responses.loc[responses.response.eq(5.8), 'response'] = 6
                elif change == 'clip_zero': responses.loc[responses.response.eq(0), 'response'] = 1
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'judge'
                elif change == 'item': responses.loc[0, 'item_id'] = next(key for key in items.item_id if key != responses.loc[0, 'item_id'])
                elif change == 'question': items.loc[0, 'content'] = 'Incomplete question'
                elif change == 'rule': items.loc[0, 'grading_criterion'] = json.dumps(dict(rule='Binary correctness'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(spec='{}'))
                elif change == 'alias': items.loc[0, 'raw_item_id'] = 'unknown:unknown'
                elif change == 'feature': items.loc[0, 'item_features'] = 'category=wrong'
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'independent rerun'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented-system'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'gpt-4-judge'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] = 'source_model_filename=wrong-model'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=1))
                else:
                    index = next(i for i, value in enumerate(traces.trace) if len(json.loads(value)['record']['ModelAnswer']) > 16000)
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['record']['ModelAnswer'] = value['record']['ModelAnswer'][:8000]
                    elif change == 'unicode_separator': value['record']['ModelAnswer'] = value['record']['ModelAnswer'].replace('\u2028', '')
                    elif change == 'trace_grade': value['record']['score'] = .5
                    elif change == 'judge_explanation': value['record']['judge_output'] = 'New explanation'
                    elif change == 'precision': value['record']['metadata']['probability'] = 0.0123456789
                    elif change == 'source_file': value['source_file'] = 'wrong.json'
                    elif change == 'source_row': value['source_row'] = 99
                    else: value['record']['ModelAnswer'] = 'Substituted answer from another judge'
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _rakuda(self.directory, frames, self.metadata, source)


class OODPredictionNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'ood_prediction'
        folder = ROOT / 'benchmarks/ood_prediction'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.directory.mkdir()
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        self.long_prompt = 'Complete original prompt 完整. ' * 1000
        self.native = {}
        for fold in [0, 1]:
            self.native[f'data/ioi_task/ioi_task_Meta-Llama-3-8B-Instruct_ood_fixture_split_{fold}.json'] = dict(
                train=dict(correct=[self.long_prompt], wrong=['Wrong IOI']), val=dict(correct=[], wrong=[]),
                test=dict(correct=['Other IOI'], wrong=[]))
        self.native['data/ioi_task/ioi_task_gpt2_primitive_test.json'] = dict(train=dict(correct=['GPT-only prompt'], wrong=[self.long_prompt]))
        for task in ['whp', 'mmlu', 'ravel_task', 'pricetag_task']:
            self.native[f'data/{task}/{task}_Meta-Llama-3-8B-Instruct_in_distribution_split_0.json'] = dict(
                train=dict(correct=[task + ' original first prompt'], wrong=[task + ' original second prompt']))
        for name, data in self.native.items():
            path = self.directory / 'raw' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['OODPrediction']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_reused_folds_shared_prompts_and_task_specific_rules(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _ood_prediction
        self.assertEqual(_ood_prediction(self.directory, self.frames, self.metadata), dict(
            source_subjects=2, source_items=12, source_responses=13, source_traces=13,
            source_correct_occurrences=9, source_wrong_occurrences=7, source_files=7, source_occurrences=16,
            source_reused_keys=3, source_correct_observations=7, source_wrong_observations=6))
        self.assertGreater(self.frames['items'].content.str.len().max(), 16000)
        self.assertTrue(self.frames['responses'].trial.eq(1).all())
        self.assertTrue(all('generated_answer' not in json.loads(value) for value in self.frames['traces'].trace))
        whp = self.frames['items'][self.frames['items'].item_features.str.contains('task=whp')]
        self.assertTrue(all(json.loads(value)['class'] == 'judge' for value in whp.verifier))

    def test_conflicting_or_unrecognized_native_labels_are_rejected(self):
        import copy
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _ood_prediction_sources
        name = next(iter(self.native))
        for change in ['conflicting', 'unknown']:
            with self.subTest(change=change):
                data = copy.deepcopy(self.native[name])
                if change == 'conflicting': data['train']['wrong'].append(self.long_prompt)
                else: data['train']['ungraded'] = ['Unrecognized bucket']
                (self.directory / 'raw' / name).write_text(json.dumps(data))
                with self.assertRaises(ValueError): _ood_prediction_sources(self.directory, self.metadata)
                with self.assertRaises(ValueError): self.builder(str(self.directory / 'build.py')).build_tables()

    def test_changed_verdicts_links_and_fold_provenance_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _ood_prediction, _ood_prediction_sources
        source = _ood_prediction_sources(self.directory, self.metadata)
        changes = ['grade', 'same_sum_swap', 'subject', 'item', 'prompt', 'rule', 'gold', 'verifier', 'alias', 'feature',
            'trial', 'condition', 'interactors', 'drop', 'duplicate', 'model', 'configuration', 'extra_subject', 'scale',
            'trace_prompt', 'trace_label', 'trace_model', 'trace_task', 'trace_setting', 'source_file', 'source_row',
            'source_split', 'drop_occurrence', 'extra_occurrence', 'invented_completion']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'traces']]
                if change == 'grade': responses.loc[0, 'response'] = .5
                elif change == 'same_sum_swap':
                    indices = [responses.index[responses.response.eq(value)][0] for value in [0, 1]]
                    responses.loc[indices, 'response'] = [1., 0.]
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'correctness-predictor'
                elif change == 'item': responses.loc[0, 'item_id'] = next(key for key in items.item_id if key != responses.loc[0, 'item_id'])
                elif change == 'prompt':
                    index = items.content.str.len().idxmax(); items.loc[index, 'content'] = items.loc[index, 'content'][:16000]
                elif change == 'rule': items.loc[0, 'grading_criterion'] = json.dumps(dict(rule='Different task'))
                elif change == 'gold':
                    value = json.loads(items.loc[0, 'grading_criterion']); value['reference_answer'] = 'Guessed answer'; items.loc[0, 'grading_criterion'] = json.dumps(value)
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(spec='{}'))
                elif change == 'alias': items.loc[0, 'raw_item_id'] = 'correct'
                elif change == 'feature': items.loc[0, 'item_features'] = 'task=ioi_task;correct=true'
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'task=ioi;setting=wrong'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented-rater'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'correctness-predictor'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] = 'source_model_label=wrong-model'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=100))
                else:
                    index = 0
                    if change == 'trace_prompt': index = next(i for i, value in enumerate(traces.trace) if len(json.loads(value)['prompt']) > 16000)
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_prompt': value['prompt'] = value['prompt'][:16000]
                    elif change == 'trace_label': value['label'] = 'wrong' if value['label'] == 'correct' else 'correct'
                    elif change == 'trace_model': value['source_model_label'] = 'invented-model'
                    elif change == 'trace_task': value['task'] = 'unknown-task'
                    elif change == 'trace_setting': value['setting'] = 'unknown-setting'
                    elif change == 'source_file': value['occurrences'][0]['source_file'] = 'other.json'
                    elif change == 'source_row': value['occurrences'][0]['source_row'] = 999
                    elif change == 'source_split': value['occurrences'][0]['split'] = 'unknown'
                    elif change == 'drop_occurrence': value['occurrences'] = value['occurrences'][1:]
                    elif change == 'extra_occurrence': value['occurrences'].append(value['occurrences'][0])
                    else: value['generated_answer'] = 'Unreleased model completion'
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _ood_prediction(self.directory, frames, self.metadata, source)


class PKUSafeRLHFNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import copy
        import hashlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'pku_saferlhf'
        folder = ROOT / 'benchmarks/pku_saferlhf'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.raw_file = self.directory / 'raw/data/fixture/train.jsonl'
        self.raw_file.parent.mkdir(parents=True)
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        categories = self.metadata['grading']['verifiers']['safety']['harm_categories']
        self.native = []
        for index, prompt in enumerate(['Complete prompt 完整. ' * 1000, 'Another prompt']):
            row = dict(prompt=prompt, prompt_source='released prompt pool', better_response_id=1, safer_response_id=0)
            for side in [0, 1]:
                text = ('Full original output 完整. ' * 1000) if index == 0 else ('Output' if side == 0 else '')
                row.update({f'response_{side}': text, f'response_{side}_source': 'model-' + str(side),
                    f'is_response_{side}_safe': side == 0,
                    f'response_{side}_harm_category': {name: (side == 1 or index == 1) and i == 0 for i, name in enumerate(categories)},
                    f'response_{side}_severity_level': side * 3,
                    f'response_{side}_sha256': hashlib.sha256((prompt + text).encode()).hexdigest()})
            self.native.append(row)
        self.native.append(copy.deepcopy(self.native[0]))
        variant = copy.deepcopy(self.native[0])
        variant['prompt'] = variant['prompt'] + ' \n'
        for side in [0, 1]:
            variant[f'response_{side}_sha256'] = hashlib.sha256((variant['prompt'] + variant[f'response_{side}']).encode()).hexdigest()
        self.native.append(variant)
        self.raw_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.native))
        self.builder = runpy.run_path(str(folder / 'build.py'))['PKUSafeRLHF']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_full_outputs_empty_text_repeated_records_and_original_disagreement(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pku_saferlhf
        self.assertEqual(_pku_saferlhf(self.directory, self.frames, self.metadata), dict(
            source_pairs=4, source_subjects=2, source_items=2, source_responses=8, source_traces=8,
            source_distinct_outputs=6, source_safe=4, source_unsafe=4, source_empty_outputs=1,
            source_category_disagreements=1, source_prompt_variants=3))
        self.assertGreater(self.frames['items'].content.str.len().max(), 16000)
        self.assertTrue(self.frames['responses'].trial.eq(1).all())
        self.assertEqual(self.frames['responses'].test_condition.nunique(), 8)

    def test_nonboolean_grade_and_wrong_hash_are_rejected(self):
        import copy
        import hashlib
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pku_saferlhf_sources
        for value in [None, 1, 'true']:
            with self.subTest(value=value):
                rows = copy.deepcopy(self.native)
                rows[0]['is_response_0_safe'] = value
                self.raw_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    _pku_saferlhf_sources(self.directory, self.metadata)
                with self.assertRaisesRegex(ValueError, 'original boolean'):
                    self.builder(str(self.directory / 'build.py')).build_tables()
        rows = copy.deepcopy(self.native)
        rows[0]['response_0_sha256'] = hashlib.sha256(rows[0]['response_0'].encode()).hexdigest()
        self.raw_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'hash covers prompt'):
            _pku_saferlhf_sources(self.directory, self.metadata)

    def test_wrong_associations_grades_and_truncated_or_rewritten_source_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pku_saferlhf, _pku_saferlhf_sources
        source = _pku_saferlhf_sources(self.directory, self.metadata)
        changes = ['grade', 'same_sum_swap', 'subject', 'item', 'prompt', 'rule', 'verifier', 'alias', 'feature',
            'trial', 'condition', 'interactors', 'drop', 'duplicate', 'model', 'configuration', 'extra_subject',
            'scale', 'trace_text', 'empty_text', 'category', 'safety_flag', 'hash', 'source_file', 'source_row', 'side']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'traces']]
                if change == 'grade': responses.loc[0, 'response'] = .5
                elif change == 'same_sum_swap':
                    indices = [responses.index[responses.response.eq(value)][0] for value in [0, 1]]
                    responses.loc[indices, 'response'] = [1., 0.]
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'human-rater'
                elif change == 'item': responses.loc[0, 'item_id'] = next(key for key in items.item_id if key != responses.loc[0, 'item_id'])
                elif change == 'prompt': items.loc[0, 'content'] = items.loc[0, 'content'][:16000]
                elif change == 'rule': items.loc[0, 'grading_criterion'] = json.dumps(dict(rule='Factual correctness'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(spec='{}'))
                elif change == 'alias': items.loc[0, 'raw_item_id'] = 'different-source-position'
                elif change == 'feature': items.loc[0, 'item_features'] = 'safe=true'
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'independent rerun'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented-rater'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'GPT-4'
                elif change == 'configuration': subjects.loc[0, 'subject_features_extra'] = 'source_model_label=wrong-model'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='interval', min=0, max=100))
                else:
                    index = 0
                    if change == 'empty_text':
                        index = next(i for i, value in enumerate(traces.trace) if json.loads(value)['record']['response_1'] == '')
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_text': value['record']['response_0'] = value['record']['response_0'][:16000]
                    elif change == 'empty_text': value['record']['response_1'] = 'Invented output'
                    elif change == 'category': value['record']['response_0_harm_category'] = {}
                    elif change == 'safety_flag': value['record']['is_response_0_safe'] = False
                    elif change == 'hash': value['record']['response_0_sha256'] = '0' * 64
                    elif change == 'source_file': value['source_file'] = 'other.jsonl'
                    elif change == 'source_row': value['source_row'] = 99
                    else: value['side'] = 1 - value['side']
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _pku_saferlhf(self.directory, frames, self.metadata, source)


class PRISMNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'prism'
        self.raw = self.directory / 'raw'
        (self.raw / 'protocol/data').mkdir(parents=True)
        folder = ROOT / 'benchmarks/prism'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        self.utterances = []
        history = []
        for turn in [0, 1]:
            prompt = 'Original user question ' + str(turn)
            history.append(dict(role='user', turn=turn, content=prompt))
            for within in [0, 1]:
                model = 'prism-fixture-a' if turn or within == 0 else 'prism-fixture-b'
                text = ('Full selected output 完整. ' * 1000 if turn == 0 and within == 0 else
                    'EMPTY STRING' if turn == 0 else 'Same selected response')
                score = (90 if within == 0 else 1) if turn == 0 else 50
                chosen = bool(turn or within == 0)
                record = dict(utterance_id='u' + str(len(self.utterances)), interaction_id='i' + str(turn),
                    conversation_id='c0', user_id='anonymous-rater', turn=turn, within_turn_id=within,
                    conversation_type='unguided', user_prompt=prompt, model_response=text, model_name=model,
                    model_provider='fixture-api', score=score, if_chosen=chosen, included_in_balanced_subset=True)
                self.utterances.append(record)
                history.append(dict(role='model', turn=turn, within_turn_id=within, content=text,
                    model_name=model, model_provider='fixture-api', score=score, if_chosen=chosen))
        conversations = [dict(conversation_id='c0', user_id='anonymous-rater', conversation_type='unguided',
            included_in_balanced_subset=True, conversation_history=history)]
        annotations = [dict(column_id='model_response', user_id='anonymous-rater', conversation_id='c0',
            interaction_id=row['interaction_id'], utterance_id=row['utterance_id'], pii_flag=False,
            pii_manual_flag=float('nan'), language_flag='en', en_flag=True,
            moderation_flag=dict(flagged=False, category_scores=dict(harm=0.010779580529581414))) for row in self.utterances]
        models = [dict(long_name=model, model_provider='configuration-provider', header='Original header for ' + model,
            selected_params=dict(temperature='1.0', max_tokens='256')) for model in ['prism-fixture-a', 'prism-fixture-b']]
        for relative,rows in [('utterances.jsonl',self.utterances),('conversations.jsonl',conversations),
            ('metadata.jsonl',annotations),('protocol/data/models.jsonl',models)]:
            (self.raw / relative).write_text(''.join(json.dumps(row,ensure_ascii=False) + '\n' for row in rows))
        self.builder = runpy.run_path(str(folder / 'build.py'))['PRISM']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source',str(self.raw),'--output',str(self.directory.parent / 'tables')])
        self.frames = {path.stem:pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_original_ratings_duplicate_choices_and_selected_history(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prism
        self.assertEqual(_prism(self.directory,self.frames,self.metadata), dict(source_subjects=2,source_items=4,
            source_responses=8,source_traces=8,source_utterances=4,source_conversations=1,source_turns=2,
            source_human_raters=1,source_chosen=3,source_empty_output_markers=1))
        self.assertGreater(self.frames['traces'].trace.str.len().max(),16000)
        self.assertGreater(self.frames['items'].content.str.len().max(),16000)
        self.assertEqual(self.frames['responses'].response.max(),90)

    def test_out_of_range_or_noninteger_native_ratings_are_rejected(self):
        import copy
        for value in [0,101,1.5,None,float('inf')]:
            with self.subTest(value=value):
                rows=copy.deepcopy(self.utterances)
                rows[0]['score']=value
                (self.raw/'utterances.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaisesRegex(ValueError,'finite integers from 1 through 100'):
                    self.builder(str(self.directory/'build.py')).build_tables()

    def test_wrong_context_rater_scale_or_output_associations_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prism,_prism_sources
        source=_prism_sources(self.directory,self.metadata)
        changes=['score','chosen','subject','item','context','leak','scale','rule','judge','alias','feature',
            'condition','trial','rater','drop','duplicate','model','configuration','extra_subject','trace_text',
            'trace_score','trace_chosen','metadata_precision','trace_rater','trace_dimension','source_file']
        for change in changes:
            with self.subTest(change=change):
                frames={name:frame.copy(deep=True) for name,frame in self.frames.items()}
                responses,items,subjects,traces=[frames[name] for name in ['responses','items','subjects','traces']]
                if change=='score':
                    index=responses.index[responses.test_condition.str.endswith(';measure=score')][0]
                    responses.loc[index,'response']/=100
                elif change=='chosen':
                    index=responses.index[responses.test_condition.str.endswith(';measure=if_chosen')][0]
                    responses.loc[index,'response']=1-responses.loc[index,'response']
                elif change=='subject':responses.loc[0,'subject_id']='anonymous-rater'
                elif change=='item':responses.loc[0,'item_id']=next(key for key in items.item_id if key!=responses.loc[0,'item_id'])
                elif change=='context':
                    index=items.content.str.len().idxmax();value=json.loads(items.loc[index,'content']);value[1]['content']=value[1]['content'][:16000];items.loc[index,'content']=json.dumps(value)
                elif change=='leak':
                    value=json.loads(items.loc[0,'content']);value.append(dict(role='assistant',content='Current target answer'));items.loc[0,'content']=json.dumps(value)
                elif change=='scale':items.loc[0,'grading_criterion']=json.dumps(dict(rule='normalized',response_scale=dict(kind='interval',min=0,max=1)))
                elif change=='rule':items.loc[0,'grading_criterion']=json.dumps(dict(rule='objective correctness'))
                elif change=='judge':items.loc[0,'verifier']=json.dumps(dict(spec='{}'))
                elif change=='alias':items.loc[0,'raw_item_id']='other-conversation'
                elif change=='feature':items.loc[0,'item_features']='dimension=unknown'
                elif change=='condition':responses.loc[0,'test_condition']='independent rerun'
                elif change=='trial':responses.loc[0,'trial']=2
                elif change=='rater':responses.loc[0,'interactors']=json.dumps(dict(human_rater='different-rater'))
                elif change=='drop':frames['responses']=responses.iloc[1:].copy()
                elif change=='duplicate':frames['responses']=pd.concat([responses,responses.iloc[:1]],ignore_index=True)
                elif change=='model':subjects.loc[0,'display_name']='anonymous-rater'
                elif change=='configuration':subjects.loc[0,'subject_features_extra']='source_model_name=wrong-model'
                elif change=='extra_subject':frames['subjects']=pd.concat([subjects,subjects.iloc[:1]],ignore_index=True)
                else:
                    value=json.loads(traces.loc[0,'trace'])
                    if change=='trace_text':value['utterance']['model_response']=value['utterance']['model_response'][:16000]
                    elif change=='trace_score':value['utterance']['score']=.9
                    elif change=='trace_chosen':value['utterance']['if_chosen']=False
                    elif change=='metadata_precision':value['metadata']['moderation_flag']['category_scores']['harm']=0.0107795805
                    elif change=='trace_rater':value['utterance']['user_id']='different-rater'
                    elif change=='trace_dimension':value['dimension']='score' if value['dimension']=='if_chosen' else 'if_chosen'
                    else:value['source_file']='different.jsonl'
                    traces.loc[0,'trace']=json.dumps(value)
                with self.assertRaises((ValueError,KeyError)):
                    _prism(self.directory,frames,self.metadata,source)


class PreferenceDissectionNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'preference_dissection'
        folder = ROOT / 'benchmarks/preference_dissection'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.raw_file = self.directory / 'raw' / self.metadata['build']['parameters']['layout']['data']
        self.raw_file.parent.mkdir(parents=True)
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        self.native = [dict(query='Full original query 完整. ' * 1000,
            response_1=dict(content='Full first candidate 完整. ' * 1000, model='candidate-author-a', num_words=2000),
            response_2=dict(content='Second candidate', model='candidate-author-b', num_words=2),
            preference_labels={'fixture-judge-a':'response_1', 'fixture-judge-b':'response_2', 'human':'response_2'}),
            dict(query='Another query', response_1=dict(content='A', model='author-a', num_words=1),
                 response_2=dict(content='', model='author-b', num_words=0),
                 preference_labels={'fixture-judge-a':'response_2', 'fixture-judge-b':'response_1', 'human':'response_1'})]
        import copy
        self.native.append(copy.deepcopy(self.native[0]))
        self.native[2]['response_1']['num_words'] = 1999
        pd.DataFrame(self.native).to_parquet(self.raw_file, index=False)
        self.builder = runpy.run_path(str(folder / 'build.py'))['PreferenceDissection']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem:pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_full_candidates_unordered_labels_and_repeated_source_rows(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _preference_dissection
        self.assertEqual(_preference_dissection(self.directory, self.frames, self.metadata), dict(
            source_pairs=3, source_subjects=2, source_items=2, source_responses=6, source_traces=6,
            source_response_1=3, source_response_2=3, source_empty_candidate_texts=1))
        self.assertGreater(self.frames['items'].content.str.len().max(), 16000)
        self.assertTrue(self.frames['responses'].trial.eq(1).all())
        self.assertEqual(self.frames['responses'].test_condition.nunique(), 3)
        self.assertTrue(all('trace' not in json.loads(value) for value in self.frames['traces'].trace))

    def test_unknown_or_missing_native_label_is_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _preference_dissection_sources
        import copy
        for value in [None, 'tie', 'response_A']:
            with self.subTest(value=value):
                rows = copy.deepcopy(self.native)
                rows[0]['preference_labels']['fixture-judge-a'] = value
                pd.DataFrame(rows).to_parquet(self.raw_file, index=False)
                with self.assertRaises(ValueError):
                    _preference_dissection_sources(self.directory, self.metadata)
                with self.assertRaisesRegex(ValueError, 'original response_1/response_2 label'):
                    self.builder(str(self.directory / 'build.py')).build_tables()

    def test_wrong_choices_links_clipped_stimuli_and_invented_reasoning_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _preference_dissection, _preference_dissection_sources
        source = _preference_dissection_sources(self.directory, self.metadata)
        changes = ['grade', 'same_sum_swap', 'subject', 'item', 'query_clip', 'candidate_clip', 'leak',
            'gold', 'verifier', 'alias', 'feature', 'trial', 'condition', 'interactors', 'drop', 'duplicate',
            'model', 'extra_subject', 'scale', 'trace_label', 'trace_model', 'source_file', 'source_row', 'reasoning']
        for change in changes:
            with self.subTest(change=change):
                frames = {name:frame.copy(deep=True) for name,frame in self.frames.items()}
                responses,items,subjects,traces = [frames[name] for name in ['responses','items','subjects','traces']]
                if change == 'grade': responses.loc[0, 'response'] = .5
                elif change == 'same_sum_swap': responses.loc[[0, 1], 'response'] = responses.loc[[1, 0], 'response'].to_numpy()
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'human'
                elif change == 'item': responses.loc[0, 'item_id'] = next(key for key in items.item_id if key != responses.loc[0, 'item_id'])
                elif change in ['query_clip', 'candidate_clip', 'leak']:
                    value = json.loads(items.loc[0, 'content'])
                    if change == 'query_clip': value['query'] = value['query'][:4000]
                    elif change == 'candidate_clip': value['response_1'] = value['response_1'][:8000]
                    else: value['preferred_response'] = 'response_1'
                    items.loc[0, 'content'] = json.dumps(value)
                elif change == 'gold': items.loc[0, 'grading_criterion'] = json.dumps(dict(reference_answer='A selected candidate'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(spec='{}'))
                elif change == 'alias': items.loc[0, 'raw_item_id'] = '99'
                elif change == 'feature': items.loc[0, 'item_features'] = 'preferred=response_1'
                elif change == 'trial': responses.loc[0, 'trial'] = 2
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'independent rerun'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'human'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses,responses.iloc[:1]],ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'candidate-author-a'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects,subjects.iloc[:1]],ignore_index=True)
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='discrete',values=[0,1],direction='higher_is_better'))
                else:
                    value = json.loads(traces.loc[0, 'trace'])
                    if change == 'trace_label': value['preference_label'] = 'response_2'
                    elif change == 'trace_model': value['source_model_label'] = 'candidate-author-a'
                    elif change == 'source_file': value['source_file'] = 'other.parquet'
                    elif change == 'source_row': value['source_row'] = 99
                    else: value['trace'] = 'Endorsed candidate falsely called judge reasoning'
                    traces.loc[0, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _preference_dissection(self.directory, frames, self.metadata, source)


class PredictionArenaNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'prediction_arena'
        self.raw = self.directory / 'raw'
        (self.raw / 'settlements').mkdir(parents=True)
        (self.raw / 'markets').mkdir()
        folder = ROOT / 'benchmarks/prediction_arena'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        agents = [dict(id='agent-a', model_id='prediction-arena-fixture-a', account_value=100, current_beliefs='Later state'),
                  dict(id='agent-b', model_id='prediction-arena-fixture-b', account_value=200, current_beliefs='Another later state')]
        (self.raw / 'agents.json').write_text(json.dumps(agents))
        records = [dict(id='settlement-a', ticker='market-a', result='yes', payout=20, realized_pnl=3.5,
                       settled_at='2026-01-01T00:00:00Z', created_at='2026-01-01T00:00:01Z'),
                   dict(id='settlement-b', ticker='market-b', result='yes', payout=20, realized_pnl=0.,
                       settled_at='2026-01-02T00:00:00Z', created_at='2026-01-02T00:00:01Z'),
                   dict(id='settlement-c', ticker='market-a', result='yes', payout=0, realized_pnl=-2.5,
                       settled_at='2026-01-01T00:00:00Z', created_at='2026-01-01T00:00:01Z')]
        (self.raw / 'settlements/agent-a.json').write_text(json.dumps(records[:2]))
        (self.raw / 'settlements/agent-b.json').write_text(json.dumps(records[2:]))
        for record in records:
            market = dict(ticker=record['ticker'], event_ticker='event-' + record['ticker'], title='Original market question ' + record['ticker'],
                subtitle='Original subtitle', yes_sub_title='Yes', no_sub_title='No', close_time='2026-01-01T00:00:00Z',
                rules_primary='Original complete rules 完整. ' * 1000, rules_secondary='Original supplementary rules',
                result=record['result'], last_price_dollars='0.99', settlement_value_dollars='1.00')
            (self.raw / 'markets' / (record['id'] + '.json')).write_text(json.dumps(dict(markets=[market], cursor=''), ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['PredictionArena']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.raw), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_settlement_profit_is_distinct_from_resolution_and_context_is_preserved(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prediction_arena
        self.assertEqual(_prediction_arena(self.directory, self.frames, self.metadata),
            dict(source_subjects=2, source_markets=2, source_responses=3, source_traces=3,
                 source_positive_profit=1, source_zero_profit=1, source_negative_profit=1))
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)
        self.assertGreater(self.frames['items'].content.str.len().max(), 16000)
        self.assertEqual(self.frames['responses'].response.sum(), 1.)

    def test_invalid_source_profit_is_not_silently_converted_to_failure(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prediction_arena_sources
        path = self.raw / 'settlements/agent-a.json'
        original = json.loads(path.read_text())
        for value in [None, float('nan'), float('inf'), float('-inf')]:
            with self.subTest(value=value):
                records = [dict(row) for row in original]
                records[0]['realized_pnl'] = value
                path.write_text(json.dumps(records))
                with self.assertRaises(ValueError):
                    _prediction_arena_sources(self.directory, self.metadata)
                with self.assertRaisesRegex(ValueError, 'finite realized profit'):
                    self.builder(str(self.directory / 'build.py')).build_tables()
        path.write_text(json.dumps(original))

    def test_corruptions_of_profit_market_subject_and_source_coverage_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prediction_arena, _prediction_arena_sources
        source = _prediction_arena_sources(self.directory, self.metadata)
        changes = ['grade', 'zero', 'subject', 'market', 'question', 'rules', 'leak', 'criterion', 'verifier',
            'alias', 'feature', 'trial', 'condition', 'interactors', 'drop', 'duplicate', 'clip', 'native_profit',
            'native_resolution', 'native_market', 'source_file', 'source_row', 'model', 'extra_subject', 'scale']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'traces']]
                if change == 'grade': responses.loc[0, 'response'] = .5
                elif change == 'zero': responses.loc[responses.response.eq(0), 'response'] = 1.
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'invented-model'
                elif change == 'market': responses.loc[0, 'item_id'] = next(value for value in items.item_id if value != responses.loc[0, 'item_id'])
                elif change in ['question', 'rules', 'leak']:
                    value = json.loads(items.loc[0, 'content'])
                    if change == 'question': value['title'] = 'Question guessed from a ticker'
                    elif change == 'rules': value['rules_primary'] = value['rules_primary'][:16000]
                    else: value['result'] = 'yes'
                    items.loc[0, 'content'] = json.dumps(value)
                elif change == 'criterion': items.loc[0, 'grading_criterion'] = json.dumps(dict(rule='Predict yes correctly'))
                elif change == 'verifier': items.loc[0, 'verifier'] = json.dumps(dict(class_='judge', spec='{}'))
                elif change == 'alias': items.loc[0, 'raw_item_id'] = 'absent-market'
                elif change == 'feature': items.loc[0, 'item_features'] = 'platform=polymarket'
                elif change == 'trial': responses.loc[0, 'trial'] = 99
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'invented occasion'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented actor'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = 'model-inferred-from-profit'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
                elif change == 'scale': frames['benchmarks'].loc[0, 'response_scale'] = json.dumps(dict(kind='discrete', values=[0, 2]))
                else:
                    value = json.loads(traces.loc[0, 'trace'])
                    if change == 'clip': value['market']['rules_primary'] = value['market']['rules_primary'][:16000]
                    elif change == 'native_profit': value['settlement']['realized_pnl'] += .01
                    elif change == 'native_resolution': value['settlement']['result'] = 'no'
                    elif change == 'native_market': value['market']['ticker'] = 'wrong-market'
                    elif change == 'source_file': value['source_file'] = 'wrong.json'
                    else: value['source_row'] = 99
                    traces.loc[0, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _prediction_arena(self.directory, frames, self.metadata, source)


class PRM800KNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import copy
        import io
        import runpy
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'prm800k'
        self.raw = self.directory / 'raw/prm800k/data'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/prm800k'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        long = 'A complete generated step 完整. ' * 1500
        first = dict(labeler='original-rater-1', timestamp='2023-01-01T00:00:00', generation=None,
            is_quality_control_question=False, is_initial_screening_question=False,
            question=dict(problem='A fixture math problem.', ground_truth_solution='The released solution.', ground_truth_answer='42'),
            label=dict(total_time=111, finish_reason='solution', steps=[
                dict(completions=[dict(text=long, rating=1, flagged=False), dict(text='Neutral', rating=0, flagged=None),
                    dict(text='Incorrect', rating=-1, flagged=False), dict(text='', rating=None, flagged=True)],
                    chosen_completion=None, human_completion=dict(text='A human replacement, not an AI response.')),
                dict(completions=None, chosen_completion=None, human_completion=dict(text='A human-only continuation.')),
                dict(completions=[dict(text='After the human replacement', rating=1, flagged=None)],
                    chosen_completion=0, human_completion=None)]))
        second = dict(labeler='original-rater-2', timestamp='2023-02-01T00:00:00', generation=7,
            is_quality_control_question=True, is_initial_screening_question=False,
            question=dict(problem='Another fixture math problem.', ground_truth_solution='Another released solution.',
                ground_truth_answer='24', pre_generated_steps=['Generated first', 'Original second', 'Unannotated tail'],
                pre_generated_answer='23', pre_generated_verifier_score=0.010779580529581414),
            label=dict(total_time=222, finish_reason='give_up', steps=[
                dict(completions=[dict(text='Generated first', rating=0, flagged=False)], chosen_completion=None, human_completion=None),
                dict(completions=[dict(text='Alternative second', rating=-1, flagged=True),
                    dict(text='Other alternative', rating=None, flagged=False)], chosen_completion=1, human_completion=None)]))
        repeated = copy.deepcopy(second)
        repeated.update(labeler='original-rater-3', generation=8, is_initial_screening_question=True)
        repeated['label']['steps'][0]['completions'][0]['rating'] = 1
        del second['question']['ground_truth_solution']
        repeated['question']['ground_truth_solution'] = None
        del repeated['question']['ground_truth_answer']
        (self.raw / 'phase1_train.jsonl').write_text(json.dumps(first, ensure_ascii=False) + '\n')
        (self.raw / 'phase2_train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in [second, repeated]))
        builder = runpy.run_path(str(folder / 'build.py'))['PRM800K']
        with contextlib.redirect_stdout(io.StringIO()):
            builder(str(self.directory / 'build.py')).main_from_args([
                '--source', str(self.directory / 'raw'), '--output', str(self.directory.parent / 'tables')])
        self.frames = {path.stem: pd.read_parquet(path) for path in (self.directory.parent / 'tables').glob('*.parquet')}

    def test_native_scale_human_prefix_ungraded_tails_and_repeated_annotations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prm800k
        counts = _prm800k(self.directory, self.frames, self.metadata)
        self.assertEqual(counts['source_records'], 3)
        self.assertEqual(counts['source_subjects'], 1)
        self.assertEqual(counts['source_candidates'], 11)
        self.assertEqual(counts['source_responses'], 15)
        self.assertEqual(counts['source_traces'], 15)
        self.assertEqual(counts['source_neutral_ratings'], 2)
        self.assertEqual(counts['source_ungraded_candidates'], 3)
        self.assertEqual(counts['source_ungraded_pregenerated_steps'], 4)
        self.assertEqual(counts['source_human_replacements'], 2)
        self.assertEqual(counts['source_missing_nonfinal_selections'], 2)
        self.assertEqual(counts['source_pregenerated_candidate_mismatches'], 2)
        self.assertEqual(self.frames['responses'].response.isna().sum(), 7)
        self.assertGreater(self.frames['traces'].trace.str.len().max(), 16000)

    def test_corruptions_of_context_scale_rater_or_source_coverage_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _prm800k, _prm800k_sources
        source = _prm800k_sources(self.directory, self.metadata)
        changes = ['grade', 'neutral', 'null_to_zero', 'subject', 'item', 'history', 'answer_leak',
            'scale', 'reference', 'rule', 'judge', 'instructions', 'alias', 'feature', 'trial', 'condition', 'interactors',
            'drop', 'duplicate', 'trace_clip', 'trace_rater', 'trace_round', 'trace_origin', 'trace_grade',
            'trace_selected', 'trace_pregenerated', 'model', 'extra_subject']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                responses, items, subjects, traces = [frames[name] for name in ['responses', 'items', 'subjects', 'traces']]
                if change == 'grade': responses.loc[0, 'response'] = 0.5
                elif change == 'neutral': responses.loc[responses.response.eq(0), 'response'] = -1.
                elif change == 'null_to_zero': responses.loc[responses.response.isna(), 'response'] = 0.
                elif change == 'subject': responses.loc[0, 'subject_id'] = 'invented-generator'
                elif change == 'item': responses.loc[0, 'item_id'] = next(value for value in items.item_id if value != responses.loc[0, 'item_id'])
                elif change in ['history', 'answer_leak']:
                    index = next(i for i, value in enumerate(items.content) if json.loads(value)['prior_solution_steps'])
                    value = json.loads(items.loc[index, 'content'])
                    if change == 'history': value['prior_solution_steps'] = []
                    else: value['prior_solution_steps'].append('The current answer must not be part of the input')
                    items.loc[index, 'content'] = json.dumps(value)
                elif change in ['scale', 'reference', 'rule']:
                    value = json.loads(items.loc[0, 'grading_criterion'])
                    if change == 'scale': value['response_scale'] = dict(kind='discrete', values=[0, 1])
                    elif change == 'reference': value['reference_answer'] = 'A model answer is not a reference solution'
                    else: value['rule'] = 'Grade only the final answer'
                    items.loc[0, 'grading_criterion'] = json.dumps(value)
                elif change in ['judge', 'instructions']:
                    value = json.loads(items.loc[0, 'verifier'])
                    if change == 'judge': value['judged_by'] = 'llm'
                    else: value['spec'] = json.dumps({'phase': 'invented'})
                    items.loc[0, 'verifier'] = json.dumps(value)
                elif change == 'alias': items.loc[0, 'raw_item_id'] = 'absent.jsonl:row=999:step=999'
                elif change == 'feature': items.loc[0, 'item_features'] = 'phase=phase1;step_index=999'
                elif change == 'trial': responses.loc[0, 'trial'] = 999
                elif change == 'condition': responses.loc[0, 'test_condition'] = 'invented annotation'
                elif change == 'interactors': responses.loc[0, 'interactors'] = 'invented actor'
                elif change == 'drop': frames['responses'] = responses.iloc[1:].copy()
                elif change == 'duplicate': frames['responses'] = pd.concat([responses, responses.iloc[:1]], ignore_index=True)
                elif change == 'model': subjects.loc[0, 'display_name'] = '7'
                elif change == 'extra_subject': frames['subjects'] = pd.concat([subjects, subjects], ignore_index=True)
                else:
                    index = traces.trace.str.len().idxmax()
                    value = json.loads(traces.loc[index, 'trace'])
                    if change == 'trace_clip': value['text'] = value['text'][:4000]
                    elif change == 'trace_rater': value['annotation']['labeler'] = 'wrong-rater'
                    elif change == 'trace_round': value['annotation']['generation'] = 99
                    elif change == 'trace_origin': value['origin'] = 'human_completion'
                    elif change == 'trace_grade': value['native_candidate']['rating'] = -1
                    elif change == 'trace_selected': value['native_step']['chosen_completion'] = 0
                    else: value['pregenerated_step'] = 'Wrong original step'
                    traces.loc[index, 'trace'] = json.dumps(value)
                with self.assertRaises((ValueError, KeyError)):
                    _prm800k(self.directory, frames, self.metadata, source)


class PsychosisBenchNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'psychosis_bench'
        self.raw = self.directory / 'raw'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/psychosis_bench'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        case = dict(id='scenario_one', name='Original scenario', theme='Fixture theme',
                    condition='Explicit', harm_type='Original harm context', prompts=['Prompt '+str(turn) for turn in range(1, 13)])
        sections = ['# Original batch log']
        for experiment, model in enumerate(['fixture/model-a', 'fixture/model-b'], 1):
            header = ['## Experiment '+str(experiment)+': '+case['name']+' × '+model, '## Experiment Details']
            header += ['- **'+label+':** '+value for label,value in [('Model',model),('Test Case',case['name']),
                ('Theme',case['theme']),('Condition',case['condition']),('Harm Type',case['harm_type'])]]
            turns, scores = [], {name:[] for name in ['DCS','HES','SIS']}
            for turn, prompt in enumerate(case['prompts'],1):
                answer = '' if (experiment,turn)==(1,1) else model+' 完整\n' * (5000 if turn==7 else 1)
                body = ['### Turn '+str(turn), '', '**👤 User:**', prompt, '', '**🤖 Assistant:**', answer, '']
                applicable = ['DCS'] if turn>=4 else []
                if turn>=7:applicable += ['HES','SIS']
                if applicable:
                    body += ['**📊 Scores:**']
                    for dimension in applicable:
                        if (experiment,turn,dimension)==(2,9,'DCS'):continue
                        value = turn % (2 if dimension=='SIS' else 3)
                        scores[dimension].append(value)
                        body += ['- '+dimension+' (Published rating): '+str(value)]
                    body += ['']
                turns += body + ['---','']
            summary = [('Total Turns','12'),('Average DCS (Delusion Confirmation)',f'{sum(scores["DCS"])/len(scores["DCS"]):.3f}'),
                ('Average HES (Harm Enablement)',f'{sum(scores["HES"])/len(scores["HES"]):.3f}'),
                ('Total SIS (Safety Intervention)',str(sum(scores['SIS']))),('Max DCS',str(max(scores['DCS']))),
                ('Max HES',str(max(scores['HES'])))]
            sections += header + ['', '## Summary Scores'] + ['- **'+name+':** '+value for name,value in summary]
            sections += ['', '## Conversation'] + turns + ['## Metadata','- **judge_model:** fixture/original-judge','','---','']
        layout = self.metadata['build']['parameters']['layout']
        self.documents = {layout['cases']:json.dumps(dict(cases=[case])), layout['log']:'\n'.join(sections),
                          layout['scorer']:'Original grading implementation retained as text, never executed.'}
        with tarfile.open(self.raw/layout['archive'],'w:gz') as archive:
            for filename, content in self.documents.items():
                data=content.encode(); member=tarfile.TarInfo('original/'+filename); member.size=len(data)
                archive.addfile(member,io.BytesIO(data))
        builder = runpy.run_path(str(folder/'build.py'))['PsychosisBench']
        with contextlib.redirect_stdout(io.StringIO()):
            builder(str(self.directory/'build.py')).main_from_args([
                '--source',str(self.raw),'--output',str(self.directory.parent/'tables')])
        self.frames={path.stem:pd.read_parquet(path) for path in (self.directory.parent/'tables').glob('*.parquet')}

    def test_full_histories_distinct_rubrics_and_unavailable_grade(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _psychosis_bench
        self.assertEqual(_psychosis_bench(self.directory,self.frames,self.metadata),dict(source_subjects=2,
            source_items=42,source_responses=42,source_traces=42,source_scenarios=1,source_experiments=2,
            source_turns=24,source_blank_generations=1,source_graded_responses=41,source_dcs_responses=18,
            source_hes_responses=12,source_sis_responses=12,source_ungraded_responses=1))
        self.assertGreater(self.frames['traces'].trace.str.len().max(),16000)
        self.assertEqual(self.frames['responses'].response.isna().sum(),1)

    def test_changed_associations_grades_history_and_judge_are_rejected(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _psychosis_bench, _psychosis_bench_sources
        source = _psychosis_bench_sources(self.directory,self.metadata)
        changes = ['grade','null_to_zero','subject','item','history_clip','answer_leak','alias','dimension',
            'scale','direction','rule','judge','sis_mode','trial','condition','interactors','drop','duplicate',
            'trace_clip','trace_configuration','trace_grade','model']
        for change in changes:
            with self.subTest(change=change):
                frames={name:frame.copy(deep=True) for name,frame in self.frames.items()}
                responses,items,subjects,traces=[frames[name] for name in ['responses','items','subjects','traces']]
                if change=='grade':responses.loc[0,'response']=float(responses.loc[0,'response'])+0.5
                elif change=='null_to_zero':responses.loc[responses.response.isna(),'response']=0.
                elif change=='subject':responses.loc[0,'subject_id']=next(value for value in subjects.subject_id if value!=responses.loc[0,'subject_id'])
                elif change=='item':responses.loc[0,'item_id']=next(value for value in items.item_id if value!=responses.loc[0,'item_id'])
                elif change in ['history_clip','answer_leak']:
                    value=json.loads(items.loc[0,'content'])
                    if change=='history_clip':value=value[-1:]
                    else:value.append(dict(role='assistant',content='Leaked current answer'))
                    items.loc[0,'content']=json.dumps(value)
                elif change=='alias':items.loc[0,'raw_item_id']='999:turn_4:DCS'
                elif change=='dimension':items.loc[0,'item_features']=items.loc[0,'item_features'].replace('dimension=DCS','dimension=HES')
                elif change in ['scale','direction','rule']:
                    value=json.loads(items.loc[0,'grading_criterion'])
                    if change=='scale':value['response_scale']['values']=[0,1]
                    elif change=='direction':value['response_scale']['direction']='higher_is_better'
                    else:value['rule']='An invented binary success criterion'
                    items.loc[0,'grading_criterion']=json.dumps(value)
                elif change in ['judge','sis_mode']:
                    index=0 if change=='judge' else next(i for i,v in enumerate(items.raw_item_id) if v.endswith(':SIS'))
                    value=json.loads(items.loc[index,'verifier'])
                    if change=='judge':value['judge']='A guessed paper judge'
                    else:value['judged_by']='llm'
                    items.loc[index,'verifier']=json.dumps(value)
                elif change=='trial':responses.loc[0,'trial']=2
                elif change=='condition':responses.loc[0,'test_condition']='Guessed temperature'
                elif change=='interactors':responses.loc[0,'interactors']=json.dumps(dict(user_sim='A different user model'))
                elif change=='drop':frames['responses']=responses.iloc[1:].copy()
                elif change=='duplicate':frames['responses']=pd.concat([responses,responses.iloc[:1]],ignore_index=True)
                elif change.startswith('trace_'):
                    index=traces.trace.str.len().idxmax(); value=json.loads(traces.loc[index,'trace'])
                    if change=='trace_clip':value['assistant']=value['assistant'][:16000]
                    elif change=='trace_configuration':value['configuration']='Guessed runtime config'
                    else:value['released_rating']=0.5
                    traces.loc[index,'trace']=json.dumps(value)
                else:subjects.loc[0,'display_name']='A guessed checkpoint'
                with self.assertRaises((ValueError,KeyError)):_psychosis_bench(self.directory,frames,self.metadata,source)


class ProgramBenchNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import io
        import runpy
        import tarfile
        from measurement_db.build_base import _tables

        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'programbench'
        self.raw = self.directory / 'raw'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/programbench'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.metadata['build']['parameters']['runs'] = {'run_a':'run_a.tar.gz', 'run_b':'run_b.tar.gz'}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        self.documents = {'registry': {'ignored_tests.json': {'task_one':['suite/ignored']}}}
        for run in ['run_a','run_b']:
            manifest = dict(schema_version=1, submission_id=run,
                system=dict(model='Fictional '+run,provider='Fixture provider'),eval=dict(programbench_version='1.0'))
            records = {'submission.yaml':manifest}
            self.documents['registry'][f'submissions/{run}/submission.yaml'] = manifest
            scores = {}
            for task in ['task_one','task_two']:
                config = dict(agent=dict(system_template='Fixture system',instance_template='Fixture task',step_limit=1000,
                    output_path='/source/'+run+'/'+task),agent_type='FixtureAgent',model=dict(model_name='fixture/'+run,
                    model_kwargs=dict(reasoning_effort='medium',temperature=0.939,max_tokens=512 if run=='run_a' and task=='task_two' else 1024)),
                    model_type='FixtureModel',environment=dict(image='fixture/'+task+':original',cwd='/workspace',timeout=180),
                    environment_type='FixtureEnvironment')
                trajectory = dict(info=dict(config=config,mini_version='2.0',instance_id=task,exit_status='Submitted'),
                    messages=[dict(role='system',content='Exact source system instructions'),
                        dict(role='user',content='Original task '+task+' 完整'),
                        dict(object='response',output=[dict(content='Full output 完整\n' * 2000)])])
                records[task+'/'+task+'.traj.json'] = trajectory
                if run=='run_b' and task=='task_two':continue
                verdicts = {'suite/pass':True,'suite/fail':False,'suite/ignored':False} if task=='task_one' else {}
                scores[task] = verdicts
                results = [dict(branch='suite',name='pass',status='failure',extra={'message':'Earlier failed retry'})] if verdicts else []
                results += [dict(branch='suite',name=name.split('/')[1],status='passed' if value else 'failure',extra={}) for name,value in verdicts.items()]
                records[task+'/'+task+'.eval.json'] = dict(test_results=results,error_code='compile_failed' if not verdicts else None,log=[])
            self.documents[run] = records
            self.documents['registry'][f'submissions/{run}/_stats/score.json'] = scores
        for name, documents in self.documents.items():
            with tarfile.open(self.raw/(name+'.tar.gz'),'w:gz') as archive:
                for filename, content in documents.items():
                    data = (yaml.safe_dump(content) if filename.endswith('.yaml') else json.dumps(content,ensure_ascii=False)).encode()
                    member = tarfile.TarInfo('original/'+filename)
                    member.size = len(data)
                    archive.addfile(member,io.BytesIO(data))
        self.builder = runpy.run_path(str(folder/'build.py'))['ProgramBench']
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory/'build.py')).main_from_args([
                '--source',str(self.raw),'--output',str(self.directory.parent/'tables')])
        self.frames = {path.stem:pd.read_parquet(path) for path in (self.directory.parent/'tables').glob('*.parquet')}

    def test_native_fractions_failed_retries_ungraded_attempt_and_complete_traces(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _programbench
        self.assertEqual(_programbench(self.directory,self.frames,self.metadata),dict(source_responses=4,
            source_subjects=3,source_items=3,source_traces=4,source_runs=2,source_tasks=2,
            source_native_test_verdicts=6,source_graded_attempts=3,source_evaluation_errors=1,source_ungraded_attempts=1))
        self.assertEqual(sorted(self.frames['responses'].response.dropna()),[0.,0.5,0.5])
        self.assertEqual(self.frames['responses'].loc[self.frames['responses'].response.eq(0.5),'item_id'].nunique(),1)

    def test_audit_rejects_changed_inputs_scores_and_native_records(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _programbench, _programbench_sources
        source = _programbench_sources(self.directory,self.metadata)
        changes = ['fraction','null_to_zero','subject','item','prompt','image','verifier','grading_rule','trace_clip',
            'trace_task','trace_grade','trace_settings','trace_retry','trial','condition','drop','duplicate','model','version','effort']
        for change in changes:
            with self.subTest(change=change):
                frames = {name:frame.copy(deep=True) for name,frame in self.frames.items()}
                responses,items,subjects,traces = [frames[name] for name in ['responses','items','subjects','traces']]
                if change=='fraction': responses.loc[responses.response.eq(0.5),'response']=1.
                elif change=='null_to_zero': responses.loc[responses.response.isna(),'response']=0.
                elif change=='subject': responses.loc[0,'subject_id']=next(value for value in subjects.subject_id if value!=responses.loc[0,'subject_id'])
                elif change=='item': responses.loc[0,'item_id']=next(value for value in items.item_id if value!=responses.loc[0,'item_id'])
                elif change in {'prompt','image'}:
                    value=json.loads(items.loc[0,'content'])
                    if change=='prompt':value['messages'][1]['content']='A reconstructed placeholder'
                    else:value['environment']['image']='fixture/current:latest'
                    items.loc[0,'content']=json.dumps(value)
                elif change=='verifier':
                    index=next(i for i,value in enumerate(items.verifier) if json.loads(json.loads(value)['spec'])['active_test_names'])
                    value=json.loads(items.loc[index,'verifier']);spec=json.loads(value['spec']);spec['active_test_names'].pop();value['spec']=json.dumps(spec);items.loc[index,'verifier']=json.dumps(value)
                elif change=='grading_rule':items.loc[0,'grading_criterion']=json.dumps(dict(rule='Binary threshold'))
                elif change.startswith('trace_'):
                    index=next(i for i,value in enumerate(traces.trace) if json.loads(value)['published_tests'])
                    value=json.loads(traces.loc[index,'trace'])
                    if change=='trace_clip':value['trajectory']['messages'][2]['output'][0]['content']='clipped'
                    elif change=='trace_task':value['task_id']='task_two'
                    elif change=='trace_grade':value['published_tests']['suite/pass']=False
                    elif change=='trace_settings':value['trajectory']['info']['config']['model']['model_kwargs']['temperature']=1.
                    else:value['evaluation']['test_results'].pop(0)
                    traces.loc[index,'trace']=json.dumps(value)
                elif change=='trial':responses.loc[0,'trial']=2
                elif change=='condition':responses.loc[0,'test_condition']='different'
                elif change=='drop':frames['responses']=responses.iloc[1:].copy()
                elif change=='duplicate':frames['responses']=pd.concat([responses,responses.iloc[:1]],ignore_index=True)
                elif change=='model':subjects.loc[0,'display_name']='A guessed checkpoint'
                elif change=='version':subjects.loc[0,'harness_version']='current'
                elif change=='effort':subjects.loc[0,'reasoning_effort']='high'
                with self.assertRaises((ValueError,KeyError)):
                    _programbench(self.directory,frames,self.metadata,source)


class PxploreNativeAuditTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import copy
        import io
        import runpy
        from measurement_db.build_base import _tables

        scratch = ROOT / 'artifacts'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_tables.reload)
        _tables.reload()
        self.directory = Path(temporary.name) / 'pxplore'
        self.raw = self.directory / 'raw/release'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/pxplore'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        for name in ['service/scripts/prompts/snippet_selection.txt', 'model/prompts/eval_profile.txt']:
            path = self.raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Complete original fixture prompt 完整\n' + name)
        for model in ['prompt_stub', 'steering_stub', 'grpo_stub']:
            inputs, evaluations = [], []
            for index in range(2):
                initial = dict(state_description='Full original learner state',
                    long_term_objective=[dict(description='Goal A | metric: recall | threshold: >=0.8 | evidence: before', is_aligned=False),
                                         dict(description='Goal B | evidence: before', is_aligned=False)],
                    short_term_objective=[dict(description='Goal C | evidence: before', is_aligned=False)],
                    implicit_motivation=[], explicit_motivation=[])
                recommendation = dict(title='Selected lesson', content='Entire lesson')
                after = copy.deepcopy(initial)
                if index == 1 and model != 'grpo_stub':
                    recommendation = None
                elif model == 'prompt_stub':
                    after['long_term_objective'][0].update(description='Goal A | metric: recall | threshold: >=0.8 | evidence: after', is_aligned=True)
                    after['long_term_objective'][1] = dict(description='Added Goal D | evidence: after', is_aligned=False)
                    after['short_term_objective'][0].pop('is_aligned')
                elif model == 'steering_stub':
                    after['long_term_objective'][1]['is_aligned'] = True
                else:
                    after['long_term_objective'][index]['is_aligned'] = True
                record = dict(course='Course ' + str(index), student_profile=initial,
                    interaction_history=[dict(role='Fictional learner', content='Original question 完整 ' + str(index)),
                                         dict(role='Tutor', content='Full original reply')],
                    recommend_candidates=[dict(content='Original candidate ' + str(index), score=0.939, metadata=dict(id='candidate-1'))],
                    recommend_snippet_id='candidate-1', recommend_content=recommendation,
                    recommend_reason='Full recorded recommendation explanation 完整 ' * 2000)
                inputs.append(record)
                evaluations.append(dict(course=record['course'], initial_state=initial, next_lesson=recommendation, next_state=after))
            for subfolder, records in [('test', inputs), ('eval', evaluations)]:
                path = self.raw / 'model/data' / subfolder / (model + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(records, ensure_ascii=False))
        self.builder = runpy.run_path(str(folder / 'build.py'))['Pxplore']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_complete_inputs_omitted_added_and_ungraded_criteria(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pxplore
        self.assertEqual(_pxplore(self.directory, self.frames, self.metadata), dict(source_responses=19,
            source_subjects=3, source_items=13, source_traces=19, source_calls=6, source_sessions=2,
            source_initial_components=18, source_next_components=18, source_matched_criteria=17,
            source_omitted_criteria=1, source_added_or_rewritten_criteria=1, source_missing_grade_fields=1,
            source_aligned=4, source_not_aligned=13, source_ungraded=2, source_fallback_calls=2))

    def test_audit_rejects_corrupted_grades_inputs_and_records(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pxplore, _pxplore_sources
        source = _pxplore_sources(self.directory, self.metadata)
        changes = ['grade_swap', 'null_to_zero', 'subject', 'item', 'prompt', 'candidate', 'axis', 'rule', 'verifier',
            'answer', 'history', 'state', 'judgment', 'source_row', 'source_file', 'status', 'condition', 'trial',
            'drop', 'duplicate', 'drop_trace', 'settings']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; old = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(old), column].iloc[0]
                elif change in ['prompt', 'candidate']:
                    content = json.loads(frames['items'].loc[0, 'content'])
                    if change == 'prompt': content['messages'][0]['content'] = 'Wrong system prompt'
                    else:
                        user = json.loads(content['messages'][1]['content']); user['candidates'][0]['content'] = 'Wrong candidate'
                        content['messages'][1]['content'] = json.dumps(user)
                    frames['items'].loc[0, 'content'] = json.dumps(content, ensure_ascii=False)
                elif change == 'axis': frames['items'].loc[0, 'item_features'] = frames['items'].loc[0, 'item_features'].replace('long_term_objective', 'short_term_objective')
                elif change == 'rule':
                    rule = json.loads(frames['items'].loc[0, 'grading_criterion']); rule['rule'] += ' Wrong criterion'
                    frames['items'].loc[0, 'grading_criterion'] = json.dumps(rule)
                elif change == 'verifier':
                    verifier = json.loads(frames['items'].loc[0, 'verifier']); verifier['judged_by'] = 'human'
                    frames['items'].loc[0, 'verifier'] = json.dumps(verifier)
                elif change in ['answer', 'history', 'state', 'judgment', 'source_row', 'source_file', 'status']:
                    trace = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'answer': trace['native_input']['recommend_reason'] = trace['native_input']['recommend_reason'][:16000]
                    elif change == 'history': trace['native_input']['interaction_history'] = []
                    elif change == 'state': trace['native_input']['student_profile']['state_description'] = 'Wrong learner state'
                    elif change == 'judgment': trace['native_evaluation']['next_state']['state_description'] = 'Wrong judged state'
                    elif change == 'source_row': trace['source_row'] = 1 - trace['source_row']
                    elif change == 'source_file': trace['source_file'] = 'unrelated.json'
                    else: trace['component_status'] = 'wrong_status'
                    frames['traces'].loc[0, 'trace'] = json.dumps(trace)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'wrong_method'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                else: frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                with self.assertRaises((ValueError, KeyError)):
                    _pxplore(self.directory, frames, self.metadata, source)

    def test_source_checks_reject_ambiguous_keys_wrong_associations_and_nonboolean_grades(self):
        import copy
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _pxplore_sources
        path = self.raw / 'model/data/eval/prompt_stub.json'
        original = path.read_text()
        for change in ['duplicate', 'grade', 'course', 'profile', 'recommendation', 'fallback']:
            records = json.loads(original)
            if change == 'duplicate': records[0]['next_state']['long_term_objective'].append(copy.deepcopy(records[0]['next_state']['long_term_objective'][0]))
            elif change == 'grade': records[0]['next_state']['long_term_objective'][0]['is_aligned'] = 'false'
            elif change == 'course': records[0]['course'] = 'Wrong course'
            elif change == 'profile': records[0]['initial_state']['state_description'] = 'Wrong profile'
            elif change == 'recommendation': records[0]['next_lesson']['title'] = 'Wrong lesson'
            else: records[1]['next_state']['long_term_objective'][0]['is_aligned'] = True
            path.write_text(json.dumps(records))
            with self.assertRaises(ValueError): _pxplore_sources(self.directory, self.metadata)
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()


class SynthPAINativeAuditTests(unittest.TestCase):
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
        self.directory = Path(temporary.name) / 'synthpai'
        self.raw = self.directory / 'raw/release'
        self.raw.mkdir(parents=True)
        folder = ROOT / 'benchmarks/synthpai'
        self.metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        self.models = ['gpt-4', 'meta-llama/Llama-2-13b-chat-hf']
        for name in ['inputs', 'logs']:
            self.metadata['build']['parameters'][name] = {model: self.metadata['build']['parameters'][name][model] for model in self.models}
        (self.directory / 'metadata.yaml').write_text(yaml.safe_dump(self.metadata))
        profiles, source_rows = [], {model: [] for model in self.models}
        grades = [[[1, .5], [0, None]], [[0, 1], [.5, 0]]]
        for index, username in enumerate(['profile-a', 'profile-b']):
            comments = [dict(text=' Complete comment 完整 ' * 1300 + str(index), username=username, pii={}),
                        dict(text='Second original comment ' + str(index), username=username, pii={})]
            review = {attribute: dict(estimate='GOLD_DO_NOT_INCLUDE_' + attribute, hardness=2, certainty=3, acc_gt=1)
                      for attribute in ['age', 'sex']}
            profile = dict(username=username, comments=comments, num_comments=2,
                reviews=dict(human=review, human_evaluated=review), predictions={}, evaluations={})
            for model_index, model in enumerate(self.models):
                answer = 'Full original answer 完整 ' * 1200 + username + model
                prediction = dict(full_answer=answer, age=dict(inference='Original reasoning', guess=['35', '40']),
                                  sex=dict(inference='Other reasoning', guess=['male', 'female']))
                profile['predictions'][model] = prediction
                profile['evaluations'][model] = dict(human_evaluated={attribute: [] if grade is None else [grade, 0., .5]
                    for attribute, grade in zip(['age', 'sex'], grades[index][model_index])})
                source_rows[model].append(dict(username='wrong-original-name' if (index, model_index) == (0, 1) else username,
                    comments=comments[:1] if (index, model_index) == (0, 1) else comments,
                    reviews=profile['reviews'], predictions={model: prediction}))
            profiles.append(profile)
        path = self.raw / 'data/synthpai_merged_evals.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in profiles))
        for model in self.models:
            # Unrelated skipped profiles share an answer; they must not make the measured join ambiguous.
            source_rows[model] += [dict(username='unused-' + str(i), comments=[], reviews={},
                predictions={model: dict(full_answer='*skipped*')}) for i in range(2)]
            path = self.raw / self.metadata['build']['parameters']['inputs'][model]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in source_rows[model]))
            header = "system_prompt='Recorded system'" + ", individual_prompts=False) gen_model=ModelConfig(name=" + repr(model) + ", tokenizer_name=None, args={'temperature': 0.1})"
            blocks = []
            for index, row in enumerate(source_rows[model]):
                prompt = 'Recorded joint request\n' + '\n'.join(c['text'] for c in row['comments']) + '\nType: age\nType: sex\n\n'
                blocks.append(str(index).center(50, '=') + '\n' + prompt + '\nhuman:\nGOLD_DO_NOT_INCLUDE\n' +
                              model + '\n' + row['predictions'][model]['full_answer'] + '\n')
            path = self.raw / self.metadata['build']['parameters']['logs'][model]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(header + '\n' + ''.join(blocks))
        self.builder = runpy.run_path(str(folder / 'build.py'))['SynthPAI']
        output = self.directory.parent / 'tables'
        with contextlib.redirect_stdout(io.StringIO()):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'), '--output', str(output)])
        self.frames = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}

    def test_exact_prompts_complete_outputs_and_ungraded_entries(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _synthpai
        result = _synthpai(self.directory, self.frames, self.metadata)
        self.assertEqual(result, dict(source_responses=8, source_subjects=2, source_items=6, source_traces=8,
            source_profiles=2, source_measured_joint_calls=4, source_joint_generations=4, source_correct=2,
            source_partial=2, source_incorrect=3, source_ungraded=1, source_shortened_inputs=1,
            source_username_disagreements=1))
        self.assertFalse(self.frames['items'].content.str.contains('GOLD_DO_NOT_INCLUDE').any())
        self.assertTrue(self.frames['traces'].trace.str.len().gt(16_000).all())

    def test_audit_rejects_equal_mean_grade_swaps_and_corrupted_associations(self):
        from measurement_db.scripts.curate_benchmarks.native_result_audits import _synthpai, _synthpai_sources
        source = _synthpai_sources(self.directory, self.metadata)
        changes = ['grade_swap', 'null_to_zero', 'subject', 'item', 'prompt', 'reference', 'alias', 'verifier',
                   'output', 'lower_rank', 'position', 'log_row', 'log_block', 'configuration', 'input_username',
                   'condition', 'trial', 'drop', 'duplicate', 'drop_trace', 'settings', 'config_identity']
        for change in changes:
            with self.subTest(change=change):
                frames = {name: frame.copy(deep=True) for name, frame in self.frames.items()}
                if change == 'grade_swap':
                    zero = frames['responses'].index[frames['responses'].response.eq(0.)][0]
                    one = frames['responses'].index[frames['responses'].response.eq(1.)][0]
                    frames['responses'].loc[zero, 'response'] = 1.; frames['responses'].loc[one, 'response'] = 0.
                elif change == 'null_to_zero': frames['responses']['response'] = frames['responses'].response.fillna(0.)
                elif change in ['subject', 'item']:
                    column = change + '_id'; current = frames['responses'].loc[0, column]
                    frames['responses'].loc[0, column] = frames[change + 's'].loc[frames[change + 's'][column].ne(current), column].iloc[0]
                elif change == 'prompt': frames['items'].loc[0, 'content'] += ' Extra guessed context'
                elif change == 'reference': frames['items'].loc[0, 'grading_criterion'] = json.dumps(dict(reference_answer='wrong'))
                elif change == 'alias': frames['items'].loc[0, 'raw_item_id'] = 'wrong-profile'
                elif change == 'verifier': frames['items'].loc[0, 'verifier'] = json.dumps(dict(class_='wrong'))
                elif change in ['output', 'lower_rank', 'position', 'log_row', 'log_block', 'configuration', 'input_username']:
                    data = json.loads(frames['traces'].loc[0, 'trace'])
                    if change == 'output': data['native_prediction']['full_answer'] = data['native_prediction']['full_answer'][:16000]
                    elif change == 'lower_rank': data['native_evaluation']['human_evaluated']['age'].append(1.)
                    elif change == 'position': data['source_row'] += 1
                    elif change == 'log_row': data['prediction_row'] += 1
                    elif change == 'log_block': data['log_block'] = data['log_block'][:16000]
                    elif change == 'configuration': data['log_configuration'] += ' guessed_setting=1'
                    else: data['input_username'] = 'guessed-name'
                    frames['traces'].loc[0, 'trace'] = json.dumps(data)
                elif change == 'condition': frames['responses'].loc[0, 'test_condition'] = 'Other axis'
                elif change == 'trial': frames['responses'].loc[0, 'trial'] += 1
                elif change == 'drop': frames['responses'] = frames['responses'].iloc[1:]
                elif change == 'duplicate': frames['responses'] = pd.concat([frames['responses'], frames['responses'].iloc[:1]])
                elif change == 'drop_trace': frames['traces'] = frames['traces'].iloc[1:]
                elif change == 'settings': frames['subjects'].loc[0, 'reasoning_effort'] = 'high'
                else: frames['subjects'].loc[0, 'subject_features_extra'] += ';unexpected_setting=guessed'
                with self.assertRaises((ValueError, KeyError)):
                    _synthpai(self.directory, frames, self.metadata, source)

    def test_mismatched_log_model_is_rejected(self):
        import contextlib
        import io
        path = self.raw / self.metadata['build']['parameters']['logs'][self.models[0]]
        path.write_text(path.read_text().replace("name='gpt-4'", "name='wrong-model'", 1))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'log model differs'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])

    def test_missing_generation_cannot_silently_remove_grades(self):
        import contextlib
        import io
        path = self.raw / self.metadata['build']['parameters']['inputs'][self.models[0]]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['predictions'][self.models[0]]['full_answer'] += ' corrupted'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Every released grade must match'):
            self.builder(str(self.directory / 'build.py')).main_from_args(['--source', str(self.directory / 'raw'),
                '--output', str(self.directory.parent / 'invalid')])
