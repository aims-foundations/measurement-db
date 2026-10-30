"""Join CORE-Bench's released capsule results to its task and grading definitions."""

import ast
import copy
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.build_measurement_tables.load_source_files import read_gpg_json
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


class CoreBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("tasks", "results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read task definitions and native result tables without extracting archives.
        root = self.raw_dir / "core-bench"
        train = json.loads((root / "benchmark/dataset/core_train.json").read_text())
        test = read_gpg_json(
            root / "benchmark/dataset/core_test.json.gpg",
            password=self.build_parameters["task_bank"]["public_password"], scratch_dir=self.dir,
        )
        tasks = pd.concat([
            pd.json_normalize(train, max_level=0).assign(task_split="train"),
            pd.json_normalize(test, max_level=0).assign(task_split="test"),
        ], ignore_index=True)
        templates = json.loads((root / "benchmark/benchmark_prompts.json").read_text())
        experiments = []
        with tarfile.open(root / "agent_results.tar.gz") as archive:
            for member in sorted(archive.getmembers(), key=lambda entry: entry.name):
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                if member.name in self.build_parameters["excluded_files"]:
                    continue
                table = pd.json_normalize(json.load(archive.extractfile(member))["capsule_results"], max_level=0)
                experiments.append(table.assign(source_file=member.name, source_row=range(len(table))))
        observations = pd.concat(experiments, ignore_index=True)

        # 2. Decode the source filenames into model, scaffold, cost limit and difficulty.
        configuration = observations.source_file.str.extract(
            r"^results/(?P<split>train|test)_(?P<agent>coreagent|autogpt)_(?P<model>.+)_"
            r"(?P<cost_limit>c-[0-9.]+)/[^/]+_codeocean_(?P<difficulty>easy|medium|hard)\.json$"
        )
        configuration["raw_label"] = configuration.model.map(self.build_parameters["models"])
        configuration["harness"] = configuration.agent.map(self.build_parameters["agents"])
        if configuration.isna().any().any():
            raise ValueError("A CORE-Bench result has an unrecognized model, scaffold or difficulty")
        observations = observations.join(configuration).assign(
            item_key=lambda frame: frame.capsule_id + "__" + frame.difficulty,
            subject_key=configuration[["raw_label", "harness", "cost_limit"]].apply(lambda row: json.dumps(row.tolist()), axis=1),
        )
        observations = observations.merge(tasks, on="capsule_id", how="left", validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all() or not observations.split.eq(observations.task_split).all():
            raise ValueError("A CORE-Bench result has no matching capsule definition in its declared split")

        # 3. Render the native task template and attach released reference answers.
        items = observations.drop_duplicates("item_key").copy()
        items["content"] = items.apply(lambda row: templates[f"codeocean_{row.difficulty}"]
            .replace("{task_prompt}", row.task_prompt)
            .replace("{json_fields}", str(row.results[0].keys()))
            + "\n\nScientific code capsule: " + row.capsule_id + "\nCapsule DOI: " + row.capsule_doi, axis=1)
        items = items.assign(
            raw_item_id=items.item_key,
            testlet_id=items.capsule_id,
            grading_criterion=items.results.map(lambda values: {
                "reference_answer": json.dumps(values, ensure_ascii=False, sort_keys=True), "rule": self.grading["rule"],
                "response_scale": dict(kind="interval", min=0, max=1, direction="higher_is_better"),
            }),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["provider_result"], sort_keys=True)),
            features=items[["difficulty"]].to_dict("records"),
        )
        subjects = observations[["subject_key", "raw_label", "harness", "cost_limit"]].drop_duplicates()
        subjects = subjects.assign(features=subjects[["harness", "cost_limit"]].to_dict("records"))

        # 4. Preserve fractional question accuracy and the complete result reports.
        total = observations.total_written_questions + observations.total_vision_questions
        correct = observations.correct_written_answers + observations.correct_vision_answers
        if not total.gt(0).all() or not correct.between(0, total).all():
            raise ValueError("CORE-Bench has invalid correct/total question counts")
        responses = observations.assign(
            response_key=observations.source_file + ":" + observations.source_row.astype(str), response=correct / total,
        )
        traces = responses.copy()
        traces = traces.assign(trace=traces.result_report.map(lambda report: json.dumps(report, ensure_ascii=False)))
        # 5. Replay only the captured deterministic scoring function. Prefixes retain
        # its early-abort behavior: evaluating each answer in isolation would change it.
        source = (root / 'benchmark/evaluations.py').read_text()
        function = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
            and node.name == 'eval_result_json']
        if len(function) != 1:
            raise ValueError('The pinned CORE-Bench evaluator is missing')
        namespace = dict(Dict=dict, np=np, math=math, t=t)
        exec(compile(ast.Module(body=function, type_ignores=[]), 'evaluations.py', 'exec'), namespace)
        evaluate = namespace['eval_result_json']
        components, outcomes = [], []
        score_fields = ('correct_written_answers', 'correct_vision_answers',
                        'total_written_questions', 'total_vision_questions')
        for row in items.itertuples():
            for question, answer in row.results[0].items():
                if not isinstance(answer, (int, float, list, str)):
                    continue
                component = hashlib.sha256(question.encode()).hexdigest()
                components.append(dict(parent_item_key=row.item_key, component_id=component,
                    grading_criterion=dict(rule=json.dumps(dict(question=question,
                        rule='Exact replay of the pinned evaluator, preserving reported answer order and early-abort behavior'),
                        ensure_ascii=False, sort_keys=True),
                        reference_answer=json.dumps([result[question] for result in row.results], ensure_ascii=False),
                        response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better')),
                    verifier=ExactMatcher(spec=json.dumps(dict(source=self.grading['verifiers']['provider_result']['source'],
                        function='eval_result_json', selected_question=question), sort_keys=True))))
        for row in responses.itertuples():
            full = evaluate(row.results, copy.deepcopy(row.result_report))
            if any(full[field] != getattr(row, field) for field in score_fields):
                continue  # Preserve historical aggregates when exact replay is not established.
            grades = {key: 0. for key, answer in row.results[0].items() if isinstance(answer, (int, float, list, str))}
            prefix, previous = {}, 0
            for question, answer in (row.result_report.items() if isinstance(row.result_report, dict) else []):
                prefix[question] = answer
                counts = evaluate(row.results, copy.deepcopy(prefix))
                correct = counts['correct_written_answers'] + counts['correct_vision_answers']
                if correct - previous not in (0, 1):
                    raise ValueError('The evaluator does not produce binary per-question increments')
                if question in grades:
                    grades[question] = float(correct - previous)
                previous = correct
            for question, grade in grades.items():
                outcomes.append(dict(parent_response_key=row.response_key,
                    component_id=hashlib.sha256(question.encode()).hexdigest(), response=grade))
        tables = {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }
        return component_tables(tables, pd.DataFrame(components), pd.DataFrame(outcomes))


if __name__ == "__main__":
    CoreBench(__file__).main_from_args()
