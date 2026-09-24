"""Join EDIT-Bench's released result matrix, task definitions and generated code."""

import ast
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class EditBenchBuild(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("items", "release", "leaderboards")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the native item table, model-by-question result matrix and available code.
        layout = self.build_parameters["layout"]
        tasks = pd.read_json(self.raw_dir / layout["items"], lines=True, dtype=False, convert_dates=False).astype(object)
        tasks = tasks.where(tasks.notna(), None)
        if tasks.problem_id.duplicated().any() or not tasks.test_code.map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError("EDIT-Bench tasks need unique IDs and executable grading definitions")
        if not tasks.test_harness.map(lambda value: isinstance(value, dict)).all():
            raise ValueError("An EDIT-Bench task has no test harness")
        with tarfile.open(self.raw_dir / layout["archive"]) as archive:
            files = {m.name.split("/", 1)[1]: m for m in archive.getmembers() if m.isfile()}
            result_paths = sorted(p for p in files if p.startswith(layout["results_prefix"]) and p.endswith(".json"))
            results = pd.DataFrame.from_dict({Path(p).stem: json.load(archive.extractfile(files[p])) for p in result_paths}, orient="index")
            prompt = archive.extractfile(files[layout["prompt_member"]]).read().decode().replace("\r\n", "\n").replace("\r", "\n")
            trace_paths = sorted(p for p in files if p.startswith(layout["traces_prefix"]))
            traces = pd.DataFrame({
                "subject_key": [Path(p).parent.name for p in trace_paths],
                "item_key": [Path(p).name for p in trace_paths],
                "trace": [archive.extractfile(files[p]).read().decode("utf-8") for p in trace_paths],
            })
            adapters = []
            for path, mapping_name in self.build_parameters["adapter_modules"].items():
                # Read literal model-name maps, without importing or running the upstream adapters.
                assignments = {n.targets[0].id: n.value for n in ast.parse(archive.extractfile(files[path]).read()).body
                               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
                adapters.append(pd.Series(ast.literal_eval(assignments[mapping_name]), name="model").rename_axis(
                    "upstream_model").reset_index().assign(inference_api=self.build_parameters["adapter_apis"][path]))

        # 2. Unpivot observed question scores; reconcile their published aggregates before binarizing.
        question_columns = results.columns[results.columns.str.fullmatch(r"question_\d+")]
        if set(results.columns) != {*question_columns, "pass_rate", "average_test_rate"}:
            raise ValueError("Unexpected EDIT-Bench result fields")
        observations = results[question_columns].rename_axis("subject_key").reset_index().melt(
            id_vars="subject_key", var_name="question", value_name="test_fraction",
        ).dropna(subset=["test_fraction"])
        if not observations.test_fraction.between(0, 1).all():
            raise ValueError("EDIT-Bench test fractions must be finite values between zero and one")
        observations = observations.assign(
            item_key=observations.question.str.removeprefix("question_"),
            response=observations.test_fraction.eq(1).astype(float),
        ).sort_values(["subject_key", "item_key"])
        aggregates = observations.groupby("subject_key").agg(pass_rate=("response", "mean"), average_test_rate=("test_fraction", "mean"))
        if not (results[aggregates.columns] - aggregates).abs().le(1e-12).all().all():
            raise ValueError("EDIT-Bench per-question scores disagree with the released aggregates")
        observations["response_key"] = observations.subject_key + ":" + observations.item_key

        # 3. Render the provider's prompt and keep every task's complete executable grading definition.
        items = observations[["item_key"]].drop_duplicates().merge(
            tasks.assign(item_key=tasks.problem_id.astype(str)), on="item_key", how="left", validate="one_to_one", indicator=True,
        )
        if not items._merge.eq("both").all():
            raise ValueError("An EDIT-Bench result has no matching task")
        execution = self.grading["verifiers"]["execution"]
        items["execution"] = items.apply(lambda row: {
            **execution[row.programming_language],
            "commands": [[*execution[row.programming_language]["commands"][0][:-1], row.python_version],
                         *execution[row.programming_language]["commands"][1:]]
                if row.programming_language == "python" else execution[row.programming_language]["commands"],
        }, axis=1)
        verifier_fields = ["problem_id", "pair_id", "programming_language", "python_version", "original_code",
                           "requirements", "test_code", "test_harness", "execution"]
        items = items.assign(
            raw_item_id=items.item_key,
            content=items.apply(lambda row: prompt.format(lang=row.programming_language, original_code=row.original_code,
                instruction=row.instruction, highlighted_code=row.highlighted_code), axis=1),
            grading_criterion=[{"rule": self.grading["rule"]} for _ in range(len(items))],
            verifier=items[verifier_fields].apply(lambda row: ExactMatcher(spec=json.dumps({
                **self.grading["verifiers"]["test_suite"], **row.to_dict(),
                "special_files": self.grading["verifiers"]["special_files"].get(row.pair_id, {}),
            }, ensure_ascii=False, sort_keys=True)), axis=1),
            features=items[["natural_language", "programming_language"]].to_dict("records"),
        )

        # 4. Join model configurations and full traces to observations; do not infer API from vendor name.
        subjects = observations[["subject_key"]].drop_duplicates().assign(
            model=lambda frame: frame.subject_key.str.removesuffix("-high"),
        ).merge(pd.concat(adapters, ignore_index=True), on="model", how="left", validate="many_to_one")
        if subjects.inference_api.isna().any():
            raise ValueError("An EDIT-Bench model is absent from the released adapter maps")
        subjects = subjects.assign(raw_label=subjects.subject_key, features=subjects.apply(lambda row: {
            **self.build_parameters["subject_features"], "inference_api": row.inference_api,
            **({"reasoning_effort": self.build_parameters["reasoning_effort"][row.subject_key]}
               if row.subject_key in self.build_parameters["reasoning_effort"] else {}),
        }, axis=1))
        traces = traces.merge(observations[["subject_key", "item_key", "response_key"]],
                              on=["subject_key", "item_key"], how="left", validate="one_to_one")
        if traces.response_key.isna().any():
            raise ValueError("Released EDIT-Bench code has no corresponding result")
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EditBenchBuild(__file__).main_from_args()
