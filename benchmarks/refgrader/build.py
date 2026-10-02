"""Tabulate original RefGrader requests, judgments and additional samples."""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class RefGrader(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        patterns, labels = parameters["patterns"], parameters["labels"]

        # 1. Normalize the original problem, model, workflow and student tables.
        problems = []
        for path in sorted(self.raw_dir.glob(parameters["layout"]["results"])):
            records = json.loads(path.read_text(),
                parse_constant=lambda value: {"native_nonfinite_number": value})["data"]
            frame = pd.json_normalize(records, max_level=0)
            frame["source_file"] = str(path.relative_to(self.raw_dir))
            frame["source_row"] = frame.index
            frame["dataset"] = parameters["datasets"][path.name]
            problems.append(frame)
        problems = pd.concat(problems, ignore_index=True)
        problems["problem_key"] = problems.index
        problems["model_config"] = problems.stage_cache.map(list)
        cells = problems.explode("model_config").copy()
        cells["workflows"] = [cache[model] for cache, model in zip(cells.stage_cache, cells.model_config)]
        cells["workflow"] = cells.workflows.map(list)
        cells = cells.explode("workflow")
        cells = cells.loc[cells.workflow.str.match(patterns["grading_workflows"])].copy()
        cells["students"] = [cache[stage] for cache, stage in zip(cells.workflows, cells.workflow)]
        cells["student_id"] = cells.students.map(list)
        cells = cells.explode("student_id")
        cells = cells.loc[cells.student_id.str.match(patterns["student_id"])].reset_index(drop=True)
        cells["cell"] = [students[sid] for students, sid in zip(cells.students, cells.student_id)]
        cells["cell_key"] = cells.index
        cells["inputs"] = cells.cell.map(lambda cell: cell["stage_inputs"])
        inputs = pd.json_normalize(cells.inputs).set_index(cells.index)
        inputs = inputs.reindex(columns=["problem", "student_solution", "remove_tags", "model_solution",
                                       "rubric", "similar_solution", "grader_type", "reference_selection"])

        # 2. Reconstruct source-described requests, preserving its tag-removal behavior.
        # The source removes matched span pairs repeatedly. Its index behavior for
        # nested or malformed tags is retained; replacing this with HTML cleaning
        # would silently change the text supplied to the grader.
        processed = {}
        for original in inputs.loc[inputs.remove_tags.eq(True), "student_solution"].unique():
            value = original
            while True:
                stack, pairs = [], []
                for match in re.finditer(patterns["span_token"], value, re.IGNORECASE):
                    if match.group().lower().startswith("</"):
                        if stack:
                            start, content_start = stack.pop()
                            pairs.append((start, match.end(), value[content_start:match.start()]))
                    elif re.search(patterns["span_class"], match.group(), re.IGNORECASE):
                        stack.append((match.start(), match.end()))
                if not pairs:
                    break
                for start, end, content in sorted(pairs, reverse=True):
                    value = value[:start] + content + value[end:]
            processed[original] = value
        student_text = inputs.student_solution.where(~inputs.remove_tags.eq(True), inputs.student_solution.map(processed))
        prompt_key = inputs.grader_type.copy()
        three_stage = cells.workflow.str.startswith("stage3_grade_")
        prompt_key.loc[three_stage] = "three_stage"
        prompt_key.loc[cells.workflow.str.startswith("absolute_grade_")] = "absolute"
        templates = {key: (self.raw_dir / name).read_text() for key, name in parameters["prompts"].items()}
        cells["prompt_file"] = prompt_key.map(parameters["prompts"])
        cells["content"] = prompt_key.map(templates) + labels["input_header"] + inputs.problem + "\n"
        cells["content"] += (labels["student_header"] + student_text + "\n").where(
            ~three_stage, labels["three_stage_student_header"] + student_text + "\n")
        cells.loc[three_stage, "content"] += labels["reference_header"] + inputs.loc[three_stage, "similar_solution"] + "\n"
        five_stage = cells.workflow.str.startswith("stage5_")
        cells.loc[five_stage, "content"] += (labels["model_solution_header"] + inputs.loc[five_stage, "model_solution"] + "\n"
            + labels["rubric_header"] + inputs.loc[five_stage, "rubric"] + "\n")
        if cells.content.isna().any():
            raise ValueError("An original grading request is missing its input or prompt template")

        # 3. Join human references, keeping them separate from the grader's prediction.
        cells["student_metadata"] = [problem["student_metadata"].get(sid, {})
            for problem, sid in zip(cells.problem_data, cells.student_id)]
        cells["reference_student_solution"] = [problem["student_solutions"][sid]
            for problem, sid in zip(cells.problem_data, cells.student_id)]
        matches_reference = inputs.student_solution.eq(cells.reference_student_solution)
        cells["reference_association"] = matches_reference.map({True: "matching_source_solution", False: "different_cached_solution"})
        human = pd.json_normalize(cells.student_metadata).reindex(columns=["correctness", "grade"])
        references = human.correctness.str.strip().str.lower().map(parameters["human_references"])
        matharena = cells.dataset.eq("matharena")
        math_grades = pd.to_numeric(human.loc[matharena, "grade"], errors="raise")
        if not (math_grades.isna() | math_grades.isin(range(8))).all():
            raise ValueError("Unexpected native human-reference grade")
        references.loc[matharena] = math_grades.map(lambda value: None if pd.isna(value) else str(int(value)))
        cells["reference"] = references.astype(object).where(references.notna() & matches_reference, None)
        cells["grading_criterion"] = [dict(reference_answer=reference, rule=self.grading["rule"])
                                      for reference in cells.reference]
        cells["identity"] = [json.dumps([content, reference], ensure_ascii=False)
                             for content, reference in zip(cells.content, cells.reference)]
        items = cells.drop_duplicates("identity").copy()
        items["item_key"] = items.cell_key.astype(str)
        items["raw_item_id"] = items.dataset + ":" + items.problem_id + ":" + items.student_id + ":" + items.workflow
        items["features"] = [dict(dataset=row.dataset, workflow=row.workflow, prompt_source=row.prompt_file)
                             for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["reported"], sort_keys=True)) for _ in items.index]
        cells = cells.merge(items[["identity", "item_key"]], on="identity", how="left", validate="many_to_one")

        # 4. Expand each recorded call, including additional sampling results.
        cells["sample_index"] = cells.cell.map(lambda cell: list(range(1 + len(cell.get("sampling_results", [])))))
        attempts = cells.explode("sample_index").reset_index(drop=True)
        attempts["native_result"] = [cell["result"] if index == 0 else cell["sampling_results"][index - 1]
                                     for cell, index in zip(attempts.cell, attempts.sample_index)]
        decoded, statuses = [], []
        for value in attempts.native_result:
            if isinstance(value, dict):
                decoded.append(value); statuses.append("native_structured_result")
                continue
            match = re.fullmatch(patterns["json_fence"], value, re.DOTALL) if isinstance(value, str) else None
            try:
                decoded.append(json.loads(match.group(1) if match else value))
                statuses.append("decoded_native_json_string")
            except (ValueError, TypeError):
                decoded.append({}); statuses.append("unparseable_native_result")
        scores = pd.json_normalize(decoded).reindex(columns=["overall_assessment.score"])
        grades = pd.to_numeric(scores["overall_assessment.score"], errors="coerce")
        valid = grades.isin(range(8))
        attempts["response"] = grades.where(valid, None)
        attempts["grade_status"] = statuses
        attempts.loc[~valid & attempts.grade_status.ne("unparseable_native_result"), "grade_status"] = "invalid_or_missing_native_grade"
        attempts["sampling_mode"] = attempts.sample_index.eq(0).map({True: "initial", False: "additional"})
        attempts["subject_key"] = attempts.model_config + ":" + attempts.workflow + ":" + attempts.sampling_mode
        subjects = attempts.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_configuration=row.model_config,
            workflow=row.workflow, sampling_mode=row.sampling_mode) for row in subjects.itertuples()]
        attempts["response_key"] = (attempts.source_file + "#" + attempts.source_row.astype(str) + "#" + attempts.model_config
            + "#" + attempts.workflow + "#" + attempts.student_id + "#" + attempts.sample_index.astype(str))
        attempts["test_condition"] = "dataset=" + attempts.dataset + ";method=" + attempts.workflow

        # 5. Preserve complete selected outputs and cache inputs without attaching another sample's output.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_problem_id=row.problem_id, source_student_id=row.student_id, model_configuration=row.model_config,
            workflow=row.workflow, sample_index=int(row.sample_index), grade_status=row.grade_status,
            native_result=row.native_result, student_metadata=row.student_metadata,
            reference_association=row.reference_association,
            different_reference_solution=row.reference_student_solution if row.reference_association == "different_cached_solution" else None,
            cache_metadata={key: value for key, value in row.cell.items() if key not in ("result", "sampling_results")}),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    RefGrader(__file__).main_from_args()
