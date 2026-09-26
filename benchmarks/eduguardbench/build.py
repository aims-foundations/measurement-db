"""Curate published EduGuardBench answers and grading records as tables."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class EduGuardBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate published worksheets, retaining each decoded native row.
        components = {}
        for component in ("sata", "adversarial"):
            frames = []
            for path in sorted(self.raw_dir.glob(parameters["paths"][component])):
                frame = pd.read_excel(path, dtype=object, keep_default_na=False)
                frame["native_record"] = frame.to_dict("records")
                frame["source_row"] = frame.index
                frame["source_file"] = str(path.relative_to(self.raw_dir))
                frame["subject_key"] = parameters[component + "_models"][path.stem]
                frames.append(frame)
            components[component] = pd.concat(frames, ignore_index=True)

        # 2. Apply the author option parser and unpivot exact and partial-credit scores.
        sata = components["sata"].copy()
        selected = sata.LLM_Answer_EN.astype(str).str.replace(";", ",").str.replace(" and ", ",").str.replace("、", ",")
        selected = selected.str.upper().str.findall(parameters["option_parser"]["pattern"]).map(set)
        ideal = sata.Answer.map(lambda value: {part.strip().upper() for part in str(value).split(",") if part.strip()})
        exact = selected.eq(ideal)
        partial = pd.Series([bool(answer) and answer < gold for answer, gold in zip(selected, ideal)], index=sata.index)
        sata["sata_exact"] = exact.astype(float)
        sata["sata_fidelity"] = pd.Series(0., index=sata.index).mask(partial, 0.5).mask(exact, 1.)
        sata["content"] = sata.Question_English
        sata["reference_answer"] = ideal.map(lambda values: ",".join(sorted(values)))
        sata["prompt_record"] = None
        identifiers = ["ID", "subject_key", "content", "reference_answer", "source_file", "source_row", "native_record", "prompt_record"]
        sata_scores = sata.melt(id_vars=identifiers, value_vars=list(parameters["sata_metrics"]),
            var_name="metric", value_name="response")

        # 3. Join original adversarial inputs and retain published verdicts and applicable refusal grades.
        prompts = pd.read_excel(self.raw_dir / parameters["paths"]["prompts"], dtype=object, keep_default_na=False)
        prompts["prompt_record"] = prompts.to_dict("records")
        prompts["content"] = [json.dumps(dict(teacher_prompt=row.Teacher_Prompt_EN,
            student_request=row.Student_Statement_EN), ensure_ascii=False) for row in prompts.itertuples()]
        adversarial = components["adversarial"].merge(prompts[["ID", "content", "prompt_record"]],
            on="ID", how="left", validate="many_to_one", indicator=True)
        if not adversarial._merge.eq("both").all():
            raise ValueError("An EduGuardBench result has no released prompt")
        adversarial["reference_answer"] = ""
        adversarial["harmful"] = pd.to_numeric(adversarial.Final_Verdict, errors="raise")
        adversarial["refusal_quality"] = pd.to_numeric(adversarial.Refusal_Quality.map(parameters["refusal_categories"]))
        verdicts = adversarial[identifiers + ["harmful"]].rename(columns={"harmful": "response"}).assign(metric="harmful")
        refusals = adversarial.loc[adversarial.harmful.eq(0), identifiers + ["refusal_quality"]]
        refusals = refusals.rename(columns={"refusal_quality": "response"}).assign(metric="refusal_quality")

        # 4. Keep available grading draws; they judge the same generated answer, not new executions.
        votes = adversarial.melt(id_vars=identifiers, value_vars=list(parameters["expert_columns"].values()),
            var_name="expert_column", value_name="response")
        votes = votes.loc[votes.response.ne("")].copy()
        votes["response"] = pd.to_numeric(votes.response, errors="raise")
        votes["draw"] = votes.expert_column.map({column: int(draw) for draw, column in parameters["expert_columns"].items()})
        votes["metric"] = "harmful_vote"
        observations = pd.concat([sata_scores, verdicts, refusals, votes], ignore_index=True)

        # 5. Share identical inputs and grading protocols while preserving every reported observation.
        identity = ["content", "reference_answer", "metric"]
        items = observations[identity + ["ID"]].drop_duplicates(identity).reset_index(drop=True)
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.ID.astype(str) + "/" + items.metric
        items["grading_criterion"] = [dict(self.grading["verifiers"][row.metric]["criterion"],
            reference_answer=row.reference_answer or None) for row in items.itertuples()]
        items["verifier"] = items.metric.map(lambda name: (ExactMatcher if name.startswith("sata_") else Judge)(
            spec=json.dumps(self.grading["verifiers"][name]["implementation"], sort_keys=True)))
        items["features"] = [dict(input_scope=parameters["input_scope"]["description"]) for _ in items.index]
        observations = observations.merge(items[identity + ["item_key"]], on=identity, how="left", validate="many_to_one")
        observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str) + ":" + observations.metric
        observations["response_key"] += observations.draw.map(lambda value: ":" + str(int(value)) if pd.notna(value) else "")
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(parameters["subject_features"],
            reasoning_mode=parameters["reasoning_modes"].get(name)) for name in subjects.subject_key]

        # 6. Retain complete source records, including invalid category text and empty model answers.
        traces = observations[["response_key", "source_file", "source_row", "metric", "draw", "native_record", "prompt_record"]]
        traces = traces.astype(object).where(traces.notna(), None)
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EduGuardBench(__file__).main_from_args()
