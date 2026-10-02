"""Normalize the released MCQA records without rerunning their evaluators."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class ElicitationGame(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        mcqa = self.raw_dir / parameters["paths"]["mcqa"]

        # 1. Load the original ordered question table and Boolean result arrays.
        questions = pd.read_csv(mcqa / parameters["paths"]["questions"])
        questions["source_row"] = questions.index
        frames = []
        for technique, pattern in parameters["arrays"].items():
            paths = sorted(mcqa.glob(pattern))
            results = pd.json_normalize([json.loads(path.read_text()) for path in paths])
            results["source_file"] = [str(path.relative_to(mcqa)) for path in paths]
            labels = results.source_file.str.extract(
                r"results_(?P<model>.+)_(?P<organism>pw-locked|circuit-broken)_(?P<shots>\d+)_shot_(?P<seed>\d+)_seed\.json$")
            if labels.isna().any().any() or not results.results.str.len().eq(len(questions)).all():
                raise ValueError("Unrecognized result configuration or incomplete Boolean array")
            results = pd.concat([results, labels], axis=1).explode("results", ignore_index=True)
            if not results.results.map(type).eq(bool).all():
                raise ValueError("Correctness arrays must contain explicit Booleans")
            results["source_row"] = results.groupby("source_file", sort=False).cumcount()
            if not results.groupby("source_file").results.transform("mean").astype(float).sub(results.accuracy).abs().le(1e-12).all():
                raise ValueError("Published aggregate accuracy disagrees with the original flags")
            results["response"] = results.results.astype(float)
            results["technique"] = technique
            results["native_record"] = results.results
            frames.append(results)

        # 2. Read all released answer records, including the unconfirmed short artifact.
        paths = sorted((mcqa / "anti_refusal_training_results").glob("*.jsonl"))
        if {path.name for path in paths} != set(parameters["anti_refusal_models"]):
            raise ValueError("Released anti-refusal files differ from the reviewed mappings")
        answers = pd.concat([pd.read_json(path, lines=True).assign(source_file=str(path.relative_to(mcqa)))
            for path in paths], ignore_index=True)
        answers["source_row"] = answers.groupby("source_file", sort=False).cumcount()
        answers["native_record"] = answers[["model_answer", "parsed_answer", "correct_answer"]].to_dict("records")
        configurations = pd.DataFrame(dict(model=parameters["anti_refusal_models"], organism=parameters["anti_refusal_organisms"]))
        configurations = configurations.rename_axis("filename").reset_index()
        answers["filename"] = answers.source_file.map(lambda value: Path(value).name)
        answers = answers.merge(configurations, on="filename", validate="many_to_one")
        answers = answers.merge(questions[["source_row", "answer"]], on="source_row", validate="many_to_one")
        if not answers.correct_answer.eq(answers.answer).all():
            raise ValueError("Answer records do not align with the published question order")
        answers["response"] = answers.parsed_answer.str.strip().eq(answers.correct_answer).astype(float)
        answers["technique"] = "anti_refusal"
        attempts = pd.concat([*frames, answers], ignore_index=True)
        attempts = attempts.astype(object).where(attempts.notna(), None)

        # 3. Separate model interventions from few-shot item variants and trial seeds.
        attempts["subject_key"] = attempts.model + ":" + attempts.organism + ":" + attempts.technique
        subjects = attempts.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = subjects.model.map(parameters["model_labels"]) + " (" + subjects.organism + ")"
        features = subjects[["organism", "technique"]]
        subjects["features"] = features.assign(**parameters["subject_features"]).to_dict("records")

        # 4. Cross target questions with their distinct original grading protocols.
        protocols = pd.DataFrame.from_dict(self.grading["verifiers"], orient="index").rename_axis("technique").reset_index()
        variants = attempts[["technique", "shots"]].drop_duplicates().merge(protocols, on="technique", validate="many_to_one")
        items = questions.merge(variants, how="cross")
        items["item_key"] = items.technique + ":" + items.shots.fillna("unspecified") + ":" + items.source_row.astype(str)
        items["raw_item_id"] = "wmdp_test:" + items.source_row.astype(str)
        items["content"] = items.question_prompt
        items["grading_criterion"] = [dict(reference_answer=answer, rule=rule) for answer, rule in zip(items.answer, items.rule)]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][technique], sort_keys=True),
            judge=self.grading["verifiers"][technique]["judge"], judged_by=self.grading["verifiers"][technique]["judged_by"])
            if technique == "anti_refusal" else ExactMatcher(spec=json.dumps(self.grading["verifiers"][technique], sort_keys=True)) for technique in items.technique]
        items["features"] = items[["subject", "technique", "shots"]].rename(columns={"subject": "domain"}).assign(**parameters["item_features"]).to_dict("records")

        # 5. Preserve every native record with its file and zero-based source position.
        attempts["item_key"] = attempts.technique + ":" + attempts.shots.fillna("unspecified") + ":" + attempts.source_row.astype(str)
        attempts["response_key"] = attempts.source_file + ":" + attempts.source_row.astype(str)
        attempts["test_condition"] = attempts.seed.map(lambda seed: "seed=" + seed if seed is not None else None)
        attempts["trace"] = [json.dumps(dict(source_file=path, source_row=int(row), record=record), ensure_ascii=False, allow_nan=False)
            for path, row, record in zip(attempts.source_file, attempts.source_row, attempts.native_record)]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": attempts[["response_key", "trace"]],
        }


if __name__ == "__main__":
    ElicitationGame(__file__).main_from_args()
