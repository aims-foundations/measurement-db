"""Join MLR-Bench's idea/proposal reviews, generated artifacts and research inputs."""

import ast
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MLRBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the selected native text files; upstream Python is parsed, never executed.
        text = {}
        with tarfile.open(self.raw_dir / self.build_parameters["source"]["archive"]) as archive:
            for member in archive:
                path = member.name.partition("/")[2]
                if member.isfile() and path.startswith(("tasks/", "agent_results/ideas_and_proposals/",
                        "agent_reviews/idea_proposal_reviews_", "mlrbench/agent/", "mlrbench/evals/")):
                    text[path] = archive.extractfile(member).read().decode("utf-8")
        files = pd.DataFrame({"path": list(text), "text": list(text.values())})
        bank = files.loc[files.path.str.match(r"^tasks/[^/]+\.md$")].assign(
            task_id=lambda frame: frame.path.str.removeprefix("tasks/").str.removesuffix(".md")
        ).rename(columns={"text": "task"})
        context = files.loc[files.path.str.match(r"^agent_results/ideas_and_proposals/[^/]+/(idea|related_work)\.md$")]
        context = context.join(context.path.str.extract(r"/(?P<task_id>[^/]+)/(?P<document>idea|related_work)\.md$"))
        context = context.pivot(index="task_id", columns="document", values="text").reset_index()

        # 2. Read the provider's literal prompts and rubrics, then render each stage's input.
        forms = []
        for stage in ("idea", "proposal"):
            config = self.build_parameters[stage]
            module = ast.parse(text[config["generator"]])
            function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == config["function"])
            prompt = next(ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets))
            suffix = next(n.value for n in function.body if isinstance(n, ast.AugAssign) and n.target.id == "prompt")
            if not isinstance(suffix, ast.JoinedStr) or any(not isinstance(n, ast.Constant) and (
                    not isinstance(n, ast.FormattedValue) or not isinstance(n.value, ast.Name)
                    or n.value.id not in {"task", "idea", "related_work"} or n.conversion != -1 or n.format_spec)
                    for n in suffix.values):
                raise ValueError("The native generation prompt needs a reviewed template parser")
            template = "".join(n.value if isinstance(n, ast.Constant) else "{" + n.value.id + "}" for n in suffix.values)
            rubric = next(ast.literal_eval(n.value) for n in ast.parse(text[config["rubric_file"]]).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == config["rubric_symbol"] for t in n.targets))
            forms.append(dict(stage=stage, instruction=prompt, template=template, rubric=rubric, rubric_file=config["rubric_file"]))
        stimuli = bank[["task_id", "task"]].merge(context, on="task_id", how="left", validate="one_to_one")
        if stimuli.isna().any().any():
            raise ValueError("A research task lacks its released proposal context")
        stimuli = stimuli.merge(pd.DataFrame(forms), how="cross")
        stimuli["content"] = stimuli.apply(lambda row: row.instruction + row.template.format_map(row.to_dict()), axis=1)

        # 3. Expand each review into its native rubric dimensions and join the exact generation.
        reviews = []
        for path in sorted(p for p in text if p.startswith("agent_reviews/idea_proposal_reviews_") and p.endswith(".json")):
            reviews.append(pd.DataFrame({"assessment": json.loads(text[path])}).rename_axis("metric").reset_index().assign(source_review=path))
        observations = pd.concat(reviews, ignore_index=True)
        coordinates = observations.source_review.str.extract(
            r"^agent_reviews/idea_proposal_reviews_(?P<reviewer>[^/]+)/(?P<task_id>[^/]+)/"
            r"(?P<stage>idea|proposal)/(?:idea|proposal)_(?P<raw_label>[^/]+)\.json$"
        )
        if coordinates.isna().any().any():
            raise ValueError("An MLR-Bench review has an unrecognized source path")
        observations = observations.join(coordinates).assign(
            source_generation=lambda frame: "agent_results/ideas_and_proposals/" + frame.task_id
                + "/" + frame.stage + "/" + frame.stage + "_" + frame.raw_label + ".md",
            response=lambda frame: frame.assessment.map(lambda value: value["score"]),
        )
        observations = observations.merge(files.rename(columns={"path": "source_generation", "text": "generation"}),
            on="source_generation", how="left", validate="many_to_one").merge(
            stimuli[["task_id", "stage", "content", "rubric", "rubric_file"]],
            on=["task_id", "stage"], how="left", validate="many_to_one")
        if observations[["generation", "content", "rubric"]].isna().any().any():
            raise ValueError("A review lacks its native generation, task input or rubric")
        observations = observations.assign(
            item_key=lambda frame: frame.task_id + "/" + frame.stage + "/" + frame.reviewer + "/" + frame.metric,
            subject_key=observations.raw_label,
            response_key=observations.source_review + "/" + observations.metric,
        )

        # 4. Each task/stage/reviewer/dimension is a grading item; generators are subjects.
        items = observations.drop_duplicates("item_key").assign(
            raw_item_id=lambda frame: frame.item_key,
            features=lambda frame: frame[["task_id", "stage"]].to_dict("records"),
            grading_criterion=lambda frame: frame.apply(lambda row: {"rule": json.dumps({
                "description": self.grading["rule"], "metric": row.metric, "rubric": row.rubric,
            }, sort_keys=True)}, axis=1),
            verifier=lambda frame: frame.apply(lambda row: Judge(spec=json.dumps({
                **self.grading["verifiers"]["review"], "model": row.reviewer,
                "rubric_file": row.rubric_file, "grade_field": row.metric + ".score",
            }, sort_keys=True), judged_by="llm"), axis=1),
        )
        subjects = observations[["subject_key", "raw_label"]].drop_duplicates().assign(
            features=lambda frame: [{"harness": self.build_parameters["run"]["harness"]} for _ in frame.index]
        )
        traces = observations.assign(trace=observations.apply(lambda row: json.dumps({
            "source_review": row.source_review, "source_generation": row.source_generation,
            "generation": row.generation, "assessment": row.assessment,
        }, ensure_ascii=False), axis=1))
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    MLRBench(__file__).main_from_args()
