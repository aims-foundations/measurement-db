"""Curate native medical-watermark generations and cached paired quality ratings."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class MedWatermarkFWS(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Load original task tables, generation triples and cached judge records.
        generations, judgments = [], []
        with ZipFile(self.raw_dir / layout["archive"]) as archive:
            banks = {dataset: pd.json_normalize(json.loads(archive.read(layout["prefix"] + path)), max_level=0)
                .assign(source_row=lambda frame: frame.index) for dataset, path in parameters["banks"].items()}
            paths = pd.DataFrame({"source_file": sorted(name.removeprefix(layout["prefix"])
                for name in archive.namelist() if name.startswith(layout["prefix"] + "logs/")
                and name.endswith(".pkl") and not name.endswith("-SCORES.pkl"))})
            tags = paths.source_file.str.extract(parameters["patterns"]["generation"])
            if tags.isna().any().any():
                raise ValueError("Unrecognized original generation filename")
            for entry in paths.join(tags).itertuples():
                with archive.open(layout["prefix"] + entry.source_file) as stream:
                    watermarked, unwatermarked, natural = read_native_pickle(stream)
                with archive.open(layout["prefix"] + entry.source_file + "-SCORES.pkl") as stream:
                    watermarked_scores, control_scores, natural_scores = read_native_pickle(stream)
                frame = pd.DataFrame(dict(watermarked=watermarked, unwatermarked=unwatermarked,
                    natural_text=natural, source_row=range(len(natural))))
                frame = frame.merge(banks[entry.dataset], on="source_row", how="left", validate="one_to_one")
                if frame.prompt.isna().any() or not frame.natural_text.eq(frame.natural).all():
                    raise ValueError("Original generation/reference does not match its task bank")
                # The detector skips outputs empty after the documented word-prefix removal.
                kept = [bool(text.split()[len(prompt.split()):]) for text, prompt in zip(frame.watermarked, frame.prompt)]
                if sum(kept) != len(watermarked_scores) or len(natural_scores) != len(frame) or control_scores:
                    raise ValueError("Detector score positions disagree with the source skip-empty rule")
                frame["detector_score"] = None
                frame.loc[kept, "detector_score"] = [float(value) for value in watermarked_scores]
                frame["natural_detector_score"] = [float(value) for value in natural_scores]
                generations.append(frame.assign(source_file=entry.source_file, experiment=entry.experiment,
                    model=entry.model, dataset=entry.dataset))
            for name in sorted(archive.namelist()):
                if name.startswith(layout["prefix"] + "logs/GPTJUDGE-Results/") and name.endswith(".json"):
                    records = json.loads(archive.read(name))
                    frame = pd.json_normalize(records, max_level=0)
                    frame["source_judgment"] = records
                    judgments.append(frame.assign(judgment_file=name.removeprefix(layout["prefix"]),
                        judgment_row=range(len(frame))))
        generations = pd.concat(generations, ignore_index=True)
        judgments = pd.concat(judgments, ignore_index=True)

        # 2. Unpivot generated variants and join each rating to the generation actually judged.
        generations = generations.melt(id_vars=[column for column in generations
            if column not in ["watermarked", "unwatermarked", "natural"]],
            value_vars=["watermarked", "unwatermarked"], var_name="variant", value_name="native_output")
        generations.loc[generations.variant.eq("unwatermarked"), "detector_score"] = None
        generations["subject_key"] = generations.source_file + "#" + generations.variant
        keys = ["source_file", "source_row", "variant"]
        ratings = judgments.melt(id_vars=[column for column in judgments if column not in ["scores_U", "scores_W"]],
            value_vars=["scores_U", "scores_W"], var_name="rating_variant", value_name="ratings")
        ratings["variant"] = ratings.rating_variant.map(parameters["variants"])
        filenames = ratings.judgment_file.str.extract(parameters["patterns"]["judgment"])
        if filenames.isna().any().any():
            raise ValueError("Unrecognized original judgment filename")
        ratings["source_file"] = "logs/" + filenames.method + "/" + filenames.filename
        # The notebook explicitly substitutes KGW controls for EXPEdit summarization.
        reuse = ratings.source_file.eq(layout["reused_control_experiment"]) & ratings.variant.eq("unwatermarked")
        ratings.loc[reuse, "source_file"] = layout["reused_control_source"]
        ratings["source_row"] = ratings.sample_id
        ratings = ratings.merge(generations, on=keys, how="left", validate="many_to_one", suffixes=("_judge", ""))
        if ratings.subject_key.isna().any() or not ratings.prompt.eq(ratings.prompt_judge).all():
            raise ValueError("Judged output has no matching original generation/task")
        expected = ratings.w_output.where(ratings.variant.eq("watermarked"), ratings.uw_output)
        completion = pd.Series([text[len(prompt):] for text, prompt in zip(ratings.native_output, ratings.prompt)], index=ratings.index)
        if not completion.eq(expected).all() or not ratings.ratings.map(len).eq(3).all():
            raise ValueError("Native judged output or rating dimensions differ")
        ratings["dimension"] = ratings.dataset.map(parameters["dimensions"]).str.split("|")
        ratings = ratings.explode(["ratings", "dimension"], ignore_index=True).rename(columns={"ratings": "response"})
        ratings["response"] = pd.to_numeric(ratings.response)
        if not ratings.response.isin([1, 2, 3, 4, 5]).all():
            raise ValueError("Cached quality rating is outside its declared ordinal scale")

        # 3. Keep generations lacking a published quality assessment as ungraded attempts.
        rated_keys = ratings[keys].drop_duplicates().assign(has_rating=True)
        ungraded = generations.merge(rated_keys, on=keys, how="left", validate="one_to_one")
        ungraded = ungraded.loc[ungraded.has_rating.isna()].assign(dimension="ungraded", response=None)
        observations = pd.concat([ratings, ungraded], ignore_index=True)
        observations = observations.astype(object).where(observations.notna(), None)
        subjects = generations[["subject_key", "source_file", "model", "experiment", "dataset", "variant"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.model + " / " + subjects.experiment + " / " + subjects.dataset + " / " + subjects.variant
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=row.model,
            declared_model_identifier=parameters["models"][row.model], experiment=row.experiment,
            source_generation_file=row.source_file, variant=row.variant) for row in subjects.itertuples()]
        identity = ["dataset", "prompt", "natural_text", "dimension"]
        items = observations.drop_duplicates(identity).copy()
        items["item_key"] = range(len(items))
        items["raw_item_id"] = items.dataset + "#" + items.source_row.astype(str)
        items["content"] = items.prompt
        items["features"] = [dict(source_dataset=row.dataset, dimension=row.dimension) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.natural_text,
            rule=self.grading["fallback_rule"] if row.dimension == "ungraded" else
            self.grading["rule"] + " " + self.grading["verifiers"][row.dataset]["criteria"][row.dimension])
            for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][row.dataset], sort_keys=True), judged_by="llm")
            for row in items.itertuples()]
        observations = observations.merge(items[identity + ["item_key"]], on=identity, how="left", validate="many_to_one")
        observations["response_key"] = range(len(observations))

        # 4. Preserve full outputs, paired judge context, source positions and auxiliary scores.
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            variant=row.variant, output=row.native_output, literal_prompt_prefix=row.native_output.startswith(row.prompt), natural_text=row.natural_text,
            detector_score=row.detector_score, natural_detector_score=row.natural_detector_score,
            judgment_file=row.judgment_file, judgment_row=None if row.judgment_row is None else int(row.judgment_row),
            source_judgment=row.source_judgment, dimension=row.dimension), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    MedWatermarkFWS(__file__).main_from_args()
