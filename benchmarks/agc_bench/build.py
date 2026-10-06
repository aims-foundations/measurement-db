#!/usr/bin/env python3
"""Curate AGC-Bench's native attempts and distinct per-judge measurements."""

import ast
import json
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class AGCBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = {name: self.raw_dir / path for name, path in parameters["layout"].items()}
        templates = parameters["templates"]
        protocols = self.grading["verifiers"]

        # 1. Concatenate the released attempts and join their fixed prompt banks.
        frames = []
        for path in sorted((paths["release"] / "generations").glob("*/*/*.parquet")):
            frame = pd.read_parquet(path)
            frame["native_record"] = frame.astype(object).where(frame.notna(), None).to_dict("records")
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        generations = pd.concat(frames, ignore_index=True)
        prompts = pd.concat([pd.read_parquet(path).assign(benchmark=path.stem)
                             for path in sorted((paths["release"] / "generations/prompts").glob("*.parquet"))],
                            ignore_index=True)
        if generations.duplicated(["model", "benchmark", "instance_id"]).any():
            raise ValueError("Duplicate native model/component/item attempts")
        prompts["reference_key"] = prompts.prompt.str.strip()
        for name, prefix in parameters["prefixes"].items():
            selected = prompts.benchmark.eq(name)
            prompts.loc[selected, "reference_key"] = prompts.loc[selected, "reference_key"].str.removeprefix(prefix)
        # MOH-X contains identical stimuli with conflicting labels; verify both
        # the source row identifier and its text rather than choosing one label.
        selected = prompts.benchmark.eq("moh_x")
        prompts.loc[selected, "reference_key"] = prompts.loc[selected, "instance_id"] + "\n" + prompts.loc[selected, "reference_key"]
        selected = prompts.benchmark.eq("fig_qa")
        tail = prompts.loc[selected, "reference_key"].str.rsplit("\n\n", n=2)
        prompts.loc[selected, "reference_key"] = tail.str[-2] + "\n\n" + tail.str[-1]
        selected = prompts.benchmark.isin(["arn", "pun_eval", "humor_transfer", "slang_generation", "ttcw", "irfl"])
        prompts.loc[selected, "reference_key"] = prompts.loc[selected, "instance_id"]

        # 2. Project the upstream reference banks into one reference table.
        references = []
        bank = pd.read_excel(paths["arn"])
        references.append(pd.DataFrame({"benchmark": "arn", "reference_key": "arn_all_" + bank.index.astype(str),
                                        "reference_answer": bank.correct_answer.astype(int).astype(str)}))
        bank = pd.read_json(paths["humor_transfer"] / "test.json", lines=True)
        references.append(pd.DataFrame({"benchmark": "humor_transfer",
                                        "reference_key": "humor_transfer_sarcasm_headlines_" + bank.index.astype(str).str.zfill(6),
                                        "reference_answer": bank.is_sarcastic.map({0: "No", 1: "Yes"})}))
        bank = pd.concat([pd.read_json(paths["pun_eval"] / "dataset" / filename, orient="index")
                          for filename in ["hom_dataset.json", "het_dataset.json"]])
        bank = bank.loc[bank.pun_word.notna()]
        references.append(pd.DataFrame({"benchmark": "pun_eval", "reference_key": bank.index,
                                        "reference_answer": bank.human_text.to_numpy()}))
        bank = pd.DataFrame(ast.literal_eval((paths["slang_generation"] / "data/conv_slang.txt").read_text()),
                            columns=["term", "definition"])
        bank = bank.loc[bank.term.str.strip().ne("") & bank.definition.str.strip().ne("")].reset_index(drop=True)
        references.append(pd.DataFrame({"benchmark": "slang_generation",
                                        "reference_key": "slang_generation_" + bank.index.astype(str),
                                        "reference_answer": bank.term.str.strip()}))
        bank = pd.read_json(paths["ttcw"] / "Art_or_Artifice/annotations/ttcw_annotations.json")
        bank["source_row"] = bank.index
        votes = bank.groupby(["story_id", "ttcw_idx", "binary_verdict"], sort=False).source_row.agg(["size", "min"]).reset_index()
        votes = votes.sort_values(["size", "min"], ascending=[False, True]).drop_duplicates(["story_id", "ttcw_idx"])
        references.append(pd.DataFrame({"benchmark": "ttcw",
                                        "reference_key": votes.story_id + "_ttcw_" + votes.ttcw_idx.astype(str).str.zfill(2),
                                        "reference_answer": votes.binary_verdict}))

        bank = pd.read_csv(paths["analobench"] / "AnaloBench-T1-Subset-S1.csv").rename(columns={"Sentence": "sentence", "Options": "options"})
        bank["reference_answer"] = bank.Label
        text_banks = {"analobench": bank}
        bank = pd.concat([pd.read_csv(path) for path in sorted(paths["fig_qa"].glob("*.csv"))], ignore_index=True)
        bank = bank.loc[bank.valid.eq(1) & bank.labels.isin([0, 1])].copy()
        bank["reference_answer"] = bank.labels.map({0: "A", 1: "B"})
        text_banks["fig_qa"] = bank
        bank = pd.read_json(paths["chinese_homophonic_puns"] / "data/task_1.json")
        bank["text"] = bank.text.str.strip()
        bank["reference_answer"] = bank.punchline.str.strip()
        text_banks["chinese_homophonic_puns"] = bank
        bank = pd.read_csv(paths["lcc_metaphor"] / "data/multi_ling/lcc_en/test.csv")
        spans = bank.span1.map(ast.literal_eval)
        bank["word"] = [" ".join(text.split()[start:end]) for text, (start, end) in zip(bank.text, spans)]
        bank["sentence"] = bank.text
        bank["reference_answer"] = bank.label.eq("Metaphor").map({True: "Yes", False: "No"})
        text_banks["lcc_metaphor"] = bank
        with ZipFile(paths["moh_x"]) as archive:
            with archive.open("data/MOH-X/MOH-X_formatted_svo_cleaned.csv") as handle:
                bank = pd.read_csv(handle)
        bank["verb"] = bank.verb.str.strip()
        bank["sentence"] = bank.sentence.str.strip()
        bank["reference_answer"] = bank.label.map({1: "Yes", 0: "No"})
        text_banks["moh_x"] = bank
        bank = pd.read_json(paths["munch"] / "tasks/word_judge.json").rename(columns={"s0": "sentence"})
        first = pd.json_normalize(bank.options.str[0]).set_axis(bank.index)
        second = pd.json_normalize(bank.options.str[1]).set_axis(bank.index)
        bank["a"], bank["b"] = first.text, second.text
        bank["reference_answer"] = (first.label.eq("apt").astype(int) + second.label.eq("apt").astype(int) * 2).map({0: "D", 1: "A", 2: "B", 3: "C"})
        text_banks["munch"] = bank
        bank = pd.read_json(paths["nyt_connections"] / "ConnectionsFinalDataset.json")
        bank["words"] = bank.words.str.join(", ")
        bank["reference_answer"] = ["\n".join(sorted(", ".join(sorted(group["words"])) for group in groups)) for groups in bank.answers]
        text_banks["nyt_connections"] = bank
        for name, bank in text_banks.items():
            keys = [templates[name].format(**row).strip() for row in bank.to_dict("records")]
            if name == "moh_x":
                keys = [f"id{index}\n{text}" for index, text in zip(bank.index, keys)]
            references.append(pd.DataFrame({"benchmark": name,
                "reference_key": keys,
                "reference_answer": bank.reference_answer.to_numpy()}))

        # 3. Restore IRFL's missing multimodal prompts with the source's seeded ordering.
        bank = pd.read_csv(paths["irfl"] / "idiom_detection_task.csv", keep_default_na=False)
        bank["instance_id"] = "idiom-detection-task_" + bank.index.astype(str)
        bank["definition"] = bank.definition.map(json.loads).str[0]
        image_orders, golds = [], []
        for index, row in bank.iterrows():
            answer = json.loads(row.answer)[0]
            order = [answer] + json.loads(row.distractors)
            random.Random(f"idiom-detection-task:{index}:{row.phrase}").shuffle(order)
            image_orders.append(order)
            golds.append(chr(65 + order.index(answer)))
        bank["images"], bank["reference_answer"] = image_orders, golds
        references.append(bank[["instance_id", "reference_answer"]].rename(columns={"instance_id": "reference_key"}).assign(benchmark="irfl"))
        stimuli, attachments = [], []
        with ZipFile(paths["irfl"] / "IRFL_images.zip") as archive:
            for row in bank.to_dict("records"):
                elements = [{"content_type": "text/plain", "text": templates["irfl"].format(**row)}]
                files = []
                for index, image in enumerate(row["images"]):
                    filename = f"images/{image}.jpeg"
                    elements.extend([{"content_type": "text/plain", "text": "\n" + chr(65 + index) + ")"},
                                     {"content_type": "image/jpeg", "location": filename}])
                    files.append({"data": archive.read(filename), "path": filename, "media_type": "image/jpeg", "role": "input"})
                elements.append({"content_type": "text/plain", "text": "\n\nAnswer:"})
                stimuli.append(json.dumps({"multimedia_elements": elements}, ensure_ascii=False))
                attachments.append(files)
        restored = pd.DataFrame({"instance_id": bank.instance_id, "restored_prompt": stimuli, "attachments": attachments})
        prompts = prompts.merge(restored.assign(benchmark="irfl"), on=["benchmark", "instance_id"], how="left", validate="one_to_one")
        selected = prompts.benchmark.eq("irfl")
        prompts.loc[selected, "prompt"] = prompts.loc[selected, "restored_prompt"]
        references = pd.concat(references, ignore_index=True).drop_duplicates()
        references = references.merge(prompts[["benchmark", "reference_key"]].drop_duplicates(), on=["benchmark", "reference_key"], how="inner")
        prompts = prompts.merge(references, on=["benchmark", "reference_key"], how="left", validate="many_to_one")
        expected_references = set(paths) - {"release"}
        if prompts.loc[prompts.benchmark.isin(expected_references), "reference_answer"].isna().any():
            missing = prompts.loc[prompts.benchmark.isin(expected_references) & prompts.reference_answer.isna(), ["benchmark", "instance_id"]]
            raise ValueError(f"Unmatched upstream references: {missing.head(12).to_dict('records')}")
        if prompts.prompt.isna().any() or prompts.prompt.str.strip().eq("").any():
            raise ValueError("Every released task needs actual text or its reconstructed multimodal stimulus")

        # 4. Join original panel judgments without turning raters into extra model trials.
        generations = generations.merge(prompts[["benchmark", "instance_id", "prompt", "reference_answer", "attachments"]],
                                        on=["benchmark", "instance_id"], how="left", validate="many_to_one")
        canonical = generations.assign(protocol="canonical_" + generations.benchmark, rater="", response=generations.canonical_score)
        ratings = pd.read_parquet(paths["release"] / "analysis/jrt_complete_ratings.parquet")
        ratings["rating_record"] = ratings.astype(object).where(ratings.notna(), None).to_dict("records")
        ratings = ratings.rename(columns={"item_id": "instance_id", "score": "response"})
        if not ratings.rater.isin(parameters["panel_raters"]).all():
            raise ValueError("Unidentified panel judge")
        panel = ratings.merge(generations, on=["model", "benchmark", "instance_id"], how="left", validate="many_to_one")
        if panel.source_file.isna().any():
            raise ValueError("A panel judgment has no matching captured model attempt")
        panel["protocol"] = "panel_" + panel.benchmark
        observations = pd.concat([canonical, panel], ignore_index=True)
        observations["grade_status"] = observations.response.isna().map({True: "not_released", False: "released_grade"})
        for name, selected in observations.groupby("protocol", sort=False).groups.items():
            spec = protocols[name]
            metric_column = "metric" if name.startswith("panel_") else "canonical_metric"
            measured = observations.loc[selected, metric_column]
            absent = measured.isna() & observations.loc[selected, "response"].isna()
            if not (measured.eq(spec["metric"]) | absent).all():
                raise ValueError(f"Unexpected metric for {name}")
            values, scale = observations.loc[selected, "response"], spec["response_scale"]
            if values.isin([float("inf"), float("-inf")]).any():
                raise ValueError(f"Nonfinite native grade in {name}")
            valid = values.isin(scale["values"]) if scale["kind"] == "discrete" else values.between(
                float("-inf") if scale["min"] is None else scale["min"],
                float("inf") if scale["max"] is None else scale["max"])
            invalid = values.notna() & ~valid
            if spec["grade_policy"] == "non_grade_statistic":
                observations.loc[selected, "response"] = float("nan")
                observations.loc[selected, "grade_status"] = "non_grade_statistic"
            elif invalid.any():
                if spec["grade_policy"] != "flag_invalid_rating":
                    raise ValueError(f"Native grades outside their declared scale: {name}")
                bad = invalid.index[invalid]
                observations.loc[bad, "response"] = float("nan")
                observations.loc[bad, "grade_status"] = "invalid_rating_preserved_in_trace"

        # 5. Project the four output tables with full provenance and untruncated traces.
        observations["item_key"] = observations.benchmark + "::" + observations.instance_id + "::" + observations.protocol + "::" + observations.rater
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.benchmark + "::" + items.instance_id
        items["features"] = [{"component_dataset": row.benchmark, "source_item_id": row.instance_id,
                              "grading_channel": row.protocol, "prompt_source": "reconstructed_multimodal" if row.benchmark == "irfl" else "recorded"}
                             for row in items.itertuples()]
        items["grading_criterion"] = [{"reference_answer": row.reference_answer if pd.notna(row.reference_answer) else None,
                                        "rule": protocols[row.protocol]["rule"], "response_scale": protocols[row.protocol]["response_scale"]}
                                       for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(protocols[row.protocol]["implementation"], sort_keys=True),
                                    judge=parameters["panel_raters"].get(row.rater), judged_by="llm")
                             if protocols[row.protocol]["kind"] == "judge" else
                             ExactMatcher(spec=json.dumps(protocols[row.protocol]["implementation"], sort_keys=True))
                             for row in items.itertuples()]
        items = items.rename(columns={"prompt": "content"})
        subjects = observations[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [{**parameters["subject_features"], "model_identifier": name} for name in subjects.subject_key]
        observations["response_key"] = observations.index
        observations["subject_key"] = observations.model
        observations["test_condition"] = "source_run=" + observations.source_run_dir
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": int(row.source_row),
                                      "generation": row.native_record, "grade_status": row.grade_status,
                                      "panel_rating": row.rating_record if isinstance(row.rating_record, dict) else None},
                                     ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "grading_criterion", "verifier", "features"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AGCBench(__file__).main_from_args()
