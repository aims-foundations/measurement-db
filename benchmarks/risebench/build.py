#!/usr/bin/env python3
"""Curate native RISEBench releases without conflating their grading protocols."""

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class RISEBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "tasks", "harness", "current_results", "current_harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read tasks and model profiles; preserve known settings, leaving others unknown.
        parameters = self.build_parameters
        tasks = pd.read_json(self.raw_dir / "tasks/datav2_total_w_subtask.json")
        tasks = tasks.astype(object).where(tasks.notna(), None)
        subjects = pd.DataFrame({
            "raw_label": parameters["subjects"],
            "thinking": parameters["thinking"],
            "cfg_img_scale": parameters["cfg_img_scale"],
        }).rename_axis("subject_key").reset_index()
        subjects["features"] = [
            {**parameters["subject_features"], **profile}
            for profile in subjects[["thinking", "cfg_img_scale"]].to_dict("records")
        ]
        current_subjects = pd.DataFrame.from_dict(
            parameters["current_subjects"], orient="index", columns=["raw_label"]
        ).rename_axis("subject_key").reset_index()
        current_subjects["features"] = [
            {**parameters["current_subject_features"], "upstream_profile": profile,
             **({"thinking": parameters["current_thinking"][profile]}
                if profile in parameters["current_thinking"] else {})}
            for profile in current_subjects.subject_key
        ]

        # 2. Load native Excel result tables and original model outputs.
        # The older release is packaged in ZIPs; the newer release uses directories.
        sheets = []
        for profile in subjects.subject_key:
            archive = f"results/{profile}.zip"
            with zipfile.ZipFile(self.raw_dir / archive) as release:
                member = f"{profile}/bagel_judge.xlsx"
                sheet = pd.read_excel(io.BytesIO(release.read(member)), keep_default_na=False)
                sheet["native_judgment"] = sheet.to_dict("records")
                output_paths = profile + "/" + sheet.category + "/" + sheet["index"] + ".png"
                sheet["output_image_base64"] = output_paths.map(
                    lambda path: base64.b64encode(release.read(path)).decode("ascii")
                )
                sheet["output_member"] = output_paths
                if parameters["thinking"][profile] == "enabled":
                    sheet["reasoning_trace"] = output_paths.str.removesuffix(".png").add(".txt").map(
                        lambda path: release.read(path).decode("utf-8")
                    )
                else:
                    sheet["reasoning_trace"] = None
                sheets.append(sheet.assign(subject_key=profile, source_archive=archive,
                                           source_member=member, grading_version="legacy"))
        # Filenames supply task identifiers. The join rejects ambiguous image matches.
        paths = sorted(path for path in (self.raw_dir / "current").rglob("*")
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".txt"})
        files = pd.DataFrame({"path": [str(path.relative_to(self.raw_dir)) for path in paths]})
        files["subject_key"] = files.path.str.split("/").str[1]
        files["index"] = files.path.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
        is_text = files.path.str.endswith(".txt")
        images = files.loc[~is_text].rename(columns={"path": "output_member"})
        thoughts = files.loc[is_text].rename(columns={"path": "reasoning_file"})
        for profile in current_subjects.subject_key:
            spreadsheet, = sorted((self.raw_dir / "current" / profile).glob("*_judge.xlsx"))
            sheet = pd.read_excel(spreadsheet, keep_default_na=False)
            sheet["native_judgment"] = sheet.to_dict("records")
            sheet["subject_key"] = profile
            sheet = sheet.merge(images, on=["subject_key", "index"], how="left", validate="one_to_one")
            sheet = sheet.merge(thoughts, on=["subject_key", "index"], how="left", validate="one_to_one")
            sheet["output_image_base64"] = sheet.output_member.map(
                lambda path: base64.b64encode((self.raw_dir / path).read_bytes()).decode("ascii"), na_action="ignore"
            )
            sheet["reasoning_trace"] = sheet.reasoning_file.map(
                lambda path: (self.raw_dir / path).read_bytes().decode("utf-8"), na_action="ignore"
            )
            if profile in parameters["video_inputs"]:
                records = json.loads((self.raw_dir / parameters["video_inputs"][profile]).read_text())
                video = pd.json_normalize(records, max_level=0)
                video["native_video_request"] = records
                video["video_source_file"] = "current/" + profile + "/" + video.video_path.str.removeprefix("./")
                video["generated_video_base64"] = video.video_source_file.map(
                    lambda path: base64.b64encode((self.raw_dir / path).read_bytes()).decode("ascii")
                )
                sheet = sheet.merge(video[["index", "video_text", "native_video_request",
                                           "video_source_file", "generated_video_base64"]],
                                    on="index", how="left", validate="one_to_one")
            sheets.append(sheet.assign(subject_key=profile, source_archive=None,
                                       source_member=str(spreadsheet.relative_to(self.raw_dir)),
                                       grading_version="current"))
        observations = pd.concat(sheets, ignore_index=True).rename(columns={"index": "item_key"})
        observations = observations.astype(object).where(observations.notna(), None)
        observations["response_key"] = observations.index
        subjects = pd.concat([subjects, current_subjects], ignore_index=True)

        # 3. Associate instructions and grading references with their exact images.
        # Reference images are grader inputs, distinguished from the model stimulus.
        items = tasks.rename(columns={"index": "item_key", "instruction": "content"})
        file_links = items[["item_key", "image", "reference_img"]].melt(
            id_vars="item_key", var_name="kind", value_name="path"
        ).dropna(subset=["path"])
        with zipfile.ZipFile(self.raw_dir / "tasks/data.zip") as images:
            file_links["data"] = file_links.path.map(lambda path: images.read("data/" + path))
        file_links["media_type"] = file_links.path.str.rsplit(".", n=1).str[-1].map(
            {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
        )
        file_links["role"] = file_links.kind.map({"image": "input_image", "reference_img": "grading_reference"})
        file_links["attachment"] = file_links[["path", "data", "media_type", "role"]].to_dict("records")
        attachments = file_links.groupby("item_key", sort=False).attachment.agg(list).rename("attachments")
        items = items.merge(attachments, on="item_key", validate="one_to_one")

        # 4. Each grading protocol defines its own item; legacy item identities remain intact.
        # Released video-generation prompts are additional known model inputs.
        video_inputs = observations.loc[observations.native_video_request.notna(),
                                        ["item_key", "subject_key", "video_text"]]
        video_items = items.merge(video_inputs, on="item_key", validate="one_to_many")
        video_items["content"] = [json.dumps({"instruction": row["content"], "generation_prompt": row["video_text"]},
                                             ensure_ascii=False, allow_nan=False)
                                  for row in video_items[["content", "video_text"]].to_dict("records")]
        video_items["grading_version"] = "current"
        video_items["input_variant"] = "video/" + video_items.subject_key
        items = items.merge(pd.DataFrame({"grading_version": ["legacy", "current"]}), how="cross")
        items["input_variant"] = items.grading_version
        items = pd.concat([items, video_items], ignore_index=True)
        rubric = items.reindex(columns=list(parameters["grading_fields"])).copy()
        rubric["completion_rule"] = items.grading_version.map({
            "legacy": self.grading["rule"],
            "current": self.grading["verifiers"]["current_judgments"]["rule"],
        })
        rules = rubric.to_dict("records")
        references = items.reference.combine_first(items.reference_txt)
        items["grading_criterion"] = [
            {"reference_answer": reference, "rule": json.dumps(rule, ensure_ascii=False, allow_nan=False)}
            for reference, rule in zip(references, rules, strict=True)
        ]
        specs = {"legacy": self.grading["verifiers"]["published_judgments"],
                 "current": self.grading["verifiers"]["current_judgments"]}
        items["verifier"] = items.grading_version.map({
            version: Judge(spec=json.dumps(spec, sort_keys=True), judge=spec.get("judge"), judged_by="llm")
            for version, spec in specs.items()
        })
        items["features"] = items[["category", "subtask"]].to_dict("records")
        items["raw_item_id"] = items.item_key
        items["item_key"] = items.input_variant + ":" + items.item_key
        observations["input_variant"] = observations.grading_version
        has_video = observations.native_video_request.notna()
        observations.loc[has_video, "input_variant"] = "video/" + observations.loc[has_video, "subject_key"]
        observations["item_key"] = observations.input_variant + ":" + observations.item_key

        # 5. Preserve native completion grades, with failed judge parses left ungraded.
        responses = observations[["response_key", "subject_key", "item_key", "complete"]].rename(
            columns={"complete": "response"}
        )
        responses.loc[observations.match_log.ne("succeed"), "response"] = None

        # 6. Retain complete judgments and available outputs, including native failure messages.
        trace_fields = ["native_judgment", "source_archive", "source_member", "output_member",
                        "output_image_base64", "reasoning_trace"]
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(row, ensure_ascii=False, allow_nan=False)
                           for row in observations[trace_fields].to_dict("records")]
        video_trace_fields = trace_fields + ["native_video_request", "video_source_file", "generated_video_base64"]
        traces.loc[has_video, "trace"] = [json.dumps(row, ensure_ascii=False, allow_nan=False)
                                         for row in observations.loc[has_video, video_trace_fields].to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion",
                            "verifier", "features", "attachments"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    RISEBench(__file__).main_from_args()
