#!/usr/bin/env python3
"""Curate released PhyBlock assembly outputs and derive their native per-scene F1."""

import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PhyBlock(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        layout = self.build_parameters["layout"]
        with zipfile.ZipFile(self.raw_dir / layout["archive"]) as archive:
            # 1. Read native output envelopes and retain their exact source text.
            files = pd.DataFrame({"member": [n for n in archive.namelist() if not n.endswith("/")]})
            files["source_file"] = files.member.str.split("/", n=1).str[1]
            members = files.set_index("source_file").member
            observations = files.join(files.source_file.str.extract(layout["outputs_pattern"]))
            observations = observations.dropna(subset=["model_directory", "scene"]).reset_index(drop=True)
            observations["native_record"] = observations.member.map(lambda n: archive.read(n).decode("utf-8"))
            native = pd.json_normalize(observations.native_record.map(json.loads), max_level=0)
            observations["raw_label"] = native.model.combine_first(native.model_version).fillna(observations.model_directory)
            # Match the author's extraction precedence, including the text block after
            # Claude's thinking block. Do not reinterpret other parts as final answers.
            texts = pd.concat([
                native.choices.str[0].str["message"].str["content"],
                native.content.str[0].str["text"], native.content.str[1].str["text"],
                native.candidates.str[0].str["content"].str["parts"].str[0].str["text"],
                native.predict,
            ], axis=1)
            observations["message"] = texts.bfill(axis=1).iloc[:, 0].fillna("")
            observations["content"] = native.instruction.fillna(self.build_parameters["instructions"]["paper_definition"])
            observations["prompt_source"] = native.instruction.notna().map({True: "native_record", False: "paper_definition_not_recorded_request"})

            # 2. Join the scene's reference geometry, candidate order and two images.
            scenes = files.join(files.source_file.str.extract(layout["goals_pattern"]))
            scenes = scenes.dropna(subset=["scene"])
            scenes["goal"] = scenes.member.map(lambda n: json.loads(archive.read(n)))
            scenes["candidates"] = [json.loads(archive.read(members[layout["candidates"].format(scene=s)])) for s in scenes.scene]
            scenes["level"] = scenes.goal.map(lambda goal: goal["level"])
            scenes["block_count"] = scenes.goal.map(lambda goal: len(goal["blocks"]))
            # Recorded open-model inputs identify these exact two task-bank images.
            for scene, paths in zip(observations.scene, native.image):
                if isinstance(paths, list) and paths != [p.format(scene=scene) for p in self.build_parameters["native_images"].values()]:
                    raise ValueError("Native image references disagree with the scene join")
            observations = observations.merge(scenes[["scene", "goal", "candidates", "level", "block_count"]],
                                              on="scene", validate="many_to_one")
            if len(observations) != len(native):
                raise ValueError("A released output has no matching scene definition")
            observations["indices"] = observations.message.str.findall(self.grading["verifiers"]["pose_constrained"]["index_pattern"])
            observations["indices"] = [[int(n) for n in indices if 0 < int(n) <= len(candidates)]
                                       for indices, candidates in zip(observations.indices, observations.candidates)]

            # 3. Keep both native grading conditions. The simpler condition reuses the
            # same output on level-2 scenes with fewer than seven reference blocks.
            specifications = self.grading["verifiers"]
            simple = specifications["topology_only"]
            topology = observations.loc[observations.level.eq(simple["source_level"]) &
                                        observations.block_count.lt(simple["max_blocks_exclusive"])]
            responses = pd.concat([observations.assign(condition="pose_constrained"),
                                   topology.assign(condition="topology_only")], ignore_index=True)
            # Native matching is sequential: each placement can enable later blocks.
            # This small stateful replay is necessary; it runs no model or simulator.
            grades = []
            for row in responses.itertuples():
                spec = specifications[row.condition]
                blocks, placed = row.goal["blocks"], set()
                for index in row.indices:
                    candidate = row.candidates[index - 1]
                    for position, block in enumerate(blocks):
                        dependencies = block.get("depend", [])
                        legal = (not spec["check_dependencies"] or dependencies == [0] or
                                 all(n - 1 in placed for n in dependencies))
                        if position not in placed and legal and all(candidate[k] == block[k] for k in spec["matching_fields"]):
                            placed.add(position)
                            break
                tp, fp, fn = len(placed), len(row.indices) - len(placed), len(blocks) - len(placed)
                precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                grades.append({"TP": tp, "FP": fp, "FN": fn, "response": round(f1, spec["decimal_places"])})
            responses = responses.join(pd.DataFrame(grades))
            responses["response_key"] = responses.index
            responses["trial"] = 1
            responses["test_condition"] = responses.condition

            # 4. Register distinct grading protocols and known prompt variants. For
            # API envelopes, the paper supplies a task definition, not an API transcript.
            keys = ["scene", "content", "condition", "prompt_source"]
            items = responses.drop_duplicates(keys).reset_index(drop=True)
            items["item_key"] = items.index
            items["raw_item_id"] = items.scene
            items["features"] = [{"scene": row.scene, "source_level": row.level, "grading_condition": row.condition,
                                  "prompt_source": row.prompt_source} for row in items.itertuples()]
            items["grading_criterion"] = [{"reference_answer": json.dumps({"goal": row.goal, "candidates": row.candidates}),
                                          "rule": self.grading["rule"]} for row in items.itertuples()]
            items["verifier"] = [ExactMatcher(spec=json.dumps(specifications[c], sort_keys=True)) for c in items.condition]
            items["attachments"] = [[{"path": path, "role": "input", "media_type": "image/png", "data": archive.read(members[path])}
                                      for path in [layout["goal_image"].format(scene=scene), layout["candidate_image"].format(scene=scene)]]
                                    for scene in items.scene]
            subjects = observations[["model_directory", "raw_label", "prompt_source"]].drop_duplicates().reset_index(drop=True)
            subjects["subject_key"] = subjects.model_directory
            subjects["features"] = [{"harness": "PhyBlock", "planning_strategy": "one_time",
                                     "native_model_directory": row.model_directory, "prompt_source": row.prompt_source,
                                     **({"reasoning_effort": self.build_parameters["reasoning_effort"][row.model_directory]}
                                        if row.model_directory in self.build_parameters["reasoning_effort"] else {})}
                                    for row in subjects.itertuples()]
            responses = responses.merge(items[keys + ["item_key"]], on=keys, validate="many_to_one")
            responses["subject_key"] = responses.model_directory

            # 5. Retain the complete native record and the derived grade's provenance.
            traces = responses[["response_key"]].copy()
            traces["trace"] = [json.dumps({"source_file": row.source_file, "native_record": row.native_record,
                                          "derived_grade": {"condition": row.condition, "TP": row.TP, "FP": row.FP,
                                                            "FN": row.FN, "F1": row.response}}, ensure_ascii=False)
                               for row in responses.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    PhyBlock(__file__).main_from_args()
