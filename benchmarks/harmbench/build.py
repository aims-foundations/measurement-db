"""Tabulate released HarmBench generations, task inputs and native judgments."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class HarmBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("website", "author")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]

        # 1. Read the released manifests and nested result mappings as tables.
        blocks, banks, models = [], [], []
        with tarfile.open(self.raw_dir / paths["website_archive"]) as archive:
            root = paths["website_root"] + "/" + paths["playground"] + "/"
            for modality, directory in [("text", "standard"), ("multimodal", "multimodal")]:
                manifest = json.load(archive.extractfile(root + "metadata_" + modality + ".json"))
                models.append(pd.json_normalize(manifest["models"]))
                records = json.load(archive.extractfile(root + modality + "_behaviors.json"))
                bank = pd.json_normalize(records)
                bank["website_behavior"] = records
                banks.append(bank[["BehaviorID", "website_behavior"]].assign(modality=modality))
                for method in pd.json_normalize(manifest["methods"]).value:
                    source_file = directory + "/" + method + ".json"
                    payload = archive.extractfile(root + source_file).read()
                    aliases = [source_file]
                    if modality == "text":
                        aliases += [category + "/" + method + ".json" for category in ["contextual", "copyright"]]
                        if any(archive.extractfile(root + alias).read() != payload for alias in aliases):
                            raise ValueError("Text-directory copies differ; review their observation identities")
                    frame = pd.DataFrame.from_dict(json.loads(payload), orient="index")
                    frame = frame.rename_axis(index="BehaviorID", columns="subject_key").stack().dropna().rename("source_record").reset_index()
                    frame = frame.join(pd.json_normalize(frame.source_record))
                    frame["source_aliases"] = [aliases] * len(frame)
                    blocks.append(frame.assign(modality=modality, method=method, source_file=source_file))
            results = pd.concat(blocks, ignore_index=True)
            results["image_name"] = results.test_case.map(lambda value: value[0] if isinstance(value, list) else None)
            wanted = set(results.image_name.dropna())
            images = {Path(member.name).name: archive.extractfile(member).read() for member in archive.getmembers()
                if member.isfile() and member.name.startswith(root + "multimodal/images/") and Path(member.name).name in wanted}
            if set(images) != wanted:
                raise ValueError("A released multimodal test case has no image")

        # 2. Join grading definitions; the website's multimodal category is mislabeled.
        definitions = []
        with tarfile.open(self.raw_dir / paths["author_archive"]) as archive:
            root = paths["author_root"] + "/"
            for modality, filename in self.build_parameters["banks"].items():
                bank = pd.read_csv(archive.extractfile(root + filename), dtype=str, keep_default_na=False)
                bank["author_behavior"] = bank.to_dict("records")
                definitions.append(bank[["BehaviorID", "FunctionalCategory", "SemanticCategory", "Tags", "author_behavior"]].assign(modality=modality))
        results = results.merge(pd.concat(banks), on=["modality", "BehaviorID"], how="left", validate="many_to_one")
        results = results.merge(pd.concat(definitions), on=["modality", "BehaviorID"], how="left", validate="many_to_one")
        if results.author_behavior.isna().any() or results.website_behavior.isna().any():
            raise ValueError("An observed behavior is absent from a captured definition bank")
        results["attempt_key"] = results.source_file + ":" + results.BehaviorID + ":" + results.subject_key
        if results.attempt_key.duplicated().any():
            raise ValueError("Repeated source generation")

        # 3. Melt only annotations that actually exist; two judgments are not two runs.
        judgments = results.melt(id_vars=[key for key in results if key not in ["label", "advbench_label"]],
            value_vars=["label", "advbench_label"], var_name="source_field", value_name="response")
        judgments = judgments.loc[[row.source_field in row.source_record for row in judgments.itertuples()]].copy()
        if not judgments.response.isin([0, 1]).all():
            raise ValueError("Expected explicit binary released judgments")
        judgments["protocol"] = judgments.modality
        judgments.loc[judgments.Tags.str.contains("hash_check", regex=False), "protocol"] = "copyright"
        judgments.loc[judgments.source_field.eq("advbench_label"), "protocol"] = "advbench"
        judgments["response_key"] = judgments.attempt_key + ":" + judgments.source_field
        judgments["item_key"] = judgments.response_key
        judgments["interactors"] = "attacker=" + judgments.method
        # Shared registration numbers distinct source entries when canonical
        # inputs coincide; that ordinal does not establish independent reruns.

        # 4. Describe target models and actual task inputs, with graders in item identity.
        subjects = pd.concat(models).drop_duplicates("value").rename(columns={"value": "subject_key", "label": "raw_label"})
        subjects = subjects.loc[subjects.subject_key.isin(results.subject_key)].copy()
        if set(subjects.subject_key) != set(results.subject_key):
            raise ValueError("An observed target model is absent from the released manifest")
        subjects["features"] = [dict(source_model=key, **self.build_parameters["subject_features"]) for key in subjects.subject_key]
        items = judgments.copy()
        items["raw_item_id"] = items.BehaviorID
        items["image_path"] = items.image_name.map({name: "images/" + hashlib.sha256(blob).hexdigest() + ".png" for name, blob in images.items()})
        items["attachments"] = [[dict(data=images[row.image_name], path=row.image_path, media_type="image/png", role="input")]
            if row.modality == "multimodal" else [] for row in items.itertuples()]
        items["content"] = [json.dumps(dict(multimedia_elements=[dict(content_type="text/plain", text=row.test_case)]
            if row.modality == "text" else [dict(content_type="image/png", location=row.image_path),
                dict(content_type="text/plain", text=row.test_case[1])]), ensure_ascii=False) for row in items.itertuples()]
        items["features"] = [dict(modality=row.modality, functional_category=row.FunctionalCategory, semantic_category=row.SemanticCategory,
            input_scope=self.build_parameters["presentation"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"] + "\n" + json.dumps(dict(
            behavior=row.website_behavior["Behavior"], context=row.author_behavior.get("ContextString", ""),
            image_description=row.author_behavior.get("RedactedImageDescription", ""), tags=row.Tags), ensure_ascii=False)) for row in items.itertuples()]
        protocols = self.grading["verifiers"]
        items["verifier"] = [Judge(judged_by="llm", judge=protocols[key]["judge"], spec=json.dumps(protocols[key], sort_keys=True))
            if protocols[key]["kind"] == "llm" else ExactMatcher(spec=json.dumps(protocols[key], sort_keys=True)) for key in items.protocol]

        # 5. Keep full source records and definition versions linked to every judgment.
        fields = ["response_key", "source_file", "source_aliases", "BehaviorID", "subject_key", "source_field", "source_record", "website_behavior", "author_behavior"]
        traces = judgments[fields].copy()
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in traces.drop(columns="response_key").to_dict("records")]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": judgments[["response_key", "subject_key", "item_key", "response", "interactors"]],
            "traces": traces[["response_key", "trace"]]}


if __name__ == "__main__":
    HarmBench(__file__).main()
