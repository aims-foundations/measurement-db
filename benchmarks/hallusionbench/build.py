"""Tabulate released HallusionBench outputs, complete images and explicit grading scope."""

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class HallusionBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, parsing = parameters["paths"], parameters["parsing"]
        coordinates = list(parameters["coordinates"])

        # 1. Concatenate the maintained model exports, preserving all native cells.
        frames = []
        for path in sorted((self.raw_dir / paths["release"]).glob(paths["predictions"])):
            frame = pd.read_excel(path, keep_default_na=False)
            frame["native_record"] = frame.to_dict("records")
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index,
                subject_key=path.name.removesuffix(parsing["suffix"]), scope="primary"))
        responses = pd.concat(frames, ignore_index=True)
        if responses.duplicated(["subject_key", "index"]).any():
            raise ValueError("Duplicate primary model/question export")

        # 2. Join each export to its original question-bank version, without fuzzy matching.
        banks = []
        for scope, key in (("primary", "bank"), ("sample", "historical_bank")):
            records = json.loads((self.raw_dir / paths[key]).read_text())
            bank = pd.json_normalize(records, max_level=0)
            bank["bank_record"] = records
            bank["index"] = bank[coordinates].astype(str).agg("_".join, axis=1)
            bank["scope"] = scope
            banks.append(bank)
        bank = pd.concat(banks, ignore_index=True)
        bank["l2-category"] = bank.category + "_" + bank.subcategory
        bank["image_path"] = bank.category + "/" + bank.subcategory + "/" + bank.set_id + "_" + bank.figure_id + ".png"
        bank["image_path"] = bank.image_path.mask(bank.visual_input.eq("0"), "")
        bank["answer"] = bank.gt_answer.map(parameters["answers"])
        bank["item_key"] = bank.scope + ":" + bank["index"]
        responses["image_lookup"] = responses.image_path.str.casefold()
        bank["image_lookup"] = bank.image_path.str.casefold()
        fields = ["scope", "index", "category", "question", "gt_answer_details", "l2-category", "image_lookup", "answer"]
        responses = responses.merge(bank[fields + ["item_key", "bank_record"]], on=fields, how="left", validate="many_to_one")
        if responses.item_key.isna().any():
            raise ValueError("An export differs from the declared question, image or reference")
        records = json.loads((self.raw_dir / paths["sample"]).read_text())
        sample = pd.json_normalize(records, max_level=0)
        sample["native_record"] = records
        sample["source_row"] = sample.index
        sample["source_file"] = paths["sample"]
        sample["scope"] = "sample"
        sample["subject_key"] = "reference_sample"
        sample["prediction"] = sample.model_prediction
        sample["index"] = sample[coordinates].astype(str).agg("_".join, axis=1)
        keys = sorted(records[0].keys() - {"model_prediction"})
        sample = sample.merge(bank.loc[bank.scope.eq("sample"), keys + ["item_key", "bank_record", "answer"]], on=keys, how="left", validate="one_to_one")
        if sample.item_key.isna().any():
            raise ValueError("The original reference sample must match the historical bank exactly")
        responses = pd.concat([responses, sample], ignore_index=True)
        responses["response_key"] = responses.source_file + ":" + responses.source_row.astype(str)

        # 3. Apply the frozen upstream exact matcher with vectorized string operations.
        original = responses.prediction.astype(str).str.lower()
        text = original.copy()
        numeric_comma = original.str.contains(parsing["numeric_comma"], regex=True)
        for punctuation in parsing["punctuation"]:
            erase = (original.str.contains(punctuation + " ", regex=False)
                | original.str.contains(" " + punctuation, regex=False) | numeric_comma)
            text = text.str.replace(punctuation, " ", regex=False).mask(erase, text.str.replace(punctuation, "", regex=False))
        text = text.str.replace(parsing["period"], "", n=int(parsing["period_limit"]), regex=True)
        words = text.str.split()
        yes, no = words.map(lambda tokens: "yes" in tokens), words.map(lambda tokens: "no" in tokens)
        responses["extracted_answer"] = "Unknown"
        responses.loc[yes & ~no, "extracted_answer"] = "Yes"
        responses.loc[no & ~yes, "extracted_answer"] = "No"
        unavailable = responses.prediction.eq("") | responses.prediction.astype(str).str.contains(parsing["failure"], regex=False)
        responses["response"] = responses.extracted_answer.eq(responses.answer).astype(float).mask(unavailable | responses.scope.eq("sample"))
        responses["grade_status"] = "derived_exact_matching"
        responses.loc[unavailable, "grade_status"] = "unavailable_output"
        responses.loc[responses.scope.eq("sample"), "grade_status"] = "unattributed_sample_without_original_grade"
        responses["extracted_answer"] = responses.extracted_answer.astype(object).mask(unavailable | responses.scope.eq("sample"), None)

        # 4. Attach the actual author-released image bytes, respecting filename case.
        files = pd.Series(sorted((self.raw_dir / paths["images"]).glob("*/*/*")), name="source_path").to_frame()
        files = files.loc[files.source_path.map(Path.is_file)].copy()
        files["image_path"] = files.source_path.map(lambda path: str(path.relative_to(self.raw_dir / paths["images"])).casefold())
        items = bank.loc[bank.item_key.isin(responses.item_key)].copy()
        items["image_lookup"] = items.image_path.str.casefold()
        items = items.merge(files.rename(columns={"image_path": "image_lookup"}), on="image_lookup", how="left", validate="many_to_one")
        if items.loc[items.visual_input.ne("0"), "source_path"].isna().any():
            raise ValueError("A visual question is missing its image")
        items["attachments"] = [[dict(source_path=row.source_path, path=row.image_path, media_type="image/png", role="input")]
            if row.visual_input != "0" else [] for row in items.itertuples()]
        items["content"] = [json.dumps(dict(multimedia_elements=(
            [dict(content_type="image/png", location=row.image_path)] if row.visual_input != "0" else [])
            + [dict(content_type="text/plain", text=row.question)]), ensure_ascii=False) for row in items.itertuples()]
        items["raw_item_id"] = items.item_key
        items["features"] = items[["category", "subcategory", "visual_input", "set_id", "figure_id", "question_id"]].to_dict("records")
        items["grading_criterion"] = [dict(reference_answer=answer, rule=self.grading["rule"]) for answer in items.answer]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_matching"], sort_keys=True)) if scope == "primary"
            else Judge(spec=json.dumps(self.grading["verifiers"]["ungraded_reference_sample"], sort_keys=True)) for scope in items.scope]

        # 5. Preserve source model identifiers and full source-linked records.
        subjects = responses[["subject_key", "scope"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key.mask(subjects.scope.eq("sample"), parameters["labels"]["sample"])
        subjects["features"] = [dict(**parameters["subject_features" if row.scope == "primary" else "sample_features"],
            model_identifier=quote(row.subject_key, safe=" /-._")) for row in subjects.itertuples()]
        traces = responses[["response_key", "source_file", "source_row", "native_record", "bank_record", "grade_status", "extracted_answer"]].copy()
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces[["response_key", "trace"]]}


if __name__ == "__main__":
    HallusionBench(__file__).main()
