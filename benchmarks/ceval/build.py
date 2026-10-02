#!/usr/bin/env python3
"""Curate C-Eval questions and the released Contamination Detector predictions."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, scope = parameters["paths"], parameters["scope"]
        formatting = parameters["formatting"]

        # 1. Load the complete official question bank as one table. The source
        # category, split and original numeric ID identify each question.
        question_tables = []
        for path in sorted((self.raw_dir / paths["official"]).glob("*/*.parquet")):
            table = pd.read_parquet(path)
            table["question_record"] = table.to_dict("records")
            table["category"], table["split"] = path.parent.name, path.name.split("-")[0]
            table["question_file"] = str(path.relative_to(self.raw_dir))
            question_tables.append(table)
        questions = pd.concat(question_tables, ignore_index=True)
        questions["raw_key"] = questions.category + "-" + questions.id.astype(str)
        questions["raw_item_id"] = questions.category + "/" + questions.split + "/" + questions.id.astype(str)
        if questions.raw_item_id.duplicated().any() or not questions.answer.isin(list("ABCD")).all():
            raise ValueError("C-Eval requires unique question keys and released A–D references")
        options = questions[list("ABCD")].rename_axis(columns="option").stack().reset_index(name="text")
        options["text"] = options.option + formatting["choice_separator"] + options.text
        questions["content"] = questions.question + formatting["question_suffix"] + options.groupby("level_0").text.agg(
            formatting["line_separator"].join)

        # 2. Read the released option records and verify their accompanying
        # question export against the official validation bank before joining.
        prediction_tables = []
        with ZipFile(self.raw_dir / paths["predictions"]) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json"):
                    continue
                records = json.loads(archive.read(member)).get(scope["prediction_key"], {})
                if records:
                    table = pd.DataFrame.from_dict(records, orient="index").rename_axis("raw_key").reset_index()
                    table["subject_key"], table["source_member"] = Path(member).stem, member
                    prediction_tables.append(table)
        predictions = pd.concat(prediction_tables, ignore_index=True)
        if predictions.duplicated(["subject_key", "raw_key"]).any() or not predictions[["gold", "pred"]].isin(list("ABCD")).all().all():
            raise ValueError("C-Eval prediction keys must be unique and option records must be A–D")
        with ZipFile(self.raw_dir / paths["question_export"]) as archive:
            with archive.open(paths["question_member"]) as stream:
                exported = pd.read_json(stream, lines=True).rename(columns={"id": "raw_key"})
        exported["question_record"] = exported.rename(columns={"raw_key": "id"}).to_dict("records")
        validation = questions.loc[questions.split.eq(scope["split"])].copy()
        fields = ["question", "A", "B", "C", "D", "answer", "explanation"]
        pd.testing.assert_index_equal(exported.set_index("raw_key").index.sort_values(),
                                      validation.set_index("raw_key").index.sort_values())
        comparison = exported[["raw_key"] + fields].melt(id_vars="raw_key", var_name="field", value_name="export").merge(
            validation[["raw_key"] + fields].melt(id_vars="raw_key", var_name="field", value_name="official"),
            on=["raw_key", "field"], validate="one_to_one")
        changes = comparison.loc[comparison["export"].ne(comparison.official)].sort_values(["raw_key", "field"])
        if changes.to_dict("records") != json.loads(parameters["alignment"]["reviewed_changes"]):
            raise ValueError("C-Eval question exports have an unreviewed content or reference change")
        validation = validation.rename(columns={"question_record": "official_question_record"})
        validation = validation.merge(exported, on="raw_key", validate="one_to_one", suffixes=("_official", ""))
        options = validation[list("ABCD")].rename_axis(columns="option").stack().reset_index(name="text")
        options["text"] = options.option + formatting["choice_separator"] + options.text
        validation["official_content"] = validation.content
        validation["content"] = validation.question + formatting["question_suffix"] + options.groupby("level_0").text.agg(
            formatting["line_separator"].join)
        attempts = predictions.merge(validation, on="raw_key", how="left", validate="many_to_one", indicator=True)

        # 3. Account for the explicitly unresolved records without inventing
        # questions. Preserve the original scoring reference when it disagrees.
        unknown = attempts.loc[attempts._merge.eq("left_only")]
        if (len(unknown) != int(scope["unknown_count"]) or
                not unknown.source_member.eq(scope["unknown_member"]).all() or
                not unknown.raw_key.str.fullmatch(scope["unknown_pattern"]).all()):
            raise ValueError("C-Eval contains unreviewed prediction records without question mappings")
        attempts = attempts.loc[attempts._merge.eq("both")].reset_index(drop=True)
        disagreements = attempts.loc[attempts.gold.ne(attempts.answer), ["raw_key", "gold", "answer"]].drop_duplicates()
        expected = parameters["reference_disagreement"]
        if disagreements.to_dict("records") != [dict(raw_key=expected["raw_item_id"],
                gold=expected["reported_gold"], answer=expected["question_bank_answer"])]:
            raise ValueError("C-Eval has an unreviewed disagreement between recorded gold and the question bank")
        questions["item_key"] = questions.raw_item_id + "/official/reference-" + questions.answer
        attempts["item_key"] = attempts.raw_item_id + "/official/reference-" + attempts.gold
        changed_text = attempts.content.ne(attempts.official_content)
        attempts.loc[changed_text, "item_key"] = attempts.loc[changed_text, "raw_item_id"] + "/export/reference-" + attempts.loc[changed_text, "gold"]
        items = pd.concat([questions.assign(gold=questions.answer), attempts], ignore_index=True).drop_duplicates("item_key")

        # 4. Declare the published five-shot protocol and compute exactly the
        # source's equality statistic, retaining both item grading variants.
        subjects = attempts[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(parameters["protocol"]) for _ in range(len(subjects))]
        items["features"] = [dict(category=row.category, split=row.split) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=gold, rule=self.grading["rule"]) for gold in items.gold]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_matching"], sort_keys=True))
        responses = attempts[["subject_key", "item_key"]].assign(response_key=attempts.index,
            response=attempts.pred.eq(attempts.gold).astype(float), test_condition="split=" + scope["split"])

        # 5. Keep each original option record and its exact question association.
        # The original ZIP also retains all unresolved shuffle records unchanged.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["predictions"], source_member=row.source_member,
            source_key=row.raw_key, prediction_record=dict(gold=row.gold, pred=row.pred),
            question_file=paths["question_export"], question_member=paths["question_member"], question_record=row.question_record,
            official_question_file=row.question_file, official_question_record=row.official_question_record,
            question_bank_answer=row.answer, reference_disagrees=row.gold != row.answer),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    CEval(__file__).main_from_args()
