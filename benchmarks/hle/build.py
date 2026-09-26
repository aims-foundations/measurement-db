"""Curate HLE's released agentic runs, original grades and complete actor streams."""

import base64
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class HLEBuild(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("supai", "deepwriter", "official", "questions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]

        # 1. Turn the released per-question dictionaries into answer and judgment tables.
        with tarfile.open(self.raw_dir / paths["supai_archive"]) as archive:
            member = next(member for member in archive if member.name == paths["supai_root"] + "/" + paths["judged"])
            summary = pd.DataFrame.from_dict(json.load(archive.extractfile(member)), orient="index").rename_axis("question_id").reset_index()
        replies = pd.DataFrame(summary.response.tolist()).assign(question_id=summary.question_id).melt(
            id_vars="question_id", var_name="actor", value_name="published_output").dropna(subset="published_output")
        grades = pd.DataFrame(summary.judge_response.tolist()).assign(question_id=summary.question_id).melt(
            id_vars="question_id", var_name="actor", value_name="judgment").dropna(subset="judgment")
        published = grades.merge(replies, on=["question_id", "actor"], how="outer", validate="one_to_one")
        published["response"] = published.judgment.str["correct"].map({"yes": 1.0, "no": 0.0})
        if (published.judgment.notna() & published.response.isna()).any():
            raise ValueError("A published judgment has an unsupported correctness label")
        published_questions = set(summary.question_id)
        questions = pd.read_parquet(self.raw_dir / paths["questions"])

        # 2. Group each stream by actor; internal retries remain inside one observation.
        # Reading one file at a time avoids holding every native event object in memory.
        observations, trace_tables = [], []
        with tarfile.open(self.raw_dir / paths["supai_archive"], "r|gz") as archive:
            for member in archive:
                prefix = paths["supai_root"] + "/" + paths["streams"]
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                question_id = Path(member.name).stem
                events = pd.DataFrame({"record": json.load(archive.extractfile(member))})
                events["position"] = events.index
                events["actor"] = events.record.str["_source"].str["key"]
                events["type"] = events.record.str["type"]
                actors = events.loc[events.actor.notna()].groupby("actor", sort=False).agg(
                    event_indices=("position", list), native_events=("record", list), event_types=("type", lambda values: sorted(set(values))))
                starts = events.loc[events.type.eq("text-start")].groupby("actor", sort=False).position.max().rename("last_start")
                text = events.loc[events.type.eq("text"), ["actor", "position", "record"]].merge(starts, on="actor", how="inner", validate="many_to_one")
                text = text.loc[text.position.gt(text.last_start)].assign(text=lambda frame: frame.record.str["text"])
                outputs = text.groupby("actor", sort=False).text.sum().reindex(starts.index, fill_value="")
                actors["final_output"] = actors.index.map(outputs)
                actors["question_id"] = question_id
                actors = actors.reset_index().merge(published.loc[published.question_id.eq(question_id)], on=["question_id", "actor"], how="left", validate="one_to_one")
                recorded = actors.published_output.notna()
                if not actors.loc[recorded, "published_output"].eq(actors.loc[recorded, "final_output"]).all():
                    raise ValueError("A summary answer differs from the corresponding complete stream")
                actors["response_key"] = "supai:" + actors.question_id + ":" + actors.actor
                actors["subject_key"] = "supai:" + actors.actor
                actors["item_key"] = "supai:" + actors.question_id
                actors["trial"] = 1
                actors["grading_status"] = "published_judgment"
                actors.loc[actors.response.isna(), "grading_status"] = "ungraded_recorded_episode"
                actors.loc[actors.response.notna() & actors.final_output.isna(), "grading_status"] = "published_judgment_without_full_answer"
                actors["execution_evidence"] = actors.event_types.map(lambda kinds: sorted(set(kinds) & set(parameters["execution_events"])))
                actors["execution_status"] = actors.execution_evidence.map(lambda kinds: "recorded_activity" if kinds else "completion_metadata_only")
                shared = events.loc[events.actor.isna(), ["position", "record"]].to_dict("records")
                actors = actors.astype(object).where(actors.notna(), None)
                traces = actors[["response_key"]].copy()
                traces["trace"] = [json.dumps(dict(source="supai", question_id=question_id, actor=row.actor,
                    source_file=member.name.removeprefix(paths["supai_root"] + "/"), event_indices=row.event_indices,
                    events=row.native_events, shared_events=shared, final_output=row.final_output,
                    summary_present=question_id in published_questions, published_output=row.published_output,
                    judgment=row.judgment, grading_status=row.grading_status, execution_status=row.execution_status,
                    execution_evidence=row.execution_evidence), ensure_ascii=False, allow_nan=False) for row in actors.itertuples()]
                trace_tables.append(traces)
                observations.append(actors[["response_key", "subject_key", "item_key", "question_id", "actor", "trial", "response"]])
        responses = pd.concat(observations, ignore_index=True)
        if not set(zip(published.question_id, published.actor)).issubset(zip(responses.question_id, responses.actor)):
            raise ValueError("A published observation has no captured stream")

        # 3. Retain DeepWriter's separate harness and saved question/reference version.
        with tarfile.open(self.raw_dir / paths["deepwriter_archive"]) as archive:
            member = next(member for member in archive if member.name == paths["deepwriter_root"] + "/" + paths["deepwriter_csv"])
            deepwriter = pd.read_csv(archive.extractfile(member), dtype=str, keep_default_na=False)
        deepwriter["source_row"] = deepwriter.index
        deepwriter["native_record"] = deepwriter.drop(columns="source_row").to_dict("records")
        deepwriter = deepwriter.loc[deepwriter.id.ne("") & deepwriter.id.ne("Totals:")].copy()
        deepwriter["response"] = pd.to_numeric(deepwriter.score, errors="raise")
        if not deepwriter.response.isin([0, 1]).all() or not deepwriter.result.isin(["pass", "fail"]).all():
            raise ValueError("DeepWriter grades must follow the released binary contract")
        deepwriter["source_discrepancy"] = deepwriter.response.ne(deepwriter.result.map({"pass":1, "fail":0}))
        deepwriter["response_key"] = "deepwriter:" + deepwriter.id
        deepwriter["item_key"] = "deepwriter:" + deepwriter.id
        deepwriter["subject_key"], deepwriter["trial"] = "deepwriter", 1
        traces = deepwriter[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source="deepwriter", source_file=paths["deepwriter_csv"], source_row=int(row.source_row),
            native_record=row.native_record, grading_status="published_numeric_score", score_result_disagreement=bool(row.source_discrepancy)),
            ensure_ascii=False, allow_nan=False) for row in deepwriter.itertuples()]
        trace_tables.append(traces)

        # 4. Preserve original questions and image bytes, excluding reference answers from inputs.
        items = questions.loc[questions.id.isin(responses.question_id), ["id", "question", "answer", "image"]].copy()
        if set(items.id) != set(responses.question_id):
            raise ValueError("A Sup AI episode lacks its official question content")
        items["protocol"], items["item_key"] = "supai", "supai:" + items.id
        items = pd.concat([items, deepwriter[["id", "question", "answer", "item_key"]].assign(image="", protocol="deepwriter")], ignore_index=True)
        items["raw_item_id"] = items.id
        images = items.loc[items.image.ne(""), ["item_key", "image"]].copy()
        images[["media_type", "base64"]] = images.image.str.extract(r"^data:([^;]+);base64,(.*)$")
        if images.base64.isna().any():
            raise ValueError("A released question image is not a complete data URI")
        images["data"] = images.base64.map(lambda value: base64.b64decode(value, validate=True))
        images["path"] = images.data.map(lambda value: "images/" + hashlib.sha256(value).hexdigest())
        images["attachment"] = [dict(data=row.data, path=row.path, media_type=row.media_type, role="input") for row in images.itertuples()]
        items = items.merge(images[["item_key", "path", "media_type", "attachment"]], on="item_key", how="left", validate="one_to_one")
        items["content"] = [json.dumps(dict(multimedia_elements=[dict(content_type="text/plain", text=row.question)] +
            ([dict(content_type=row.media_type, location=row.path)] if isinstance(row.path, str) else [])), ensure_ascii=False) for row in items.itertuples()]
        items["attachments"] = items.attachment.map(lambda value: [value] if isinstance(value, dict) else [])
        items["features"] = [dict(source_protocol=row.protocol, input_scope=parameters["presentation"][row.protocol + "_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.answer, rule=self.grading["rule"] + "\n" + self.grading["verifiers"][row.protocol]["protocol"]) for row in items.itertuples()]
        items["verifier"] = [Judge(judged_by="llm", judge=self.grading["verifiers"][row.protocol].get("judge"),
            spec=json.dumps(self.grading["verifiers"][row.protocol], sort_keys=True)) for row in items.itertuples()]

        # 5. Define subjects by both the published model alias and the execution system.
        subjects = responses[["subject_key", "actor"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.actor.map(parameters["model_labels"])
        if subjects.raw_label.isna().any():
            raise ValueError("A recorded actor has no declared model identity")
        system = parameters["supai"]
        subjects["features"] = [dict(harness=system["harness"], model_identifier=actor,
            execution_scope=system["execution_scope"], system_prompt=quote(system["system_prompt"], safe=" /-._")) for actor in subjects.actor]
        system = parameters["deepwriter"]
        subjects = pd.concat([subjects[["subject_key", "raw_label", "features"]], pd.DataFrame([dict(subject_key="deepwriter", raw_label=system["label"],
            features=dict(harness=system["harness"], model_identifier=system["model_identifier"], execution_scope=system["execution_scope"]))])], ignore_index=True)

        # 6. Return linked tables; the shared writer enforces IDs, scales and relationships.
        columns = ["response_key", "subject_key", "item_key", "trial", "response"]
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": pd.concat([responses[columns], deepwriter[columns]], ignore_index=True),
            "traces": pd.concat(trace_tables, ignore_index=True)}


if __name__ == "__main__":
    HLEBuild(__file__).main_from_args()
