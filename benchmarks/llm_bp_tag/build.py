"""Tabulate the released LLM-BP node-classification inputs and predictions."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class LLMBPTag(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, prompt = parameters["layout"], parameters["prompt"]
        frames = []
        for dataset, spec in parameters.items():
            if spec.get("role") != "dataset":
                continue

            # 1. Load node tables, original test order and passive message records.
            labels = pd.read_csv(self.raw_dir / layout["categories"].format(dataset=dataset),
                keep_default_na=False).iloc[:, 0].tolist()
            if spec["format"] == "csv":
                nodes = pd.read_csv(self.raw_dir / spec["nodes"], keep_default_na=False)
                if nodes.node_id.tolist() != list(range(len(nodes))):
                    raise ValueError("CSV row order must equal the original graph node order")
                nodes = nodes[["node_id", "raw_text", "label"]]
                test_ids = nodes.node_id.to_numpy()
            else:
                graph = read_native_pickle(self.raw_dir / layout["graph"].format(dataset=dataset))
                data = graph.state["_store"].state["_mapping"]
                texts = read_native_pickle(self.raw_dir / layout["texts"].format(dataset=dataset))
                nodes = pd.DataFrame(dict(node_id=range(len(texts)), raw_text=texts, label=data["y"]))
                test_ids = np.flatnonzero(data[spec["test_field"]]) if spec["test_field"] == "test_mask" else data[spec["test_field"]]
            test = pd.DataFrame(dict(source_position=range(len(test_ids)), node_id=test_ids))
            filename = layout["results"].format(dataset=dataset)
            messages = read_native_pickle(self.raw_dir / filename)
            records = [{**message.state, "__pydantic_fields_set__": sorted(message.state["__pydantic_fields_set__"])}
                for message in messages]
            predictions = pd.DataFrame(dict(source_position=range(len(records)), message=records))
            if len(test) != len(predictions) or test.node_id.duplicated().any():
                raise ValueError("Every original test node must have exactly one released prediction")

            # 2. Join by the source's documented positions; never truncate to fit.
            frame = test.merge(nodes, on="node_id", how="left", validate="one_to_one")
            frame = frame.merge(predictions, on="source_position", validate="one_to_one")
            if frame[["raw_text", "label", "message"]].isna().any().any() or not frame.label.isin(range(len(labels))).all():
                raise ValueError("A test node has no text or a label outside the declared categories")
            frame["output"] = frame.message.map(lambda record: record["__dict__"]["content"])
            if not frame.output.map(lambda value: isinstance(value, str)).all():
                raise ValueError("The published parser requires a text response")

            # 3. Apply the published parser with column operations and retain its ambiguities.
            matches = pd.DataFrame({index: frame.output.str.contains(re.sub(r"\(.*?\)", "", label),
                flags=re.IGNORECASE, regex=True) for index, label in enumerate(labels)})
            frame["prediction"] = matches.idxmax(axis=1).where(matches.sum(axis=1).eq(1), -1)
            frame.loc[matches.sum(axis=1).gt(1), "prediction"] = -2
            frame["response"] = frame.prediction.eq(frame.label).astype(float)
            frame["reference"] = frame.label.map(dict(enumerate(labels)))
            frame["content"] = frame.raw_text.str[:int(prompt["max_chars"])].map(lambda text: json.dumps([
                dict(role="system", content=prompt["system"]), dict(role="user", content=prompt["user"].format(
                    description=spec["description"], n_classes=len(labels), labels=labels, text=text))], ensure_ascii=False))
            frame = frame.assign(dataset=dataset, source_file=filename)
            frame["item_key"] = dataset + "::node" + frame.node_id.astype(str)
            frame["response_key"] = filename + "#" + frame.source_position.astype(str)
            frames.append(frame)
        attempts = pd.concat(frames, ignore_index=True)

        # 4. Define source-specific subjects, actual prompts and the original grading rule.
        subject = parameters["subject"]
        subjects = pd.DataFrame([dict(subject_key=subject["key"], raw_label=subject["label"],
            features=parameters["subject_features"])])
        attempts["subject_key"] = subject["key"]
        attempts["test_condition"] = attempts.dataset.map(lambda dataset: json.dumps(dict(dataset=dataset,
            native_trial=0, grading="original_category_parser"), sort_keys=True))
        items = attempts[["item_key", "content", "reference", "dataset"]].copy()
        items["raw_item_id"] = items.item_key
        items["features"] = items.dataset.map(lambda dataset: dict(graph_dataset=dataset))
        items["grading_criterion"] = items.reference.map(lambda label: dict(reference_answer=label,
            rule=self.grading["verifiers"]["classification"]["rule"]))
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["classification"], sort_keys=True))

        # 5. Keep complete message states and exact node/response positions in traces.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_position=row.source_position,
            dataset=row.dataset, node_id=row.node_id, native_trial=0, message=row.message,
            parsed_category_index=row.prediction, reference_category_index=row.label), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        # The shared registrar numbers occurrences after identical items resolve.
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LLMBPTag(__file__).main_from_args()
