"""Curate released DevBench choice scores with complete, available image inputs."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DevBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "trog_images", "vocabulary_images")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        components = {name: parameters[group] for name, group in parameters["components"].items()}
        image_columns = list(parameters["image_columns"].values())

        # 1. Concatenate the author manifests; preserve their original row order.
        manifests = pd.concat([pd.read_csv(self.raw_dir / component["manifest"], keep_default_na=False)
            .assign(component=name, source_row=lambda frame: frame.index, choices=int(component["choices"]))
            for name, component in components.items()], ignore_index=True)
        manifests["item_key"] = manifests.component + "/" + manifests.source_row.astype(str)
        choices = manifests.melt(id_vars=["item_key", "component", "source_row", "text1", "choices"],
            value_vars=image_columns, var_name="option", value_name="image").dropna(subset="image")

        # 2. Join actual image bytes, preserving choice order and excluding incomplete trials.
        sources = []
        for name, group in parameters["file_images"].items():
            source = parameters[group]
            paths = sorted((self.raw_dir / source["root"]).glob(source["glob"]))
            sources.append(pd.DataFrame(dict(component=name,
                image=[source["prefix"] + str(path.relative_to(self.raw_dir / source["root"])) for path in paths],
                data=[path.read_bytes() for path in paths])))
        source = parameters["zip_images"]
        with ZipFile(self.raw_dir / source["file"]) as archive:
            paths = sorted(name for name in archive.namelist() if name.endswith(source["suffix"]))
            sources.append(pd.DataFrame(dict(component=source["component"],
                image=[source["prefix"] + Path(name).name for name in paths], data=[archive.read(name) for name in paths])))
        images = pd.concat(sources, ignore_index=True)
        choices = choices.merge(images, on=["component", "image"], how="left", validate="many_to_one")
        complete = choices.groupby("item_key").data.agg(lambda values: values.notna().all())
        choices = choices.loc[choices.item_key.map(complete)].sort_values(["item_key", "option"]).copy()
        choices["path"] = choices.component + "/" + choices.image
        choices["media_type"] = choices.image.map(lambda value: parameters["media_types"][Path(value).suffix])
        choices["attachment"] = [dict(data=row.data, path=row.path, media_type=row.media_type, role="input")
            for row in choices.itertuples()]
        choices["element"] = [dict(content_type=row.media_type, location=row.path) for row in choices.itertuples()]
        inputs = choices.groupby("item_key", sort=False).agg(attachments=("attachment", list), elements=("element", list))
        items = manifests.merge(inputs, on="item_key", how="inner", validate="one_to_one")

        # 3. Decode only published numeric arrays; the R scorer subtracts yes/no scores as doubles.
        scores = []
        for name, component in components.items():
            for path in sorted(self.raw_dir.glob(component["arrays"])):
                original = np.load(path, allow_pickle=False)
                values = original.squeeze().astype(float)
                if values.ndim == 3:
                    if values.shape[2] != 2:
                        raise ValueError("Unexpected DevBench yes/no axis")
                    values = values[:, :, 0] - values[:, :, 1]
                columns = image_columns[:int(component["choices"])]
                if values.shape != (int(manifests.component.eq(name).sum()), len(columns)) or not np.isfinite(values).all():
                    raise ValueError("DevBench score shape or values do not match the manifest")
                frame = pd.DataFrame(values, columns=columns).assign(component=name, source_row=lambda table: table.index,
                    subject_key=path.stem.removeprefix(component["prefix"]), source_file=str(path.relative_to(self.raw_dir)))
                frame["original_scores"] = original.tolist()
                frame["array_shape"] = [list(original.shape)] * len(frame)
                frame["array_dtype"] = str(original.dtype)
                frame["response"] = frame[columns[0]].gt(frame[columns[1:]].max(axis=1)).astype(float)
                scores.append(frame)
        observations = pd.concat(scores, ignore_index=True).merge(items[["component", "source_row", "item_key"]],
            on=["component", "source_row"], how="inner", validate="many_to_one")
        observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str)
        observations["test_condition"] = "component=" + observations.component

        # 4. Keep released model variants distinct and attach the original trial inputs.
        subjects = observations[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [parameters["subject_features"] for _ in subjects.index]
        items["raw_item_id"] = items.item_key
        items["content"] = [json.dumps(dict(multimedia_elements=[dict(content_type="text/plain", text=row.text1)]
            + row.elements), ensure_ascii=False) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=image_columns[0], rule=self.grading["rule"])
            for _ in items.index]
        items["verifier"] = items.component.map(lambda name: ExactMatcher(
            spec=json.dumps(self.grading["verifiers"][name], sort_keys=True)))
        items["features"] = [dict(component=row.component, input_scope=parameters["input_scope"]["description"])
            for row in items.itertuples()]

        # 5. Preserve each complete native numeric row, its shape, dtype and source position.
        traces = observations[["response_key", "source_file", "source_row", "component", "subject_key",
            "array_shape", "array_dtype", "original_scores"]].copy()
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
            for record in traces.drop(columns="response_key").to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features", "attachments"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]],
        }


if __name__ == "__main__":
    DevBench(__file__).main_from_args()
