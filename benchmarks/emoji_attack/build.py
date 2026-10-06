"""Keep original input variants, outputs and success annotations together."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class EmojiAttack(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "reference")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["paths"]["results"]

        # 1. Read the original JSONL tables and retain each complete native line.
        frames = []
        for path in sorted(root.glob("*/*.jsonl")):
            text = path.read_text()
            frame = pd.read_json(io.StringIO(text), lines=True, dtype=False)
            frame["native_line"] = [line for line in text.splitlines() if line.strip()]
            frame["source_file"] = str(path.relative_to(root))
            frame["source_row"] = frame.index
            frame["attack_family"] = path.parent.name
            frame["model_label"] = path.stem
            frames.append(frame)
        attempts = pd.concat(frames, ignore_index=True)
        attempts = attempts.astype(object).where(attempts.notna(), None)

        # 2. Normalize the sole published verdict, preserving its original trace.
        if not attempts.eval_results.str.len().eq(1).all() or not attempts.target_responses.str.len().eq(1).all():
            raise ValueError("Each native row must pair exactly one output with one annotation")
        verdicts = attempts.eval_results.str[0]
        if not verdicts.map(lambda value: type(value) is bool or type(value) is str and value in {"True", "False"}).all():
            raise ValueError("Unrecognized published success annotation")
        attempts["response"] = verdicts.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

        # 3. Preserve the full released input components, including empty values.
        inputs = attempts[["query", "jailbreak_prompt", "translated_query"]].to_dict("records")
        attempts["content"] = [json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) for record in inputs]
        attempts["input_hash"] = attempts.content.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        attempts["item_key"] = attempts.attack_family + ":" + attempts.input_hash
        items = attempts.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [Judge(spec=json.dumps(dict(attack_family=method, **self.grading["verifiers"]["published"]), sort_keys=True))
            for method in items.attack_family]
        items["features"] = [dict(parameters["item_features"]) for _ in items.index]

        # 4. Retain native model labels and the documented InternLM spelling alias.
        attempts["subject_key"] = attempts.model_label.replace(parameters["model_aliases"])
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [dict(parameters["subject_features"]) for _ in subjects.index]

        # 5. Link all attempts to their full source record; number repeats centrally.
        attempts["response_key"] = attempts.source_file + ":" + attempts.source_row.astype(str)
        attempts["interactors"] = "attacker=" + attempts.attack_family
        attempts["trace"] = [json.dumps(dict(source_file=path, source_row=int(row), record=json.loads(record)), ensure_ascii=False, allow_nan=False)
            for path, row, record in zip(attempts.source_file, attempts.source_row, attempts.native_line)]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "interactors"]],
            "traces": attempts[["response_key", "trace"]],
        }


if __name__ == "__main__":
    EmojiAttack(__file__).main_from_args()
