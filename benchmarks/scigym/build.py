#!/usr/bin/env python3
"""Curate released SciGym trajectories and their structural reaction-recovery scores."""

import json
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


class SciGym(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "tasks", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        # 1. Keep distinct trajectories, retaining the positions of identical copies.
        native = pd.read_parquet(self.raw_dir / parameters["layout"]["results"])
        fields = ["biomodel_id", "model_name", "chat_history", "final_model"]
        observations = native.drop_duplicates(fields).reset_index(drop=True)
        keys = ["biomodel_id", "model_name"]
        if observations.duplicated(keys).any():
            raise ValueError("Different trajectories need separate, source-supported trial identities")
        positions = native.assign(source_row=native.index).groupby(keys, sort=False).source_row.agg(list)
        observations = observations.merge(positions.rename("source_rows"), on=keys, validate="one_to_one")
        observations["run_key"] = observations.index.astype(str)
        end = parameters["transcript"]["initial_prompt_end"]
        if not observations.chat_history.str.contains(end, regex=False).all():
            raise ValueError("Cannot identify the initial user prompt in a released transcript")
        observations["initial_prompt"] = observations.chat_history.str.split(end, n=1, regex=False).str[0] + end
        observations["max_iterations"] = observations.initial_prompt.str.extract(parameters["transcript"]["iteration_pattern"])[0]
        if observations.max_iterations.isna().any():
            raise ValueError("The released initial prompt lacks an iteration allowance")
        tasks = pd.read_parquet(self.raw_dir / parameters["layout"]["tasks"]).rename(columns={"folder_name": "biomodel_id"})
        observations = observations.merge(tasks, on="biomodel_id", validate="many_to_one")
        if len(observations) != len(native.drop_duplicates(fields)):
            raise ValueError("A released trajectory has no matching task reference")
        if not all(partial in prompt for partial, prompt in zip(observations.partial, observations.initial_prompt)):
            raise ValueError("A transcript's starting model disagrees with the task bank")

        # 2. Normalize reference, partial and predicted XML into reaction tables.
        documents = pd.concat([
            tasks[["biomodel_id", "truth_xml"]].rename(columns={"truth_xml": "xml"}).assign(
                model_key=lambda table: "truth:" + table.biomodel_id),
            tasks[["biomodel_id", "partial"]].rename(columns={"partial": "xml"}).assign(
                model_key=lambda table: "partial:" + table.biomodel_id),
            observations[["biomodel_id", "run_key", "final_model"]].rename(columns={"final_model": "xml"}).assign(
                model_key=lambda table: "prediction:" + table.run_key),
        ], ignore_index=True)
        documents["reaction"] = documents.xml.map(ET.fromstring).map(lambda root: root.findall(parameters["xml"]["reactions"]))
        reactions = documents[["model_key", "reaction"]].explode("reaction").dropna(subset=["reaction"])
        for role in ["reactants", "products", "modifiers"]:
            path = parameters["xml"][role]
            reactions[role] = reactions.reaction.map(lambda reaction: tuple(sorted({node.attrib["species"] for node in reaction.findall(path)})))
        signatures = pd.concat([
            reactions.assign(condition="with_modifiers"),
            reactions.assign(condition="without_modifiers", modifiers=[()] * len(reactions)),
        ], ignore_index=True)
        signatures["signature"] = list(zip(signatures.reactants, signatures.products, signatures.modifiers))
        reaction_sets = signatures.groupby(["model_key", "condition"], sort=False).signature.agg(frozenset)

        # 3. Compare added reactions with missing reference reactions under both rules.
        scores = observations[["run_key", "biomodel_id"]].merge(
            pd.DataFrame({"condition": list(parameters["conditions"])}), how="cross")
        for name, prefix, key in [("truth", "truth:", "biomodel_id"), ("partial", "partial:", "biomodel_id"),
                                  ("prediction", "prediction:", "run_key")]:
            lookup = pd.MultiIndex.from_arrays([prefix + scores[key], scores.condition], names=["model_key", "condition"])
            scores[name] = [value if isinstance(value, frozenset) else frozenset() for value in reaction_sets.reindex(lookup)]
        missing = [truth - partial for truth, partial in zip(scores.truth, scores.partial)]
        added = [prediction - partial for prediction, partial in zip(scores.prediction, scores.partial)]
        scores["missing"] = [len(values) for values in missing]
        scores["added"] = [len(values) for values in added]
        scores["matched"] = [len(actual & expected) for actual, expected in zip(added, missing)]
        scores["precision"] = scores.matched / scores.added.where(scores.added.gt(0), 1)
        scores["recall"] = scores.matched / scores.missing.where(scores.missing.gt(0), 1)
        total = scores.precision + scores.recall
        scores["f1"] = 2 * scores.precision * scores.recall / total.where(total.gt(0), 1)
        grades = scores.melt(id_vars=["run_key", "condition"], value_vars=["precision", "recall", "f1"],
                             var_name="statistic", value_name="response")
        grades["native_metric"] = grades.condition.map(parameters["conditions"]) + "_" + grades.statistic
        # STE requires unreleased numerical results or a new simulation. Keep it null.
        unavailable = observations[["run_key"]].assign(native_metric="ste", response=None)
        grades = pd.concat([grades[["run_key", "native_metric", "response"]], unavailable], ignore_index=True)
        grades["metric"] = grades.native_metric.map(parameters["metrics"])
        responses = grades.merge(observations, on="run_key", validate="many_to_one")
        responses["response_key"] = responses.index
        responses["subject_key"] = responses.model_name
        responses["item_key"] = responses.biomodel_id + ":" + responses.metric
        responses["trial"] = 1

        # 4. Each metric has its own grading protocol and explicit score direction.
        subjects = observations[["model_name", "max_iterations"]].drop_duplicates().rename(
            columns={"model_name": "raw_label"})
        subjects["subject_key"] = subjects.raw_label
        subjects["features"] = [{"harness": "SciGym", "max_iterations": int(n), "interaction": "iterative"}
                                for n in subjects.max_iterations]
        items = responses.drop_duplicates(["item_key", "initial_prompt"])
        if items.duplicated("item_key").any():
            raise ValueError("Task prompts disagree between released trajectories")
        items = items.rename(columns={"initial_prompt": "content", "biomodel_id": "raw_item_id"}).copy()
        items["features"] = [{"split": "small"} for _ in items.index]
        specs = [self.grading["verifiers"]["trajectory_error" if metric == "ste" else "reaction_recovery"]
                 for metric in items.native_metric]
        items["grading_criterion"] = [
            {"reference_answer": json.dumps({"truth_xml": row.truth_xml, "partial": row.partial, "truth_sedml": row.truth_sedml}),
             "rule": self.grading["rule"], "response_scale": spec["response_scale"]}
            for row, spec in zip(items.itertuples(), specs)
        ]
        items["verifier"] = [ExactMatcher(spec=json.dumps({**spec, "native_metric": metric}, sort_keys=True))
                             for metric, spec in zip(items.native_metric, specs)]

        # 5. Preserve complete transcripts, final models and all duplicate source positions.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_rows": row.source_rows, "chat_history": row.chat_history,
                                      "final_model": row.final_model, "native_metric": row.native_metric,
                                      "grade_origin": "not_released_not_rerun" if row.native_metric == "ste"
                                                      else "recomputed_from_released_final_model"}, ensure_ascii=False)
                           for row in responses.itertuples()]
        items["testlet_id"] = items.raw_item_id
        tables = {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": traces,
        }
        return _reaction_items(tables, responses, scores, missing, added)


def _reaction_items(tables, observations, scores, missing, added):
    """Keep one execution, its original metrics, and native reaction comparisons."""
    parents = tables['items'].loc[tables['items'].item_key.str.endswith(':rms_with_modifiers_f1')].copy()
    parents['item_key'] = parents.raw_item_id
    parents['attachments'] = parents.grading_criterion.map(lambda value: [dict(path='reaction-reference.json',
        data=value['reference_answer'].encode(), role='grading_reference', media_type='application/json')])
    runs = observations.loc[observations.native_metric.eq('rpm_f1')].copy()
    runs['response_key'] = runs.run_key
    runs['item_key'] = runs.biomodel_id
    metrics = {key: [dict(native_metric=row.native_metric, score=None if pd.isna(row.response) else row.response,
        grade_origin='not_released_not_rerun' if row.native_metric == 'ste' else 'recomputed_from_released_final_model')
        for row in group.itertuples()] for key, group in observations.groupby('run_key', sort=False)}
    traces = pd.DataFrame([dict(response_key=row.run_key, trace=json.dumps(dict(source_rows=row.source_rows,
        chat_history=row.chat_history, final_model=row.final_model, measurements=metrics[row.run_key]), ensure_ascii=False, allow_nan=False))
        for row in runs.itertuples()])
    definitions, outcomes = {}, []
    protocol = json.loads(tables['items'].iloc[0].verifier.spec)
    del protocol['native_metric']
    for row, targets, predictions in zip(scores.itertuples(), missing, added, strict=True):
        checks = []
        for signature in sorted(targets):
            reference = json.dumps(dict(zip(('reactants', 'products', 'modifiers'), signature)), sort_keys=True)
            name = row.condition + ':reaction:' + hashlib.sha256(reference.encode()).hexdigest()
            checks.append((name, float(signature in predictions), dict(reference_answer=reference,
                rule=json.dumps(dict(condition=row.condition, rule='The final model recovers this missing reference reaction under the native species-set comparison.',
                    reference_asset='reaction-reference.json'), sort_keys=True),
                response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))))
        name = row.condition + ':extra'
        checks.append((name, float(len(predictions - targets)), dict(reference_answer=None,
            rule=json.dumps(dict(condition=row.condition, rule='Number of added reaction signatures absent from the missing reference set.',
                reference_asset='reaction-reference.json'), sort_keys=True),
            response_scale=dict(kind='interval', min=0, max=None, direction='lower_is_better'))))
        for name, grade, criterion in checks:
            definitions[row.biomodel_id, name] = dict(parent_item_key=row.biomodel_id, component_id=name,
                grading_criterion=criterion, verifier=ExactMatcher(spec=json.dumps(dict(
                    kind='native_reaction_set_comparison', condition=row.condition, original_protocol=protocol), sort_keys=True)))
            outcomes.append(dict(parent_response_key=row.run_key, component_id=name, response=grade))
    ste = tables['items'].loc[tables['items'].item_key.str.endswith(':simulation_trajectory_error')].set_index('raw_item_id')
    for row in runs.itertuples():
        original = ste.loc[row.biomodel_id]
        definitions[row.biomodel_id, 'ste'] = dict(parent_item_key=row.biomodel_id, component_id='ste',
            grading_criterion=original.grading_criterion, verifier=original.verifier)
        outcomes.append(dict(parent_response_key=row.run_key, component_id='ste', response=None))
    return component_tables(dict(subjects=tables['subjects'], items=parents,
        responses=runs[['response_key', 'subject_key', 'item_key', 'trial', 'response']], traces=traces),
        pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))


if __name__ == "__main__":
    SciGym(__file__).main_from_args()
