"""Explicitly typed two-identity pilot tables, separate from the production schema.

Item and subject descriptions remain provisional. Source references and missing
metadata are retained, so these files must not be passed off as canonical imports.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

VERSION = "intervention-pilot-v2"
S, I, F, B = pa.string(), pa.int64(), pa.float64(), pa.bool_()
L = pa.list_(pa.field("element", pa.string()))


def schema(*columns):
    return pa.schema([pa.field(name, kind, nullable=nullable) for name, kind, nullable in columns],
                     metadata={b"measurement_db.pilot_schema": VERSION.encode()})


SCHEMAS = {
    "benchmarks": schema(("benchmark_id", S, False), ("name", S, False), ("version", S, True),
        ("protocol_ref", S, False), ("source_revision", S, False), ("scope", S, False),
        ("n_available_item_records", I, False), ("identity_status", S, False)),
    "experiments": schema(("experiment_id", S, False), ("study_id", S, False), ("name", S, False),
        ("status", S, False), ("paper_url", S, False), ("repository_url", S, False),
        ("source_revision", S, False), ("evaluation_date", S, True),
        ("n_responses", I, False), ("n_known_grades", I, False), ("n_observed_subjects", I, False)),
    "experiment_benchmarks": schema(("experiment_id", S, False), ("benchmark_id", S, False),
        ("analysis_scope_id", S, False), ("n_cohort_items", I, False), ("n_responses", I, False),
        ("n_known_grades", I, False), ("n_observed_subjects", I, False)),
    "items": schema(("item_id", S, False), ("benchmark_id", S, False), ("raw_item_id", S, False),
        ("content", S, True), ("source_reference_answer", S, True), ("protocol_ref", S, False),
        ("source_locator", S, False), ("metadata_status", S, False)),
    "subjects": schema(("subject_id", S, False), ("model", S, False), ("harness", S, False),
        ("harness_version", S, True), ("configuration_json", S, False), ("metadata_status", S, False)),
    "arms": schema(("arm_id", S, False), ("experiment_id", S, False), ("source_arm_id", S, False),
        ("definition_json", S, False), ("definition_source", S, False), ("status", S, False)),
    "cohort_items": schema(("experiment_id", S, False), ("benchmark_id", S, False), ("item_id", S, False)),
    "responses": schema(("response_id", S, False), ("experiment_id", S, False),
        ("benchmark_id", S, False), ("subject_id", S, False), ("item_id", S, False),
        ("trial", I, False), ("test_condition", S, True), ("interactors", S, True), ("response", F, True)),
    "response_provenance": schema(("response_id", S, False), ("arm_id", S, False),
        ("source_repeat", S, False), ("source_run", S, True), ("source_locator", S, False),
        ("source_revision", S, False), ("trace_member", S, True)),
    "response_metrics": schema(("response_id", S, False), ("metric", S, False), ("value", F, True),
        ("unit", S, False), ("policy", S, False)),
    "comparisons": schema(("comparison_id", S, False), ("experiment_id", S, False),
        ("benchmark_id", S, False), ("analysis_scope_id", S, False), ("model", S, False),
        ("reference_arm_id", S, False), ("target_arm_id", S, False),
        ("reference_subject_ids", L, False), ("target_subject_ids", L, False),
        ("n_tasks", I, False), ("matching_unit", S, False), ("repeat_pairing_assumed", B, False),
        ("origin", S, False)),
    "design_configurations": schema(("configuration_id", S, False), ("experiment_id", S, False),
        ("configuration_json", S, False), ("status", S, False), ("source_locator", S, False)),
    "design_relations": schema(("experiment_id", S, False), ("reference_configuration_id", S, False),
        ("target_configuration_id", S, False), ("changed_axis", S, False),
        ("reference_value_json", S, False), ("target_value_json", S, False), ("status", S, False)),
}
PRIMARY_KEYS = {
    "benchmarks": ["benchmark_id"], "experiments": ["experiment_id"],
    "experiment_benchmarks": ["experiment_id", "benchmark_id"], "items": ["item_id"],
    "subjects": ["subject_id"], "arms": ["arm_id"],
    "cohort_items": ["experiment_id", "benchmark_id", "item_id"], "responses": ["response_id"],
    "response_provenance": ["response_id"], "response_metrics": ["response_id", "metric"],
    "comparisons": ["comparison_id"], "design_configurations": ["configuration_id"],
    "design_relations": ["reference_configuration_id", "target_configuration_id"],
}
RESPONSE_KEY = ["experiment_id", "subject_id", "item_id", "trial", "test_condition", "interactors"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    def encode_scalar(scalar):
        if isinstance(scalar, (date, datetime)):
            return scalar.isoformat()
        raise TypeError(f"Unsupported JSON value: {type(scalar).__name__}")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=encode_scalar)


def identity(kind, value):
    return f"pilot_{kind}_" + hashlib.sha256((VERSION + ":" + kind + ":" + encode(value)).encode()).hexdigest()[:24]


def item_identity(benchmark_id, raw_item_id, protocol_ref):
    # Deliberately excludes study and experiment. Not canonical stimulus hashing.
    return identity("item", [benchmark_id, raw_item_id, protocol_ref])


def response_identity(row):
    # Grade is a measurement, not part of the identity of an attempt.
    return identity("response", [row[key] for key in RESPONSE_KEY])


def typed_table(name, rows):
    expected = set(SCHEMAS[name].names)
    for row in rows:
        require(set(row) == expected, f"{name}: columns differ from pilot schema")
    table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
    for field in table.schema:
        require(field.nullable or table[field.name].null_count == 0, f"{name}.{field.name}: unexpected null")
    return table


def validate(tables):
    require(set(tables) == set(SCHEMAS), "Missing or unexpected pilot tables")
    frames = {}
    for name, table in tables.items():
        require(table.schema.equals(SCHEMAS[name], check_metadata=True), f"{name}: schema mismatch")
        for field in table.schema:
            require(field.nullable or table[field.name].null_count == 0, f"{name}.{field.name}: unexpected null")
        frame = table.to_pandas()
        require(not frame.duplicated(PRIMARY_KEYS[name]).any(), f"{name}: duplicate primary key")
        frames[name] = frame
    benchmarks = set(frames["benchmarks"].benchmark_id)
    experiments = set(frames["experiments"].experiment_id)
    subjects = set(frames["subjects"].subject_id)
    items = frames["items"].set_index("item_id")
    arms = frames["arms"].set_index("arm_id")
    links = set(zip(frames["experiment_benchmarks"].experiment_id, frames["experiment_benchmarks"].benchmark_id))
    cohorts = set(frames["cohort_items"].itertuples(index=False, name=None))
    for name, frame in frames.items():
        if "benchmark_id" in frame:
            require(set(frame.benchmark_id) <= benchmarks, f"{name}: unknown benchmark")
        if "experiment_id" in frame:
            require(set(frame.experiment_id) <= experiments, f"{name}: unknown experiment")
        if "experiment_id" in frame and "benchmark_id" in frame:
            require(set(zip(frame.experiment_id, frame.benchmark_id)) <= links, f"{name}: unknown experiment/benchmark pair")
    for row in frames["cohort_items"].itertuples():
        require(row.item_id in items.index and items.loc[row.item_id, "benchmark_id"] == row.benchmark_id,
                "cohort_items: item ownership mismatch")
    responses = frames["responses"]
    require(not responses.duplicated(RESPONSE_KEY).any(), "responses: duplicate observation key")
    require(set(responses.subject_id) <= subjects, "responses: unknown subject")
    require(responses.trial.gt(0).all(), "responses: trial must be positive")
    require(responses.response.dropna().isin([0.0, 1.0]).all(), "responses: invalid binary grade")
    for row in tables["responses"].to_pylist():
        require((row["experiment_id"], row["benchmark_id"], row["item_id"]) in cohorts, "responses: unknown cohort item")
        require(row["response_id"] == response_identity(row), "responses: identity mismatch")
    observed_ids = set(responses.response_id)
    require(set(frames["response_provenance"].response_id) == observed_ids, "response provenance coverage mismatch")
    require(set(frames["response_metrics"].response_id) <= observed_ids, "metrics: unknown response")
    response_lookup = responses.set_index("response_id")
    for row in frames["response_provenance"].itertuples():
        require(row.arm_id in arms.index, "provenance: unknown arm")
        require(arms.loc[row.arm_id, "experiment_id"] == response_lookup.loc[row.response_id, "experiment_id"],
                "provenance: arm experiment mismatch")
    for row in frames["comparisons"].itertuples():
        matched_items = []
        for arm_id, ids in [(row.reference_arm_id, row.reference_subject_ids), (row.target_arm_id, row.target_subject_ids)]:
            require(arm_id in arms.index and arms.loc[arm_id, "experiment_id"] == row.experiment_id,
                    "comparison: arm experiment mismatch")
            require(len(ids) > 0 and set(ids) <= subjects, "comparison: invalid subjects")
            arm_response_ids = set(frames["response_provenance"].loc[frames["response_provenance"].arm_id.eq(arm_id), "response_id"])
            panel = responses[responses.experiment_id.eq(row.experiment_id) & responses.benchmark_id.eq(row.benchmark_id)
                              & responses.response_id.isin(arm_response_ids) & responses.subject_id.isin(ids)]
            require(set(panel.subject_id) == set(ids), "comparison: subjects missing from arm/scope")
            matched_items.append(set(panel.item_id))
        require(matched_items[0] == matched_items[1] and len(matched_items[0]) == row.n_tasks,
                "comparison: cohort mismatch")
    for row in frames["benchmarks"].itertuples():
        require(int(frames["items"].benchmark_id.eq(row.benchmark_id).sum()) == row.n_available_item_records,
                "benchmark item counts mismatch")
    for row in frames["experiments"].itertuples():
        panel = responses[responses.experiment_id.eq(row.experiment_id)]
        require((len(panel), int(panel.response.notna().sum()), panel.subject_id.nunique()) ==
                (row.n_responses, row.n_known_grades, row.n_observed_subjects), "experiment counts mismatch")
        require(not row.status.startswith("design_only") or panel.empty, "design-only experiment has fabricated responses")
    for row in frames["experiment_benchmarks"].itertuples():
        panel = responses[responses.experiment_id.eq(row.experiment_id) & responses.benchmark_id.eq(row.benchmark_id)]
        count = sum(e == row.experiment_id and b == row.benchmark_id for e, b, _ in cohorts)
        require((count, len(panel), int(panel.response.notna().sum()), panel.subject_id.nunique()) ==
                (row.n_cohort_items, row.n_responses, row.n_known_grades, row.n_observed_subjects), "cohort counts mismatch")
    configs = frames["design_configurations"].set_index("configuration_id")
    for row in frames["design_relations"].itertuples():
        require(row.reference_configuration_id in configs.index and row.target_configuration_id in configs.index,
                "design relation: unknown configuration")
        require(configs.loc[row.reference_configuration_id, "experiment_id"] == row.experiment_id ==
                configs.loc[row.target_configuration_id, "experiment_id"], "design relation: experiment mismatch")
    return {"status": "passed", "table_rows": {name: len(frame) for name, frame in frames.items()}}


def build_tables(output_root, docs, paths):
    mapping = json.loads((docs / "catalog-mapping.json").read_text())
    manifests = {name: json.loads((Path(__file__).resolve().parents[2] / "benchmarks" / name / "study.json").read_text())
                 for name in ["onlycodes", "tool_architecture", "agentarch"]}
    benchmarks = {row["benchmark_id"]: row for row in mapping["benchmarks"]}
    rows = {name: [] for name in SCHEMAS}
    item_rows, subject_rows = {}, {}
    archive = zipfile.ZipFile(paths["tool_archive"])
    try:
        for experiment in mapping["experiments"]:
            eid, study = experiment["experiment_id"], experiment["study_id"]
            cohort_ids = set()
            manifest = manifests[study]
            arm_definitions = {arm["arm_id"]: arm for arm in manifest["arms"]}
            for aid, arm in arm_definitions.items():
                rows["arms"].append(dict(arm_id=f"{eid}:{aid}", experiment_id=eid, source_arm_id=aid,
                    definition_json=encode(arm), definition_source=manifest["repository_url"] + "/tree/" + manifest["repository_revision"],
                    status="observed"))
            if study != "agentarch":
                data = pd.read_parquet(output_root / "studies" / study / "observations.parquet")
                require(set(data.experiment_id) == set(experiment["analysis_scopes"]), "Unexpected analysis scopes")
                source_config_subjects = {}
                for record in data.to_dict("records"):
                    scope, aid = record["experiment_id"], record["arm_id"]
                    bid = experiment["analysis_scopes"][scope]
                    benchmark = benchmarks[bid]
                    item_id = item_identity(bid, record["task_id"], benchmark["protocol_ref"])
                    cohort_ids.add(item_id)
                    if item_id not in item_rows:
                        content = None
                        if study == "tool_architecture":
                            content = json.loads(archive.read(record["trace_member"]))["task"]
                        item_rows[item_id] = dict(item_id=item_id, benchmark_id=bid, raw_item_id=record["task_id"],
                            content=content, source_reference_answer=None, protocol_ref=benchmark["protocol_ref"],
                            source_locator=record.get("trace_member", record["source_locator"]),
                            metadata_status="released_task_text_grader_audit_pending" if content else "source_reference_only")
                    version = record["harness_version"] if pd.notna(record["harness_version"]) else None
                    spec = {"model": record["model"], "harness": record["harness"], "harness_version": version,
                            "implementation_ref": manifest["repository_url"] + "/tree/" + manifest["repository_revision"],
                            "configuration_selector": aid,
                            "configuration_definition": {k: v for k, v in arm_definitions[aid].items()
                                                         if k not in {"arm_id", "source_labels", "relative_to"}}}
                    subject_id = identity("subject", spec)
                    subject_rows[subject_id] = dict(subject_id=subject_id, model=record["model"],
                        harness=record["harness"], harness_version=version, configuration_json=encode(spec),
                        metadata_status="available_description_full_configuration_audit_pending")
                    source_config_subjects[record["configuration_id"]] = subject_id
                    row = dict(experiment_id=eid, benchmark_id=bid, subject_id=subject_id, item_id=item_id,
                               trial=int(record["source_repeat"]) + (1 if study == "tool_architecture" else 0),
                               test_condition=None, interactors=None,
                               response=float(record["grade_observed"]) if pd.notna(record["grade_observed"]) else None)
                    row["response_id"] = response_identity(row)
                    rows["responses"].append(row)
                    rows["response_provenance"].append(dict(response_id=row["response_id"], arm_id=f"{eid}:{aid}",
                        source_repeat=str(record["source_repeat"]), source_run=str(record["run"]) if "run" in record else None,
                        source_locator=record["source_locator"], source_revision=manifest["repository_revision"],
                        trace_member=record.get("trace_member")))
                    metrics = {"success_all_attempts": ("binary", "Recorded grades; Tool-interface Table 2 counts unknown/unfinished as failure."),
                               "input_tokens": ("tokens", "source_reported"), "output_tokens": ("tokens", "source_reported")}
                    metrics.update({"cost_usd_adjusted": ("USD", "source_cache_adjusted_accounting")} if study == "onlycodes"
                                   else {"steps": ("steps", "source_reported")})
                    for metric, (unit, policy) in metrics.items():
                        rows["response_metrics"].append(dict(response_id=row["response_id"], metric=metric,
                            value=float(record[metric]) if pd.notna(record[metric]) else None, unit=unit, policy=policy))
                source_comparisons = pd.read_parquet(output_root / "studies" / study / "comparisons.parquet")
                for comp in source_comparisons.to_dict("records"):
                    scope = comp["experiment_id"]
                    rows["comparisons"].append(dict(comparison_id=comp["comparison_id"], experiment_id=eid,
                        benchmark_id=experiment["analysis_scopes"][scope], analysis_scope_id=scope, model=comp["model"],
                        reference_arm_id=f"{eid}:{comp['reference_arm_id']}", target_arm_id=f"{eid}:{comp['target_arm_id']}",
                        reference_subject_ids=sorted(source_config_subjects[c] for c in comp["reference_configuration_ids"]),
                        target_subject_ids=sorted(source_config_subjects[c] for c in comp["target_configuration_ids"]),
                        n_tasks=int(comp["n_tasks"]), matching_unit="item_id", repeat_pairing_assumed=False,
                        origin=comp["origin"]))
            else:
                for scope, bid in experiment["analysis_scopes"].items():
                    source_id = benchmarks[bid]["protocol_ref"]
                    source = yaml.safe_load(paths[source_id].read_text())
                    for index, task in enumerate(source["utterances_and_ground_truths"]):
                        item_id = item_identity(bid, str(task["id"]), source_id)
                        require(item_id not in item_rows, "Duplicate AgentArch task identifier")
                        item_rows[item_id] = dict(item_id=item_id, benchmark_id=bid, raw_item_id=str(task["id"]),
                            content=task["user_utterance"], source_reference_answer=encode(task["ground_truth"]),
                            protocol_ref=source_id, source_locator=f"{source_id}#/utterances_and_ground_truths/{index}",
                            metadata_status="released_case_full_prompt_and_grader_audit_pending")
                        cohort_ids.add(item_id)
                configs = []
                for config in manifest["configurations"]:
                    spec = {axis: config[axis] for axis in manifest["axes"]}
                    # The source calls this axis agent_style; the manifest uses the same name.
                    cid = identity("design", {"implementation": manifest["repository_revision"], **spec})
                    configs.append((cid, spec))
                    rows["design_configurations"].append(dict(configuration_id=cid, experiment_id=eid,
                        configuration_json=encode(spec), status="supported_design_not_observed_run",
                        source_locator="AgentArch:entrypoint.sh"))
                for (left_id, left), (right_id, right) in itertools.combinations(configs, 2):
                    changed = [key for key in left if left[key] != right[key]]
                    if len(changed) == 1:
                        axis = changed[0]
                        rows["design_relations"].append(dict(experiment_id=eid,
                            reference_configuration_id=left_id, target_configuration_id=right_id, changed_axis=axis,
                            reference_value_json=encode(left[axis]), target_value_json=encode(right[axis]),
                            status="analyst_enumerated_single_axis_change_not_measured_effect"))
            for scope, bid in experiment["analysis_scopes"].items():
                cohort = [item for item in item_rows.values() if item["benchmark_id"] == bid and item["item_id"] in cohort_ids]
                rows["cohort_items"].extend(dict(experiment_id=eid, benchmark_id=bid, item_id=item["item_id"]) for item in cohort)
                panel = [row for row in rows["responses"] if row["experiment_id"] == eid and row["benchmark_id"] == bid]
                rows["experiment_benchmarks"].append(dict(experiment_id=eid, benchmark_id=bid, analysis_scope_id=scope,
                    n_cohort_items=len(cohort), n_responses=len(panel), n_known_grades=sum(row["response"] is not None for row in panel),
                    n_observed_subjects=len({row["subject_id"] for row in panel})))
            panel = [row for row in rows["responses"] if row["experiment_id"] == eid]
            rows["experiments"].append(dict(experiment_id=eid, study_id=study, name=experiment["name"], status=experiment["status"],
                paper_url=manifest["paper_url"], repository_url=manifest["repository_url"], source_revision=manifest["repository_revision"],
                evaluation_date=None, n_responses=len(panel), n_known_grades=sum(row["response"] is not None for row in panel),
                n_observed_subjects=len({row["subject_id"] for row in panel})))
    finally:
        archive.close()
    rows["items"] = list(item_rows.values())
    rows["subjects"] = list(subject_rows.values())
    for benchmark in benchmarks.values():
        rows["benchmarks"].append({**benchmark, "version": None,
            "n_available_item_records": sum(item["benchmark_id"] == benchmark["benchmark_id"] for item in rows["items"]),
            "identity_status": mapping["identity_status"]})
    return {name: typed_table(name, sorted(data, key=lambda row: encode([row[key] for key in PRIMARY_KEYS[name]])))
            for name, data in rows.items()}


def export_study_packages(tables, package_root, catalog_dir, study=None):
    """Partition validated tables by source study, preserving all logical IDs.

    Catalogs stay shared. A package may contain several benchmarks and experiments.
    Its generated manifest points to the catalog rather than copying metadata rows.
    """
    rows = {name: table.to_pylist() for name, table in tables.items()}
    study_ids = sorted({row["study_id"] for row in rows["experiments"]})
    require(study is None or study in study_ids, f"Unknown study package: {study}")
    summaries = {}
    for sid in study_ids:
        if study is not None and sid != study:
            continue
        eids = {row["experiment_id"] for row in rows["experiments"] if row["study_id"] == sid}
        selected = {name: [row for row in data if row["experiment_id"] in eids]
                    for name, data in rows.items()
                    if "experiment_id" in SCHEMAS[name].names and name != "experiments"}
        item_ids = {row["item_id"] for row in selected["cohort_items"]}
        subject_ids = {row["subject_id"] for row in selected["responses"]}
        response_ids = {row["response_id"] for row in selected["responses"]}
        selected["items"] = [row for row in rows["items"] if row["item_id"] in item_ids]
        selected["subjects"] = [row for row in rows["subjects"] if row["subject_id"] in subject_ids]
        for name in ("response_provenance", "response_metrics"):
            selected[name] = [row for row in rows[name] if row["response_id"] in response_ids]
        # Membership is part of the shared catalog, alongside benchmark/experiment definitions.
        del selected["experiment_benchmarks"]
        out = package_root / sid / "formatted_tables"
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for name, data in selected.items():
            path = out / f"{name}.parquet"
            if not data:
                # Remove only this exporter's known generated tables on subsequent builds.
                path.unlink(missing_ok=True)
                continue
            table = typed_table(name, data)
            pq.write_table(table, path)
            require(pq.read_table(path).equals(table, check_metadata=True), f"Round-trip mismatch: {path}")
            files.append(dict(path=path.name, table=name, rows=len(data),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = dict(schema_version=VERSION, study_id=sid, experiment_ids=sorted(eids),
                        benchmark_ids=sorted({row["benchmark_id"] for row in selected["cohort_items"]}),
                        catalog_dir=os.path.relpath(catalog_dir, out), files=files)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summaries[sid] = {"path": str(out), "table_rows": {entry["table"]: entry["rows"] for entry in files}}
    return summaries


def export(output_root, docs, paths, package_root=None, package_study=None):
    tables = build_tables(output_root, docs, paths)
    verification = validate(tables)
    release = output_root / "release_v2"
    release.mkdir(parents=True, exist_ok=True)
    files = []
    def write(path, name, table):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        require(pq.read_table(path).equals(table, check_metadata=True), f"Round-trip mismatch: {path}")
        files.append({"path": str(path.relative_to(release)), "table": name, "rows": table.num_rows,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    catalogs = ["benchmarks", "experiments", "experiment_benchmarks", "arms", "comparisons", "design_configurations", "design_relations"]
    for name in catalogs:
        write(release / "catalog" / f"{name}.parquet", name, tables[name])
    for benchmark in tables["benchmarks"].to_pylist():
        bid = benchmark["benchmark_id"]
        root = release / "benchmarks" / bid
        items = [row for row in tables["items"].to_pylist() if row["benchmark_id"] == bid]
        write(root / "formatted_tables/items.parquet", "items", typed_table("items", items))
        for link in tables["experiment_benchmarks"].to_pylist():
            if link["benchmark_id"] != bid:
                continue
            eid = link["experiment_id"]
            partition = root / "experiments" / quote(eid, safe="") / "formatted_tables"
            cohort = [row for row in tables["cohort_items"].to_pylist() if row["experiment_id"] == eid and row["benchmark_id"] == bid]
            write(partition / "cohort_items.parquet", "cohort_items", typed_table("cohort_items", cohort))
            responses = [row for row in tables["responses"].to_pylist() if row["experiment_id"] == eid and row["benchmark_id"] == bid]
            if not responses:
                continue  # Unavailable original results do not become null or zero observations.
            response_ids = {row["response_id"] for row in responses}
            subject_ids = {row["subject_id"] for row in responses}
            write(partition / "responses.parquet", "responses", typed_table("responses", responses))
            write(partition / "subjects.parquet", "subjects", typed_table("subjects", [row for row in tables["subjects"].to_pylist() if row["subject_id"] in subject_ids]))
            for name in ["response_provenance", "response_metrics"]:
                write(partition / f"{name}.parquet", name, typed_table(name, [row for row in tables[name].to_pylist() if row["response_id"] in response_ids]))
    verification.update(schema_version=VERSION, scope="Prototype schema validated; full canonical stimulus/configuration/grader audit pending.",
                        unknown_grades=tables["responses"]["response"].null_count, files=files)
    (release / "schema.json").write_text(json.dumps({name: [{"name": f.name, "type": str(f.type), "nullable": f.nullable}
        for f in s] for name, s in SCHEMAS.items()}, indent=2) + "\n")
    (release / "sources.lock.json").write_text((docs / "sources.lock.json").read_text())
    (release / "catalog-mapping.json").write_text((docs / "catalog-mapping.json").read_text())
    verification["study_packages"] = export_study_packages(tables, package_root, release / "catalog", package_study) if package_root is not None else {}
    (release / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification
