"""Reconcile component observations against native releases, independently of builders."""

from collections import Counter
from importlib import import_module
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml


SUPPORTED = {"reasoning_gym", "lawbench", "appworld", "openbiorq", "theagentcompany", "csedb", "healthadminbench", "researchclawbench", "tau2_bench", "dpai", "taubench", "corebench", "swepolybench", "atmossci_bench", "llmail_inject", "mmdocrag", "infibench", "weavebench", "afrieval", "care_enzymes", "arcagi"}
SUPPORTED.update({'cybench', 'swebench_live', 'chi_bench', 'scbench_long', 'real_webagents', 'risebench', 'scivisagentbench'})
SUPPORTED.update({'bigfinancebench', 'multi_swebench', 'proteininvbench'})
SUPPORTED.update({'wikihow_agent', 'goodai_ltm_benchmark'})
SUPPORTED.update({'bountybench', 'aider', 'statqa', 'das_med_hallucination', 'classroom_ai', 'imgedit', 'complexbench'})
SUPPORTED.add('engdesign')
SUPPORTED.add('pku_saferlhf')
SUPPORTED.add('autoresearchbench')
SUPPORTED.add('mathconstraint')
SUPPORTED.add('phyblock')
SUPPORTED.add('decodingtrust')
SUPPORTED.add('embodied_agent_interface')
SUPPORTED.add('scigym')
SUPPORTED.add('agc_bench')
SUPPORTED.add('qatch')
SUPPORTED.add('llm4ir')
SUPPORTED.add('nis3d')


def source_attempts(directory, metadata):
    """Yield (native run ID, original task grade, [(raw item, testlet, grade), ...])."""
    raw, slug = directory / "raw", directory.name
    parameters = metadata["build"].get("parameters", {})
    if slug == 'goodai_ltm_benchmark':
        from .testlet_memory_audit import source_attempts as memory_sources
        yield from memory_sources(directory, metadata)
    elif slug == 'wikihow_agent':
        yield from _wikihow(directory, metadata)
    elif slug in {'risebench', 'scivisagentbench'}:
        from .testlet_visual_audits import source_attempts as visual_sources
        yield from visual_sources(directory, metadata)
    elif slug in {'cybench', 'swebench_live', 'chi_bench', 'scbench_long', 'real_webagents', 'multi_swebench'}:
        from .testlet_agent_audits import source_attempts as agent_sources
        yield from agent_sources(directory, metadata)
    elif slug in {'afrieval', 'care_enzymes', 'arcagi', 'proteininvbench'}:
        from .testlet_label_audits import source_attempts as label_sources
        yield from label_sources(directory, metadata)
    elif slug == 'infibench':
        from .testlet_infi_audits import source_attempts as infi_sources
        yield from infi_sources(directory, metadata)
    elif slug == 'weavebench':
        import tarfile
        paths = parameters['paths']
        with tarfile.open(raw / paths['archive']) as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(paths['root'] + '/' + paths['records']) or not member.name.endswith('.json'):
                    continue
                record = json.load(archive.extractfile(member))
                task = record['task_id']
                score = record['score']
                aggregate = None if score is None else float(score >= metadata['grading']['verifiers']['published_judge']['pass_threshold'])
                checks = [(task + '::' + name, task, detail['score']) for name, detail in record['dimensions'].items()]
                if not checks and aggregate is not None:
                    checks = [(task, task, aggregate)]
                yield task, aggregate, checks
    elif slug in {'atmossci_bench', 'llmail_inject', 'mmdocrag'}:
        from .testlet_checklist_audits import source_attempts as checklist_sources
        yield from checklist_sources(directory, metadata)
    elif slug == 'swepolybench':
        import re
        release = raw / parameters['layout']['release']
        attempts, results = set(), {}
        for path in release.glob('evaluation/*/*/all_preds.jsonl'):
            run = str(path.parent.relative_to(release))
            attempts.update((run, json.loads(line)['instance_id']) for line in path.read_text().splitlines() if line.strip())
        for path in release.glob('evaluation/*/*/trajs/**/*'):
            if path.is_file():
                source = str(path.relative_to(raw))
                instance = re.search(parameters['patterns']['instance_id'], source).group(1)
                attempts.add((str(path.relative_to(release)).split('/trajs/', 1)[0], instance))
        for path in release.glob('evaluation/*/*/logs/*_result.json'):
            record = json.loads(path.read_text())
            key = str(path.parent.parent.relative_to(release)), record['instance_id']
            assert key not in results
            results[key] = record
            if any(record[field] for field in ('generation', 'patch_applied', 'with_logs')):
                attempts.add(key)
        for run, task in sorted(attempts):
            record = results.get((run, task)) or {}
            passed, failed = set(record.get('passed_tests', [])), set(record.get('failed_tests', []))
            parent = run.split('/')[1] + '/' + task
            checks = [(parent + '::' + hashlib.sha256(name.encode()).hexdigest(), task,
                None if name in passed and name in failed else float(name in passed)) for name in sorted(passed | failed)]
            score = float(record['resolved']) if record else None
            if not checks:
                checks = [(parent, task, score)]
            yield run + '::' + task, score, checks
    elif slug in {"taubench", "corebench"}:
        from .testlet_replay_audits import source_attempts as replay_sources
        yield from replay_sources(directory, metadata)
    elif slug == "openbiorq":
        protocol = metadata["grading"]["verifiers"]["checklist"]
        for model in parameters["models"]:
            folder = raw / parameters["layout"]["results"] / model
            for line in (folder / "checklist.jsonl").read_text().splitlines():
                record = json.loads(line)
                task, score = record["task_id"], record.get("checklist_score")
                aggregate = None if score is None else float(score >= protocol["success_threshold"])
                components = [(task + "::" + str(v["id"]), task, protocol["verdict_values"][v["v"]])
                    for v in record.get("verdicts") or []]
                yield model + "::" + task, aggregate, components
    elif slug == "theagentcompany":
        root = raw / parameters["layout"]["runs"]
        for model in parameters["models"]:
            scored = set()
            for path in sorted((root / model / "results").glob("eval_*.json")):
                task = path.stem.removeprefix("eval_").removesuffix("-image")
                scored.add(task)
                record = json.loads(path.read_text())
                components = [(task + f"::checkpoint:{i}:out-of:{c['total']}", "1.0.0/" + task,
                    c["result"] / c["total"]) for i, c in enumerate(record["checkpoints"], 1) if c["total"]]
                score = record["final_score"]
                yield str(path.relative_to(root)), score["result"] / score["total"], components
            for path in sorted((root / model / "trajectories").glob("traj_*")):
                task = path.name.removeprefix("traj_").removesuffix(".gz").removesuffix(".json").removesuffix(".txt").removesuffix("-image")
                if task not in scored:
                    scored.add(task)
                    yield str(path.relative_to(root)), None, [(task + '::ungraded_task_outcome', '1.0.0/' + task, None)]
    elif slug == "appworld":
        from .appworld_bundle import open_bundle
        for path in sorted((raw / "bundles").glob("*.bundle")):
            with open_bundle(path) as archive:
                split = json.loads(archive.read(path.stem + "/metadata.json"))["dataset"]
                records = json.loads(archive.read(f"{path.stem}/evaluations/{split}.json"))["individual"]
                for task, record in records.items():
                    checks = [(task + "::" + hashlib.sha256(c["requirement"].encode()).hexdigest(), task, float(passed))
                        for passed, key in ((True, "passes"), (False, "failures")) for c in record[key]]
                    assert len(checks) == record["num_tests"], "Incomplete native AppWorld checks"
                    assert bool(record["success"]) == all(c[2] for c in checks), "AppWorld native aggregation mismatch"
                    yield path.name + "/" + task, float(record["success"]), checks
    elif slug == "csedb":
        from .native_result_audits import _csedb_source_records
        _, records = _csedb_source_records(directory, metadata)
        for key, entry in records.items():
            task, record = entry["case_id"], entry["record"]
            checks = []
            if entry["status"] == "released_grade":
                if record.get("规则判断列表"):
                    model = key[-1]
                    field = parameters["judgment_fields"][model]
                    result_key = parameters["judgment_keys"][model]
                    text = record[field].strip().removeprefix("```json").removesuffix("```")
                    verdicts = json.loads(text)[result_key]
                    checks = [(task + f"::criterion:{i}", task, float(v == "yes")) for i, v in enumerate(verdicts, 1)]
                else:
                    checks = [(task, task, entry["grade"])]
            elif record.get('规则判断列表'):
                checks = [(task + f'::criterion:{i}', task, None) for i in range(1, len(record['规则判断列表']) + 1)]
            else:
                checks = [(task, task, None)]
            yield ":".join(map(str, key)), entry["grade"], checks
    elif slug == "healthadminbench":
        from .native_result_audits import _healthadmin_source_records
        records, _, _, website, _ = _healthadmin_source_records(directory, metadata)
        for run, entry in records.items():
            record, path = entry["record"], entry["path"]
            task = record["domain"] + "/" + record["task_id"]
            exported = website[run]["trajectory_json"]
            if exported:
                verdicts = json.loads(exported)["evaluation_result"]["eval_results"]
                checks = [(task + "::" + str(i), path, float(v["success"])) for i, v in enumerate(verdicts)]
            else:
                checks = [(task, path, float(record[field])) for field in ("pass_orig", "pass_new")]
            yield run + ":pass_orig", float(record["pass_orig"]), checks
    elif slug == "researchclawbench":
        from .native_result_audits import _researchclawbench_sources
        source = _researchclawbench_sources(directory, metadata)
        for run, record in source["index"].items():
            task = record["task_id"]
            if run in source["details"]:
                criteria = source["details"][run]["native_detail"]["score"]["items"]
                checks = [(task + "::" + str(c["index"]), task, c["score"]) for c in criteria]
            else:
                checks = [(task, task, record["total_score"])]
            yield run, record["total_score"], checks
    elif slug == "dpai":
        for path in sorted(raw.rglob("*.json")):
            filename = str(path.relative_to(raw))
            if not filename.startswith(parameters["reports"]["prefix"]):
                continue
            for task, phases in json.loads(path.read_text())["results"].items():
                for phase, record in phases.items():
                    checks = []
                    for stage in ("all_tests_validation", "all_tests_after_prediction"):
                        result = ((record["evaluations"].get(stage) or {}).get("data") or {}).get("test_result") or {}
                        for name, test in (result.get("tests") or {}).items():
                            assert test["status"] in {"success", "failed", "skipped"}
                            grade = None if test["status"] == "skipped" else float(test["status"] == "success")
                            checks.append((task + ":" + phase + "::" + stage + ":" + name, task, grade))
                    score = record["score"]["normalized_score"]
                    if not checks:
                        checks = [(task + ":" + phase, task, score)]
                    yield filename + ":" + task + ":" + phase, score, checks
    elif slug == "tau2_bench":
        fields = {"action_checks": ("ACTION", "action", "actions", "action_match"),
            "env_assertions": ("ENV_ASSERTION", "env_assertion", "env_assertions", "met"),
            "nl_assertions": ("NL_ASSERTION", "nl_assertion", "nl_assertions", "met"),
            "communicate_checks": ("COMMUNICATE", "info", "communicate_info", "met")}
        for pattern in ("repository_runs", "submitted_runs"):
            for path in sorted(raw.glob(parameters["layout"][pattern])):
                if path.name == "submission.json":
                    continue
                run = json.loads(path.read_text())
                tasks = {str(t["id"]): t for t in run["tasks"]}
                for simulation in run["simulations"]:
                    task_id = str(simulation["task_id"])
                    task = run["info"]["environment_info"]["domain_name"] + "/" + task_id
                    criteria = tasks[task_id]["evaluation_criteria"]
                    reward = simulation.get("reward_info") or {}
                    evidence = []
                    if reward.get("db_check") is not None:
                        evidence.append(("DB", dict(rule="Released database-match verdict; individual field comparisons are unavailable"),
                            None, reward["db_check"]["db_match"]))
                    for field, (kind, identity, task_field, verdict) in fields.items():
                        for check in reward.get(field) or []:
                            evidence.append((kind, check[identity], check[identity] in (criteria.get(task_field) or []), check[verdict]))
                    checks = []
                    for kind, definition, matches, verdict in evidence:
                        assert type(verdict) is bool
                        rule = dict(checker=kind, criterion=definition,
                            contributes_to_task_reward=kind in (reward.get("reward_basis") or []),
                            matches_embedded_task_definition=matches)
                        component = kind + ":" + hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
                        checks.append((task + "::" + component, task, float(verdict)))
                    if not checks and reward.get("reward") is not None:
                        checks = [(task, task, reward["reward"])]
                    elif not checks:
                        checks = [(task + '::ungraded_task_outcome', task, None)]
                    yield simulation["id"], reward.get("reward"), checks
    else:
        raise ValueError("No component source audit for " + slug)


def _wikihow(directory, metadata):
    rubrics = json.loads((directory / 'raw/protocol/data/evaluation_rubrics.json').read_text())
    assert set(rubrics) == {'Clarity', 'Truthfulness', 'Engagement', 'Coherence', 'Depth', 'Relevance', 'Progress', 'Naturalness'}
    for path in sorted((directory / 'raw/conversations').glob('T-*_corrected.json')):
        source = str(path.relative_to(directory / 'raw'))
        for index, row in enumerate(json.loads(path.read_text())['total_conversations']):
            task = str(row['doc_id']) + ':' + str(row['method_id'])
            score = row['evaluation']['Completion Achieved']
            assert score in (0, 1)
            checks = [(task + '::Completion Achieved', task, float(score))]
            for name in rubrics:
                grade = row['evaluation'].get(name)
                if type(grade) not in (int, float) or not 1 <= grade <= 5:
                    grade = None
                checks.append((task + '::' + name, task, grade))
            yield source + '#' + str(index), float(score), checks


DIRECT_AUDITS = {
    'nis3d': 'testlet_nis3d_audit',
    'agc_bench': 'testlet_agc_audit',
    'qatch': 'testlet_qatch_audit',
    'llm4ir': 'testlet_llm4ir_audit',
    'aider': 'testlet_coding_audit',
    'autoresearchbench': 'testlet_autoresearch_audit',
    'bountybench': 'testlet_bounty_audit',
    'classroom_ai': 'testlet_classroom_audit',
    'complexbench': 'testlet_complex_audit',
    'das_med_hallucination': 'testlet_das_audit',
    'decodingtrust': 'testlet_decodingtrust_audit',
    'embodied_agent_interface': 'testlet_embodied_audit',
    'engdesign': 'testlet_engdesign_audit',
    'imgedit': 'testlet_imgedit_audit',
    'lawbench': 'testlet_lawbench_audit',
    'mathconstraint': 'testlet_mathconstraint_audit',
    'openbiorq': 'testlet_openbio_audit',
    'phyblock': 'testlet_phyblock_audit',
    'pku_saferlhf': 'testlet_pku_audit',
    'reasoning_gym': 'testlet_reasoning_audit',
    'scigym': 'testlet_scigym_audit',
    'statqa': 'testlet_statqa_audit',
}

COMPLETE_AUDITS = {
    'appworld': 'testlet_appworld_audit',
    'arcagi': 'testlet_task_audits',
    'atmossci_bench': 'testlet_atmossci_audit',
    'care_enzymes': 'testlet_task_audits',
    'chi_bench': 'testlet_chi_audit',
    'corebench': 'testlet_core_audit',
    'csedb': 'testlet_csedb_audit',
    'cybench': 'testlet_cybench_audit',
    'dpai': 'testlet_dpai_audit',
    'healthadminbench': 'testlet_healthadmin_audit',
    'llmail_inject': 'testlet_llmail_audit',
    'mmdocrag': 'testlet_mmdocrag_audit',
    'multi_swebench': 'testlet_multiswe_audit',
    'real_webagents': 'testlet_real_audit',
    'researchclawbench': 'testlet_researchclaw_audit',
    'risebench': 'testlet_rise_audit',
    'scbench_long': 'testlet_scbench_audit',
    'scivisagentbench': 'testlet_scivis_audit',
    'swebench_live': 'testlet_swe_audit',
    'swepolybench': 'testlet_swe_audit',
    'tau2_bench': 'testlet_tau2_audit',
    'taubench': 'testlet_tau_audit',
    'theagentcompany': 'testlet_company_audit',
}

def verify_component_results(directory, tables_directory=None, *, tables=None):
    directory = Path(directory)
    root = Path(tables_directory) if tables_directory else directory / "formatted_tables"
    metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
    def full_audit(module):
        complete = tables if tables is not None else {
            path.stem: pd.read_parquet(path) for path in root.glob('*.parquet')}
        return import_module('.' + module, __package__).verify(directory, complete, metadata)

    if directory.name in DIRECT_AUDITS:
        return full_audit(DIRECT_AUDITS[directory.name])
    if directory.name == 'bigfinancebench':
        from .testlet_finance_audit import verify
        return verify(directory, root, metadata)
    expected, grades = Counter(), {}
    for run, aggregate, checks in source_attempts(directory, metadata):
        assert run not in grades, "Repeated native execution ID"
        grades[run] = aggregate
        expected.update((run, item, testlet, None if grade is None else float(grade)) for item, testlet, grade in checks)
    def read(name, columns=None):
        if tables is not None:
            return tables[name][columns] if columns else tables[name]
        return pd.read_parquet(root / (name + '.parquet'), columns=columns)

    attempts = read('attempts', ["attempt_id", "raw_attempt_id", "subject_id"])
    assert attempts.raw_attempt_id.is_unique, "Repeated imported execution ID"
    assert set(attempts.raw_attempt_id) == set(grades), "Missing or extra source executions"
    responses = read('responses')
    assert responses.attempt_id.notna().all(), "A component lost its execution link"
    assert set(responses.attempt_id) == set(attempts.attempt_id), "An execution lost its item observation"
    cells = responses.merge(attempts, on="attempt_id", validate="many_to_one", suffixes=("", "_attempt"))
    assert cells.subject_id.eq(cells.subject_id_attempt).all(), "Component attributed to a different subject"
    items = read('items', ["item_id", "raw_item_id", "testlet_id"])
    cells = cells.merge(items, on="item_id", validate="many_to_one")
    actual = Counter((r.raw_attempt_id, r.raw_item_id, r.testlet_id, None if pd.isna(r.response) else float(r.response))
        for r in cells.itertuples())
    assert actual == expected, f"Component mismatch: {sum((expected-actual).values())} missing, {sum((actual-expected).values())} extra"
    if tables is not None:
        batches = [tables['attempts'][['raw_attempt_id', 'trace']].to_dict('records')]
    else:
        batches = (batch.to_pylist() for batch in pq.ParquetFile(root / 'attempts.parquet').iter_batches(
            batch_size=32, columns=['raw_attempt_id', 'trace']))
    for batch in batches:
        for row in batch:
            trace = json.loads(row["trace"])
            assert trace["recorded_task_score"] == grades[row["raw_attempt_id"]], "Original task score was lost or changed"
            assert isinstance(trace["source_trace"], str), "Missing original execution evidence"
    module = COMPLETE_AUDITS.get(directory.name)
    details = full_audit(module) if module else {}
    return dict(details, source_attempts=len(grades), source_component_observations=sum(expected.values()),
        imported_items=len(items), source_observed_testlets=len({key[2] for key in expected}))
