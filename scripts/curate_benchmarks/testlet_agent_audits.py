"""Native run and checkpoint reconciliation for agent benchmarks."""

from collections import defaultdict
import hashlib
import json
import re


def source_attempts(directory, metadata):
    if directory.name == 'multi_swebench':
        yield from _multi_swe(directory)
    elif directory.name == 'real_webagents':
        index = 0
        for model in json.loads((directory / 'raw/model_data.json').read_text())['aiModels']:
            if not model.get('name'):
                continue
            for site in model.get('websites') or []:
                for row in site.get('tasks') or [None]:
                    run = str(index)
                    index += 1
                    if row is None or row.get('id') is None or row.get('accuracy') is None:
                        continue
                    task = str(row['id'])
                    passed, failed = set(row.get('evalsPassed') or []), set(row.get('evalsFailed') or [])
                    aggregate = float(not failed) if row.get('evalsFailed') is not None else float(float(row['accuracy']) >= 100)
                    checks = [(task + '::' + name, task, None if name in passed and name in failed else float(name in passed))
                        for name in sorted(passed | failed)]
                    if not checks:
                        checks = [(task, task, aggregate)]
                    yield run, aggregate, checks
    elif directory.name == 'chi_bench':
        from .native_result_audits import _chi_source_records
        native = _chi_source_records(directory, metadata)['native']
        for run, entry in native.items():
            task = entry['task']
            scorecards = []
            for name in entry['artifacts']:
                if name.endswith('/verifier/scorecard.json'):
                    scorecards.append(json.loads((directory / 'raw' / name).read_text()))
            if scorecards:
                assert all(card == scorecards[0] for card in scorecards), 'Conflicting source scorecards'
                checks = [(task + '::' + name, task, grade) for name, grade in scorecards[0]['check_scores'].items()]
            else:
                checks = []
            aggregate = ((entry['record'].get('verifier_result') or {}).get('rewards') or {}).get('reward')
            if not checks:
                checks = [(task, task, aggregate)]
            yield run, aggregate, checks
    elif directory.name == 'scbench_long':
        layout = metadata['build']['parameters']['layout']
        raw = directory / 'raw'
        tasks = {json.loads(path.read_text())['id'] for path in (raw / layout['tasks']).glob('*/eval.json')}
        for row in json.loads((raw / layout['index']).read_text())['runResults']:
            task = row['task']
            if task not in tasks:
                continue
            checks = []
            for check in row.get('graderChecks') or []:
                identity = hashlib.sha256(json.dumps([check['kind'], check['name']]).encode()).hexdigest()
                assert type(check['passed']) is bool
                checks.append((task + '::' + identity, task, float(check['passed'])))
            if not checks:
                checks = [(task, task, float(row['passed']))]
            yield json.dumps([row['modelId'], task, row['trialIndex']]), float(row['passed']), checks
    elif directory.name == 'cybench':
        from .native_result_audits import _cybench_source_records
        expected, native, _, _ = _cybench_source_records(directory, metadata)
        for (model, task, mode), entry in expected.items():
            if mode == 'subtask_fractional':
                continue
            parent = task + ':' + mode
            checks = [(parent + '::flag_recovery', task, entry['grade'])]
            if mode == 'subtask_guided':
                record = native[entry['source_file']]
                checks.extend((parent + '::' + name, task, check['score'])
                    for name, check in record['subtask_completions'].items())
            yield '/'.join((model, task, mode)), entry['grade'], checks
    elif directory.name == 'swebench_live':
        yield from _swe_live(directory, metadata)
    else:
        raise ValueError(directory.name)


def _swe_live(directory, metadata):
    from .native_result_audits import _swe_live_sources
    native = _swe_live_sources(directory, metadata)['native']
    parameters = metadata['build']['parameters']
    release = directory / 'raw' / parameters['layout']['release']
    runs = sorted({run for run, _ in native}, key=len, reverse=True)
    outcomes = defaultdict(lambda: defaultdict(set))
    for path in sorted(release.rglob('*.json')):
        if not (path.name == 'report.json' or path.name.endswith(('.report.json', '.results.json'))):
            continue
        relative = str(path.relative_to(release))
        run = next((r for r in runs if relative.startswith(r + '/')), None)
        if run is None:
            continue
        remaining = relative[len(run) + 1:]
        match = re.search(r'(?:^|/)([^/]+?__[^/]+?-\d+)(?=/|\.|$)', remaining)
        if not match or (run, match[1]) not in native:
            continue
        task = match[1]
        payload = json.loads(path.read_text())
        record = payload.get(task, payload)
        assert record.get('instance_id', task) == task
        for group, statuses in record.get('tests_status', record).items():
            if group not in {'FAIL_TO_PASS', 'PASS_TO_PASS', 'FAIL_TO_FAIL', 'PASS_TO_FAIL'}:
                continue
            assert set(statuses) <= {'success', 'failure'}
            for verdict, names in statuses.items():
                assert isinstance(names, list)
                for name in names:
                    outcomes[run, task][group, name].add(float(verdict == 'success'))
    for (run, task), entry in native.items():
        testlet = entry['track'] + '/' + task
        checks = []
        for (group, name), values in outcomes[run, task].items():
            component = hashlib.sha256(json.dumps([group, name]).encode()).hexdigest()
            checks.append((testlet + '::' + component, testlet, next(iter(values)) if len(values) == 1 else None))
        if not checks:
            checks = [(testlet, testlet, entry['grade'])]
        yield run + '::' + task, entry['grade'], checks


def _multi_swe(directory):
    raw = directory / 'raw'
    for path in sorted((raw / 'results').glob('*.json')):
        run = re.sub(r'_x([0-9a-f]{2})_', lambda m: chr(int(m[1], 16)), path.stem)
        record = json.loads(path.read_text())
        predictions = raw / 'preds' / (path.stem + '.jsonl')
        patches = {}
        if predictions.is_file():
            text = predictions.read_text()
            if not text.startswith('version https://git-lfs.github.com/spec/v1'):
                for line in text.split('\n'):
                    if line.strip():
                        prediction = json.loads(line)
                        if prediction.get('instance_id') is not None:
                            patches[prediction['instance_id']] = prediction.get('model_patch')
        for grade, ids in ((1., record.get('resolved') or record.get('resolved_ids') or []),
                           (0., record.get('unresolved_ids') or record.get('unresolved') or [])):
            for identifier in ids:
                task = re.sub(r'^([^/]+)/([^:]+):pr-(\d+)$', r'\1__\2-\3', identifier)
                folder = raw / 'checks' / path.stem / task
                checks = []
                if (folder / 'report.json').is_file():
                    report = json.loads((folder / 'report.json').read_text())
                    assert (folder / 'fix.patch').read_text() == patches[task], 'Report did not evaluate the submitted patch'
                    assert task == f"{report['org']}__{report['repo']}-{report['number']}"
                    outcomes = defaultdict(set)
                    for field, value in (('passed_tests', 1.), ('failed_tests', 0.), ('skipped_tests', None)):
                        for name in (report.get('fix_patch_result') or {}).get(field, []):
                            outcomes[name].add(value)
                    checks = [(task + '::' + name, task, next(iter(values)) if len(values) == 1 else None)
                        for name, values in outcomes.items()]
                if not checks:
                    checks = [(task, task, grade)]
                yield run + '::' + task, grade, checks
