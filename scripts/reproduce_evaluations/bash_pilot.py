"""Bookkeeping for visible Bash procedures; does not download or run benchmarks.

Records immutable input hashes, the exact procedure, frozen random selections,
and a common result. The credential wrapper supplies secrets only to inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from bounded_luna import now, save
from select_pilot_tasks import select_tasks


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    records = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            key = str(path.relative_to(root))
            if path.is_symlink():
                records[key] = {'link': os.readlink(path)}
            elif path.is_file():
                records[key] = {'size': path.stat().st_size, 'sha256': digest(path)}
    return records


def unchanged(before, after, *, allow_additions=False):
    changes = [p for p, value in before.items() if after.get(p) != value]
    if not allow_additions:
        changes += sorted(after.keys() - before.keys())
    if changes:
        raise ValueError('Raw inputs changed: ' + ', '.join(changes[:10]))


def selection(bench):
    value = json.loads((bench / 'reproduction_checks/selection.json').read_text())
    expected = select_tasks(bench.name, value['eligible_tasks'], seed=value['seed'],
                            scope=value['scope'], strata=value['design'].startswith('Three random strata'))
    if value != expected:
        raise ValueError('Selection is not the recorded random sample')
    return [row['task_id'] for row in value['selected_tasks']]


def begin(bench, output):
    selection(bench)
    procedure = output / 'procedure'
    procedure.mkdir()
    files = [bench / 'metadata.yaml', bench / 'reproducibility.yaml']
    for directory, dirs, names in os.walk(bench / 'reproduction_checks'):
        dirs[:] = [d for d in dirs if d not in ('runs', 'run_results', '__pycache__', '.cache')]
        files.extend(Path(directory) / name for name in names
                     if Path(name).suffix in ('.sh', '.py', '.json', '.yaml', '.patch', '.txt', '.sha256')
                     or name == 'Dockerfile')
    for path in files:
        target = procedure / path.relative_to(bench)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shared = Path(__file__).parent
    for name in ('bash_pilot.py', 'procedure.sh', 'bounded_luna.py',
                 'pilot_runtime.py', 'select_pilot_tasks.py', 'native_outputs.py', 'inspect_budget.py', 'chat_budget.py', 'capture_receipt.py'):
        target = procedure / 'shared' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(shared / name, target)
    save(output / 'raw.before.json', inventory(bench / 'raw'))
    save(output / 'result.json', dict(format_version=1, benchmark=bench.name,
         started_at=now(), status='running', tasks=selection(bench), stages={},
         procedure_sha256=inventory(procedure), raw_unchanged=None,
         scope='Three random tasks; execution and grading pilot, not a benchmark score.'))


def finish(bench, output, code, prepare_only):
    result = json.loads((output / 'result.json').read_text())
    after = inventory(bench / 'raw')
    save(output / 'raw.after.json', after)
    baseline = output / 'raw.sealed.json'
    try:
        unchanged(json.loads((output / 'raw.before.json').read_text()), after, allow_additions=True)
        if baseline.exists():
            unchanged(json.loads(baseline.read_text()), after)
        result['raw_unchanged'] = True
    except ValueError as exc:
        result.update(raw_unchanged=False, integrity_error=str(exc))
        code = 1
    report = output / 'fresh_results.json'
    tasks = json.loads(report.read_text()).get('tasks', []) if report.exists() else []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    reported = {r['task'] for r in tasks}
    tasks.extend(dict(task=task, status='not_attempted', response=None,
                      api_attempts=0, reason='An earlier stage or task stopped the procedure')
                 for task in result['tasks'] if task not in reported)
    complete = (len(tasks) == len(result['tasks']) and
                {r['task'] for r in tasks} == set(result['tasks']) and
                all(r.get('status') == 'graded' and r.get('api_attempts', 0) > 0 and
                    r.get('response') == r.get('replayed_response') for r in tasks))
    result.update(finished_at=now(), exit_code=code,
                  status=('prepared' if prepare_only else 'passed') if code == 0
                  and (prepare_only or complete) else 'blocked' if code == 77 else 'failed')
    if code == 0 and not prepare_only and not complete:
        result['exit_code'] = code = 1
        result['error'] = 'Not every selected task has fresh inference and matching replayed grades'
    save(output / 'task_results.json', tasks)
    save(output / 'result.json', result)
    print(f"{bench.name}: {result['status']}; raw unchanged={result['raw_unchanged']}; {output}")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['begin', 'seal', 'finish', 'tasks', 'stage', 'credential-run'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bench = Path(os.environ['REPRO_BENCHMARK_DIR'])
    output = Path(os.environ['REPRO_RUN_DIR'])
    if args.action == 'begin':
        begin(bench, output)
    elif args.action == 'tasks':
        print(json.dumps(selection(bench)))
    elif args.action == 'seal':
        after = inventory(bench / 'raw')
        unchanged(json.loads((output / 'raw.before.json').read_text()), after, allow_additions=True)
        save(output / 'raw.sealed.json', after)
    elif args.action == 'finish':
        return finish(bench, output, int(args.arguments[0]), args.arguments[1] == '1')
    elif args.action == 'stage':
        name, status = args.arguments
        result = json.loads((output / 'result.json').read_text())
        result['stages'][name] = dict(status=status, recorded_at=now())
        save(output / 'result.json', result)
    elif args.action == 'credential-run':
        env = dict(os.environ)
        if not env.get('OPENAI_API_KEY'):
            path = env.get('REPRO_COMPETITION_CONFIG')
            if not path:
                raise ValueError('Set OPENAI_API_KEY or REPRO_COMPETITION_CONFIG')
            env['OPENAI_API_KEY'] = json.loads(Path(path).read_text())['api_keys']['OPENAI_API_KEY']
        return subprocess.run(args.arguments, env=env, check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
