#!/usr/bin/env python3
"""Validate, restore and exercise a benchmark's declared reproducibility checks.

Source receipts are generated evidence, not another hand-maintained metadata
inventory. No credentials are recorded. A partial/blocked run exits nonzero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

import jsonschema
import yaml

ROOT = Path(__file__).absolute().parents[2]
STAGES = ('prepare', 'exercise', 'regrade', 'rerun')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def local_path(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Path leaves benchmark directory: {relative}')
    return path


def load_contract(benchmark):
    spec = yaml.safe_load((benchmark / 'reproducibility.yaml').read_text())
    schema = yaml.safe_load((ROOT / 'reproducibility_schema.yaml').read_text())
    jsonschema.Draft202012Validator(schema).validate(spec)
    metadata = yaml.safe_load((benchmark / 'metadata.yaml').read_text())
    for name, artifact in spec['artifacts'].items():
        for field in ('path', 'receipt'):
            local_path(benchmark, artifact[field])
        if artifact['source_index'] >= len(metadata['sources']['upstream']):
            raise ValueError(f'{name}: source_index is absent from metadata.yaml')
    for profile in spec['profiles'].values():
        if profile['agent'] is not None:
            local_path(benchmark, profile['agent']['configuration'])
        if profile['source_artifact'] not in spec['artifacts']:
            raise ValueError('Unknown source artifact')
        source = spec['artifacts'][profile['source_artifact']]
        if source['kind'] not in ('github_archive', 'file') or 'code' not in source['roles']:
            raise ValueError('Pilot source_artifact must be a code archive or code file')
    if spec['verification']['default_profile'] not in spec['profiles']:
        raise ValueError('Unknown default profile')
    for stage, command in spec['commands'].items():
        if any(STAGES.index(dependency) >= STAGES.index(stage) for dependency in command['requires']):
            raise ValueError(f'{stage}: dependencies must be earlier stages')
    return spec, metadata


def stage_environment(base, agent, stage):
    """Expose only explicitly declared credentials, and only during inference."""
    env = {key: value for key, value in base.items()
           if not any(word in key.upper() for word in
                      ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'CREDENTIAL'))}
    config_path = env.pop('REPRO_COMPETITION_CONFIG', None)
    if stage != 'rerun' or agent is None:
        return env
    stored = {}
    if config_path:
        stored = json.loads(Path(config_path).read_text()).get('api_keys', {})
    for name in agent['credential_names']:
        value = base.get(name) or stored.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Missing required credential: {name}')
        env[name] = value
    return env


def capture_github(benchmark, artifact, source):
    """Fetch a fixed GitHub source revision; never overwrite a captured archive."""
    import re
    url, revision = source['url'], source['revision']
    if not re.fullmatch(r'https://github.com/[\w.-]+/[\w.-]+', url):
        raise ValueError('Pilot source capture supports GitHub repository URLs')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Code capture requires an immutable Git commit in metadata.yaml')
    path = local_path(benchmark, artifact['path'])
    receipt_path = local_path(benchmark, artifact['receipt'])
    if path.exists() and receipt_path.exists():
        return verify_capture(benchmark, artifact, source)
    if path.exists() or receipt_path.exists():
        raise ValueError('Incomplete capture exists; inspect it before retrying')
    path.parent.mkdir(parents=True, exist_ok=True)
    download_url = f'https://codeload.github.com/{url.removeprefix("https://github.com/")}/tar.gz/{revision}'
    temporary = path.with_suffix('.part')
    try:
        with urllib.request.urlopen(download_url, timeout=90) as response, temporary.open('xb') as out:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 600 * 1024 * 1024:
                    raise ValueError('Archive exceeds the 600 MiB pilot capture limit')
                out.write(chunk)
        receipt = dict(format_version=1, source_url=url, revision=revision,
                       download_url=download_url, path=artifact['path'], size=size,
                       sha256=digest(temporary), captured_at=now())
        temporary.replace(path)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    finally:
        temporary.unlink(missing_ok=True)
    return verify_capture(benchmark, artifact, source)


def verify_capture(benchmark, artifact, source):
    path = local_path(benchmark, artifact['path'])
    receipt = json.loads(local_path(benchmark, artifact['receipt']).read_text())
    if (receipt['source_url'], receipt['revision']) != (source['url'], source['revision']):
        raise ValueError('Capture source/revision differs from metadata.yaml')
    if receipt['path'] != artifact['path'] or path.stat().st_size != receipt['size']:
        raise ValueError('Capture path or size differs from receipt')
    if digest(path) != receipt['sha256']:
        raise ValueError('Captured source checksum mismatch')
    return receipt


def extract_source(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        # The data filter rejects escaping links and device files. Extraction
        # never targets raw/ or an existing checkout.
        bundle.extractall(destination, filter='data')
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError('Expected a GitHub archive with one root directory')
    return roots[0]


def restore_source(benchmark, artifact, destination):
    path = local_path(benchmark, artifact['path'])
    if artifact['kind'] == 'github_archive':
        return extract_source(path, destination)
    # Some benchmark authors release their native evaluator as one source file.
    # Keep that file unchanged; do not import the leaderboard web application.
    if artifact['kind'] != 'file' or 'code' not in artifact['roles']:
        raise ValueError('Unsupported source artifact')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, destination / path.name)
    return destination


def execute(argv, *, cwd, env, timeout, stdout, stderr):
    started = time.monotonic()
    with stdout.open('w') as out, stderr.open('w') as err:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            status = 'passed' if code == 0 else 'blocked' if code == 77 else 'failed'
            reason = None if code == 0 else f'Exit {code}; see {stderr.name} and {stdout.name}'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code, status, reason = process.returncode, 'failed', f'Timeout after {timeout}s'
    return dict(status=status, exit_code=code, reason=reason,
                elapsed_seconds=round(time.monotonic() - started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--profile')
    parser.add_argument('--stages', nargs='+', choices=STAGES)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--work-dir', type=Path, help='New work directory; defaults to work/ inside the run output')
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    spec, metadata = load_contract(benchmark)
    if args.validate_only:
        print(f'{benchmark.name}: contract valid')
        return 0
    profile_name = args.profile or spec['verification']['default_profile']
    profile = spec['profiles'][profile_name]
    tasks = args.tasks or profile['tasks']
    if not set(tasks).issubset(profile['tasks']):
        raise ValueError('Select tasks from the declared pilot profile')
    stages = args.stages or spec['verification']['default_stages']
    if len(set(stages)) != len(stages):
        raise ValueError('Duplicate stages')
    for stage in stages:
        if any(dependency not in stages[:stages.index(stage)] for dependency in spec['commands'][stage]['requires']):
            raise ValueError(f'{stage}: include its dependencies first')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or benchmark / 'reproduction_checks/run_results' / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = args.work_dir.resolve() if args.work_dir else output / 'work'
    work.mkdir(parents=True, exist_ok=False)
    result = dict(format_version=1, benchmark=benchmark.name, started_at=now(), profile=profile_name,
                  tasks=tasks, historical_match=profile['historical_match'], profile_definition=profile,
                  contract_sha256=digest(benchmark / 'reproducibility.yaml'),
                  metadata_sha256=digest(benchmark / 'metadata.yaml'),
                  host=dict(platform=platform.platform(), machine=platform.machine(), python=sys.version),
                  work_directory=str(work), limitations=spec['verification']['limitations'], captures={}, stages={})
    evidence = output / 'procedure'
    evidence.mkdir()
    procedure_files = [benchmark / 'reproducibility.yaml', benchmark / 'metadata.yaml',
                       Path(__file__), ROOT / 'reproducibility_schema.yaml',
                       ROOT / 'scripts/reproduce_evaluations/bounded_luna.py',
                       *[ROOT / 'scripts/reproduce_evaluations' / name for name in
                         ('select_pilot_tasks.py', 'pilot_runtime.py', 'swe_pilot.py', 'resume_interrupted_pilot.py')
                         if (ROOT / 'scripts/reproduce_evaluations' / name).exists()],
                       *sorted((benchmark / 'reproduction_checks').glob('selection.json')),
                       *sorted((benchmark / 'reproduction_checks').glob('*.sh')),
                       *sorted((benchmark / 'reproduction_checks').glob('Dockerfile*')),
                       *sorted((benchmark / 'reproduction_checks').glob('.dockerignore')),
                       *sorted((benchmark / 'reproduction_checks').glob('*.py'))]
    for task_file in sorted((benchmark / 'reproduction_checks' / 'task_data').glob('*.json')):
        task_evidence = evidence / 'task_data'
        task_evidence.mkdir(exist_ok=True)
        shutil.copyfile(task_file, task_evidence / task_file.name)
    if profile['agent'] is not None:
        procedure_files.append(local_path(benchmark, profile['agent']['configuration']))
    result['procedure_sha256'] = {}
    for file in procedure_files:
        shutil.copyfile(file, evidence / file.name)
        result['procedure_sha256'][file.name] = digest(file)

    for task_file in sorted((evidence / 'task_data').glob('*.json')):
        result['procedure_sha256']['task_data/' + task_file.name] = digest(task_file)

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        for name, artifact in spec['artifacts'].items():
            source = metadata['sources']['upstream'][artifact['source_index']]
            capture = capture_github if artifact['kind'] == 'github_archive' else verify_capture
            result['captures'][name] = capture(benchmark, artifact, source)
        result['stages']['restore'] = dict(status='passed', description='Source bytes verified against capture receipts; this is not full environment restoration.')
        save()
        if args.capture_only:
            return 0
        artifact = spec['artifacts'][profile['source_artifact']]
        source = restore_source(benchmark, artifact, work / 'source')
        env = os.environ.copy()
        env.update(REPRO_BENCHMARK_DIR=str(benchmark), REPRO_SOURCE_DIR=str(source),
                   REPRO_RUN_DIR=str(output), REPRO_WORK_DIR=str(work),
                   REPRO_TASKS=json.dumps(tasks), REPRO_PYTHON=sys.executable,
                   REPRO_ARTIFACTS=json.dumps(spec['artifacts']),
                   PYTHONDONTWRITEBYTECODE='1', UV_CACHE_DIR=str(work / 'uv-cache'),
                   PIP_CACHE_DIR=str(work / 'pip-cache'))
        if profile['agent'] is not None:
            env['REPRO_AGENT_CONFIG'] = str(local_path(benchmark, profile['agent']['configuration']))
        for stage in STAGES:
            command = spec['commands'][stage]
            if stage not in stages or command['argv'] is None:
                entry = dict(status='not_attempted', reason=command['unavailable_reason'] or 'Not selected')
            elif any(result['stages'].get(dep, {}).get('status') != 'passed' for dep in command['requires']):
                entry = dict(status='blocked', reason='A required preceding stage did not pass')
            elif stage == 'rerun' and profile['agent'] is None:
                entry = dict(status='blocked', reason='Select a profile with an explicit agent configuration')
            else:
                print(f'{benchmark.name}: {stage}', flush=True)
                entry = execute(command['argv'], cwd=benchmark, env=stage_environment(env, profile['agent'], stage), timeout=command['timeout_seconds'],
                                stdout=output / f'{stage}.stdout.log', stderr=output / f'{stage}.stderr.log')
                entry.update(argv=command['argv'], stdout=f'{stage}.stdout.log', stderr=f'{stage}.stderr.log')
            entry['description'] = command['description']
            result['stages'][stage] = entry
            save()
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['finished_at'] = now()
        save()
        print(f'Record: {output / "result.json"}', flush=True)
    return 1 if any(result['stages'][stage]['status'] != 'passed' for stage in stages) else 0


if __name__ == '__main__':
    raise SystemExit(main())
