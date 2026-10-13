"""Run and measure an explicitly preview-only deployment using Actions credentials."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def api(path, method='GET', payload=None):
    url = 'https://api.vercel.com' + path + '?' + urlencode({'teamId': os.environ['VERCEL_ORG_ID']})
    with urlopen(Request(url, method=method, data=None if payload is None else json.dumps(payload).encode(),
                        headers={'Authorization': 'Bearer ' + os.environ['VERCEL_TOKEN'], 'Content-Type': 'application/json'}), timeout=60) as r:
        return json.load(r)


def preflight(root):
    project = api('/v9/projects/' + os.environ['VERCEL_PROJECT_ID'])
    assert project['rootDirectory'] == 'website' and project['framework'] == 'nextjs'
    assert project.get('targets', {}).get('production'), 'Experiment requires an existing production project'
    (root / 'project-before.json').write_text(json.dumps({
        'id': project['id'], 'name': project['name'],
        'production_deployment': project['targets']['production']['id'],
    }, indent=2))
    print('Verified existing Vercel project; experiment target is preview.', flush=True)


def deploy(root, label):
    assert os.environ.get('GITHUB_REF') == 'refs/heads/perf/gallery-packed-preview'
    output = Path('.vercel/output')
    files = [p for p in output.rglob('*') if p.is_file()]
    assert 0 < len(files) < 10000, f'Experiment upload budget exceeded: {len(files)}'
    metrics = root / f'{label}-requests.jsonl'
    env = dict(os.environ)
    env['VERCEL_UPLOAD_METRICS'] = str(metrics)
    env['NODE_OPTIONS'] = ' '.join('--require=' + str(Path(p).resolve()) for p in [
        'website/scripts/vercel-fetch.cjs', 'website/scripts/vercel-upload-metrics.cjs'])
    command = ['pnpm', 'dlx', 'vercel@62.2.0', 'deploy', '--prebuilt', '--target=preview', '--yes',
               '--meta', 'githubDeployment=1', '--meta', 'githubCommitRef=perf/gallery-packed-preview',
               '--meta', f'packedGalleryExperiment={label}',
               '--env', 'HF_TOKEN=' + os.environ['HF_TOKEN'], '--token=' + os.environ['VERCEL_TOKEN']]
    begun = time.monotonic()
    print(f'Starting {label} preview: {len(files)} build-output files, {sum(p.stat().st_size for p in files)} bytes.', flush=True)
    with (root / f'{label}-stdout.log').open('w') as stdout, (root / f'{label}-stderr.log').open('w') as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    elapsed = time.monotonic() - begun
    calls = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    uploads = [row for row in calls if row['method'] == 'POST' and row['path'].endswith('/files')]
    summary = {'label': label, 'exit_code': result.returncode, 'seconds': elapsed,
               'build_output_files': len(files), 'build_output_bytes': sum(p.stat().st_size for p in files),
               'file_upload_requests': len(uploads),
               'successful_upload_bytes': sum(r['bodyBytes'] for r in uploads if 200 <= r.get('status', 0) < 300),
               'successful_file_uploads': sum(200 <= r.get('status', 0) < 300 for r in uploads)}
    urls = re.findall(r'https://[^\s]+\.vercel\.app', (root / f'{label}-stdout.log').read_text())
    if result.returncode == 0 and urls:
        summary['url'] = urls[-1]
        deployment = api('/v13/deployments/' + urls[-1].split('://')[1])
        summary.update(deployment_id=deployment['id'], state=deployment.get('readyState'), target=deployment.get('target'))
        assert deployment.get('target') != 'production'
        assert deployment.get('readyState') == 'READY'
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
                stream.write(f"url={summary['url']}\n")
    (root / f'{label}-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write('\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    if result.returncode:
        print((root / f'{label}-stderr.log').read_text()[-6000:], flush=True)
        raise SystemExit(result.returncode)


def edit(root):
    base = Path('.vercel/output/static/measurement-db/benchmark-data/xstest')
    source_path = base / 'source.json.gz'
    source = json.loads(gzip.decompress(source_path.read_bytes()))
    old_index_path = base / source['lookupPack']['index']
    index = json.loads(old_index_path.read_bytes())
    bucket, selected = next(iter(index['families']['item']['chunks'].items()))
    pack_id, offset, length, _ = selected
    pack = index['packs'][pack_id]
    old_pack_path = base / 'lookup-pack' / pack['file']
    payload = old_pack_path.read_bytes()
    entries = json.loads(gzip.decompress(payload[offset:offset + length]))
    key, value = next((k, v) for k, v in entries.items() if isinstance(v.get('content'), str))
    value['content'] += '\n\n[Preview experiment: testing incremental uploads.]'
    encoded = gzip.compress(json.dumps(entries, ensure_ascii=False, separators=(',', ':')).encode(), compresslevel=6, mtime=0)
    replacement = payload[:offset] + encoded + payload[offset + length:]
    delta = len(encoded) - length
    for family in index['families'].values():
        for chunk in family['chunks'].values():
            if chunk[0] == pack_id and chunk[1] > offset:
                chunk[1] += delta
    index['families']['item']['chunks'][bucket] = [pack_id, offset, len(encoded), hashlib.sha256(encoded).hexdigest()]
    pack.update(bytes=len(replacement), sha256=hashlib.sha256(replacement).hexdigest())
    pack['file'] = pack['sha256'] + '.bin'
    (old_pack_path.parent / pack['file']).write_bytes(replacement)
    index_bytes = json.dumps(index, sort_keys=True, separators=(',', ':')).encode()
    index_name = 'index.' + hashlib.sha256(index_bytes).hexdigest() + '.json'
    (old_index_path.parent / index_name).write_bytes(index_bytes)
    source['lookupPack']['index'] = 'lookup-pack/' + index_name
    source_path.write_bytes(gzip.compress(json.dumps(source, separators=(',', ':')).encode(), mtime=0))
    old_index_path.unlink()
    if all(p['file'] != old_pack_path.name for p in index['packs']):
        old_pack_path.unlink()
    (root / 'controlled-edit.json').write_text(json.dumps({
        'benchmark': 'xstest', 'key': json.loads(key), 'pack_bytes': len(replacement),
        'description': 'One prompt contains an explicit preview experiment note; scientific outputs unchanged.',
        'expected_changed_files': ['one pack', 'one index', 'source descriptor'],
    }, indent=2))
    print('Changed one XSTest prompt in preview build output for the incremental-upload measurement.', flush=True)


def share(root):
    summary = json.loads((root / 'repeat-summary.json').read_text())
    hostname = 'measurement-db-packed-20261012.vercel.app'
    try:
        existing = api('/v4/aliases/' + hostname)
    except HTTPError as error:
        if error.code != 404:
            raise
    else:
        deployment = api('/v13/deployments/' + existing['deploymentId'])
        assert existing.get('projectId') == os.environ['VERCEL_PROJECT_ID']
        assert deployment.get('target') != 'production'
        assert deployment.get('meta', {}).get('githubCommitRef') == 'perf/gallery-packed-preview'
    api('/v2/deployments/' + summary['deployment_id'] + '/aliases', 'POST', {'alias': hostname})
    alias = api('/v4/aliases/' + hostname)
    previous = set(alias.get('protectionBypass', {}))
    api('/aliases/' + alias['uid'] + '/protection-bypass', 'PATCH', {'ttl': 604800})
    updated = api('/v4/aliases/' + hostname)
    keys = set(updated.get('protectionBypass', {})) - previous
    assert len(keys) == 1
    link = 'https://' + hostname + '/measurement-db?' + urlencode({'_vercel_share': keys.pop()})
    private = root / 'review-link.private.txt'
    private.write_text(link + '\n')
    private.chmod(0o600)
    current = api('/v9/projects/' + os.environ['VERCEL_PROJECT_ID'])
    before = json.loads((root / 'project-before.json').read_text())
    state = {'preview_alias': hostname, 'production_before': before['production_deployment'],
             'production_after': current['targets']['production']['id'],
             'review_link_file': str(private)}
    (root / 'final-state.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['preflight', 'deploy', 'edit', 'share'])
    p.add_argument('--label', default='initial')
    args = p.parse_args()
    root = Path(os.environ['EXPERIMENT_RESULTS'])
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'deploy':
        deploy(root, args.label)
    else:
        {'preflight': preflight, 'edit': edit, 'share': share}[args.phase](root)
