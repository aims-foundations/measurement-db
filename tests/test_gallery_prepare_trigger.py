"""Exercise the real gallery selector against Git changes, without rendering."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('event,paths,expected', [
    ('pull_request', ['benchmarks/agentdojo/build.py',
                      'scripts/build_measurement_tables/trace_conversion.py',
                      'scripts/curate_benchmarks/chat_trace_audit.py',
                      '.github/workflows/metadata.yml', '.github/workflows/prepare-gallery.yml',
                      'docs/structured-trace-viewer-plan.md'], False),
    ('pull_request', ['website/components/trace-viewer.tsx'], False),
    ('pull_request', ['benchmarks/agentdojo/metadata.yaml'], True),
    ('pull_request', ['scripts/ci/canonical_gallery.py'], True),
    ('pull_request', ['scripts/render_website/generate_benchmark_gallery.py'], True),
    ('pull_request', ['README.md'], False),
    ('pull_request', ['scripts/trace_previews/export.py', 'website/content/generated/trace-previews.json'], False),
    ('pull_request', ['website/content/curated/benchmark-overrides.json'], True),
    ('push', ['website/components/trace-viewer.tsx'], True),
    ('push', ['benchmarks/agentdojo/build.py'], True),
    ('push', ['.github/workflows/prepare-gallery.yml'], True),
    ('workflow_dispatch', ['README.md'], True),
])
def test_gallery_preparation_follows_published_website_inputs(tmp_path, event, paths, expected):
    workflow = yaml.safe_load((ROOT / '.github/workflows/prepare-gallery.yml').read_text())
    script = next(step['run'] for step in workflow['jobs']['changes']['steps'] if step.get('id') == 'check')

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()

    git('init', '-q')
    (tmp_path / 'README.md').write_text('baseline\n')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'baseline')
    before = git('rev-parse', 'HEAD')
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('changed\n')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'change')
    after = git('rev-parse', 'HEAD')
    output = tmp_path / 'outputs'
    env = {**os.environ, 'GITHUB_EVENT_NAME': event, 'BEFORE': before, 'AFTER': after,
           'GITHUB_OUTPUT': str(output)}
    subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=tmp_path, env=env, check=True)
    assert output.read_text().strip() == f'deploy={str(expected).lower()}'
