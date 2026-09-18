"""Native SDK arguments must not leak into the provider's JSON body."""
import importlib.util
from pathlib import Path
import sys


def test_native_sdk_payload_preserves_images_and_excludes_client_timeout(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'scripts/reproduce_evaluations'))
    for name in ('REPRO_BENCHMARK_DIR', 'REPRO_RUN_DIR', 'REPRO_SOURCE_DIR'):
        monkeypatch.setenv(name, str(tmp_path))
    spec = importlib.util.spec_from_file_location('math_pilot', root /
        'benchmarks/matharena/reproduction_checks/helpers/checks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = {'model': 'gpt-5.6-luna', 'timeout': 30000, 'tools': None,
        'messages': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'unchanged task'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,ABCD', 'detail': 'high'}]}]}
    wire = module.provider_payload(original)
    assert 'timeout' not in wire and 'tools' not in wire
    assert wire['messages'][0]['content'] == [
        {'type': 'text', 'text': 'unchanged task'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,ABCD', 'detail': 'high'}}]
    assert original['messages'][0]['content'][0]['type'] == 'input_text'
