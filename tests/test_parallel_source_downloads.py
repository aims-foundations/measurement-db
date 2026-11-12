"""Concurrent source transfers retain ordering, retry limits, and exact integrity."""
import hashlib
import io
import threading
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from build_base import BenchmarkBuild
from scripts.build_measurement_tables.load_source_files import SourceDataError

PAYLOAD = bytes(range(64))
URL = 'https://zenodo.org/api/records/123/files/source.zip/content'


def response(data, start, end):
    result = io.BytesIO(data)
    result.status = 206
    result.headers = {'Content-Range': f'bytes {start}-{end}/{len(PAYLOAD)}'}
    return result


def download(path, opener, **kwargs):
    with patch('urllib.request.urlopen', side_effect=opener), \
         patch('scripts.build_measurement_tables.load_source_files.time.sleep'):
        return BenchmarkBuild._download(SimpleNamespace(slug='fixture'), URL, path,
            expected_size=len(PAYLOAD), expected_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            chunk_size=8, **kwargs)


def limits(request):
    return map(int, request.get_header('Range').removeprefix('bytes=').split('-'))


def test_ranges_run_concurrently_but_assemble_in_order_and_clean_up(tmp_path):
    barrier = threading.Barrier(4)
    late_first = threading.Event()
    completed = []
    lock = threading.Lock()

    def fetch(request, **kwargs):
        start, end = limits(request)
        if start < 32:
            barrier.wait(timeout=5)
            if start == 0:
                assert late_first.wait(timeout=5)
        with lock:
            completed.append(start)
        if 0 < start < 32:
            late_first.set()
        return response(PAYLOAD[start:end + 1], start, end)

    path = tmp_path / 'source.zip'
    download(path, fetch)
    assert path.read_bytes() == PAYLOAD
    assert sorted(completed) == list(range(0, 64, 8))
    assert completed.index(0) > 0
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_range_resumes_without_repeating_or_skipping_bytes(tmp_path):
    requested = []
    lock = threading.Lock()

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise IncompleteRead(b'', 5)
            return super().read(3)

    def fetch(request, **kwargs):
        start, end = limits(request)
        with lock:
            requested.append((start, end))
        if start == 8:
            result = Interrupted(PAYLOAD[start:end + 1])
            result.status = 206
            result.headers = {'Content-Range': f'bytes {start}-{end}/64'}
            return result
        return response(PAYLOAD[start:end + 1], start, end)

    path = tmp_path / 'source.zip'
    download(path, fetch)
    assert path.read_bytes() == PAYLOAD
    assert requested.count((8, 15)) == requested.count((11, 15)) == 1
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('failure', ['wrong-range', 'overrun', 'corrupt-hash', 'access-denied', 'empty'])
def test_invalid_transfers_preserve_existing_file_and_remove_partials(tmp_path, failure):
    path = tmp_path / 'source.zip'
    path.write_bytes(b'previous source')
    attempts = []
    lock = threading.Lock()

    def fetch(request, **kwargs):
        start, end = limits(request)
        with lock:
            attempts.append(start)
        if failure == 'access-denied':
            raise HTTPError(URL, 403, 'Denied', {}, None)
        data = PAYLOAD[start:end + 1]
        if failure == 'overrun':
            data += b'x'
        elif failure == 'corrupt-hash':
            data = b'x' * len(data)
        elif failure == 'empty':
            data = b''
        result = response(data, start, end)
        if failure == 'wrong-range':
            result.headers['Content-Range'] = f'bytes {start}-{end}/65'
        return result

    expected = HTTPError if failure == 'access-denied' else IncompleteRead if failure == 'empty' else SourceDataError
    with pytest.raises(expected):
        download(path, fetch)
    assert path.read_bytes() == b'previous source'
    assert list(tmp_path.iterdir()) == [path]
    if failure == 'access-denied':
        assert max(attempts.count(start) for start in attempts) == 1
    if failure == 'empty':
        assert max(attempts.count(start) for start in attempts) == 8


def test_server_without_ranges_falls_back_to_verified_complete_transfer(tmp_path):
    requests = []

    def fetch(request, **kwargs):
        requests.append(request.get_header('Range'))
        result = io.BytesIO(PAYLOAD)
        result.status, result.headers = 200, {'Content-Length': str(len(PAYLOAD))}
        return result

    path = tmp_path / 'source.zip'
    download(path, fetch)
    assert path.read_bytes() == PAYLOAD
    assert requests[-1] is None
    assert list(tmp_path.iterdir()) == [path]
