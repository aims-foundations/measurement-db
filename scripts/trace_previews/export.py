"""Prepare bounded display pages before the browser reads a source trace.

Canonical Parquet files are read only. Native provenance stays in those files;
every displayed field has either its complete value or a prefix plus exact parts.
This exporter is independent of gallery chart/model preparation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow.parquet as pq

from scripts.render_website import record_previews as storage
from scripts.render_website.record_previews import RecordPacks, encoded

FORMAT = 'trace-preview-v1'
PAGE_BYTES = 1024 * 1024
PAGE_STEPS = 32
FIELD_CHARS = 20000
PART_CHARS = 32768
TESTS_SHOWN = 50


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity():
    return hashlib.sha256(encoded(dict(code=digest(__file__), transport=digest(storage.__file__), format=FORMAT,
        page_bytes=PAGE_BYTES, page_steps=PAGE_STEPS, field_chars=FIELD_CHARS,
        part_chars=PART_CHARS, tests_shown=TESTS_SHOWN))).hexdigest()


def search_text(step):
    if step['type'] == 'llm':
        return '\n'.join([step.get('content') or '', step.get('reasoning') or '',
                          *[call['arguments'] for call in step.get('toolCalls', [])]])
    if step['type'] == 'tool':
        return '\n'.join([step['output'], step.get('error') or ''])
    return step.get('content') or ''


class Writer:
    def __init__(self, directory):
        self.packs = RecordPacks(directory)
        self.maximum = 0

    def put(self, value, *, text=False):
        raw = value.encode('utf-8') if text else encoded(value)
        if len(raw) > PAGE_BYTES:
            raise ValueError('A trace preview page exceeds its expanded byte budget')
        self.maximum = max(self.maximum, len(raw))
        location = self.packs.put(raw)
        if location[2] > PAGE_BYTES:
            raise ValueError('A trace preview page exceeds its compressed byte budget')
        return location

    def text(self, value):
        return [self.put(value[i:i+PART_CHARS], text=True)
                for i in range(0, len(value), PART_CHARS)]

    def project(self, value, fields, path=()):
        if isinstance(value, dict):
            return {key: self.project(child, fields, (*path, key))
                    for key, child in value.items() if key != 'native'}
        if isinstance(value, list):
            return [self.project(child, fields, (*path, i)) for i, child in enumerate(value)]
        if isinstance(value, str) and len(value) > FIELD_CHARS:
            fields.append(dict(path=list(path), characters=len(value.encode('utf-16-le'))//2,
                               parts=self.text(value)))
            return value[:FIELD_CHARS]
        return value

    def trace(self, response_id, raw):
        original = json.loads(raw)
        if original.get('schema') != 'trace/1' or not isinstance(original.get('steps'), list):
            raise ValueError(f'{response_id}: expected a canonical trace/1 record')
        fields = []
        root = {key: deepcopy(value) for key, value in original.items() if key not in {'steps', 'native'}}
        tests = root.get('evaluation', {}).get('tests')
        if tests:
            root['evaluation']['testsTotal'] = len(tests)
            root['evaluation']['testCounts'] = dict(Counter(t['status'] for t in tests))
            if len(tests) > TESTS_SHOWN:
                fields.append(dict(path=['evaluation', 'tests'], encoding='json',
                    characters=len(encoded(tests).decode('utf-8').encode('utf-16-le'))//2,
                    parts=self.text(encoded(tests).decode('utf-8'))))
                root['evaluation']['tests'] = (sorted(tests, key=lambda t: t['status'] == 'passed'))[:TESTS_SHOWN]
        root = self.project(root, fields)
        counts = Counter(s['type'] for s in original['steps'])
        model = next((s['model'] for s in original['steps'] if s['type'] == 'llm' and s.get('model')), None)
        summary = dict(steps=len(original['steps']), modelTurns=counts['llm'], toolResults=counts['tool'],
                       hasReasoning=any(s.get('reasoning') for s in original['steps']))
        if model:
            summary['model'] = model
        pages, batch, page_fields, search, search_batch = [], [], [], [], []
        start = turns = turn_start = 0

        def flush_page():
            nonlocal batch, page_fields, start, turn_start
            if not batch:
                return
            value = dict(response_id=response_id, start=start, steps=batch, fields=page_fields)
            pages.append(dict(start=start, end=start+len(batch), turnStart=turn_start, range=self.put(value)))
            start += len(batch); turn_start = turns; batch, page_fields = [], []

        def flush_search():
            nonlocal search_batch
            if search_batch:
                search.append(self.put(dict(response_id=response_id, entries=search_batch)))
                search_batch = []

        for number, source_step in enumerate(original['steps']):
            step_fields = []
            step = self.project(source_step, step_fields)
            # Prefix lengths remain visible in the existing trace renderer.
            for field in step_fields:
                if len(field['path']) == 1:
                    step.setdefault('truncated', {})[field['path'][0]] = field['characters']
            for field in step_fields:
                field['step'] = number
            candidate = dict(response_id=response_id, start=start, steps=[*batch, step], fields=[*page_fields, *step_fields])
            if batch and (len(batch) >= PAGE_STEPS or len(encoded(candidate)) > PAGE_BYTES):
                flush_page()
            batch.append(step); page_fields.extend(step_fields)
            turns += source_step['type'] == 'llm'
            # Search covers uncapped display text, never native provenance. A
            # continuation flag lets the browser match across chunk boundaries.
            text = search_text(source_step)
            for part, offset in enumerate(range(0, len(text), PART_CHARS)):
                entry = [number, text[offset:offset+PART_CHARS], part > 0]
                if search_batch and len(encoded(dict(response_id=response_id, entries=[*search_batch, entry]))) > PAGE_BYTES:
                    flush_search()
                search_batch.append(entry)
        flush_page(); flush_search()
        header = dict(kind=FORMAT, response_id=response_id, trace=root, summary=summary,
                      pages=pages, fields=fields, search=search,
                      source_trace_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest())
        return self.put(header)


def export(path, output):
    """Atomically export a complete table, cached by source bytes and exporter identity."""
    path, output = Path(path), Path(output)
    source_sha, exporter = digest(path), identity()
    if output.exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['source_sha256'] != source_sha or manifest['exporter'] != exporter:
            raise ValueError('Preview cache does not match source bytes and exporter version')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.trace-preview-', dir=output.parent))
    writer = Writer(staging)
    buckets, count = [{} for _ in range(256)], 0
    try:
        source = pq.ParquetFile(path)
        if (source.schema_arrow.metadata or {}).get(b'measurement_db.trace_format') != b'trace/1':
            raise ValueError('Structured trace previews require the trace/1 Parquet marker')
        for batch in source.iter_batches(batch_size=1, columns=['response_id', 'trace']):
            for record in batch.to_pylist():
                key = record['response_id']
                bucket = buckets[hashlib.sha256(key.encode()).digest()[0]]
                if key in bucket:
                    raise ValueError(f'Duplicate trace identity: {key}')
                bucket[key] = writer.trace(key, record['trace'])
                count += 1
                if count % 100 == 0:
                    print(f'Exported {count}/{source.metadata.num_rows} structured trace previews', flush=True)
        indexes = [writer.put(bucket) for bucket in buckets]
        writer.packs.close()
        if digest(path) != source_sha or identity() != exporter:
            raise ValueError('Source or exporter changed during preview preparation')
        manifest = dict(format=FORMAT, source_sha256=source_sha, exporter=exporter,
                        source_bytes=path.stat().st_size, rows=count, packs=writer.packs.files,
                        indexes=indexes, max_payload_bytes=writer.maximum, page_bytes=PAGE_BYTES)
        (staging/'manifest.json').write_bytes(encoded(manifest))
        staging.rename(output)
        return manifest
    except BaseException:
        writer.packs.close()
        shutil.rmtree(staging)
        raise


def verify(path, directory):
    """Independently reconstruct all display fields and compare against canonical rows."""
    path, directory = Path(path), Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if digest(path) != manifest['source_sha256']:
        raise ValueError('Canonical source changed')
    handles = [(directory/f'pack-{i}.bin').open('rb') for i in range(len(manifest['packs']))]

    def read(location, *, text=False):
        pack, offset, size, expanded, sha = location
        if max(size, expanded) > manifest['page_bytes']:
            raise ValueError('Preview payload exceeds budget')
        handle = handles[pack]; handle.seek(offset); compressed = handle.read(size)
        if hashlib.sha256(compressed).hexdigest() != sha:
            raise ValueError('Preview checksum mismatch')
        raw = gzip.decompress(compressed)
        if len(raw) != expanded:
            raise ValueError('Preview expanded size mismatch')
        return raw.decode('utf-8') if text else json.loads(raw)

    def restore(value, fields):
        for field in fields:
            parent = value
            for key in field['path'][:-1]:
                parent = parent[key]
            text = ''.join(read(part, text=True) for part in field['parts'])
            parent[field['path'][-1]] = json.loads(text) if field.get('encoding') == 'json' else text

    count = 0
    try:
        indexes = [read(location) for location in manifest['indexes']]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=['response_id', 'trace']):
            for row in batch.to_pylist():
                key = row['response_id']; header = read(indexes[hashlib.sha256(key.encode()).digest()[0]][key])
                if header['response_id'] != key or header['source_trace_sha256'] != hashlib.sha256(row['trace'].encode()).hexdigest():
                    raise ValueError('Preview identity or canonical trace digest mismatch')
                original = json.loads(row['trace'])
                expected_root = {k:v for k,v in original.items() if k not in {'native', 'steps'}}
                root = header['trace']; restore(root, header['fields'])
                root.get('evaluation', {}).pop('testsTotal', None); root.get('evaluation', {}).pop('testCounts', None)
                if root != expected_root:
                    raise ValueError('Preview header differs from canonical fields')
                steps = []
                for descriptor in header['pages']:
                    page = read(descriptor['range'])
                    if page['response_id'] != key or page['start'] != len(steps) or descriptor['start'] != len(steps):
                        raise ValueError('Preview step order or identity mismatch')
                    for field in page['fields']:
                        restore(page['steps'][field['step']-page['start']], [field])
                    for step in page['steps']:
                        step.pop('truncated', None)
                    steps.extend(page['steps'])
                    if len(steps) != descriptor['end']:
                        raise ValueError('Preview step range mismatch')
                expected_steps = [{k:v for k,v in s.items() if k!='native'} for s in original['steps']]
                if steps != expected_steps or header['summary']['steps'] != len(steps):
                    raise ValueError('Displayed steps differ from canonical fields')
                text = {}
                for location in header['search']:
                    page = read(location)
                    if page['response_id'] != key:
                        raise ValueError('Search identity mismatch')
                    for index, part, continuation in page['entries']:
                        if continuation != (index in text):
                            raise ValueError('Search continuation mismatch')
                        text[index] = text.get(index, '') + part
                if any(text.get(i, '') != search_text(step) for i,step in enumerate(expected_steps)):
                    raise ValueError('Search text differs from complete displayed source fields')
                count += 1
        if count != manifest['rows']:
            raise ValueError('Preview record count mismatch')
        return dict(rows=count, source_sha256=manifest['source_sha256'], max_payload_bytes=manifest['max_payload_bytes'],
                    exact_display_fields=True, complete_search_text=True, canonical_source_unchanged=True)
    finally:
        for handle in handles:
            handle.close()
