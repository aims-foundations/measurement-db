"""Named upstream downloads and explicit snapshot restoration, without network access."""
import copy
import hashlib
import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from build_base import BenchmarkBuild, BuildContractError, ExactMatcher
from scripts.build_measurement_tables import reload
from scripts.build_measurement_tables.load_source_files import SourceDataError
from scripts.build_measurement_tables.source_snapshots import (
    DEFAULT_SOURCE_REPOSITORY,
    DEFAULT_SOURCE_REVISION,
    restore_snapshot,
    snapshot_artifacts,
    snapshot_location,
    verify_snapshot_file,
)
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    BenchmarkMetadataError,
    load_benchmark_metadata,
    validate_benchmark_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b'{"released":true}\n'


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.metadata = yaml.safe_load((ROOT / 'benchmarks/real_webagents/metadata.yaml').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'fixture'
        self.archive = snapshot_location(self.folder)
        self.raw = self.folder / 'raw'
        self.raw.mkdir(parents=True)
        self.metadata_path = self.folder / 'metadata.yaml'
        self.metadata_path.write_text(yaml.safe_dump(self.metadata))
        self.artifact = dict(file='source.json', size=len(PAYLOAD), hash_kind='sha256',
                             digest=hashlib.sha256(PAYLOAD).hexdigest(), url='https://example.org/source')

    def test_compact_metadata_and_unknown_upstream_revision(self):
        load_benchmark_metadata(self.metadata_path)

    def test_local_unpublished_snapshot_is_verified_without_network(self):
        manifest = self.folder / 'inputs.json'
        manifest.write_text(json.dumps({'format_version': 1,
            'benchmarks': {'fixture': [self.artifact]}}))
        with patch.dict('os.environ', {'MEASUREMENT_DB_SOURCE_MANIFEST': str(manifest)}):
            location = snapshot_location(self.folder)
        artifacts = snapshot_artifacts(location)
        with self.assertRaisesRegex(SourceDataError, 'absent'):
            restore_snapshot(location, self.raw, artifacts)
        target = self.raw / 'source.json'
        target.write_bytes(PAYLOAD)
        with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('network')):
            restore_snapshot(location, self.raw, artifacts)
        target.write_bytes(b'x' * len(PAYLOAD))
        with self.assertRaisesRegex(SourceDataError, 'differs'):
            restore_snapshot(location, self.raw, artifacts)

    def test_invalid_local_manifest_is_rejected(self):
        manifest = self.folder / 'inputs.json'
        location = {'manifest': str(manifest), 'benchmark': 'fixture'}
        for artifacts in ([], [self.artifact, self.artifact],
                          [dict(self.artifact, file='../outside')],
                          [dict(self.artifact, size=True)],
                          [dict(self.artifact, digest='wrong')]):
            manifest.write_text(json.dumps({'format_version': 1,
                'benchmarks': {'fixture': artifacts}}))
            with self.assertRaises(SourceDataError):
                snapshot_artifacts(location)

    def test_unknown_fields_and_embedded_file_lists_are_rejected(self):
        for location, key, value in (
            (('sources',), 'downloads', {'artifact_0001': {}}),
            (('sources',), 'inputs', {'artifact_0001': {}}),
            (('sources',), 'archive', {'repo_id': 'example/data'}),
            (('sources',), 'notes', 'Generic archive explanation'),
            ((), 'archive_layout', {'files': []}),
            ((), 'expectations', {}),
        ):
            with self.subTest(key=key):
                metadata=copy.deepcopy(self.metadata);target=metadata
                for part in location: target=target[part]
                target[key]=value
                with self.assertRaises(BenchmarkMetadataError):
                    validate_benchmark_metadata(metadata, path=self.metadata_path)

    def test_shared_snapshot_defaults_and_runtime_overrides(self):
        self.assertEqual(self.archive, {'repo_id': DEFAULT_SOURCE_REPOSITORY,
                         'revision': DEFAULT_SOURCE_REVISION, 'path': 'fixture/raw'})
        with patch.dict('os.environ', {'MEASUREMENT_DB_SOURCE_REPO': 'example/data',
                                      'MEASUREMENT_DB_SOURCE_REVISION': 'b'*40}):
            self.assertEqual(snapshot_location(self.folder), {
                'repo_id': 'example/data', 'revision': 'b'*40, 'path': 'fixture/raw'})
        for value in ('main', 'c969fab', ''):
            with patch.dict('os.environ', {'MEASUREMENT_DB_SOURCE_REVISION': value}):
                with self.assertRaises(SourceDataError): snapshot_location(self.folder)
        with patch.dict('os.environ', {'MEASUREMENT_DB_SOURCE_REPO': 'example/private'}, clear=True):
            with self.assertRaises(SourceDataError): snapshot_location(self.folder)

    def test_missing_required_upstream_fields(self):
        for key in ('url','revision'):
            metadata=copy.deepcopy(self.metadata)
            del metadata['sources']['upstream'][0][key]
            with self.assertRaises(BenchmarkMetadataError):
                validate_benchmark_metadata(metadata,path=self.metadata_path)

    def test_inventory_uses_pinned_tree_and_both_hash_types(self):
        entries=[SimpleNamespace(path='fixture/raw/subdir'),
                 SimpleNamespace(path='fixture/raw/a', size=len(PAYLOAD), blob_id='a'*40, lfs=None),
                 SimpleNamespace(path='fixture/raw/b', size=len(PAYLOAD), blob_id='b'*40,
                                 lfs=SimpleNamespace(sha256=self.artifact['digest']))]
        with patch('huggingface_hub.HfApi.list_repo_tree',return_value=entries) as tree:
            actual=snapshot_artifacts(self.archive)
        self.assertEqual([a['hash_kind'] for a in actual],['git_sha1','sha256'])
        self.assertEqual(tree.call_args.kwargs['revision'],self.archive['revision'])
        self.assertEqual(tree.call_args.kwargs['path_in_repo'],'fixture/raw')
        self.assertTrue(all(self.archive['revision'] in a['url'] for a in actual))

    def test_empty_and_escaping_snapshot_paths_are_rejected(self):
        for entries in ([],[SimpleNamespace(path='fixture/raw/../outside',size=1,blob_id='a'*40,lfs=None)]):
            with patch('huggingface_hub.HfApi.list_repo_tree',return_value=entries):
                with self.assertRaises(SourceDataError): snapshot_artifacts(self.archive)

    def test_restoration_fetches_only_missing_files_at_pinned_revision(self):
        cached=Path(self.temp.name)/'downloaded';cached.write_bytes(PAYLOAD)
        with patch('huggingface_hub.hf_hub_download',return_value=str(cached)) as download:
            restore_snapshot(self.archive,self.raw,[self.artifact])
            restore_snapshot(self.archive,self.raw,[self.artifact])
        self.assertEqual((self.raw/'source.json').read_bytes(),PAYLOAD)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs['revision'],self.archive['revision'])

    def test_corrupt_cached_input_is_rejected_without_network_or_replacement(self):
        target=self.raw/'source.json';target.write_bytes(b'x'*len(PAYLOAD))
        with patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaisesRegex(SourceDataError,'pinned source'):
                restore_snapshot(self.archive,self.raw,[self.artifact])
            download.assert_not_called()
        self.assertEqual(target.read_bytes(),b'x'*len(PAYLOAD))

    def test_git_blob_verification(self):
        target=self.raw/'source.json';target.write_bytes(PAYLOAD)
        artifact={**self.artifact,'hash_kind':'git_sha1',
                  'digest':hashlib.sha1(b'blob %d\0'%len(PAYLOAD)+PAYLOAD).hexdigest()}
        verify_snapshot_file(target,artifact)
        target.write_bytes(b'x'*len(PAYLOAD))
        with self.assertRaises(SourceDataError):verify_snapshot_file(target,artifact)

    def test_symlink_cannot_escape_raw(self):
        outside=Path(self.temp.name)/'outside';outside.write_bytes(PAYLOAD)
        (self.raw/'source.json').symlink_to(outside)
        with self.assertRaisesRegex(SourceDataError,'escapes raw'):
            restore_snapshot(self.archive,self.raw,[self.artifact])

    def test_changed_input_during_build_preserves_previous_outputs(self):
        class MutationBuild(BenchmarkBuild):
            def build_subject_item_response_rows(self):
                (self.raw_dir/'source.json').write_bytes(b'x'*len(PAYLOAD))
        (self.raw/'source.json').write_bytes(PAYLOAD)
        tables = self.folder / 'formatted_tables'
        tables.mkdir()
        output=tables/'responses.parquet';output.write_bytes(b'previous output')
        with patch('scripts.build_measurement_tables.source_snapshots.snapshot_artifacts',
                   return_value=[self.artifact]):
            with self.assertRaisesRegex(BuildContractError,'pinned source'):
                MutationBuild(str(self.folder/'build.py')).main_from_args(['--archive'])
        self.assertEqual(output.read_bytes(),b'previous output')

    def test_archive_build_keeps_tables_separate_from_sources_and_metadata(self):
        class FixtureBuild(BenchmarkBuild):
            def build_subject_item_response_rows(self):
                subject = self.add_subject('gpt-4o')
                item = self.add_item(raw_item_id='q1', content='Return one.',
                    grading_criterion={'reference_answer': '1'},
                    verifier=ExactMatcher(spec='Exact binary match'))
                self.add_response(subject_id=subject, item_id=item,
                                  response=1, trace='1')

        reload()
        self.addCleanup(reload)
        source = self.raw / 'source.json'
        source.write_bytes(PAYLOAD)
        metadata = self.metadata_path.read_bytes()
        with patch('scripts.build_measurement_tables.source_snapshots.snapshot_artifacts',
                   return_value=[self.artifact]):
            FixtureBuild(str(self.folder / 'build.py')).main_from_args(['--archive'])
        self.assertEqual({p.stem for p in (self.folder / 'formatted_tables').glob('*.parquet')},
                         {'items', 'subjects', 'benchmarks', 'responses', 'traces'})
        self.assertFalse(list(self.folder.glob('*.parquet')))
        self.assertEqual(source.read_bytes(), PAYLOAD)
        self.assertEqual(self.metadata_path.read_bytes(), metadata)

    def test_upstream_is_default_and_cache_is_verified_without_archive_access(self):
        class FixtureBuild(BenchmarkBuild):
            def download(self):
                return self.fetch_sources('results')

            def build_subject_item_response_rows(self):
                subject = self.add_subject('gpt-4o')
                item = self.add_item(raw_item_id='q1', content='Return one.',
                    grading_criterion={'reference_answer': '1'}, verifier=ExactMatcher(spec='Equality'))
                self.add_response(subject_id=subject, item_id=item, response=1, trace='1')

        self.metadata['sources']['upstream'] = [dict(
            name='results', url='https://provider.example/results.json', revision=None,
            file='source.json', size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())]
        self.metadata_path.write_text(yaml.safe_dump(self.metadata))
        reload()
        self.addCleanup(reload)
        with patch('scripts.build_measurement_tables.source_snapshots.snapshot_artifacts',
                   side_effect=AssertionError('archive must not be consulted')):
            with patch('urllib.request.urlopen', return_value=io.BytesIO(PAYLOAD)) as download:
                FixtureBuild(str(self.folder/'build.py')).main()
            self.assertEqual(download.call_count, 1)
            self.assertEqual((self.raw/'source.json').read_bytes(), PAYLOAD)
            with patch('urllib.request.urlopen', side_effect=AssertionError('cached file')):
                reload()
                FixtureBuild(str(self.folder/'build.py')).main()
            output = self.folder/'formatted_tables/responses.parquet'
            before = output.read_bytes()
            (self.raw/'source.json').write_bytes(b'x' * len(PAYLOAD))
            with self.assertRaises(SourceDataError):
                FixtureBuild(str(self.folder/'build.py')).main()
            self.assertEqual(output.read_bytes(), before)

    def test_upstream_repository_selection_and_missing_paths(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        sources = [dict(name='tasks', url='https://github.com/provider/benchmark', revision='a'*40,
            files=[{'match': r'tasks/(?P<name>[^/]+\.json)', 'path': 'tasks/{name}'}])]
        tree = dict(tree=[dict(type='blob', path='tasks/one.json', size=2, sha='b'*40),
                         dict(type='blob', path='README.md', size=10, sha='c'*40)], truncated=False)
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   return_value=io.BytesIO(json.dumps(tree).encode())):
            artifacts = upstream_artifacts(sources, ('tasks',))
        self.assertEqual([a['file'] for a in artifacts], ['tasks/one.json'])
        self.assertEqual(artifacts[0]['digest'], 'b'*40)
        self.assertIn('/'+'a'*40+'/', artifacts[0]['url'])
        with self.assertRaisesRegex(SourceDataError, 'declared upstream'):
            upstream_artifacts(sources, ('missing',))
        for rule, message in (({'match': 'absent', 'path': 'absent'}, 'no upstream files'),
                              ({'match': r'tasks/.*', 'path': '../outside'}, 'unsafe raw destination')):
            sources[0]['files'] = [rule]
            with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                       return_value=io.BytesIO(json.dumps(tree).encode())):
                with self.assertRaisesRegex(SourceDataError, message):
                    upstream_artifacts(sources, ('tasks',))
        sources[0]['files'] = [{'match': '.*', 'path': 'same'}]
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   return_value=io.BytesIO(json.dumps(tree).encode())):
            with self.assertRaisesRegex(SourceDataError, 'Duplicate upstream destination'):
                upstream_artifacts(sources, ('tasks',))

    def test_upstream_metadata_requires_pins_and_complete_download_specification(self):
        for entry in (dict(name='broken', url='https://provider.example/results', revision=None),
                      dict(name='moving', url='https://github.com/provider/benchmark', revision='main',
                           files=[{'match': '.*', 'path': '{path}'}])):
            self.metadata['sources']['upstream'] = [entry]
            with self.assertRaises(BenchmarkMetadataError):
                validate_benchmark_metadata(self.metadata, path=self.metadata_path)


if __name__=='__main__': unittest.main()
