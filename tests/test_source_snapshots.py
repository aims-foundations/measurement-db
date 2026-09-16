"""Contract-2 metadata and snapshot restoration, without network access."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml

from build_base import BenchmarkBuild, BuildContractError
from scripts.build_measurement_tables.load_source_files import SourceDataError
from scripts.build_measurement_tables.source_snapshots import (
    snapshot_artifacts, restore_snapshot, verify_snapshot_file, snapshot_location,
    DEFAULT_SOURCE_REPOSITORY, DEFAULT_SOURCE_REVISION,
)
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    BenchmarkMetadataError, load_benchmark_metadata, validate_benchmark_metadata,
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
            with self.assertRaisesRegex(SourceDataError,'pinned archive'):
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
        output=self.folder/'responses.parquet';output.write_bytes(b'previous output')
        with patch('scripts.build_measurement_tables.source_snapshots.snapshot_artifacts',
                   return_value=[self.artifact]):
            with self.assertRaisesRegex(BuildContractError,'pinned archive'):
                MutationBuild(str(self.folder/'build.py')).main()
        self.assertEqual(output.read_bytes(),b'previous output')


if __name__=='__main__': unittest.main()
