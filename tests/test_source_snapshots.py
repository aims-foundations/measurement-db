"""Named upstream downloads and explicit snapshot restoration, without network access."""
import copy
import base64
import gzip
import hashlib
import json
import io
import shutil
import subprocess
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

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG is required for encrypted upstream releases")
    def test_gpg_json_preserves_raw_and_rejects_wrong_password(self):
        from scripts.build_measurement_tables.load_source_files import read_gpg_json
        with tempfile.TemporaryDirectory(dir=self.folder) as home:
            encrypted = subprocess.run(
                ["gpg", "--no-options", "--homedir", home, "--batch", "--no-tty",
                 "--pinentry-mode", "loopback", "--passphrase", "public-test-password",
                 "--symmetric", "--output", "-"],
                input=PAYLOAD, capture_output=True, check=True,
            ).stdout
        source = self.raw / "bank.json.gpg"
        source.write_bytes(encrypted)
        self.assertEqual(read_gpg_json(source, password="public-test-password", scratch_dir=self.folder),
                         json.loads(PAYLOAD))
        with self.assertRaisesRegex(SourceDataError, "Cannot decrypt"):
            read_gpg_json(source, password="wrong-password", scratch_dir=self.folder)
        self.assertEqual(source.read_bytes(), encrypted)
        self.assertEqual(list(self.raw.iterdir()), [source])
        self.assertFalse(list(self.folder.glob(".gpg-*")))

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

    def test_scoped_github_trees_keep_hashes_and_skip_unselected_assets(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        source = dict(name='results', url='https://github.com/provider/benchmark', revision='a'*40,
                      tree_paths=['runs/*/results', 'runs/one/results/item.json'],
                      files=[dict(match=r'runs/(?P<run>[^/]+)/results/(?P<file>.+)', path='{run}/{file}')])
        self.metadata['sources']['upstream'] = [source]
        validate_benchmark_metadata(self.metadata, path=self.metadata_path)
        tree = lambda path, sha: dict(type='tree', path=path, sha=sha)
        blob = dict(type='blob', path='item.json', sha='f'*40, size=12)
        payloads = {
            'a'*40: [tree('runs', 'b'*40), tree('large-assets', '0'*40)],
            'b'*40: [tree('one', 'c'*40)],
            'c'*40: [tree('results', 'd'*40), tree('screenshots', 'e'*40)],
            'd'*40: [blob], 'd'*40+'?recursive=1': [blob],
        }
        calls = []
        def respond(request, **kwargs):
            key = request.full_url.rsplit('/', 1)[-1]
            calls.append(key)
            return io.BytesIO(json.dumps(dict(tree=payloads[key], truncated=False)).encode())
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=respond):
            artifacts = upstream_artifacts([source], ('results',))
        self.assertEqual([a['file'] for a in artifacts], ['one/item.json'])
        self.assertEqual(artifacts[0]['digest'], 'f'*40)
        self.assertIn('/'+'a'*40+'/runs/one/results/item.json', artifacts[0]['url'])
        self.assertEqual(calls.count('a'*40), 1)
        self.assertNotIn('e'*40, calls)
        for paths in (['../outside'], ['/absolute'], ['runs/**'], ['runs/missing']):
            with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=respond):
                with self.assertRaises(SourceDataError):
                    upstream_artifacts([dict(source, tree_paths=paths)], ('results',))
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   return_value=io.BytesIO(json.dumps(dict(tree=[], truncated=True)).encode())):
            with self.assertRaisesRegex(SourceDataError, 'truncated'):
                upstream_artifacts([source], ('results',))

    def test_github_lfs_verifies_pointer_and_downloaded_content(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        pointer = (f'version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(PAYLOAD).hexdigest()}\n'
                   f'size {len(PAYLOAD)}\n').encode()
        pointer_hash = hashlib.sha1(f'blob {len(pointer)}\0'.encode() + pointer).hexdigest()
        source = dict(name='bundles', url='https://github.com/provider/benchmark', revision='a'*40,
                      git_lfs=True, files=[dict(match=r'outputs/one\.bundle', path='one.bundle')])
        self.metadata['sources']['upstream'] = [source]
        validate_benchmark_metadata(self.metadata, path=self.metadata_path)
        tree = dict(tree=[dict(type='blob', path='outputs/one.bundle', size=len(pointer), sha=pointer_hash)], truncated=False)
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   side_effect=[io.BytesIO(json.dumps(tree).encode()), io.BytesIO(pointer)]):
            artifact, = upstream_artifacts([source], ('bundles',))
        self.assertEqual(artifact['size'], len(PAYLOAD))
        self.assertEqual(artifact['hash_kind'], 'sha256')
        self.assertEqual(artifact['digest'], hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(artifact['url'], 'https://media.githubusercontent.com/media/provider/benchmark/' + 'a'*40 + '/outputs/one.bundle')
        target = self.raw / 'one.bundle'
        target.write_bytes(PAYLOAD)
        verify_snapshot_file(target, artifact)
        target.write_bytes(pointer)
        with self.assertRaises(SourceDataError):
            verify_snapshot_file(target, artifact)
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   side_effect=[io.BytesIO(json.dumps(tree).encode()), io.BytesIO(pointer.replace(b'size ', b'Size '))]):
            with self.assertRaisesRegex(SourceDataError, 'pointer differs'):
                upstream_artifacts([source], ('bundles',))

    def test_public_gcs_selection_pins_versions_and_keeps_encoded_bytes(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        encoded = gzip.compress(PAYLOAD, mtime=0)
        checksum = hashlib.md5(encoded).hexdigest()
        entry = dict(name='release/run:one/source.json', generation='123', size=str(len(encoded)),
                     md5Hash=base64.b64encode(bytes.fromhex(checksum)).decode(), contentEncoding='gzip')
        identity = [dict(path='run:one/source.json', generation='123', size=len(encoded),
                         digest=checksum, content_encoding='gzip')]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        source = dict(name='results', url='https://storage.googleapis.com/provider-bucket', revision='v1',
                      prefix='release/', tree_sha256=fingerprint,
                      files=[dict(match=r'[^/]+/source\.json', path='{path}')])
        self.metadata['sources']['upstream'] = [source]
        validate_benchmark_metadata(self.metadata, path=self.metadata_path)
        pages = [dict(items=[], nextPageToken='next'), dict(items=[entry])]
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   side_effect=[io.BytesIO(json.dumps(page).encode()) for page in pages]) as fetch:
            artifacts = upstream_artifacts([source], ('results',))
        self.assertIn('pageToken=next', fetch.call_args_list[1].args[0].full_url)
        self.assertEqual(artifacts[0]['file'], 'run_x3a_one/source.json.gz')
        self.assertTrue(artifacts[0]['url'].endswith('?generation=123'))
        target = self.raw / 'captured.json.gz'
        target.write_bytes(encoded)
        verify_snapshot_file(target, artifacts[0])
        target.write_bytes(b'x' * len(encoded))
        with self.assertRaisesRegex(SourceDataError, 'content differs'):
            verify_snapshot_file(target, artifacts[0])
        changed = {**entry, 'generation': '124'}
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   return_value=io.BytesIO(json.dumps(dict(items=[changed])).encode())):
            with self.assertRaisesRegex(SourceDataError, 'pinned tree'):
                upstream_artifacts([source], ('results',))
        del source['tree_sha256']
        with self.assertRaises(BenchmarkMetadataError):
            validate_benchmark_metadata(self.metadata, path=self.metadata_path)

    def test_gcs_download_requests_original_gzip_representation(self):
        encoded = gzip.compress(PAYLOAD, mtime=0)
        class FixtureBuild(BenchmarkBuild):
            def build_tables(self):
                raise AssertionError('No table build is needed for this transport check')
        self.metadata_path.write_text(yaml.safe_dump(self.metadata))
        builder = FixtureBuild(str(self.folder / 'build.py'))
        artifact = dict(file='new.json.gz', size=len(encoded), hash_kind='md5',
                        digest=hashlib.md5(encoded).hexdigest(), content_encoding='gzip',
                        url='https://storage.googleapis.com/bucket/object?generation=123')
        with patch('build_base._source_files.upstream_artifacts', return_value=[artifact]), \
             patch('build_base.urllib.request.urlopen', return_value=io.BytesIO(encoded)) as fetch:
            builder.fetch_sources('results')
        self.assertEqual(fetch.call_args.args[0].get_header('Accept-encoding'), 'gzip')
        self.assertEqual(fetch.call_args.args[0].get_header('User-agent'), 'measurement-db')
        self.assertEqual((self.raw/'new.json.gz').read_bytes(), encoded)

    def test_helm_release_index_selects_versioned_runs_and_checks_its_bytes(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        payload = json.dumps([
            {'run_spec': {'groups': ['chosen']},
             'run_path': '/old/machine/benchmark_output/runs/v1/scenario:model=a'},
            {'run_spec': {'groups': ['other']},
             'run_path': 'benchmark_output/runs/v2/other:model=b'},
        ]).encode()
        index = dict(name='release', url='https://provider.example/runs.json', revision='v2',
                     file='release.json', size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        relative = 'benchmark_output/runs/v1/scenario:model=a/instances.json'
        entry = dict(name='safety/'+relative, generation='123', size='2',
                     md5Hash=base64.b64encode(hashlib.md5(b'[]').digest()).decode())
        identity = [dict(path=relative, generation='123', size=2,
                         digest=hashlib.md5(b'[]').hexdigest(), content_encoding='')]
        source = dict(name='runs', url='https://storage.googleapis.com/provider-bucket', revision='v2',
                      prefix='safety/', helm_index={'source': 'release', 'group': 'chosen'},
                      tree_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      files=[dict(match=r'benchmark_output/runs/(?P<version>[^/]+)/(?P<run>[^/]+)/instances\.json',
                                  path='runs/{version}/{run}/instances.json')])
        with patch('scripts.build_measurement_tables.load_source_files.urlopen',
                   side_effect=[io.BytesIO(payload), io.BytesIO(json.dumps({'items': [entry]}).encode())]) as fetch:
            artifacts = upstream_artifacts([index, source], ('release', 'runs'))
        self.assertEqual([a['file'] for a in artifacts], ['release.json', 'runs/v1/scenario_x3a_model_x3d_a/instances.json'])
        self.assertIn('scenario%3Amodel%3Da%2F', fetch.call_args.args[0].full_url)
        # With no group restriction, a multi-task release selects both runs.
        source['helm_index'].pop('group')
        other = dict(entry, name='safety/benchmark_output/runs/v2/other:model=b/instances.json')
        identity.append(dict(identity[0], path=other['name'].removeprefix('safety/')))
        source['tree_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=[
                io.BytesIO(payload), io.BytesIO(json.dumps({'items': [entry]}).encode()),
                io.BytesIO(json.dumps({'items': [other]}).encode())]):
            artifacts = upstream_artifacts([index, source], ('runs',))
        self.assertEqual(len(artifacts), 2)
        index['sha256'] = '0'*64
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', return_value=io.BytesIO(payload)):
            with self.assertRaisesRegex(SourceDataError, 'index differs'):
                upstream_artifacts([index, source], ('runs',))

    def test_html_index_pins_linked_contents_and_reuses_verified_raw_files(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        page = b'<div class="message">Original &amp; complete transcript</div>'
        index_bytes = (b'<a href="run.html">Run</a><a href="run.html">Duplicate link</a>'
                       b'<a href="https://elsewhere.example/run.html">External</a>'
                       b'<a href="../run.html">Outside prefix</a><a href="style.css">Style</a>')
        index = dict(name='index', url='https://provider.example/release/index.html', revision=None,
                     file='site/index.html', size=len(index_bytes), sha256=hashlib.sha256(index_bytes).hexdigest())
        identity = [dict(path='run.html', size=len(page), digest=hashlib.sha256(page).hexdigest())]
        source = dict(name='logs', url='https://provider.example/release/', revision=None, html_index='index',
                      tree_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      files=[dict(match=r'[^/]+\.html', path='site/{path}')])
        self.metadata['sources']['upstream'] = [index, source]
        validate_benchmark_metadata(self.metadata, path=self.metadata_path)

        def fetch(request, **kwargs):
            self.assertEqual(request.get_header('Accept-encoding'), 'identity')
            return io.BytesIO({index['url']: index_bytes, 'https://provider.example/release/run.html': page}[request.full_url])

        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=fetch) as download:
            artifacts = upstream_artifacts([index, source], ('index', 'logs'))
        self.assertEqual(download.call_count, 2)
        self.assertEqual([a['file'] for a in artifacts], ['site/index.html', 'site/run.html'])
        (self.raw/'site').mkdir()
        (self.raw/'site/index.html').write_bytes(index_bytes)
        target = self.raw/'site/run.html'
        target.write_bytes(page)
        verify_snapshot_file(target, artifacts[1])
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=AssertionError('network')):
            self.assertEqual(upstream_artifacts([index, source], ('index', 'logs'), raw_dir=self.raw), artifacts)
            target.write_bytes(page.replace(b'Original', b'Modified'))
            with self.assertRaisesRegex(SourceDataError, 'pinned tree'):
                upstream_artifacts([index, source], ('logs',), raw_dir=self.raw)
            target.write_bytes(page)
            (self.raw/'site/index.html').write_bytes(index_bytes+b' ')
            with self.assertRaisesRegex(SourceDataError, 'index differs'):
                upstream_artifacts([index, source], ('logs',), raw_dir=self.raw)
        source['files'][0]['path'] = '../outside.html'
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=fetch):
            with self.assertRaisesRegex(SourceDataError, 'unsafe raw destination'):
                upstream_artifacts([index, source], ('logs',))
        del source['tree_sha256']
        with self.assertRaises(BenchmarkMetadataError):
            validate_benchmark_metadata(self.metadata, path=self.metadata_path)


    def test_public_drive_folder_pins_membership_contents_and_cached_inputs(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        root = 'https://drive.google.com/embeddedfolderview?id=root123'
        child = 'https://drive.google.com/embeddedfolderview?id=child456'
        download = 'https://drive.usercontent.google.com/download?id=file789&export=download'
        pages = {root: b'<a href="https://drive.google.com/drive/folders/child456">task &amp; run</a>',
                 child: b'<a href="https://drive.google.com/file/d/file789/view?usp=drive_web">result.json</a>',
                 download: PAYLOAD}
        identity = [dict(path='task & run/result.json', drive_id='file789', size=len(PAYLOAD),
                         digest=hashlib.sha256(PAYLOAD).hexdigest())]
        source = dict(name='runs', url='https://drive.google.com/drive/folders/root123', revision=None,
                      tree_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      files=[dict(match=r'.*\.json', path='drive/{path}')])
        self.metadata['sources']['upstream'] = [source]
        validate_benchmark_metadata(self.metadata, path=self.metadata_path)

        def fetch(request, **kwargs):
            self.assertEqual(request.get_header('Accept-encoding'), 'identity')
            return io.BytesIO(pages[request.full_url])

        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=fetch):
            artifacts = upstream_artifacts([source], ('runs',))
        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact['file'], 'drive/task_x20__x26__x20_run/result.json')
        self.assertEqual(artifact['url'], download)
        target = self.raw / artifact['file']
        target.parent.mkdir(parents=True)
        target.write_bytes(PAYLOAD)
        pages.pop(download)
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=fetch):
            self.assertEqual(upstream_artifacts([source], ('runs',), raw_dir=self.raw), artifacts)
            target.write_bytes(PAYLOAD + b' ')
            with self.assertRaisesRegex(SourceDataError, 'pinned tree'):
                upstream_artifacts([source], ('runs',), raw_dir=self.raw)
            target.write_bytes(PAYLOAD)
            pages[child] = b'<a href="https://drive.google.com/file/d/otherID/view">result.json</a>'
            with self.assertRaisesRegex(SourceDataError, 'pinned tree'):
                upstream_artifacts([source], ('runs',), raw_dir=self.raw)
        del source['tree_sha256']
        self.metadata['sources']['upstream'] = [source]
        with self.assertRaises(BenchmarkMetadataError):
            validate_benchmark_metadata(self.metadata, path=self.metadata_path)

    def test_public_drive_rejects_incomplete_unsafe_and_cyclic_trees(self):
        from scripts.build_measurement_tables.load_source_files import upstream_artifacts
        from urllib.error import HTTPError
        root = 'https://drive.google.com/embeddedfolderview?id=root123'
        source = dict(name='runs', url='https://drive.google.com/drive/folders/root123', revision=None,
                      tree_sha256='0'*64, files=[dict(match='.*', path='drive/{path}')])
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=HTTPError(root, 403, 'Forbidden', None, None)):
            with self.assertRaises(HTTPError):
                upstream_artifacts([source], ('runs',))
        for page, error in [
            (b'<a href="https://drive.google.com/drive/folders/root123">loop</a>', 'cycle'),
            (b'<a href="https://drive.google.com/file/d/file789/view">../outside.json</a>', 'unsafe Drive filename'),
            (b'<a href="https://drive.google.com/file/d/file789/view">same.json</a><a href="https://drive.google.com/file/d/file123/view">same.json</a>', 'duplicate Drive path'),
            (b'<html>Login page instead of a file listing</html>', 'pinned tree'),
        ]:
            with self.subTest(error=error), patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=lambda *a, **k: io.BytesIO(page)):
                with self.assertRaisesRegex(SourceDataError, error):
                    upstream_artifacts([source], ('runs',))
        source['files'][0]['path'] = '../outside.json'
        page = b'<a href="https://drive.google.com/file/d/file789/view">result.json</a>'
        with patch('scripts.build_measurement_tables.load_source_files.urlopen', side_effect=lambda *a, **k: io.BytesIO(page)):
            with self.assertRaisesRegex(SourceDataError, 'unsafe raw destination'):
                upstream_artifacts([source], ('runs',))


if __name__=='__main__': unittest.main()
