"""MJ-Bench's online exports identify alignment in filenames, not folders."""

import contextlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import runpy
import shutil
import sys
import tarfile
import tempfile
import unittest

import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from measurement_db.scripts.curate_benchmarks.native_result_audits import _mj_bench_sources


class MJBenchBuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.raw = self.directory / 'raw'
        benchmark = ROOT / 'benchmarks/mj_bench'
        self.metadata = yaml.safe_load((benchmark / 'metadata.yaml').read_text())
        self.builder = runpy.run_path(str(benchmark / 'build.py'))['MJBench'](str(benchmark / 'build.py'))
        self.builder.raw_dir = self.raw

        def picture(color, format='PNG'):
            output = BytesIO()
            Image.new('RGB', (2, 2), color).save(output, format=format)
            return output.getvalue()

        tasks = self.raw / 'tasks/data/alignment.parquet'
        tasks.parent.mkdir(parents=True)
        pd.DataFrame([dict(caption='Synthetic image pair', info='', label=0,
            image0=dict(path='left.png', bytes=picture('red')),
            image1=dict(path='right.png', bytes=picture('blue')))]).to_parquet(tasks)
        layout = self.metadata['build']['parameters']['layout']
        pickapic = self.raw / layout['pickapic']
        pickapic.parent.mkdir(parents=True)
        pd.DataFrame(columns=['image_0_uid', 'jpg_0', 'image_1_uid', 'jpg_1']).to_parquet(pickapic)
        hpdv2 = self.raw / layout['hpdv2']
        hpdv2.parent.mkdir(parents=True)
        with tarfile.open(hpdv2, 'w:gz') as archive:
            payload = picture('green', 'JPEG')
            member = tarfile.TarInfo('unused.jpg')
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
        self.write_json('author/safety/nsfw/captions_nsfw.json', [dict(
            caption='Unmatched historical input', image_0='missing0.jpg', image_1='missing1.jpg', label_0=1)])

        self.records = {}
        for folder, prediction in [
            ('author/online_result/action', 1),
            ('author/online_result', 1),
            ('author/closesource_result/alignment/action', 0),
        ]:
            path = folder + '/gpt-4-turbo_alignment_number10.json'
            record = dict(caption='Synthetic image pair', image_0_path='left.png', image_1_path='right.png',
                label=0, output_0='Original left rating', output_1='Original right rating', pred=prediction)
            self.records[path] = record
            self.write_json(path, [record])

    def write_json(self, relative, records):
        path = self.raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records))

    def test_online_and_folder_exports_preserve_agreement_and_provenance(self):
        with contextlib.redirect_stdout(StringIO()):
            tables = self.builder.build_tables()
        self.assertEqual(len(tables['responses']), 1)
        self.assertEqual(tables['responses'].iloc[0].response, 1.0)
        self.assertEqual(tables['items'].iloc[0].features['dimension'], 'alignment')
        features = tables['subjects'].iloc[0].features
        self.assertEqual([features[key] for key in ['source_model', 'input_mode', 'response_style', 'declared_scale']],
            ['gpt-4-turbo', 'single_image', 'number', '10'])
        trace = json.loads(tables['traces'].iloc[0].trace)
        self.assertEqual(trace['reported_preferences'], ['0'])
        self.assertEqual({row['source_file']: row['native_record'] for row in trace['source_assessments']}, self.records)

    def test_independent_audit_recognizes_online_alignment_exports(self):
        source = _mj_bench_sources(self.directory, self.metadata)
        self.assertEqual(source['source_occurrences'], 3)
        self.assertEqual(len(source['groups']), 1)
        group = next(iter(source['groups'].values()))
        self.assertEqual(group['dimension'], 'alignment')
        self.assertEqual(group['configuration'], ('gpt-4-turbo', 'single_image', 'number', '10'))
        self.assertEqual(group['preferences'], {'0'})
        self.assertEqual({row['source_file']: row['native_record'] for row in group['assessments']}, self.records)

    def test_unknown_source_family_still_fails(self):
        for path in self.records:
            (self.raw / path).unlink()
        self.write_json('author/online_result/action/unknown.json', [next(iter(self.records.values()))])
        with self.assertRaises(KeyError):
            self.builder.build_tables()
        with self.assertRaisesRegex(ValueError, 'MJ known legacy source family'):
            _mj_bench_sources(self.directory, self.metadata)

    def test_parent_folder_names_do_not_change_dimensions_or_preferences(self):
        self.write_json('author/result/safety/example/example_number_scale10.json', [dict(
            next(iter(self.records.values())), pred=0, output_0='Safety rating')])
        with contextlib.redirect_stdout(StringIO()):
            expected = self.builder.build_tables()
        expected_audit = _mj_bench_sources(self.directory, self.metadata)
        original = self.raw
        # These are legitimate workspace names, not fields in an author release.
        for parent in ['artifacts', 'alignment', 'safety', 'bias', 'online_result']:
            with self.subTest(parent=parent):
                directory = self.directory / parent
                self.builder.raw_dir = directory / 'raw'
                shutil.copytree(original, self.builder.raw_dir)
                with contextlib.redirect_stdout(StringIO()):
                    actual = self.builder.build_tables()
                for name in expected:
                    pd.testing.assert_frame_equal(actual[name], expected[name])
                self.assertEqual(_mj_bench_sources(directory, self.metadata), expected_audit)


if __name__ == '__main__':
    unittest.main()
