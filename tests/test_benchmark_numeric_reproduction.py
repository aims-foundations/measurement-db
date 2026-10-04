"""Guard the native numeric behavior needed to reproduce published tables."""

import json
from pathlib import Path
import pickle
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.db import connect
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


class LawBenchRuntimeTests(unittest.TestCase):
    def test_wrong_interpreter_fails_before_download_or_grading(self):
        cls = runpy.run_path(str(ROOT / 'benchmarks/lawbench/build.py'))['LawBench']
        builder = object.__new__(cls)
        metadata = yaml.safe_load((ROOT / 'benchmarks/lawbench/metadata.yaml').read_text())
        builder.build_parameters = metadata['build']['parameters']
        with patch.object(sys, 'version_info', (3, 11, 17)), patch.object(builder, 'fetch_sources') as fetch:
            for method in [builder.download, builder.build_tables]:
                with self.assertRaisesRegex(ValueError, 'requires Python 3.12.12'):
                    method()
            fetch.assert_not_called()
        with patch.object(sys, 'version_info', (3, 12, 12)), patch.object(builder, 'fetch_sources') as fetch:
            builder.download()
            fetch.assert_called_once_with('upstream')


class MLIPArenaNumericTests(unittest.TestCase):
    def test_combustion_norm_preserves_published_scores_and_trace_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'mlip_arena'
            directory.mkdir()
            metadata = yaml.safe_load((ROOT / 'benchmarks/mlip_arena/metadata.yaml').read_text())
            # Keep the real combustion protocol; provide tiny source banks for
            # the unrelated families that the parser inventories first.
            grading = metadata['grading']['verifiers']['native']
            grading['families'] = {'combustion': grading['families']['combustion']}
            (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
            cls = runpy.run_path(str(ROOT / 'benchmarks/mlip_arena/build.py'))['MLIPArena']
            builder = cls(str(directory / 'build.py'))
            raw = builder.raw_dir
            for name, properties in [
                ('github/benchmarks/wbm_structures.db', {'wbm_id': 'fixture'}),
                ('github/benchmarks/c2db/c2db.db', {'uid': 'fixture'}),
                ('huggingface/stability/random-mixture.db', {}),
            ]:
                path = raw / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with connect(path) as database:
                    database.write(Atoms('H', cell=[1, 1, 1]), **properties)
            path = raw / 'github/benchmarks/mof/classification/input.pkl'
            path.parent.mkdir(parents=True)
            with pd.option_context('future.infer_string', False):
                path.write_bytes(pickle.dumps(pd.DataFrame(columns=['name', 'class', 'structure'])))
            path = raw / 'huggingface/vacancy_migration/fixture-fcc-H1.pkl'
            path.parent.mkdir(parents=True)
            path.write_bytes(pickle.dumps({'asymmetry': 0.0}))
            path = raw / 'huggingface/combustion/H256O128.extxyz'
            path.parent.mkdir(parents=True)
            path.write_text('1\nSynthetic fixture\nH 0 0 0\n')
            cases = {
                'CHGNet': ([3.38188e-05, 2.2049e-06, 1.50085e-05], 3.706518433381925e-05),
                'ORB': ([2866.1156120998, 12383.6556441052, -6059.5385201132], 14081.461319424077),
            }
            for model, (vector, expected) in cases.items():
                path = raw / f'github/benchmarks/combustion/{model}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([dict(method=model, formula='H', energies=[0.0, 1.0],
                                                com_drifts=[vector], steps_per_second=1.0)]))
            tables = builder.build_tables()
            responses = tables['responses'].set_index('response_key').response
            checked = set()
            for row in tables['traces'].itertuples():
                trace = json.loads(row.trace)
                if trace['metric'] != 'com_drift':
                    continue
                vector, expected = cases[trace['model']]
                self.assertEqual(responses[row.response_key], expected)
                self.assertEqual(trace['native_metric'], expected)
                self.assertEqual(trace['native_record']['com_drifts'], [vector])
                checked.add(trace['model'])
            self.assertEqual(checked, set(cases))


if __name__ == '__main__':
    unittest.main()
