"""Shared characterization contract: strict claims and faithful comparisons."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import yaml

from scripts.build_measurement_tables.validate_characterization import (
    CharacterizationError, characterize_tables, check_source_claims, check_tables,
    load_characterization, logical_sha256, validate_characterization,
)
from scripts.build_measurement_tables.validate_benchmark_metadata import (
    BenchmarkMetadataError, load_benchmark_metadata, validate_benchmark_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def example():
    return {
        'format_version': 1,
        'tables': {name: {'rows': 1, 'logical_sha256': 'a' * 64}
                   for name in ('subjects', 'items', 'benchmarks')},
        'source_claims': {'released_items': {
            'kind': 'count', 'origin': 'reported', 'source': 'https://arxiv.org/pdf/2504.11543',
            'locator': 'Table 1', 'scope': 'Illustrative test fixture, held-out split', 'expected': 12,
        }},
    }


class SchemaTests(unittest.TestCase):
    def test_every_public_benchmark_has_a_valid_characterization(self):
        paths = [p for p in (ROOT / 'benchmarks').glob('*/metadata.yaml')
                 if not p.parent.name.startswith('_')]
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(benchmark=path.parent.name):
                load_characterization(path.with_name('characterization.yaml'))
                self.assertFalse((path.parent / 'testdata').exists())
                self.assertNotIn('validation', load_benchmark_metadata(path))

    def test_required_fields_and_closed_mappings(self):
        for key in ('format_version', 'tables', 'source_claims'):
            with self.subTest(missing=key):
                value = example(); del value[key]
                with self.assertRaises(CharacterizationError): validate_characterization(value)
        for change in (
            lambda d: d.update(source_claims={}),
            lambda d: d.update(format_version=True),
            lambda d: d.update(format_version=2),
            lambda d: d.update(notes='unknown'),
            lambda d: d['tables'].update(extra={'rows': 0, 'logical_sha256': 'a' * 64}),
            lambda d: d['tables']['items'].update(rows=-1),
            lambda d: d['tables']['items'].update(logical_sha256='unreviewed'),
            lambda d: d['tables']['items'].update(columns=[]),
        ):
            value = example(); change(value)
            with self.assertRaises(CharacterizationError): validate_characterization(value)

    def test_claims_require_checkable_evidence(self):
        for key in ('kind', 'origin', 'source', 'locator', 'scope', 'expected'):
            value = example(); del value['source_claims']['released_items'][key]
            with self.subTest(missing=key), self.assertRaises(CharacterizationError):
                validate_characterization(value)
        for key, invalid in (
            ('source', '/local/file'), ('source', 'https://'), ('locator', '  '),
            ('scope', ''), ('origin', 'guessed'), ('expected', -1), ('expected', True),
            ('expected', 1.5), ('expected', None), ('expected', float('nan')),
            ('expected', float('inf')), ('unknown', 1), ('absolute_tolerance', 1),
        ):
            value = example(); value['source_claims']['released_items'][key] = invalid
            with self.subTest(key=key, value=invalid), self.assertRaises(CharacterizationError):
                validate_characterization(value)

    def test_reported_and_derived_sources_and_aggregate_tolerance(self):
        for origin in ('reported', 'derived_from_data'):
            for url in ('https://arxiv.org/pdf/1234.56789', 'https://example.org/blog/release',
                        'https://github.com/org/data/blob/commit/results.json'):
                value = example()
                value['source_claims']['released_items'].update(origin=origin, source=url)
                validate_characterization(value)
        value = example()
        claim = value['source_claims']['released_items']
        claim.update(kind='aggregate', expected={'model_a': 12.5}, absolute_tolerance=0.01)
        validate_characterization(value)
        for bad in ({}, {'model_a': True}, {'model_a': float('inf')}):
            claim['expected'] = bad
            with self.assertRaises(CharacterizationError): validate_characterization(value)
        claim['expected'] = 0.5
        del claim['absolute_tolerance']
        with self.assertRaises(CharacterizationError): validate_characterization(value)
        for bad in (-1, float('inf')):
            claim['absolute_tolerance'] = bad
            with self.assertRaises(CharacterizationError): validate_characterization(value)

    def test_duplicate_yaml_keys_and_nonstring_keys_fail(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'characterization.yaml'
            path.write_text(yaml.safe_dump(example()) + 'format_version: 1\n')
            with self.assertRaisesRegex(CharacterizationError, 'duplicate key'):
                load_characterization(path)
        value = example(); value['source_claims'][1] = value['source_claims'].pop('released_items')
        with self.assertRaises(CharacterizationError): validate_characterization(value)

    def test_current_metadata_rejects_old_validation_location(self):
        path = ROOT / 'benchmarks/real_webagents/metadata.yaml'
        value = load_benchmark_metadata(path)
        value['validation'] = {'source_claims': {}}
        with self.assertRaisesRegex(BenchmarkMetadataError, 'characterization.yaml'):
            validate_benchmark_metadata(value, path=path)

    def test_cli_fails_for_benchmark_without_characterization(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'example'; folder.mkdir()
            (folder / 'metadata.yaml').touch()
            result = subprocess.run([sys.executable, '-m',
                'scripts.build_measurement_tables.validate_characterization',
                '--benchmarks-dir', tmp], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('characterization.yaml', result.stdout)


class ComparisonTests(unittest.TestCase):
    def test_all_claims_must_be_checked_and_match(self):
        value = example()
        check_source_claims(value, {'released_items': 12})
        for observed in ({}, {'released_items': 12, 'extra': 0}, {'released_items': 11},
                         {'released_items': True}, {'released_items': float('inf')},
                         {'released_items': float('nan')}, {'released_items': 12.1}):
            with self.subTest(observed=observed), self.assertRaises(CharacterizationError):
                check_source_claims(value, observed)

    def test_aggregate_groups_and_absolute_tolerance(self):
        value = example()
        value['source_claims']['released_items'].update(
            kind='aggregate', expected={'a': 25.0, 'b': 12.5}, absolute_tolerance=0.01)
        check_source_claims(value, {'released_items': {'a': 25.005, 'b': 12.5}})
        for observations in ({'a': 25}, {'a': 25, 'b': 12.5, 'c': 1},
                             {'a': 25.02, 'b': 12.5}, {'a': 25, 'b': float('nan')}):
            with self.assertRaises(CharacterizationError):
                check_source_claims(value, {'released_items': observations})

    def test_hash_ignores_row_order_but_preserves_data_and_multiplicity(self):
        frame = pd.DataFrame({'id': ['a', 'b'], 'value': [0.0, None],
                              'payload': [b'abc', b'def'], 'list': [[1, 2], [3, 4]]})
        digest = logical_sha256(frame)
        self.assertEqual(digest, logical_sha256(frame.iloc[::-1]))
        for column, replacement in [('value', 1.0), ('payload', b'ABC'), ('list', [2, 1])]:
            changed = frame.copy(); changed.at[0, column] = replacement
            self.assertNotEqual(digest, logical_sha256(changed))
        self.assertNotEqual(digest, logical_sha256(pd.concat([frame, frame.iloc[:1]])))
        self.assertNotEqual(digest, logical_sha256(frame.rename(columns={'id': 'other'})))
        with TemporaryDirectory() as tmp:
            for compression in ('snappy', 'gzip'):
                path = Path(tmp) / f'{compression}.parquet'
                frame.to_parquet(path, compression=compression)
                self.assertEqual(digest, logical_sha256(pd.read_parquet(path)))

    def test_table_coverage_and_hash_changes(self):
        tables = {name: pd.DataFrame({'id': ['one']}) for name in ('subjects', 'items', 'benchmarks')}
        value = example(); value['tables'] = characterize_tables(tables)
        check_tables(value, tables)
        for observed in ({k: v for k, v in tables.items() if k != 'items'},
                         tables | {'traces': pd.DataFrame()},
                         tables | {'items': pd.DataFrame({'id': ['different']})}):
            with self.assertRaises(CharacterizationError): check_tables(value, observed)


if __name__ == '__main__':
    unittest.main()
