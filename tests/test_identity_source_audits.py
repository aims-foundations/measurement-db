"""Reviewed aliases must retain native configuration identity and attribution."""
import copy
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from measurement_db.scripts.curate_benchmarks.native_result_audits import (
    _crow, _kormedmcqa, _subject_registry,
)
from measurement_db.scripts.build_measurement_tables.normalize_evaluation_settings import features_string


def subject(identifier, label, harness, features):
    entry = _subject_registry().get(label, {})
    return dict(subject_id=identifier, display_name=label, harness=harness,
                normalized_name=entry.get('model'), provider=entry.get('company'),
                release_date=entry.get('release_date'), access_date=None,
                harness_version=None, reasoning_effort=None,
                subject_features_extra=features_string(features))


class IdentitySourceAudits(unittest.TestCase):
    def fixture(self, slug):
        tables = dict(items=pd.DataFrame(columns=['item_id']),
                      responses=pd.DataFrame(columns=['response_id']),
                      traces=pd.DataFrame(columns=['response_id', 'trace']))
        if slug == 'crow':
            model = 'CodeLLaMA-13B-Instruct'
            exports = {'a' * 64: dict(model=model), 'b' * 64: dict(model=model)}
            labels = {'CROW / ' + model + ' / result export ' + digest[:12]: model for digest in exports}
            rows = [subject(digest, model, 'CROW', dict(checkpoint='not_recorded',
                        defense='not_recorded', export_sha256=digest, reported_model=model,
                        source_display_name='CROW / ' + model + ' / result export ' + digest[:12]))
                    for digest in exports]
            scale = dict(values=[0, 1], direction='lower_is_better', meanings={'0': 'failure', '1': 'success'})
            tables['benchmarks'] = pd.DataFrame([dict(response_scale=json.dumps(scale))])
            metadata = dict(build=dict(parameters=dict(subject_labels=labels)),
                            benchmark=dict(response_scale=scale))
            audit = lambda data: _crow(Path(slug), data, metadata, (exports, {}, []))
        else:
            raw = 'claude-haiku-4-5-2025-12-11'
            model = 'claude-haiku-4-5'
            rows = [subject('one', model, 'Korean LLM Benchmark Suite',
                            dict(source_model_label=raw, source_display_name=raw,
                                 historical_configuration='not_recorded'))]
            metadata = dict(build=dict(parameters=dict(subject_labels={raw: model},
                subject_features=dict(harness='Korean LLM Benchmark Suite', historical_configuration='not_recorded'))))
            audit = lambda data: _kormedmcqa(Path(slug), data, metadata, ({}, {raw}, {}, {}))
        tables['subjects'] = pd.DataFrame(rows)
        return audit, tables

    def test_aliases_keep_separate_configurations(self):
        for slug, count in [('crow', 2), ('kormedmcqa', 1)]:
            with self.subTest(slug=slug):
                audit, tables = self.fixture(slug)
                self.assertEqual(audit(tables)['source_subjects'], count)

    def test_identity_corruption_is_rejected(self):
        for slug in ['crow', 'kormedmcqa']:
            audit, tables = self.fixture(slug)
            for field in ['display_name', 'provider', 'harness', 'access_date']:
                with self.subTest(slug=slug, field=field):
                    bad = copy.deepcopy(tables)
                    bad['subjects'].loc[0, field] = 'invented'
                    with self.assertRaises(ValueError):
                        audit(bad)
            for replace in [None, 'wrong-source']:
                with self.subTest(slug=slug, alias=replace):
                    bad = copy.deepcopy(tables)
                    value = bad['subjects'].loc[0, 'subject_features_extra']
                    fields = dict(part.split('=', 1) for part in value.split(';'))
                    if replace is None:
                        del fields['source_display_name']
                    else:
                        fields['source_display_name'] = replace
                    bad['subjects'].loc[0, 'subject_features_extra'] = features_string(fields)
                    with self.assertRaises(ValueError):
                        audit(bad)


if __name__ == '__main__':
    unittest.main()
