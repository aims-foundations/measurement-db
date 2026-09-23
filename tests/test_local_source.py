"""Fresh observations keep native inputs and released tables untouched."""
import hashlib,json,sys
from pathlib import Path
import pandas as pd
import pytest,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
# GitHub may name this checkout differently; use the file module directly.
import importlib.util
spec=importlib.util.spec_from_file_location('public_build_base',ROOT/'build_base.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Fixture(module.BenchmarkBuild):
 def build_subject_item_response_rows(self):
  source=json.loads((self.raw_dir/'attempt.json').read_text())
  subject=self.add_subject('fixture-model')
  item=self.add_item(raw_item_id='q',content='Return one.',grading_criterion={'reference_answer':'1'},verifier=module.ExactMatcher(spec='Exact binary match'))
  self.add_response(subject_id=subject,item_id=item,trial=1,response=source['response'],trace=source['trace'])

@pytest.fixture
def inputs(tmp_path,monkeypatch):
 registry=sys.modules['scripts.build_measurement_tables.register_measurements']
 for name in ('_subjects','_items','_benchmarks'):monkeypatch.setattr(registry,name,None)
 bench=tmp_path/'fixture';bench.mkdir()
 (bench/'metadata.yaml').write_text((ROOT/'benchmarks/_template/metadata.yaml').read_text())
 (bench/'formatted_tables').mkdir()
 (bench/'formatted_tables'/'responses.parquet').write_bytes(b'published-file-must-not-change')
 source=tmp_path/'native';source.mkdir()
 (source/'attempt.json').write_text(json.dumps({'response':0,'trace':'a'*20001}))
 (source/'subject_settings.json').write_text(json.dumps({'fixture-model':{'harness':'native','reasoning_effort':'medium'}}))
 return bench,source

def test_fresh_source_preserves_release_and_full_trace(inputs):
 bench,source=inputs;before={p.name:p.read_bytes() for p in source.iterdir()}
 output=source.parent/'formatted_tables'
 Fixture(str(bench/'build.py')).main_from_args(['--source',str(source),'--output',str(output)])
 assert {p.name:p.read_bytes() for p in source.iterdir()}==before
 assert (bench/'formatted_tables'/'responses.parquet').read_bytes()==b'published-file-must-not-change'
 assert {p.stem for p in output.glob('*.parquet')}=={'subjects','items','responses','traces','benchmarks'}
 assert pd.read_parquet(output/'traces.parquet').iloc[0]['trace']=='a'*20001
 assert pd.read_parquet(output/'responses.parquet').iloc[0]['response']==0
 assert pd.read_parquet(output/'subjects.parquet').iloc[0]['reasoning_effort']=='medium'

def test_invalid_grade_is_still_rejected(inputs):
 bench,source=inputs;(source/'attempt.json').write_text(json.dumps({'response':2,'trace':'invalid-grade'}))
 with pytest.raises((ValueError,RuntimeError)):
  Fixture(str(bench/'build.py')).main_from_args(['--source',str(source)])
 assert not list((source.parent/'formatted_tables').glob('*.parquet'))
