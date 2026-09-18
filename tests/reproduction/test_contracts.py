"""Offline checks for the public pilot contracts and frozen random selections."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/reproduce_evaluations'))
from run_checks import load_contract
from bash_pilot import selection,unchanged

@pytest.mark.parametrize('slug',['swe_rebench','multi_swebench','real_webagents',
                                 'matharena','mmdocrag','researchcodebench'])
def test_contract_and_frozen_selection(slug):
 benchmark=ROOT/'benchmarks'/slug
 contract,_=load_contract(benchmark)
 tasks=selection(benchmark)
 assert len(tasks)==3 and len(set(tasks))==3
 assert set(tasks)==set(contract['profiles']['luna_medium']['tasks'])
 assert (benchmark/'reproduction_checks/check.sh').is_file()
 # Every auxiliary judge must use the same complete pricing contract as the
 # predictor; otherwise its first paid request can fail before accounting.
 from inspect_budget import prices
 config=json.loads((benchmark/'reproduction_checks/luna.json').read_text())
 for model in [config,*config.get('additional_models',{}).values()]:
  assert all(rate >= 0 for rate in prices(model,1000))


def test_raw_modification_never_accepted_as_a_new_capture():
 with pytest.raises(ValueError):unchanged({'source':{'sha256':'before'}},{'source':{'sha256':'after'}},allow_additions=True)
 unchanged({'source':{'sha256':'before'}},{'source':{'sha256':'before'},'new':{'sha256':'captured'}},allow_additions=True)
