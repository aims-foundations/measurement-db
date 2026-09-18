"""Declare the new model to upstream LiteLLM, then execute native OpenHands."""
import json,os,runpy,sys
from pathlib import Path
source=Path(os.environ['REPRO_SOURCE_DIR']);sys.path.insert(0,str(source))
import litellm
config=json.loads(Path(os.environ['REPRO_AGENT_CONFIG']).read_text())
info={'max_input_tokens':1050000,'max_output_tokens':128000,'input_cost_per_token':config['pricing']['input_per_million_usd']/1e6,'output_cost_per_token':config['pricing']['output_per_million_usd']/1e6,'litellm_provider':'openai','mode':'chat','supports_function_calling':True,'supports_reasoning':True}
litellm.register_model({config['model']:info,'openai/'+config['model']:info})
sys.argv[0]=str(source/'evaluation/benchmarks/swe_bench/run_infer.py')
runpy.run_path(sys.argv[0],run_name='__main__')
