"""Register Luna's provider metadata, then invoke the unchanged native CLI.

The upstream agent predates this model. Prompts, command parser, tools, agent
loop and stopping behavior remain upstream-owned. HTTP dollar accounting and
reasoning settings are supplied by the explicitly configured local proxy.
"""
from pathlib import Path
import json,os,runpy,sys
source=Path(os.environ['REPRO_SOURCE_DIR']);sys.path.insert(0,str(source))
from sweagent.agent.models import OpenAIModel
config=json.loads(Path(os.environ['REPRO_AGENT_CONFIG']).read_text())
OpenAIModel.MODELS[config['model']]={'max_context':1050000,'cost_per_input_token':config['pricing']['input_per_million_usd']/1e6,'cost_per_output_token':config['pricing']['output_per_million_usd']/1e6}
sys.argv[0]=str(source/'run.py')
runpy.run_path(sys.argv[0],run_name='__main__')
