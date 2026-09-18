"""Execute native OpenHands on frozen tasks whose upstream grading controls pass."""
from pathlib import Path
import json,os,subprocess,sys
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR']);W=O/'work'
sys.path.insert(0,str(B.parents[1]/'scripts/reproduce_evaluations'))
from chat_budget import ChatGuard
from bounded_luna import save
import checks
config=json.loads((B/'reproduction_checks/luna.json').read_text())
guard=ChatGuard(config,os.environ['REPRO_BUDGET_FILE'],O/'provider',B.name,os.environ.pop('OPENAI_API_KEY'))
controls=json.loads((O/'exercise.json').read_text());eligible=set(controls['gold']['resolved_ids'])-set(controls['empty']['resolved_ids'])
results=[];predictions={}
with guard.server() as url:
 (W/'source/config.toml').write_text('[llm.pilot]\nmodel = "openai/'+config['model']+'"\napi_key = "pilot-local-placeholder"\nbase_url = "'+url+'"\nreasoning_effort = "medium"\nnative_tool_calling = true\nmax_input_tokens = 1050000\nmax_output_tokens = 128000\n')
 env=dict(os.environ,OPENAI_API_KEY='pilot-local-placeholder',REPRO_SOURCE_DIR=str(W/'source'),REPRO_AGENT_CONFIG=str(B/'reproduction_checks/luna.json'),PYTHONPATH=str(W/'source'),RUNTIME='docker',TOKENIZERS_PARALLELISM='false')
 for record in checks.records:
  task=record['instance_id'];row={'task':task,'response':None,'status':'blocked','api_attempts':0};results.append(row)
  if task not in eligible:row['reason']='The upstream reference-solution/empty-patch grading controls did not separate; no paid inference.'
  else:
   guard.task=task;before=len(guard.calls)
   dataset=W/'datasets'/task/'SWE-rebench';dataset.mkdir(parents=True,exist_ok=True)
   (dataset/'test.jsonl').write_text(json.dumps(record)+'\n')
   output=O/'native-agent'/task
   cmd=[sys.executable,str(B/'reproduction_checks/helpers/model_entry.py'),'--agent-cls','CodeActAgent','--llm-config','llm.pilot','--max-iterations','50','--eval-num-workers','1','--eval-output-dir',str(output),'--eval-n-limit','1','--dataset',str(dataset),'--split','test','--mode','swe']
   save(O/(task+'-command.json'),cmd)
   with (O/(task+'-agent.log')).open('w') as log:proc=subprocess.run(cmd,cwd=W/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
   row.update(exit_code=proc.returncode,api_attempts=len(guard.calls)-before,status='error')
   files=list(output.rglob('output.jsonl'))
   if proc.returncode==0 and len(files)==1:
    native=[json.loads(line) for line in files[0].read_text().splitlines()]
    if len(native)==1 and not native[0].get('error') and isinstance(native[0].get('test_result',{}).get('git_patch'),str):
     predictions[task]=native[0]['test_result']['git_patch'];row.update(status='inferred',native_output=str(files[0]))
   if task in guard.exhausted:row.update(status='budget_exhausted',response=None)
  save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
if predictions:
 first=checks.grade('fresh',predictions);second=checks.grade('replay',predictions)
 for row in results:
  if row['status']!='inferred':continue
  task=row['task']
  if task not in first['completed_ids'] or task not in second['completed_ids']:row['status']='grading_error';continue
  score=int(task in first['resolved_ids']);replay=int(task in second['resolved_ids'])
  row.update(response=score,replayed_response=replay,status='graded' if score==replay else 'grade_mismatch')
save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
if any(r['status']!='graded' for r in results):raise SystemExit(77)
