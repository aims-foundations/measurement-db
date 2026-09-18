"""Run the unchanged MSWE agent through the shared dollar guard."""
from pathlib import Path
import json,os,subprocess,sys
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR']);W=O/'work'
sys.path.insert(0,str(B.parents[1]/'scripts/reproduce_evaluations'))
from chat_budget import ChatGuard
from bounded_luna import save
import checks
config=json.loads((B/'reproduction_checks/luna.json').read_text())
guard=ChatGuard(config,os.environ['REPRO_BUDGET_FILE'],O/'provider',B.name,os.environ.pop('OPENAI_API_KEY'))
results=[];predictions={}
with guard.server() as url:
 env=dict(os.environ,OPENAI_API_KEY='pilot-local-placeholder',OPENAI_API_BASE_URL=url,REPRO_SOURCE_DIR=str(W/'source'),REPRO_AGENT_CONFIG=str(B/'reproduction_checks/luna.json'),PYTHONPATH=str(W/'source'))
 for record in checks.records:
  task=f"{record['org']}__{record['repo']}-{record['number']}";guard.task=task;before=len(guard.calls)
  row={'task':task,'response':None,'status':'error','api_attempts':0};results.append(row)
  # The upstream loader infers language from the input path, matching its release naming convention.
  selection=next(s for s in json.loads((B/'reproduction_checks/selection.json').read_text())['selected_tasks'] if s['task_id']==task)
  language=selection['source_file'].split('__')[0]
  inp=O/f'{language}-{task}.jsonl';inp.write_text(json.dumps(record)+'\n')
  cmd=[sys.executable,str(B/'reproduction_checks/helpers/model_entry.py'),'--model_name',config['model'],'--pr_file',str(inp),'--config_file','config/default.yaml','--cache_task_images','True','--pre_build_all_images','False','--remove_image','False','--max_workers_build_image','1','--max_workers_run_instance','1','--print_config','False','--raise_exceptions','True','--suffix',task]
  save(O/(task+'-command.json'),cmd)
  with (O/(task+'-agent.log')).open('w') as out:
   proc=subprocess.run(cmd,cwd=W/'source',env=env,stdout=out,stderr=subprocess.STDOUT)
  row.update(api_attempts=len(guard.calls)-before,exit_code=proc.returncode)
  files=list((W/'source/trajectories').glob(f'*/**/*{task}/all_preds.jsonl'))
  if proc.returncode==0 and len(files)==1:
   native=[json.loads(s) for s in files[0].read_text().splitlines()]
   pred=next((p for p in native if p['instance_id']==task),None)
   trajectory=json.loads((files[0].parent/(task+'.traj')).read_text())
   native_exit=trajectory.get('info',{}).get('exit_status','')
   row['native_exit_status']=native_exit
   if pred is not None and native_exit.startswith('submitted'):
    predictions[task]=pred.get('model_patch') or '';row.update(status='inferred',native_predictions=str(files[0]))
  if task in guard.exhausted:row.update(status='budget_exhausted',response=None)
  save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
# Native grading runs on exactly the attempted submissions, with no replacement tasks.
selected=[r for r in checks.records if f"{r['org']}__{r['repo']}-{r['number']}" in predictions]
if selected:
 original=checks.records;checks.records=selected
 first=checks.grade('fresh',predictions);second=checks.grade('replay',predictions)
 a=json.loads((first/'outputs/final_report.json').read_text());b=json.loads((second/'outputs/final_report.json').read_text())
 for row in results:
  if row['status']!='inferred':continue
  native_id=row['task'].replace('__','/').rsplit('-',1);native_id=f'{native_id[0]}:pr-{native_id[1]}'
  if native_id not in a['completed_ids'] or native_id not in b['completed_ids']:
   row.update(status='grading_error');continue
  score=int(native_id in a['resolved_ids']);replay=int(native_id in b['resolved_ids'])
  row.update(response=score,replayed_response=replay,status='graded' if score==replay else 'grade_mismatch')
 checks.records=original
save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
if any(r['status']!='graded' for r in results):raise SystemExit(77)
