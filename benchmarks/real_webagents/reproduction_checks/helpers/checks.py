"""Exercise native browser reset and record unavailable tasks without replacement."""
from pathlib import Path
import json, os, sys, traceback
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR'])
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(O/'work/browsers')

def exercise():
 import gymnasium as gym
 import agisdk.REAL.browsergym.webclones
 results=[]
 for available in json.loads((O/'environment-availability.json').read_text()):
  task=available['task'];row={'task':task,'status':'blocked','response':None,'api_attempts':0,**available};results.append(row)
  if available['available']:
   env=None
   try:
    env=gym.make('browsergym/webclones.'+task,headless=True)
    obs,info=env.reset(seed=20260918)
    row.update(status='prepared',observation_keys=sorted(obs.keys()))
    # Confirm that the native state endpoint works before paying for an agent.
    state=env.unwrapped.task.get_finish_json()
    (O/(task+'-initial-state.json')).write_text(json.dumps(state,indent=2)+'\n')
   except Exception as exc:
    row.update(status='blocked',error_type=type(exc).__name__,reason=str(exc));traceback.print_exc()
   finally:
    if env:env.close()
  else:row['reason']='Upstream environment unavailable; no alternate host or replacement task used.'
  (O/'exercise.json').write_text(json.dumps(results,indent=2)+'\n')
 if not any(r['status']=='prepared' for r in results):raise SystemExit(77)



def rerun():
 sys.path.insert(0,str(B.parents[1]/'scripts/reproduce_evaluations'))
 from chat_budget import ChatGuard
 from bounded_luna import save
 from agisdk import REAL
 from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
 config=json.loads((B/'reproduction_checks/luna.json').read_text())
 key=os.environ.pop('OPENAI_API_KEY')
 guard=ChatGuard(config,os.environ['REPRO_BUDGET_FILE'],O/'provider',B.name,key)
 results=[];captures=[];native_evaluate=WebCloneEvaluator.evaluate
 def record_evaluate(self,env_state=None,model_response=None):
  result=native_evaluate(self,env_state,model_response)
  captures.append({'task':guard.task,'env_state':env_state,'model_response':model_response,'result':result})
  save(O/'native-grading-inputs.json',captures)
  return result
 WebCloneEvaluator.evaluate=record_evaluate
 with guard.server() as url:
  os.environ['OPENAI_API_KEY']='pilot-local-placeholder';os.environ['OPENAI_BASE_URL']=url
  for prior in json.loads((O/'exercise.json').read_text()):
   task=prior['task'];row={'task':task,'status':'blocked','response':None,'api_attempts':0};results.append(row)
   if prior['status']!='prepared':row['reason']=prior.get('reason','Environment not prepared')
   else:
    guard.task=task;before=len(guard.calls)
    try:
     harness=REAL.harness(model=config['model'],task_name='webclones.'+task,headless=True,results_dir=str(O/'native-agent'),use_cache=False,leaderboard=False)
     value=harness.run();save(O/(task+'-native-result.json'),value)
     record=value['webclones.'+task]
     row['native_record']=record
     relevant=[c for c in captures if c['task']==task]
     if not record.get('err_msg') and not relevant and record.get('truncated') and record.get('completed'):
      row.update(status='native_step_limit',response=int(record['score']>0),
                 reason='Native maximum steps reached without submitting an answer; submission grader was not exercised.')
      row['api_attempts']=len(guard.calls)-before
      save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
      continue
     if record.get('err_msg') or not relevant:raise ValueError('No completed native grade; inspect saved experiment')
     saved=relevant[-1]
     # Replay the native evaluator on saved terminal state/answer, under the same cap.
     from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
     replay=native_evaluate(WebCloneEvaluator(TaskConfig(task)),saved['env_state'],saved['model_response'])
     score=int(saved['result'][0]>0);repeated=int(replay[0]>0)
     row.update(response=score,replayed_response=repeated,status='graded' if score==repeated else 'grade_mismatch')
    except Exception as exc:
     row.update(status='error',error_type=type(exc).__name__,reason=str(exc));traceback.print_exc()
    row['api_attempts']=len(guard.calls)-before
    if task in guard.exhausted:row.update(status='budget_exhausted',response=None)
   save(O/'fresh_results.json',{'tasks':results,'estimated_cost_upper_bound_usd':sum(c.get('estimated_cost_usd',c['reserved_cost_usd']) for c in guard.calls)})
 if any(r['status']!='graded' for r in results):raise SystemExit(77)

if __name__=='__main__':globals()[sys.argv[1]]()
