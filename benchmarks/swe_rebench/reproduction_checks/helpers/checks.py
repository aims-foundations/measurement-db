"""Invoke the pinned upstream SWE-rebench grading CLI and OpenHands CLI."""
from pathlib import Path
import json,os,subprocess,sys
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR']);W=O/'work'
records=json.loads((O/'selected-instances.json').read_text());ids=[r['instance_id'] for r in records]

def grade(label,predictions):
 directory=O/label;directory.mkdir(exist_ok=True)
 pred='gold'
 if predictions!='gold':
  pred=directory/'predictions.jsonl';pred.write_text(''.join(json.dumps({'instance_id':task,'model_name_or_path':'pilot','model_patch':predictions[task]})+'\n' for task in predictions));pred=str(pred)
 cmd=[str(W/'grader-environment/bin/python'),'-m','swebench.harness.run_evaluation','--dataset_name',str(O/'selected-instances.json'),'--predictions_path',pred,'--max_workers','1','--run_id',label,'--cache_level','instance','--namespace','swerebench','--instance_ids',*ids]
 with (directory/'stdout.log').open('w') as out:result=subprocess.run(cmd,cwd=directory,stdout=out,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'Native grader failed; see {directory}')
 reports=list(directory.glob('*.json'))
 reports=[p for p in reports if p.name!='predictions.jsonl']
 if len(reports)!=1:raise RuntimeError('Expected exactly one native grading report')
 return json.loads(reports[0].read_text())

def exercise():
 gold=grade('gold','gold');empty=grade('empty',{task:'' for task in ids})
 (O/'exercise.json').write_text(json.dumps({'gold':gold,'empty':empty},indent=2)+'\n')

def rerun():
 import runpy
 runpy.run_path(str(Path(__file__).with_name('run_agent.py')),run_name='__main__')

if __name__=='__main__':globals()[sys.argv[1]]()
