"""Call the upstream Multi-SWE-bench grader; no replacement grading rules."""
from pathlib import Path
import json, os, subprocess, sys
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR']);W=O/'work'
records=[json.loads(line) for line in (O/'selected-instances.jsonl').read_text().splitlines()]

def grade(label,patches):
 directory=O/label;directory.mkdir(exist_ok=True)
 [ (directory/name).mkdir(exist_ok=True) for name in ('workdir','outputs','repos','logs') ]
 patch_file=directory/'patches.jsonl';patch_file.write_text(''.join(json.dumps({'org':r['org'],'repo':r['repo'],'number':r['number'],'fix_patch':patches[f"{r['org']}__{r['repo']}-{r['number']}"]})+'\n' for r in records))
 config={'mode':'evaluation','workdir':str(directory/'workdir'),'patch_files':[str(patch_file)],'dataset_files':[str(O/'selected-instances.jsonl')],'force_build':False,'output_dir':str(directory/'outputs'),'specifics':[],'skips':[],'repo_dir':str(directory/'repos'),'need_clone':False,'global_env':[],'clear_env':True,'stop_on_error':True,'max_workers':1,'max_workers_build_image':1,'max_workers_run_instance':1,'log_dir':str(directory/'logs'),'log_level':'INFO'}
 path=directory/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
 with (directory/'stdout.log').open('w') as out:
  result=subprocess.run([sys.executable,'-m','multi_swe_bench.harness.run_evaluation','--config',str(path)],cwd=W/'grader',stdout=out,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'Native grading failed; inspect {directory}')
 return directory

def exercise():
 paths={}
 for label,patches in [('gold',{f"{r['org']}__{r['repo']}-{r['number']}":r['fix_patch'] for r in records}),('empty',{f"{r['org']}__{r['repo']}-{r['number']}":'' for r in records})]:
  paths[label]=str(grade(label,patches))
 (O/'controls.json').write_text(json.dumps(paths,indent=2)+'\n')
 gold=json.loads((Path(paths['gold'])/'outputs/final_report.json').read_text())
 empty=json.loads((Path(paths['empty'])/'outputs/final_report.json').read_text())
 expected={f"{r['org']}/{r['repo']}:pr-{r['number']}" for r in records}
 if set(gold['resolved_ids'])!=expected or empty['resolved_ids'] or gold['error_ids'] or empty['error_ids']:
  raise ValueError('Native reference-solution and empty-patch controls did not separate')

def rerun():
 import runpy
 runpy.run_path(str(Path(__file__).with_name('run_agent.py')),run_name='__main__')

if __name__=='__main__':globals()[sys.argv[1]]()
