"""Place completed native reports and patches in the released builder's layout.

No scores are synthesized. Cost-limited or failed executions remain in the run
record and are excluded from graded observations. Full trajectories stay beside
native agent outputs; the table's trace is the complete submitted patch.
"""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
run=a.run.resolve(strict=True);out=run/'native'
result=json.loads((run/'fresh_results.json').read_text())
rows=[r for r in result['tasks'] if r['status']=='graded' and r['response']==r['replayed_response']]
if not rows:raise ValueError('No completed, replay-verified fresh observations')
config=json.loads((run/'procedure/reproduction_checks/luna.json').read_text())
report=json.loads((run/'fresh/outputs/final_report.json').read_text())
expected={r['task'] for r in rows}
normalize=lambda x:x.replace(':pr-','-').replace('/','__')
actual={normalize(x) for x in report['resolved_ids']+report['unresolved_ids']}
if actual!=expected:raise ValueError('Native graded report differs from accepted fresh attempts')
out.mkdir(exist_ok=False);(out/'results').mkdir();(out/'preds').mkdir()
# This pilot's three frozen tasks are all Go; preserve the source records verbatim.
selection=json.loads((run/'procedure/reproduction_checks/selection.json').read_text())
languages={r['source_file'].split('__')[0] for r in selection['selected_tasks']}
if len(languages)!=1:raise ValueError('Export each language report separately')
date=json.loads((run/'result.json').read_text())['started_at'][:10].replace('-','')
language=languages.pop();stem=f'{language}__{date}_MSWE-agent_{config["model"]}'
shutil.copyfile(run/'selected-instances.jsonl',out/f'{language}__selected_dataset.jsonl')
shutil.copyfile(run/'fresh/outputs/final_report.json',out/'results'/(stem+'.json'))
with (out/'preds'/(stem+'.jsonl')).open('w') as target:
 for row in rows:
  for line in Path(row['native_predictions']).read_text().splitlines():
   if json.loads(line)['instance_id']==row['task']:target.write(line+'\n')
settings={config['model']:{'harness':'MSWE-agent','harness_version':'88217624b637646b886cb0462995c07559e96f58','reasoning_effort':config['reasoning_effort'],'pilot_api_cap_usd':config['max_estimated_cost_usd'],'grading_harness_version':'24f493f8a103e72312ded4f6b9c89f081d69cb09'}}
(out/'subject_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
(out/'export.json').write_text(json.dumps({'run':str(run),'included_tasks':sorted(expected),'excluded_tasks':[{'task':r['task'],'status':r['status']} for r in result['tasks'] if r['task'] not in expected],'trace_scope':'Complete submitted patch in tables; full native conversation and actions retained in run.'},indent=2)+'\n')
print(out)
