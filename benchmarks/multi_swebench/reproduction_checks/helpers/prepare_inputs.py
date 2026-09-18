"""Select complete upstream records without changing their grading artifacts."""
from pathlib import Path
import hashlib, json, os
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR'])
selected=json.loads((B/'reproduction_checks/selection.json').read_text())['selected_tasks'];records=[]
for r in selected:
 path=B/'raw'/r['source_file']
 rows=[json.loads(line) for line in path.open()]
 found=[x for x in rows if (x.get('instance_id') or f"{x['org']}__{x['repo']}-{x['number']}")==r['task_id']]
 if len(found)!=1:raise ValueError('Expected one source record for '+r['task_id'])
 records+=found
(O/'selected-instances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(O/'selected-images.txt').write_text(''.join(f"mswebench/{r['org']}_m_{r['repo']}:pr-{r['number']}\n" for r in records))
(O/'task-source-receipts.json').write_text(json.dumps([{'path':'raw/'+name,'sha256':hashlib.file_digest((B/'raw'/name).open('rb'),'sha256').hexdigest()} for name in sorted({r['source_file'] for r in selected})],indent=2)+'\n')
