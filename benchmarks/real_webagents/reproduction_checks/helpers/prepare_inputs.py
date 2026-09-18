"""Freeze the native task definitions and report external environment availability."""
from pathlib import Path
import json, os, urllib.request, urllib.error
B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR']); rows=[]
for task in json.loads(os.environ['REPRO_TASKS']):
 value=json.loads((B/'raw/tasks'/f'{task}.json').read_text()); url=value['website']['url']
 native=Path(os.environ['REPRO_SOURCE_DIR'])/'src/agisdk/REAL/browsergym/webclones/tasks'/f'{task}.json'
 if json.loads(native.read_text())!=value:raise ValueError('Task definition changed: '+task)
 row={'task':task,'url':url,'available':False}
 try:
  with urllib.request.urlopen(url,timeout=30) as r: row.update(http_status=r.status,available=r.status==200)
 except urllib.error.HTTPError as e: row['http_status']=e.code
 except Exception as e:row['error_type']=type(e).__name__
 rows.append(row)
(O/'environment-availability.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
