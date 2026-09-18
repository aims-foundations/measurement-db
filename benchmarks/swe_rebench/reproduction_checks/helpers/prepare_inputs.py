"""Capture pinned full task records and verify the curated task/grade definitions."""
from pathlib import Path
import hashlib, json, os, shutil
import pandas as pd
from huggingface_hub import hf_hub_download

B=Path(os.environ['REPRO_BENCHMARK_DIR']);O=Path(os.environ['REPRO_RUN_DIR'])
selected=json.loads((B/'reproduction_checks/selection.json').read_text())['selected_tasks']
ids=[r['task_id'] for r in selected]; revision='89cdfbab4ab1bd8f5a658bb212d1b63624f4f881'
records=[];sources=[]
for name in ['data/test-00000-of-00002.parquet','data/test-00001-of-00002.parquet']:
 dst=B/'raw/reproducibility'/f'SWE-rebench-{revision}'/name
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():
  cached=hf_hub_download('nebius/SWE-rebench',name,repo_type='dataset',revision=revision)
  shutil.copyfile(cached,dst)
 frame=pd.read_parquet(dst); records.extend(frame[frame.instance_id.isin(ids)].to_dict('records'))
 sources.append({'url':f'https://huggingface.co/datasets/nebius/SWE-rebench/resolve/{revision}/{name}','path':str(dst.relative_to(B)),'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest(),'size':dst.stat().st_size})
if {r['instance_id'] for r in records} != set(ids) or len(records)!=3:
 raise ValueError('Pinned upstream dataset does not contain the unchanged random selection')
bank=pd.read_parquet(B/'raw/instances.parquet').set_index('instance_id')
def plain(value):
 if hasattr(value,'tolist'):return value.tolist()
 return value
for r in records:
 old=bank.loc[r['instance_id']]
 for key in ('problem_statement','patch','test_patch','FAIL_TO_PASS','PASS_TO_PASS'):
  new,prev=plain(r[key]),plain(old[key])
  if isinstance(new,str) and key in ('FAIL_TO_PASS','PASS_TO_PASS'):new=json.loads(new)
  if new!=prev:raise ValueError(f"Upstream {key} changed for {r['instance_id']}; do not silently change task version")
(O/'selected-instances.json').write_text(json.dumps(records,default=plain,indent=2)+'\n')
(O/'task-source-receipts.json').write_text(json.dumps(sources,indent=2)+'\n')
(O/'selected-images.txt').write_text(''.join(str(bank.loc[task,'docker_image'])+':latest\n' for task in ids))
print('Verified all three task definitions against the curated raw snapshot.')
