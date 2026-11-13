"""A real Python-exported source/transport fixture for the TypeScript reader."""
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.trace_previews.export import export, verify, PART_CHARS

output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
text='Ω'*(PART_CHARS-3)+'NEEDLE across chunks'+'😺λ\n'*40000
value={'schema':'trace/1','converter':'fixture@1','native':{'source':'Native-only-large-data'*1000000},
    'steps':[{'type':'llm','content':f'Recorded answer {i}','agent':'planner' if i<65 else 'executor'}
             for i in range(130)]+[{'type':'tool','output':text,'callId':'original-id','exitCode':0}],
    'run':{'cost':0}}
source=output/'traces.parquet'
table=pa.Table.from_pylist([{'response_id':'wire-id','trace':json.dumps(value,ensure_ascii=False)}])
table=table.replace_schema_metadata({b'measurement_db.trace_format':b'trace/1'})
pq.write_table(table,source,use_dictionary=False)
manifest=export(source,output/'preview');verify(source,output/'preview')
descriptor=dict(manifest,source='https://fixture.invalid/source.parquet',revision='a'*40,directory='fixture')
(output/'descriptor.json').write_text(json.dumps(descriptor))
(output/'expected.json').write_text(json.dumps(dict(output=text,steps=131,modelTurns=130,toolResults=1),ensure_ascii=False))
print('Prepared real trace/1 wire fixture with a large native record and paged complete Unicode text.')
