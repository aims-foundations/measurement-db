"""Trace display transport bounds and fidelity, independently of converters."""
from copy import deepcopy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.trace_previews import export as previews


def write(path, values, *, marker=True):
    table = pa.Table.from_pylist([{'response_id': key, 'trace': json.dumps(value, ensure_ascii=False)}
                                 for key, value in values.items()])
    if marker:
        table = table.replace_schema_metadata({b'measurement_db.trace_format': b'trace/1'})
    pq.write_table(table, path, use_dictionary=False)


def trace():
    return {'schema':'trace/1', 'converter':'fixture@1',
        'native': {'source':'Huge native DOM Ω'*1200000},
        'steps': [{'type':'message','label':'task','content':'An exact prompt'},
            *[{'type':'llm','agent':f'agent-{i%2}','content':f'Complete answer {i}',
               'toolCalls':[{'id':f'call-{i}','name':'command','arguments':'{}'}],
               'native':{'source_fields':['retained only in canonical source']}}
              for i in range(75)],
            {'type':'tool','output':'😺λ\n'*40000,'exitCode':0}],
        'run':{'exception':'Exact exception Ω'*2000},
        'evaluation':{'tests':[{'name':f'Test {i}','status':'failure' if i%10==0 else 'passed'}
                               for i in range(100)]}}


def test_full_projection_and_search_roundtrip_with_large_native_and_unicode(tmp_path):
    source = tmp_path/'traces.parquet'; output=tmp_path/'preview'
    write(source, {'original-id':trace()}); before=source.read_bytes()
    manifest=previews.export(source,output)
    assert previews.verify(source,output)['rows']==1
    assert source.read_bytes()==before
    assert manifest['max_payload_bytes']<=previews.PAGE_BYTES
    assert sum(manifest['packs'])<len(trace()['native']['source'])/100
    # Reuse is based on exact source bytes AND exporter identity.
    assert previews.export(source,output)==manifest


def test_bound_pages_by_serialized_bytes_and_step_count(tmp_path,monkeypatch):
    monkeypatch.setattr(previews,'PAGE_BYTES',4096)
    monkeypatch.setattr(previews,'PAGE_STEPS',3)
    monkeypatch.setattr(previews,'PART_CHARS',100)
    monkeypatch.setattr(previews,'FIELD_CHARS',100)
    source=tmp_path/'traces.parquet'; output=tmp_path/'preview'
    value={'schema':'trace/1','converter':'fixture@1','steps':[
        {'type':'llm','content':str(i)+'Ω'*100} for i in range(20)]}
    write(source,{'a':value})
    manifest=previews.export(source,output)
    assert manifest['max_payload_bytes']<=4096
    assert previews.verify(source,output)['complete_search_text']


@pytest.mark.parametrize('change',['source','exporter','bytes'])
def test_cache_rejects_stale_source_exporter_and_corrupt_parts(tmp_path,monkeypatch,change):
    source=tmp_path/'traces.parquet';output=tmp_path/'preview'
    value={'schema':'trace/1','converter':'fixture@1','steps':[{'type':'llm','content':'Original'}]}
    write(source,{'stable-id':value});previews.export(source,output)
    if change=='source':
        value['steps'][0]['content']='Changed';write(source,{'stable-id':value})
        with pytest.raises(ValueError,match='cache does not match'):previews.export(source,output)
    elif change=='exporter':
        monkeypatch.setattr(previews,'FIELD_CHARS',1)
        with pytest.raises(ValueError,match='cache does not match'):previews.export(source,output)
    else:
        p=output/'pack-0.bin';data=bytearray(p.read_bytes());data[15]^=1;p.write_bytes(data)
        with pytest.raises(ValueError,match='checksum'):previews.verify(source,output)


def test_unknown_format_and_unmarked_source_never_publish_partial_cache(tmp_path):
    source=tmp_path/'traces.parquet';output=tmp_path/'preview'
    write(source,{'r':{'schema':'trace/2','steps':[]}})
    with pytest.raises(ValueError,match='trace/1'):previews.export(source,output)
    assert not output.exists() and not list(tmp_path.glob('.trace-preview-*'))
    write(source,{'r':{'schema':'trace/1','steps':[]}},marker=False)
    with pytest.raises(ValueError,match='marker'):previews.export(source,output)


def test_empty_trace_stays_empty_and_unavailable_details_stay_absent(tmp_path):
    source=tmp_path/'traces.parquet';output=tmp_path/'preview'
    write(source,{'r':{'schema':'trace/1','converter':'fixture@1','steps':[]}})
    previews.export(source,output)
    assert previews.verify(source,output)['rows']==1
