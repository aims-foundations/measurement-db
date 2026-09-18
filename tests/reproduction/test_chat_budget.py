"""Offline provider fixtures: preserved requests, persistent budget, no bypass."""
import importlib.util,json,sys
from pathlib import Path
import httpx,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/reproduce_evaluations'))
from chat_budget import ChatGuard,count_payload
from bounded_luna import PilotLimit

@pytest.fixture
def config():
 return json.loads((Path(__file__).resolve().parents[2]/'benchmarks/multi_swebench/reproduction_checks/luna.json').read_text())

def test_preserves_native_content_and_meters_retry(tmp_path,config):
 requests=[]
 def upstream(request):
  payload=json.loads(request.content);requests.append((request.url.path,payload))
  if request.url.path.endswith('input_tokens'):return httpx.Response(200,json={'input_tokens':100})
  return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'done'},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':50}})
 guard=ChatGuard(config,tmp_path/'budget.json',tmp_path/'run','fixture','not-a-real-key',httpx.MockTransport(upstream))
 payload={'model':config['model'],'messages':[{'role':'user','content':'original task'}],'temperature':0,'top_p':.95}
 code,body=guard.complete(payload)
 assert code==200 and json.loads(body)['choices'][0]['message']['content']=='done'
 forwarded=requests[-1][1]
 assert forwarded['messages']==payload['messages']
 assert forwarded['reasoning_effort']=='medium'
 assert 'temperature' not in forwarded and 'top_p' not in forwarded
 assert payload['temperature']==0
 assert guard.calls[0]['estimated_cost_usd']==pytest.approx(.00008)
 assert 'not-a-real-key' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
 guard.client.close()

def test_missing_usage_keeps_reservation_and_blocks_next_dispatch(tmp_path,config):
 paid=[]
 def upstream(request):
  if request.url.path.endswith('input_tokens'):return httpx.Response(200,json={'input_tokens':100})
  paid.append(request);return httpx.Response(500,json={'error':{'message':'fixture'}})
 guard=ChatGuard(config,tmp_path/'budget.json',tmp_path/'run','fixture','fixture',httpx.MockTransport(upstream))
 p={'model':config['model'],'messages':[{'role':'user','content':'task'}]}
 assert guard.complete(p)[0]==500
 with pytest.raises(PilotLimit):guard.complete(p)
 assert len(paid)==1
 guard.client.close()

def test_function_results_and_images_counted_without_changing_input():
 p={'model':'fixture','messages':[{'role':'assistant','content':None,'tool_calls':[{'id':'a','type':'function','function':{'name':'terminal','arguments':'{"command":"pwd"}'}}]},{'role':'tool','tool_call_id':'a','content':'/testbed'},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,fixture','detail':'high'}}]}]}
 value,_=count_payload(p)
 assert value['input'][0]['type']=='function_call'
 assert value['input'][1]['type']=='function_call_output'
 assert value['input'][2]['content'][0]['image_url'].startswith('data:image/png')
 assert 'url' in p['messages'][2]['content'][0]['image_url']

def test_responses_bridge_preserves_tools_and_opaque_reasoning(tmp_path,config):
 config={**config,'provider_interface':'responses'}
 requests=[]
 def upstream(request):
  body=json.loads(request.content);requests.append((request.url.path,body))
  if request.url.path.endswith('input_tokens'):return httpx.Response(200,json={'input_tokens':150})
  return httpx.Response(200,json={'id':'resp_fixture','status':'completed','output':[
   {'type':'reasoning','id':'rs_fixture','summary':[],'encrypted_content':'opaque-fixture'},
   {'type':'function_call','call_id':'call_fixture','name':'shell','arguments':'{"command":"pwd"}'}],
   'usage':{'input_tokens':150,'output_tokens':50}})
 guard=ChatGuard(config,tmp_path/'budget.json',tmp_path/'run','fixture','not-a-real-key',httpx.MockTransport(upstream))
 first={'model':config['model'],'messages':[{'role':'user','content':'original instruction'}],
        'tools':[{'type':'function','function':{'name':'shell','parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}]}
 code,body=guard.complete(first);reply=json.loads(body)['choices'][0]['message']
 assert code==200 and reply['tool_calls'][0]['id']=='call_fixture'
 second={**first,'messages':first['messages']+[reply,{'role':'tool','tool_call_id':'call_fixture','content':'/workspace'}]}
 guard.complete(second)
 forwarded=requests[-1][1]
 assert requests[-1][0]=='/v1/responses'
 assert forwarded['reasoning']=={'effort':'medium'}
 assert forwarded['store'] is False
 assert any(x.get('encrypted_content')=='opaque-fixture' for x in forwarded['input'])
 assert forwarded['input'][-1]=={'type':'function_call_output','call_id':'call_fixture','output':'/workspace'}
 assert guard.calls[-1]['estimated_cost_usd']>0
