"""Local dollar guard for unmodified upstream Chat Completions clients.

Only the declared model is forwarded. Native messages/tools are preserved;
Luna reasoning configuration replaces unsupported sampling parameters. Token
counting uses the Responses count endpoint plus conservative chat framing/text
allowances. Every dispatch/retry reserves funds, and missing usage retains its
reservation. This is an estimated API-cost guard, not a provider billing cap.
"""
from __future__ import annotations
import contextlib, json, math, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from bounded_luna import PilotLimit, save
from inspect_budget import DollarBudget, prices


def count_payload(payload):
    inputs=[]; text_bytes=0
    for m in payload['messages']:
        content=m.get('content') or ''
        if m['role']=='tool':
            text_bytes+=len((content if isinstance(content,str) else json.dumps(content)).encode())
            inputs.append({'type':'function_call_output','call_id':m['tool_call_id'],'output':content})
        else:
            if isinstance(content,list):
                parts=[]
                for p in content:
                    if p['type']=='text':parts.append({'type':'input_text','text':p['text']});text_bytes+=len(p['text'].encode())
                    elif p['type']=='image_url':parts.append({'type':'input_image',**p['image_url']})
                    else:raise ValueError('Unsupported input modality')
                # Chat image_url.url becomes Responses image_url.
                for p in parts:
                    if p['type']=='input_image':p['image_url']=p.pop('url')
                content=parts
            elif isinstance(content,str):text_bytes+=len(content.encode())
            else:raise ValueError('Unsupported message content')
            if content:inputs.append({'role':m['role'],'content':content})
            for call in m.get('tool_calls',[]):
                f=call['function'];inputs.append({'type':'function_call','call_id':call['id'],**f})
                text_bytes+=len(json.dumps(f,ensure_ascii=False).encode())
    tools=[]
    for tool in payload.get('tools',[]):
        if tool['type']!='function':raise ValueError('Hosted tools are not covered by this budget')
        tools.append({'type':'function',**tool['function']})
    text_bytes+=len(json.dumps(tools,ensure_ascii=False).encode())
    result={'model':payload['model'],'input':inputs}
    if tools:result['tools']=tools
    return result,text_bytes+256*(len(payload['messages'])+len(tools)+1)


class ChatGuard:
    def __init__(self,config,ledger,output,benchmark,api_key,transport=None):
        self.config=config;self.budget=DollarBudget(ledger,config['max_estimated_cost_usd'])
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.benchmark=benchmark;self.task='setup';self.calls=[];self.exhausted=set();self.reasoning_by_call={}
        self.client=httpx.Client(base_url='https://api.openai.com/v1/',headers={'Authorization':'Bearer '+api_key},transport=transport,timeout=600)
    def complete(self,payload):
        active=self.config if payload.get('model')==self.config['model'] else self.config.get('additional_models',{}).get(payload.get('model'))
        if active is None:raise ValueError('Model is outside the declared pilot configuration')
        if payload.get('stream') or payload.get('n',1)!=1:raise ValueError('Pilot guard requires one non-streaming completion')
        if payload.get('service_tier','auto') not in ('auto','default'):raise ValueError('Unsupported service tier')
        payload=dict(payload)
        # Model compatibility, not an agent-loop or prompt change.
        if active.get('reasoning_effort') is not None:
            payload['reasoning_effort']=active['reasoning_effort']
            payload.pop('temperature',None);payload.pop('top_p',None)
        counted,allowance=count_payload(payload)
        use_responses=active.get('provider_interface')=='responses'
        if use_responses:
            # Preserve opaque reasoning items belonging to previous tool calls.
            inputs=[];seen=set()
            for item in counted['input']:
                if item.get('type')=='function_call':
                    for reasoning in self.reasoning_by_call.get(item['call_id'],[]):
                        identity=reasoning.get('id') or json.dumps(reasoning,sort_keys=True)
                        if identity not in seen:inputs.append(reasoning);seen.add(identity)
                inputs.append(item)
            counted['input']=inputs
        counter=self.client.post('responses/input_tokens',json=counted)
        counter.raise_for_status();tokens=counter.json()['input_tokens']
        if type(tokens) is not int or tokens<0:raise ValueError('Invalid token count')
        reserved_tokens=tokens+allowance
        requested=payload.pop('max_tokens',None) or payload.get('max_completion_tokens') or active['provider_max_output_tokens']
        entry=self.budget.reserve_generation(reserved_tokens,requested,prices(active,reserved_tokens),self.benchmark,self.task,self.output)
        payload['max_completion_tokens']=entry['allowed_output_tokens']
        self.calls.append(entry);stem=self.output/f"api-{entry['number']:04d}"
        save(stem.with_suffix('.request.json'),payload);started=time.monotonic()
        try:
            endpoint='chat/completions';forwarded=payload
            if use_responses:
                endpoint='responses'
                forwarded={**counted,'max_output_tokens':entry['allowed_output_tokens'],
                           'reasoning':{'effort':active['reasoning_effort']},
                           'store':False,'include':['reasoning.encrypted_content']}
                for field in ('tool_choice','parallel_tool_calls'):
                    if field in payload:
                        choice=payload[field]
                        if field=='tool_choice' and isinstance(choice,dict) and choice.get('type')=='function':
                            choice={'type':'function','name':choice['function']['name']}
                        forwarded[field]=choice
                save(stem.with_suffix('.responses-request.json'),forwarded)
            r=self.client.post(endpoint,json=forwarded);entry['http_status']=r.status_code
            if r.status_code==200:
                value=r.json();save(stem.with_suffix('.response.json'),value)
                if use_responses:
                    if value.get('status') not in ('completed','incomplete'):raise ValueError('Provider did not produce a completed or token-limited response')
                    reasoning=[o for o in value.get('output',[]) if o.get('type')=='reasoning']
                    calls=[o for o in value.get('output',[]) if o.get('type')=='function_call']
                    for call in calls:self.reasoning_by_call[call['call_id']]=reasoning
                    native_usage=value.get('usage')
                    if not native_usage:raise ValueError('Missing usage; reservation retained')
                    usage={'prompt_tokens':native_usage['input_tokens'],'completion_tokens':native_usage['output_tokens'],
                           'total_tokens':native_usage['input_tokens']+native_usage['output_tokens'],
                           'prompt_tokens_details':native_usage.get('input_tokens_details',{}),
                           'completion_tokens_details':native_usage.get('output_tokens_details',{})}
                    content=''.join(part.get('text','') for o in value.get('output',[]) if o.get('type')=='message' for part in o.get('content',[]) if part.get('type')=='output_text')
                    message={'role':'assistant','content':content or None}
                    if calls:message['tool_calls']=[{'id':c['call_id'],'type':'function','function':{'name':c['name'],'arguments':c['arguments']}} for c in calls]
                    reason='length' if value.get('status')=='incomplete' else 'tool_calls' if calls else 'stop'
                    value={'id':value['id'],'object':'chat.completion','created':value.get('created_at',int(time.time())),
                           'model':payload['model'],'choices':[{'index':0,'message':message,'finish_reason':reason}], 'usage':usage}
                u=value.get('usage')
                if not u:raise ValueError('Missing usage; reservation retained')
                inp,out=u['prompt_tokens'],u['completion_tokens'];cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
                if any(type(v) is not int or v<0 for v in (inp,out,cached)) or cached>inp:raise ValueError('Invalid usage')
                rates=prices(active,inp);cost=((inp-cached)*rates[0]+cached*rates[1]+out*rates[2])/1e6
                if cost>entry['reserved_cost_usd']+1e-10:raise ValueError('Usage exceeds reservation; stop and review price contract')
                entry.update(status='completed',estimated_cost_usd=cost,usage=u)
                if any(c.get('finish_reason')=='length' for c in value.get('choices',[])) and entry['cost_limited_output']:self.exhausted.add(self.task)
            else:entry['status']='error'
            return r.status_code,json.dumps(value).encode() if r.status_code==200 and use_responses else r.content
        except BaseException as exc:
            entry.update(status='error',error_type=type(exc).__name__);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            self.budget.update(entry['number'],**{k:v for k,v in entry.items() if k!='number'})
            save(self.output/'api_usage.json',self.calls)
    @contextlib.contextmanager
    def server(self):
        guard=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                status=400;body=b'{"error":{"message":"Unsupported pilot request"}}'
                try:
                    if self.path!='/v1/chat/completions':raise ValueError('Unsupported endpoint')
                    size=int(self.headers['Content-Length']);payload=json.loads(self.rfile.read(size))
                    if guard.task in guard.exhausted:raise PilotLimit('Pilot API budget exhausted')
                    status,body=guard.complete(payload)
                except PilotLimit:
                    guard.exhausted.add(guard.task);status=402;body=b'{"error":{"message":"Pilot API dollar budget exhausted"}}'
                except Exception as exc:
                    save(guard.output/'provider_error.json',{'task':guard.task,'error_type':type(exc).__name__,'message':str(exc) if isinstance(exc,ValueError) else 'See HTTP status in API usage; no credentials recorded.'})
                    status=400;body=b'{"error":{"message":"Pilot provider configuration or accounting error"}}'
                self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:yield f'http://127.0.0.1:{server.server_port}/v1'
        finally:server.shutdown();server.server_close();thread.join();self.client.close()
