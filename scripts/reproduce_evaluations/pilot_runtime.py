"""Small, bounded local-container adapter used by reproducibility pilots.

Provider credentials remain on the host. Containers receive only their task files;
native evaluator inputs are supplied after the agent has stopped.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from bounded_luna import Luna, PilotLimit, save, answer


class Container:
    def __init__(self, image, directory, mounts=(), gpu=None, workdir='/testbed', memory='4g', network='none', capabilities=(), cpus=2, user=None):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.name='measurement-pilot-'+uuid.uuid4().hex[:12]
        self.image=image; self.mounts=mounts; self.gpu=gpu; self.workdir=workdir
        self.memory=memory; self.network=network; self.counter=0; self.capabilities=tuple(capabilities)
        self.cpus=cpus; self.user=user
    def __enter__(self):
        cmd=['docker','run','-d','--name',self.name,'--network',self.network,'--cpus',str(self.cpus),'--memory',self.memory,
             '--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--entrypoint','/bin/sh','-w',self.workdir]
        for capability in self.capabilities:cmd += ['--cap-add', capability]
        if self.user is not None:cmd += ['--user', self.user]
        for src,dst,mode in self.mounts: cmd+=['-v',f'{Path(src).resolve()}:{dst}:{mode}']
        if self.gpu is not None:cmd+=['--gpus',f'device={self.gpu}']
        cmd += [self.image,'-c','sleep infinity']
        subprocess.run(cmd,check=True,capture_output=True,text=True)
        save(self.directory/'container.json',dict(name=self.name,image=self.image,network=self.network,gpu=self.gpu,capabilities=self.capabilities,cpus=self.cpus,user=self.user,memory=self.memory))
        return self
    def run(self, command, timeout=45, limit=10000, check=False, user=None):
        self.counter+=1; stem=self.directory/f'command-{self.counter:03d}'
        stem.with_suffix('.sh').write_text(command+'\n')
        # timeout runs inside the container so an exec timeout does not leave work running.
        cmd=['docker','exec']
        if user is not None:cmd += ['--user', user]
        result=subprocess.run(cmd+[self.name,'timeout','--signal=TERM','--kill-after=5',str(timeout),'bash','-lc',command],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout+20)
        stem.with_suffix('.log').write_text(result.stdout)
        observed=dict(exit_code=result.returncode,output=result.stdout[-limit:])
        save(stem.with_suffix('.json'),observed)
        if check and result.returncode:raise RuntimeError(f'Container command failed ({result.returncode}); see {stem.name}.log')
        return observed
    def put(self, source, destination):
        subprocess.run(['docker','cp',str(source),f'{self.name}:{destination}'],check=True,capture_output=True)
    def get(self, source, destination, required=True):
        result=subprocess.run(['docker','cp',f'{self.name}:{source}',str(destination)],capture_output=True,text=True)
        if required and result.returncode:raise RuntimeError('Container output missing: '+source)
        return result.returncode==0
    def __exit__(self,*args):
        subprocess.run(['docker','rm','-f',self.name],capture_output=True)


def shell_agent(client, task, world, prompt, max_turns=None, checkpoint=None):
    client.begin(task); began=time.monotonic(); trajectory=[]
    history=[dict(role='user',content=prompt+'\nUse the terminal tool to inspect and edit the task files. '
        f'This is a short pilot with at most {client.config["max_calls_per_task"]} model turns. Do not access evaluator internals or hidden answers. '
        'Finish with a brief message when done. Each terminal command has a 45 second timeout.')]
    tool=dict(type='function',name='terminal',description='Execute a bash command in the isolated task container.',strict=True,
              parameters=dict(type='object',properties=dict(command=dict(type='string')),required=['command'],additionalProperties=False))
    if checkpoint:
        history=checkpoint['history'];client.task_calls=checkpoint['calls']
        trajectory=checkpoint.get('trajectory',[])
    stop='finished'
    try:
        for _ in range(max_turns or (client.config['max_calls_per_task']-client.task_calls)):
            if time.monotonic()-began>client.config['task_timeout_seconds']:
                stop='task_time_limit';break
            response=client.complete(history,[tool]);history.extend(response.get('output',[]))
            calls=[x for x in response.get('output',[]) if x['type']=='function_call']
            if not calls:break
            for call in calls:
                command=json.loads(call['arguments'])['command']
                obs=world.run(command)
                trajectory.append(dict(command=command,observation=obs))
                save(world.directory/'trajectory.json',trajectory)
                history.append(dict(type='function_call_output',call_id=call['call_id'],output=json.dumps(obs)))
        else:stop='turn_limit'
    except PilotLimit as exc:stop=str(exc)
    finally:
        save(world.directory/'history.json',history)
        save(world.directory/'agent.json',dict(task=task,stop=stop,elapsed_seconds=time.monotonic()-began,api_attempts=client.task_calls))
    return trajectory
