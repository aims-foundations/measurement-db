"""Recorded BrowserGym snapshots and agent outputs to trace/1."""
from collections import Counter

from .source_fields import SourceTrace

CONVERTER = 'browsergym@1'


def convert(source):
    out = SourceTrace(source, CONVERTER, json_encoded=isinstance(source, str))
    record = out.source['record']
    if not isinstance(record.get('steps'), list):
        raise TypeError('BrowserGym requires the recorded step list')
    counts = Counter(source_steps=len(record['steps']))
    for number, native in enumerate(record['steps']):
        prefix = ['record', 'steps', number]
        # These snapshots precede the action in the same saved step. The error
        # field explicitly refers to the previous action, including at the end.
        observation = {'type': 'event', 'name': 'browser_observation'}
        if native.get('url') is not None:
            out.project(observation, 'content', prefix + ['url'])
        out.add(observation, browsergym={'step': number, 'kind': 'observation'})
        counts['observations'] += 1
        if native.get('last_action_error'):
            error = {'type': 'event', 'name': 'last_action_error'}
            out.project(error, 'content', prefix + ['last_action_error'])
            out.add(error, browsergym={'step': number, 'kind': 'last_action_error'})
            counts['action_errors'] += 1
        if native.get('action') is None and native.get('reasoning') is None:
            continue
        step = {'type': 'llm'}
        for key, target in [('action', 'content'), ('reasoning', 'reasoning')]:
            if native.get(key) is not None:
                out.project(step, target, prefix + [key], json_text=True)
        if record.get('model') is not None:
            out.project(step, 'model', ['record', 'model'])
        stats = native.get('stats') or {}
        for key, target in [('input_tokens', 'prompt'), ('output_tokens', 'completion')]:
            if stats.get(key) is not None:
                step.setdefault('usage', {})
                out.project(step, ['usage', target], prefix + ['stats', key])
        if stats.get('cost') is not None:
            if stats['cost'] == 0 and any(step.get('usage', {}).values()):
                counts['zero_cost_with_tokens_omitted'] += 1
            else:
                out.project(step, 'cost', prefix + ['stats', 'cost'])
        out.add(step, browsergym={'step': number, 'kind': 'agent'})
        counts['llm'] += 1
    summary = record.get('summary_info') or {}
    for key, target in [('err_msg', 'exception'), ('stack_trace', 'traceback')]:
        if summary.get(key):
            out.project_section('run', target, ['record', 'summary_info', key], json_text=True)
    return out.value(), counts
