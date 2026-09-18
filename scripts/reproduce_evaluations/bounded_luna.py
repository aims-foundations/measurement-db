"""Small Responses client with a persistent, shared pilot spending ledger.

Only custom benchmark functions are enabled. Credentials stay in this process;
requests and responses are recorded without HTTP authorization headers.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class PilotLimit(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)


class Budget:
    def __init__(self, path, limit=0.25, max_attempts=108):
        self.path = Path(path)
        self.limit, self.max_attempts = limit, max_attempts
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if data['limit_usd'] != self.limit or data['max_attempts'] != self.max_attempts:
                    raise ValueError('Existing ledger limits differ; do not reset the budget')
            else:
                data = dict(limit_usd=self.limit, max_attempts=self.max_attempts, attempts=[])
            yield data
            save(self.path, data)

    def reserve(self, amount, benchmark, task, run):
        with self.locked() as data:
            used = sum(x.get('estimated_cost_usd', x['reserved_cost_usd']) for x in data['attempts'])
            if used + amount > self.limit or (self.max_attempts is not None and len(data['attempts']) >= self.max_attempts):
                raise PilotLimit('Shared batch budget exhausted before dispatch')
            number = len(data['attempts']) + 1
            data['attempts'].append(dict(number=number, benchmark=benchmark, task=task, run=str(run),
                status='reserved', started_at=now(), reserved_cost_usd=amount))
        return number

    def update(self, number, **changes):
        with self.locked() as data:
            data['attempts'][number - 1].update(changes)


class Luna:
    def __init__(self):
        self.config = json.loads(Path(os.environ['REPRO_AGENT_CONFIG']).read_text())
        self.output = Path(os.environ['REPRO_RUN_DIR'])
        self.benchmark = Path(os.environ['REPRO_BENCHMARK_DIR']).name
        self.budget = Budget(os.environ['REPRO_BUDGET_FILE'], self.config['max_estimated_cost_usd'],
                             self.config['max_total_attempts'])
        self.key = os.environ['OPENAI_API_KEY']
        self.calls, self.task, self.task_calls = [], None, 0

    def begin(self, task):
        self.task, self.task_calls = task, 0

    def reservation(self, payload, raw):
        cfg = self.config
        if len(raw) > cfg['max_request_bytes']:
            raise PilotLimit('Request exceeds pilot input limit')
        rates = cfg['pricing']
        return ((len(raw) + 4096) * rates['input_per_million_usd']
                + cfg['max_output_tokens'] * rates['output_per_million_usd']) / 1e6

    def complete(self, history, tools=None):
        cfg = self.config
        if self.task_calls >= cfg['max_calls_per_task']:
            raise PilotLimit('Per-task call limit reached')
        payload = dict(model=cfg['model'], reasoning={'effort': cfg['reasoning_effort']},
                       input=history, max_output_tokens=cfg['max_output_tokens'], store=False,
                       include=['reasoning.encrypted_content'])
        if tools:
            payload.update(tools=tools, tool_choice='auto')
        raw = json.dumps(payload, ensure_ascii=False).encode()
        rates = cfg['pricing']
        reserve = self.reservation(payload, raw)
        number = self.budget.reserve(reserve, self.benchmark, self.task, self.output)
        self.task_calls += 1
        entry = dict(number=number, task=self.task, status='started', reserved_cost_usd=reserve)
        self.calls.append(entry)
        stem = self.output / f'api-{number:03d}'
        save(stem.with_suffix('.request.json'), payload)
        save(self.output / 'api_usage.json', self.calls)
        request = urllib.request.Request('https://api.openai.com/v1/responses', data=raw,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=cfg['request_timeout_seconds']) as reply:
                response = json.load(reply)
                request_id = reply.headers.get('x-request-id')
            save(stem.with_suffix('.response.json'), response)
            usage = response.get('usage')
            if not usage:
                raise RuntimeError('Missing API usage; reservation retained')
            cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
            cost = ((usage['input_tokens'] - cached) * rates['input_per_million_usd']
                    + cached * rates['cached_input_per_million_usd']
                    + usage['output_tokens'] * rates['output_per_million_usd']) / 1e6
            entry.update(status='completed', usage=usage, estimated_cost_usd=cost,
                         model=response.get('model'), response_id=response.get('id'), request_id=request_id)
            if response.get('status') != 'completed':
                raise PilotLimit('API response incomplete at configured output limit')
            return response
        except urllib.error.HTTPError as exc:
            # Keep diagnostic text only after removing the credential. Never
            # persist Request objects, headers, or unsanitized exception reprs.
            body = exc.read().decode('utf-8', 'replace').replace(self.key, '[REDACTED]')
            entry.update(status='error', http_status=exc.code, error_type=type(exc).__name__, provider_error=body[:1500])
            raise RuntimeError(f'API HTTP {exc.code}; see sanitized usage record') from None
        except Exception as exc:
            entry.update(status='error', error_type=type(exc).__name__)
            raise
        finally:
            entry['elapsed_seconds'] = round(time.monotonic() - started, 3)
            self.budget.update(number, **{k:v for k,v in entry.items() if k != 'number'})
            save(self.output / 'api_usage.json', self.calls)

    def summary(self):
        return dict(api_attempts=len(self.calls), estimated_cost_usd=sum(x.get('estimated_cost_usd', 0) for x in self.calls),
                    unmetered_attempts=sum('usage' not in x for x in self.calls),
                    usage={k:sum(x.get('usage', {}).get(k, 0) for x in self.calls)
                           for k in ('input_tokens', 'output_tokens', 'total_tokens')})


def answer(response):
    return '\n'.join(p['text'] for x in response.get('output', []) if x['type'] == 'message'
                     for p in x.get('content', []) if p['type'] == 'output_text')
