"""Dollar accounting around Inspect's unmodified OpenAI Responses provider.

This HTTP transport adds no agent loop, prompts, tool policy or time/turn limit.
It counts the input using the provider, reserves funds before each generation
(including SDK retries), and bounds output only by remaining dollars. Requests
without returned usage retain their reservation. Headers/credentials are never
written. Only text and local function tools are supported by this price model.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import time

import httpx

from bounded_luna import Budget, PilotLimit, now, save


class DollarBudget(Budget):
    def __init__(self, path, limit):
        super().__init__(path, limit=limit, max_attempts=None)

    def remove_legacy_attempt_limit(self):
        """Explicit policy migration; never clear attempts or change dollars."""
        import fcntl
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            if data['limit_usd'] != self.limit:
                raise ValueError('Existing dollar limit differs; do not reset spending')
            if data['max_attempts'] is not None:
                data.setdefault('policy_changes', []).append(dict(
                    recorded_at=now(), previous_max_attempts=data['max_attempts'],
                    max_attempts=None, reason='Use upstream limits; retain dollar cap'))
                data['max_attempts'] = None
                save(self.path, data)

    def reserve_generation(self, input_tokens, requested_output, rates, benchmark, task, run):
        with self.locked() as data:
            used = sum(Decimal(str(x.get('estimated_cost_usd', x['reserved_cost_usd'])))
                       for x in data['attempts'])
            input_cost = Decimal(input_tokens) * Decimal(str(rates[0])) / 1_000_000
            output_price = Decimal(str(rates[2])) / 1_000_000
            available = Decimal(str(self.limit)) - used - input_cost
            output_tokens = min(requested_output, int((available / output_price).to_integral_value(rounding=ROUND_FLOOR)))
            # Responses API requires max_output_tokens >= 16.
            if output_tokens < 16:
                raise PilotLimit('API dollar budget exhausted before dispatch')
            number = len(data['attempts']) + 1
            entry = dict(number=number, benchmark=benchmark, task=task, run=str(run),
                         status='reserved', started_at=now(), input_tokens_counted=input_tokens,
                         requested_output_tokens=requested_output, allowed_output_tokens=output_tokens,
                         cost_limited_output=output_tokens < requested_output,
                         reserved_cost_usd=float(input_cost + output_tokens * output_price))
            data['attempts'].append(entry)
        return entry


def prices(config, input_tokens):
    p = config['pricing']
    long = input_tokens > p['long_context_threshold_tokens']
    return (p['input_per_million_usd'] * (p['long_input_multiplier'] if long else 1),
            p['cached_input_per_million_usd'] * (p['long_input_multiplier'] if long else 1),
            p['output_per_million_usd'] * (p['long_output_multiplier'] if long else 1))


class DollarTransport(httpx.AsyncBaseTransport):
    def __init__(self, config, ledger, output, benchmark, transport=None):
        self.config, self.budget = config, DollarBudget(ledger, config['max_estimated_cost_usd'])
        self.output, self.benchmark = Path(output), benchmark
        self.transport = transport or httpx.AsyncHTTPTransport()
        self.task = None
        self.calls = []
        self.exhausted = set()

    def rejected(self, request, message, code=400):
        return httpx.Response(code, request=request, json={
            'error': {'type': 'reproduction_budget_error', 'message': message}})

    async def handle_async_request(self, request):
        # Do not accidentally price another endpoint, model, or hosted tool as Luna text.
        if (request.url.host != 'api.openai.com' or request.method != 'POST'
                or request.url.path not in ('/v1/responses', '/v1/responses/input_tokens')):
            return self.rejected(request, 'Endpoint is outside the recorded price contract')
        payload = json.loads(await request.aread())
        if (payload.get('model') != self.config['model']
                or payload.get('stream') or payload.get('background')
                or payload.get('service_tier') not in (None, 'auto', 'default')
                or any(t['type'] != 'function' for t in payload.get('tools', []))):
            return self.rejected(request, 'Request is outside the recorded price contract')
        if request.url.path.endswith('/input_tokens'):
            return await self.transport.handle_async_request(request)
        if self.task in self.exhausted:
            return self.rejected(request, 'API dollar budget exhausted', 402)
        # Count the same input, tools and conversation state, before reserving.
        fields = ('model', 'input', 'instructions', 'tools', 'tool_choice', 'text',
                  'reasoning', 'parallel_tool_calls', 'previous_response_id', 'conversation', 'truncation')
        count_request = self.request(request, {k: payload[k] for k in fields if k in payload},
                                     path='/v1/responses/input_tokens')
        counted = await self.transport.handle_async_request(count_request)
        await counted.aread()
        if counted.status_code != 200:
            # Preserve native retry handling for transient token-count failures.
            return counted
        input_tokens = counted.json()['input_tokens']
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError('Invalid input-token count')
        rates = prices(self.config, input_tokens)
        requested = payload.get('max_output_tokens') or self.config['provider_max_output_tokens']
        try:
            entry = self.budget.reserve_generation(input_tokens, requested, rates,
                                                   self.benchmark, self.task, self.output)
        except PilotLimit as exc:
            self.exhausted.add(self.task)
            save(self.output / 'budget_stops.json', sorted(self.exhausted))
            return self.rejected(request, str(exc), 402)
        payload['max_output_tokens'] = entry['allowed_output_tokens']
        paid_request = self.request(request, payload)
        self.calls.append(entry)
        stem = self.output / f"api-{entry['number']:03d}"
        save(stem.with_suffix('.request.json'), payload)
        started = time.monotonic()
        try:
            response = await self.transport.handle_async_request(paid_request)
            body = await response.aread()
            entry.update(http_status=response.status_code)
            if response.status_code == 200:
                result = json.loads(body)
                save(stem.with_suffix('.response.json'), result)
                usage = result.get('usage')
                if not usage:
                    raise ValueError('Missing usage; reservation retained')
                tokens, output = usage['input_tokens'], usage['output_tokens']
                cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
                if any(type(v) is not int or v < 0 for v in (tokens, output, cached)) or cached > tokens:
                    raise ValueError('Invalid usage; reservation retained')
                actual_rates = prices(self.config, tokens)
                cost = ((tokens - cached) * actual_rates[0] + cached * actual_rates[1]
                        + output * actual_rates[2]) / 1e6
                entry.update(status='completed', usage=usage, estimated_cost_usd=cost,
                             model=result.get('model'), response_id=result.get('id'),
                             request_id=response.headers.get('x-request-id'))
                if cost > entry['reserved_cost_usd'] + 1e-10:
                    raise ValueError('Usage exceeded reservation; halt and review price contract')
                if result.get('status') == 'incomplete' and entry['cost_limited_output']:
                    self.exhausted.add(self.task)
                    save(self.output / 'budget_stops.json', sorted(self.exhausted))
            else:
                # No response headers or unsanitized provider error bodies are persisted.
                entry['status'] = 'error'
            # Return the original buffered response. Reconstructing it with its
            # compression headers could try to decompress the body twice.
            return response
        except BaseException as exc:
            entry.update(status='error', error_type=type(exc).__name__)
            raise
        finally:
            entry['elapsed_seconds'] = round(time.monotonic() - started, 3)
            self.budget.update(entry['number'], **{k: v for k, v in entry.items() if k != 'number'})
            save(self.output / 'api_usage.json', self.calls)

    @staticmethod
    def request(original, payload, path=None):
        headers = dict(original.headers)
        headers.pop('content-length', None)
        return httpx.Request(original.method, original.url.copy_with(path=path) if path else original.url,
                             headers=headers, json=payload, extensions=original.extensions)

    async def aclose(self):
        await self.transport.aclose()
