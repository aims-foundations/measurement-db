"""Frozen task selection around MathArena's unchanged Runner and grader.

The helper materializes captured problems in the runner's native local layout,
registers Luna, and connects native SDK calls to the shared dollar guard. Four
attempts are evaluated sequentially to avoid competing dollar reservations.
"""
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

B = Path(os.environ['REPRO_BENCHMARK_DIR'])
O = Path(os.environ['REPRO_RUN_DIR'])
S = Path(os.environ['REPRO_SOURCE_DIR'])
from bounded_luna import save

CONFIGS = {'cmimc_2025': 'cmimc/cmimc_2025', 'smt_2025': 'smt/smt_2025',
           'kangaroo_2025_1_2': 'kangaroo/kangaroo_2025_1-2'}


def selected():
    return json.loads((B / 'reproduction_checks/selection.json').read_text())['selected_tasks']


def provider_payload(payload):
    """Translate SDK-only arguments and Responses image blocks to Chat JSON."""
    payload = {k: v for k, v in payload.items() if v is not None and k != 'timeout'}
    messages = []
    for message in payload['messages']:
        message = dict(message)
        if isinstance(message.get('content'), list):
            parts = []
            for part in message['content']:
                if part['type'] == 'input_text':
                    part = {'type': 'text', 'text': part['text']}
                elif part['type'] == 'input_image':
                    part = {'type': 'image_url', 'image_url': {
                        'url': part['image_url'], 'detail': part.get('detail', 'auto')}}
                parts.append(part)
            message['content'] = parts
        messages.append(message)
    payload['messages'] = messages
    return payload


def inputs():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    configs = O / 'work/competitions'; configs.mkdir()
    records = []
    for task in selected():
        comp = task['competition']; idx = int(task['problem_idx'])
        # Source rows only supply the question, image, and reference answer.
        found = None
        for path in sorted((B / 'raw/sources' / comp / 'data').glob('*.parquet')):
            dtype = pq.read_schema(path).field('problem_idx').type
            value = str(idx) if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) else idx
            table = pq.read_table(path, filters=[('problem_idx', '=', value)])
            if len(table):
                found = table.to_pylist()[0]; break
        if found is None:
            raise ValueError(f'Selected source problem absent: {task}')
        data = O / 'work/problems' / comp; (data / 'problems').mkdir(parents=True)
        with (data / 'answers.csv').open('w') as f:
            writer = csv.writer(f); writer.writerow(['id', 'answer']); writer.writerow([idx, found['gold_answer']])
        if found.get('problem'):
            (data / 'problems' / f'{idx}.tex').write_text(found['problem'])
        image = found.get('image')
        if image:
            (data / 'problems' / f'{idx}.png').write_bytes(image['bytes'])
        native_config = yaml.safe_load((S / 'configs/competitions' / (CONFIGS[comp] + '.yaml')).read_text())
        native_config['dataset_path'] = str(data)
        (configs / (comp + '.yaml')).write_text(yaml.safe_dump(native_config, sort_keys=False))
        records.append({'task': task['task_id'], 'source': str(path.relative_to(B)),
                        'problem_idx': idx, 'gold_answer': str(found['gold_answer'])})
    save(O / 'selected-inputs.json', records)
    models = O / 'work/models'; models.mkdir()
    (models / 'luna.yaml').write_text(yaml.safe_dump({'model': 'gpt-5.6-luna', 'api': 'openai',
        'human_readable_id': 'gpt-5.6-luna', 'reasoning_effort': 'medium',
        'base_url': 'http://127.0.0.1:1/v1', 'max_tokens_param': 'max_completion_tokens',
        'read_cost': 0.2, 'write_cost': 1.2}, sort_keys=False))


def runner(task):
    from matharena.runner import Runner
    return Runner(task['competition'], 4, [int(task['problem_idx'])],
                  str(O / 'work/competitions'), str(O / 'work/models'), str(O / 'native-runner'), False)


def exercise():
    from matharena.grader import extract_and_grade
    os.chdir(S); os.environ['OPENAI_API_KEY'] = 'pilot-local-placeholder'
    rows = []
    for task in selected():
        native = runner(task); p = native.problems[0]
        query = native.prepare_run('luna', set_request_metadata=False)['solver'].build_query(
            p['problem'], p['image'])
        messages = query + [{'role': 'assistant', 'content': '\\boxed{' + p['answer'] + '}'}]
        grade = extract_and_grade(messages, 20, p['answer'], native.competition_config, p)
        assert grade[1], 'Native reference-answer control failed'
        rows.append({'task': task['task_id'], 'status': 'prepared', 'gold_control': True})
    save(O / 'exercise.json', rows)


class StopNativeRetry(BaseException):
    pass


def rerun():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from chat_budget import ChatGuard
    from openai.types.chat import ChatCompletion
    import matharena.api_client as api
    from matharena.grader import extract_and_grade
    from matharena.utils import normalize_conversation
    os.chdir(S)
    config = json.loads((B / 'reproduction_checks/luna.json').read_text())
    guard = ChatGuard(config, os.environ['REPRO_BUDGET_FILE'], O / 'provider', B.name,
                      os.environ.pop('OPENAI_API_KEY'))
    os.environ['OPENAI_API_KEY'] = 'pilot-local-placeholder'

    def create(**payload):
        try:
            # Native MathArena also uses Responses-style image blocks on its
            # Chat path. Translate only their wire representation, preserving
            # the exact text, bytes, and image-detail setting.
            payload = provider_payload(payload)
            status, body = guard.complete(payload)
            if status != 200 or guard.task in guard.exhausted:
                raise ValueError(f'API interrupted: HTTP {status}; see dollar ledger')
            return ChatCompletion.model_validate_json(body)
        except Exception as exc:
            raise StopNativeRetry(str(exc)) from exc

    # Replace only SDK transport construction; native API/solver/runner stay in use.
    api.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    native_dir = O / 'native'; native_dir.mkdir()
    save(native_dir / 'subject_settings.json', {'gpt-5.6-luna': {'reasoning_effort': 'medium',
        'harness': 'MathArena', 'harness_version': 'b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b'}})
    rows = []
    for task in selected():
        guard.task = task['task_id']; before = len(guard.calls)
        row = {'task': guard.task, 'status': 'blocked', 'response': None}; rows.append(row)
        records = []
        try:
            native = runner(task); prepared = native.prepare_run('luna'); p = native.problems[0]
            solver = prepared['solver']; idx = int(task['problem_idx'])
            for attempt, statement in enumerate(prepared['batch']):
                responses = solver.solve_batch([statement], {0: idx}, {0: attempt})
                native.process_solver_responses('luna', solver, prepared['all_runs'], {0: idx},
                                                prepared['status_path'], responses, print_final_status=False)
                run = prepared['all_runs'][idx]
                messages = normalize_conversation(run.messages[-1])
                replay = extract_and_grade(messages, run.detailed_costs[-1].get('output_tokens', 0),
                                            p['answer'], native.competition_config, p)
                if bool(replay[1]) != bool(run.correct[-1]):
                    raise ValueError('Saved grade differs from native replay')
                prompt = messages[0]['content']
                record = {'problem_idx': idx, 'problem': p['problem'], 'model_name': 'gpt-5.6-luna',
                    'model_config': 'openai/gpt-5.6-luna--medium', 'idx_answer': attempt,
                    'user_message': json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, list) else prompt,
                    'all_messages': json.dumps(messages, ensure_ascii=False), 'answer': messages[-1]['content'],
                    'gold_answer': p['answer'], 'parsed_answer': str(run.answers[-1]), 'correct': bool(run.correct[-1])}
                records.append(record)
                dest = native_dir / 'sources' / task['competition'] / 'data'; dest.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(records), dest / 'pilot.parquet')
                # The normal builder expects already captured decoded input images.
                if p['image']:
                    payload = base64.b64decode(p['image']); assets = native_dir / 'decoded_images'; assets.mkdir(exist_ok=True)
                    (assets / hashlib.sha256(payload).hexdigest()).write_bytes(payload)
            score = sum(int(r['correct']) for r in records) / len(records)
            row.update(status='graded', response=score, replayed_response=score)
        except (StopNativeRetry, Exception) as exc:
            row.update(reason=str(exc), error_type=type(exc).__name__)
        row.update(api_attempts=len(guard.calls) - before, completed_attempts=len(records), native_attempts=4)
        save(O / 'fresh_results.json', {'tasks': rows, 'estimated_cost_upper_bound_usd': sum(
            c.get('estimated_cost_usd', c['reserved_cost_usd']) for c in guard.calls)})
    guard.client.close()
    if any(r['status'] != 'graded' for r in rows):
        raise SystemExit(77)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
