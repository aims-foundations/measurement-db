"""Native MMDocRAG text inference/judging, with an explicit dollar-stop adapter.

Upstream retries recursively forever. The adapter raises outside Exception on
budget/configuration errors, so those cannot become unbounded paid retries.
It does not change prompts, generations, or the judge's five dimensions.
"""
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

B = Path(os.environ['REPRO_BENCHMARK_DIR'])
O = Path(os.environ['REPRO_RUN_DIR'])
S = Path(os.environ['REPRO_SOURCE_DIR'])
sys.path.insert(0, str(S))
from bounded_luna import save


def inputs():
    (S / 'dataset').mkdir(exist_ok=True)
    for name in ('evaluation_15.jsonl', 'evaluation_20.jsonl'):
        shutil.copyfile(B / 'raw' / name, S / 'dataset' / name)


def selected():
    return json.loads((B / 'reproduction_checks/selection.json').read_text())['selected_tasks']


def exercise():
    os.chdir(S)
    from inference_wrapper import OpenAI_Inference, OpenAI_LLM_Judge
    from data_utils import load_jsonl
    records = load_jsonl('dataset/evaluation_20.jsonl')
    agent = OpenAI_Inference('unused', 'http://127.0.0.1:1/v1', 'gpt-5.6-luna')
    judge = OpenAI_LLM_Judge('unused', 'http://127.0.0.1:1/v1', 20)
    rows = []
    for entry in selected():
        q = int(entry['task_id']); data = records[q]
        assert data['q_id'] == q
        messages = agent.get_text_messages(data['question'], data['text_quotes'], data['img_quotes'])
        assert messages and judge.get_text_messages(q, 'Offline formatting check')
        rows.append({'task': str(q), 'status': 'prepared', 'prompt_characters': len(json.dumps(messages))})
    save(O / 'exercise.json', rows)


class StopNativeRetry(BaseException):
    pass


def rerun():
    os.chdir(S)
    from inference_wrapper import OpenAI_Inference, OpenAI_LLM_Judge
    from data_utils import load_jsonl
    from openai.types.chat import ChatCompletion
    from chat_budget import ChatGuard
    # Use exactly the shared builder's declared grade interpretation.
    import importlib.util
    spec = importlib.util.spec_from_file_location('mmdocrag_build', B / 'build.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = json.loads((B / 'reproduction_checks/luna.json').read_text())
    guard = ChatGuard(config, os.environ['REPRO_BUDGET_FILE'], O / 'provider', B.name,
                      os.environ.pop('OPENAI_API_KEY'))

    def create(**payload):
        try:
            status, body = guard.complete(payload)
            if status != 200:
                raise ValueError(f'Provider HTTP {status}; reservation retained')
            if guard.task in guard.exhausted:
                raise ValueError('Output was truncated by the dollar cap')
            return ChatCompletion.model_validate_json(body)
        except Exception as exc:
            raise StopNativeRetry(str(exc)) from exc

    native_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent = OpenAI_Inference('unused', 'http://127.0.0.1:1/v1', config['model'])
    judge = OpenAI_LLM_Judge('unused', 'http://127.0.0.1:1/v1', 20)
    agent.client.close(); judge.client.close()
    agent.client = judge.client = native_client
    records = load_jsonl('dataset/evaluation_20.jsonl')
    native = O / 'native'; (native / 'resp').mkdir(parents=True); (native / 'eval').mkdir()
    for name in ('evaluation_15.jsonl', 'evaluation_20.jsonl'):
        shutil.copyfile(S / 'dataset' / name, native / name)
    save(native / 'subject_settings.json', {config['model']: {
        'reasoning_effort': 'medium', 'harness': 'MMDocRAG',
        'harness_version': '2fd7505c6a576376b4a92aafaff6d87494765bb7',
        'mode': 'pure-text', 'quotes': 20}})
    rows = []
    for task in selected():
        q = int(task['task_id']); guard.task = str(q); before = len(guard.calls)
        row = {'task': str(q), 'status': 'blocked', 'response': None}; rows.append(row)
        try:
            data = records[q]
            answer = agent.get_api_response(q, data['question'], data['text_quotes'], data['img_quotes'])
            with (native / 'resp' / f'{config["model"]}_pure-text_response_quotes20.jsonl').open('a') as f:
                f.write(json.dumps(answer, ensure_ascii=False) + '\n')
            judgment = judge.get_api_response(q, answer['response'])
            with (native / 'eval' / f'{config["model"]}_pure-text_quotes20_llm-judge.jsonl').open('a') as f:
                f.write(json.dumps(judgment, ensure_ascii=False) + '\n')
            score = module.answer_quality(judgment['response'])
            if score is None:
                raise ValueError('Judge did not provide all five valid dimension grades')
            replay = module.answer_quality(json.loads(json.dumps(judgment))['response'])
            row.update(status='graded', response=score, replayed_response=replay,
                       replay_scope='Reparse saved native judgment; no second stochastic judgment')
        except (StopNativeRetry, Exception) as exc:
            row.update(reason=str(exc), error_type=type(exc).__name__)
        row['api_attempts'] = len(guard.calls) - before
        save(O / 'fresh_results.json', {'tasks': rows, 'estimated_cost_upper_bound_usd': sum(
            c.get('estimated_cost_usd', c['reserved_cost_usd']) for c in guard.calls)})
    guard.client.close()
    if any(row['status'] != 'graded' for row in rows):
        raise SystemExit(77)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
