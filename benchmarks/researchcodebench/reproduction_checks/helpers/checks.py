"""Run upstream prompt construction, code parsing and tests on frozen tasks.

Only model registration/provider transport and selection are custom. Native test
programs, 60-second timeout, and completion insertion remain unchanged. Gold
controls must execute before a fresh answer can be treated as a model failure.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

B = Path(os.environ['REPRO_BENCHMARK_DIR'])
O = Path(os.environ['REPRO_RUN_DIR'])
S = Path(os.environ['REPRO_SOURCE_DIR'])
sys.path.insert(0, str(S))
from bounded_luna import save


def selected():
    return json.loads((B / 'reproduction_checks/selection.json').read_text())['selected_tasks']


def inputs():
    pass  # All selected task code, papers, and tests belong to the pinned checkout.


def task_pset(task):
    from core.annotation.models.pset import PSet
    pset = PSet.parse_pset(str(S / 'pset'), selected_problems=[task['paper']])
    found = 0
    for problem in pset.problems:
        for file in problem.problem_files:
            file.snippets = [s for s in file.snippets if s.name == task['snippet']]
            found += len(file.snippets)
    assert found == 1, f'Expected one native snippet: {task}'
    return pset


def exercise():
    rows = []
    for task in selected():
        task_pset(task)
        folder = S / 'pset' / task['paper']
        # Run the actual tests, not sanity_check_tests.py which ignores exit codes.
        result = subprocess.run(['timeout', '60', sys.executable, 'paper2code_test.py'],
                                cwd=folder, capture_output=True, text=True)
        log = O / ('control-' + task['paper'] + '.json')
        save(log, {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        rows.append({'task': task['task_id'], 'status': 'prepared' if result.returncode == 0 else 'blocked',
                     'control_exit_code': result.returncode, 'control_log': log.name})
    save(O / 'exercise.json', rows)
    if not any(row['status'] == 'prepared' for row in rows):
        raise SystemExit(77)


class StopNativeRetry(BaseException):
    pass


def rerun():
    from chat_budget import ChatGuard
    from core.annotation.utils.run_inference import run_inference
    config = json.loads((B / 'reproduction_checks/luna.json').read_text())
    guard = ChatGuard(config, os.environ['REPRO_BUDGET_FILE'], O / 'provider', B.name,
                      os.environ.pop('OPENAI_API_KEY'))

    class Provider:
        async def run(self, *, llm_type, user_message, num_completions, temperature):
            assert num_completions == 1
            try:
                status, body = guard.complete({'model': config['model'],
                    'messages': [{'role': 'user', 'content': user_message}], 'temperature': temperature})
                if status != 200 or guard.task in guard.exhausted:
                    raise ValueError(f'API interrupted: HTTP {status}; see dollar ledger')
                return [json.loads(body)['choices'][0]['message']['content']]
            except Exception as exc:
                raise StopNativeRetry(str(exc)) from exc

    native = O / 'native'; native.mkdir()
    archive = B / 'raw' / 'researchcodebench-2758001c2ff84fc25c546339d65479ed058b0265.tar.gz'
    shutil.copyfile(archive, native / archive.name)
    save(native / 'subject_settings.json', {config['model']: {'reasoning_effort': 'medium',
        'harness': 'ResearchCodeBench', 'harness_version': '2758001c2ff84fc25c546339d65479ed058b0265',
        'test_timeout_seconds': 60, 'with_paper': True, 'n_completions': 1}})
    prepared = {r['task']: r for r in json.loads((O / 'exercise.json').read_text())}
    rows = []; stats = {'results': {}}
    for i, task in enumerate(selected()):
        guard.task = task['task_id']; before = len(guard.calls)
        row = {'task': guard.task, 'status': 'blocked', 'response': None, 'api_attempts': 0}; rows.append(row)
        if prepared[guard.task]['status'] != 'prepared':
            row['reason'] = 'Unmodified upstream gold control failed; no inference spent on this task.'
        else:
            try:
                pset = task_pset(task); problem = pset.problems[0]
                file = next(f for f in problem.problem_files if f.snippets); snippet = file.snippets[0]
                context = ''.join(f.flatten() for f in problem.context_files)
                masked = context + file.mask_given_snippet(snippet, placeholder_lines=None, remove_markers=True)
                prediction = asyncio.run(run_inference(masked, Path(problem.paper_tex_path).read_text(),
                    llm_type=SimpleNamespace(name='GPT_5_6_LUNA_MEDIUM'), n_completions=1,
                    temperature=0, clients=Provider(), wo_paper=False))
                snippet.predictions = {config['model']: prediction}
                output = native / f'task-{i}.json'
                pset.test_all(str(S / 'pset'), str(O / 'work' / f'test-{i}'), timeout_seconds=60,
                              output_file=str(output))
                completion = snippet.predictions[config['model']].completions[0]
                original = copy.deepcopy(completion.test_result)
                completion.test_result = None
                pset.test_all(str(S / 'pset'), str(O / 'work' / f'replay-{i}'), timeout_seconds=60,
                              output_file=str(native / f'task-{i}-replay.json'))
                replay = completion.test_result
                score = int(original.passed); repeat = int(replay.passed)
                row.update(status='graded' if score == repeat else 'grade_mismatch', response=score,
                           replayed_response=repeat, exit_code=original.exit_code, replay_exit_code=replay.exit_code)
                stats['results'][task['paper']] = {'results': {config['model']: {'results': {
                    task['snippet']: [{'passed': original.passed, 'exit_code': original.exit_code,
                        'completion_idx': 0, 'completion': completion.completion}]}}}}
                save(native / 'overall_stats.json', stats)
            except (StopNativeRetry, Exception) as exc:
                row.update(reason=str(exc), error_type=type(exc).__name__)
        row['api_attempts'] = len(guard.calls) - before
        save(O / 'fresh_results.json', {'tasks': rows, 'estimated_cost_upper_bound_usd': sum(
            c.get('estimated_cost_usd', c['reserved_cost_usd']) for c in guard.calls)})
    guard.client.close()
    if any(r['status'] != 'graded' for r in rows):
        raise SystemExit(77)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
