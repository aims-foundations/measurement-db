"""Tabulate original tau-bench attempts and attributed HAL run exports."""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import ConvertedTrace, ExactMatcher, StructuredTraceBuild
from measurement_db.scripts.build_measurement_tables.trace_converters import chat


class TauBench(StructuredTraceBuild):
    def download(self):
        return self.fetch_sources('*')

    @staticmethod
    def _hal_messages(source):
        """Group saved HAL calls only when their recorded contexts agree.

        Provider wrappers are deduplicated by response ID. Distinct planner,
        executor or retry contexts remain separate; no conversation is guessed.
        The original logs and the source of every displayed reply stay in native.
        """
        def model(value):
            return value.removeprefix('openai/') if isinstance(value, str) else value

        def comparable(message):
            value = {key: deepcopy(message[key]) for key in
                     ['role', 'content', 'tool_calls', 'tool_call_id', 'name']
                     if message.get(key) is not None}
            content = value.get('content')
            if isinstance(content, list):
                parts = [block.get('text', block.get('content')) if isinstance(block, dict) else block
                         for block in content]
                if all(isinstance(part, str) for part in parts):
                    value['content'] = '\n'.join(parts)
            if not value.get('tool_calls'):
                value.pop('tool_calls', None)
            for call in value.get('tool_calls', []):
                arguments = (call.get('function') or {}).get('arguments')
                if isinstance(arguments, str):
                    try:
                        call['function']['arguments'] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
            return value

        def compatible(reply, retained):
            left, right = comparable(reply), comparable(retained)
            proposed, used = left.pop('tool_calls', []), right.pop('tool_calls', [])
            return left == right and (not proposed or not used or proposed[:len(used)] == used)

        expected = model(source['configuration']['agent_args']['model_name'])
        calls = {}
        for index, log in enumerate(source['logs']):
            inputs, output = log.get('inputs') or {}, log.get('output') or {}
            if not isinstance(output, dict) or not isinstance(inputs.get('messages'), list):
                continue
            if expected not in {model(inputs.get('model')), model(output.get('model'))}:
                continue
            choices = output.get('choices') or []
            reply = choices[0].get('message') if choices else None
            if not isinstance(reply, dict):
                continue
            identity = output.get('id') or ('log', index)
            previous = calls.get(identity)
            # Outer records retain the harness policy roles; provider clients
            # sometimes rewrite those roles for reasoning-model APIs.
            if previous is None or ('/op/litellm.completion:' in log.get('op_name', '')
                                    and '/op/litellm.completion:' not in previous[1].get('op_name', '')):
                calls[identity] = (index, log, reply)
        remaining = list(calls.values())
        contexts = []
        while remaining:
            chosen = max(remaining, key=lambda row: (len(row[1]['inputs']['messages']), row[1].get('started_at') or ''))
            messages = deepcopy(chosen[1]['inputs']['messages']) + [deepcopy(chosen[2])]
            normalized = [comparable(m) for m in messages]
            members = [row for row in remaining
                       if [comparable(m) for m in row[1]['inputs']['messages']] ==
                          normalized[:len(row[1]['inputs']['messages'])]
                       and compatible(row[2], messages[len(row[1]['inputs']['messages'])])]
            if not members:
                raise ValueError('A HAL call cannot be associated with its own saved context')
            # Some clients emit empty IDs, while the harness keeps only the
            # executed call in the next input history. That recorded history
            # supplies the association even if the full proposal had many calls.
            pending, tool_sources = [], []
            for position, message in enumerate(messages):
                if message.get('role') == 'assistant':
                    pending.extend((position, i, call) for i, call in enumerate(message.get('tool_calls') or []))
                if message.get('role') == 'tool':
                    matches = [(i, call) for i, call in enumerate(pending)
                               if (not message.get('tool_call_id') or call[2].get('id') == message['tool_call_id'])
                               and (not message.get('name') or call[2].get('function', {}).get('name') == message['name'])]
                    if len(matches) == 1:
                        origin, ordinal, call = pending.pop(matches[0][0])
                        if not message.get('tool_call_id') and 'tool_call' not in message:
                            message['tool_call'] = deepcopy(call)
                            tool_sources.append(dict(message=position, assistant=origin, call=ordinal))
            evidence = []
            for index, log, reply in members:
                position = len(log['inputs']['messages'])
                # Retain harness-parsed calls when the provider returned text;
                # retain every proposed call when the harness executed a prefix.
                target = messages[position]
                target.update({key: deepcopy(value) for key, value in reply.items()
                               if value is not None and not (key == 'tool_calls' and not value)})
                for key in ['model', 'usage']:
                    if log['output'].get(key) is not None:
                        target[key] = deepcopy(log['output'][key])
                for key in ['started_at', 'ended_at']:
                    if log.get(key) is not None:
                        target[key] = log[key]
                evidence.append(dict(message=position, log=index))
            contexts.append(dict(messages=messages, history_log=chosen[0], replies=evidence, tool_sources=tool_sources,
                                 started_at=chosen[1].get('started_at') or ''))
            assigned = {row[0] for row in members}
            remaining = [row for row in remaining if row[0] not in assigned]
        contexts.sort(key=lambda row: row['started_at'])
        flattened, evidence = [], []
        for number, context in enumerate(contexts, 1):
            messages = context.pop('messages')
            context.pop('started_at')
            context['offset'] = len(flattened)
            if len(contexts) > 1:
                for message in messages:
                    message['agent'] = f'HAL context {number}'
            flattened.extend(messages)
            evidence.append(context)
        # Preserve a terminating simulator reply only when its recorded input
        # explicitly quotes the final agent reply.
        if flattened and isinstance(flattened[-1].get('content'), str) and flattened[-1]['content']:
            tails = []
            for index, log in enumerate(source['logs']):
                inputs, output = log.get('inputs') or {}, log.get('output') or {}
                history = inputs.get('messages') or []
                if not isinstance(output, dict) or model(inputs.get('model')) == expected or not history:
                    continue
                choices = output.get('choices') or []
                reply = choices[0].get('message') if choices else None
                if (isinstance(reply, dict) and history[-1].get('role') == 'user'
                        and history[-1].get('content') == flattened[-1]['content']):
                    tails.append((log.get('started_at') or '', index, reply))
            if tails:
                _, index, reply = max(tails)
                message = dict(role='user', content=reply.get('content'))
                if flattened[-1].get('agent'):
                    message['agent'] = flattened[-1]['agent']
                flattened.append(message)
                evidence[-1]['simulator_log'] = index
        return flattened, evidence

    def convert_trace(self, source, *, context):
        native = {'source': deepcopy(source)}
        result = source['result']
        run = {'exception': result} if isinstance(result, str) else {}
        if isinstance(result, dict) and 'traj' in result:
            messages = result['traj']
            del native['source']['result']['traj']
        else:
            messages, evidence = self._hal_messages(source)
            native['hal_message_sources'] = evidence
            run['apiCalls'] = sum(len(group['replies']) for group in evidence)
        model = source['configuration'].get('agent_args', {}).get('model_name') or source['configuration'].get('model')
        value, counts = chat.convert(messages, context=context, user_participant='simulated_user',
                                     native=native, run=run, model=model)
        return ConvertedTrace(value, counts)

    def build_tables(self):
        parameters, labels = self.build_parameters, self.build_parameters['labels']

        # 1. Read complete original Sierra trajectories directly into a table.
        histories = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['sierra_runs'])):
            records = json.loads(path.read_text())
            frame = pd.json_normalize(records, max_level=0)
            frame['native_record'] = records
            frame['source_file'] = str(path.relative_to(self.raw_dir))
            frame['source_row'] = range(len(frame))
            frame['domain'] = parameters['sierra_domains'][path.name]
            frame['configuration'] = [dict(collection='sierra', model=parameters['sierra_models'][path.name],
                agent=labels['sierra_agent'], historical_settings=labels['historical_settings'])] * len(frame)
            frame['model'] = parameters['sierra_models'][path.name]
            frame['task'] = frame['info'].map(lambda info: info['task'])
            frame['source_trial'] = frame['trial']
            frame['logs'] = [[] for _ in range(len(frame))]
            frame['task_definition_source'] = frame['source_file']
            histories.append(frame)
        sierra = pd.concat(histories, ignore_index=True)
        sierra['collection'] = 'sierra'

        # 2. Decode public HAL exports in memory, preserving the encrypted raw files.
        exports = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['hal_runs'])):
            with ZipFile(path) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError('A HAL export has an unexpected archive layout')
                envelope = json.loads(archive.read(archive.namelist()[0]))
            key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                salt=base64.b64decode(envelope['salt']), iterations=int(parameters['hal_encoding']['iterations']))
            cipher = Fernet(base64.urlsafe_b64encode(key.derive(parameters['hal_encoding']['public_password'].encode())))
            run = json.loads(cipher.decrypt(base64.b64decode(envelope['encrypted_data'])))
            frame = pd.DataFrame(run['raw_eval_results'].items(), columns=['task_id', 'native_record'])
            frame['source_file'] = str(path.relative_to(self.raw_dir))
            frame['source_row'] = range(len(frame))
            frame['domain'] = 'airline'
            frame['configuration'] = [{key: value for key, value in run['config'].items()
                if key not in {'date', 'run_id'}}] * len(frame)
            frame['run_config'] = [run['config']] * len(frame)
            frame['model'] = run['config']['agent_args']['model_name']
            frame['reward'] = frame.native_record.map(lambda record: record.get('reward') if isinstance(record, dict) else None)
            frame['task'] = frame.native_record.map(lambda record: record.get('task') if isinstance(record, dict) else None)
            logs = pd.DataFrame(run['raw_logging_results'])
            logs['record'] = run['raw_logging_results']
            logs['task_id'] = logs.weave_task_id.astype(str)
            if not logs.task_id.isin(frame.task_id).all():
                raise ValueError('A HAL trace cannot be associated with an observed attempt')
            grouped = logs.groupby('task_id', sort=False).record.agg(list).rename('logs')
            frame = frame.merge(grouped, on='task_id', how='left', validate='one_to_one')
            frame['logs'] = frame.logs.map(lambda value: value if isinstance(value, list) else [])
            exports.append(frame)
        hal = pd.concat(exports, ignore_index=True)
        hal['collection'] = 'hal'
        hal['source_trial'] = None

        # 3. Check the shared HAL task bank before associating error-only records.
        bank = hal.loc[hal.task.notna(), ['task_id', 'task', 'source_file']].copy()
        bank['definition'] = bank.task.map(lambda task: json.dumps(task, sort_keys=True))
        bank = bank.drop_duplicates(['task_id', 'definition'])
        if bank.task_id.duplicated().any():
            raise ValueError('HAL task IDs have conflicting definitions; error records require source review')
        bank = bank.rename(columns={'task': 'shared_task', 'source_file': 'shared_source'})
        hal = hal.merge(bank[['task_id', 'shared_task', 'shared_source']], on='task_id', how='left', validate='many_to_one')
        hal['task_definition_source'] = hal.source_file.where(hal.task.notna(), hal.shared_source)
        hal['task'] = hal.task.combine_first(hal.shared_task)
        attempts = pd.concat([sierra, hal], ignore_index=True)
        attempts['task_id'] = attempts.task_id.astype(str)
        if attempts.task.isna().any() or not attempts.reward.dropna().isin([0, 1]).all():
            raise ValueError('An original attempt has no task definition or an invalid reward')

        # 4. Identify literal agent configurations and complete task/grading definitions.
        attempts['subject_key'] = attempts.configuration.map(lambda config: hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        subjects = attempts[['subject_key', 'model', 'collection']].drop_duplicates('subject_key').rename(columns={'model': 'raw_label'})
        subjects['features'] = [dict(collection=row.collection, recorded_configuration_sha256=row.subject_key,
            configuration_scope=labels['configuration_scope']) for row in subjects.itertuples()]
        attempts['response_key'] = attempts.source_file + '::' + attempts.source_row.astype(str)
        items = attempts[['response_key', 'task', 'task_id', 'domain']].rename(columns={'response_key': 'item_key'})
        items['raw_item_id'] = items.domain + '/' + items.task_id
        items['content'] = items.task.map(lambda task: json.dumps(dict(user_id=task['user_id'], instruction=task['instruction']),
            sort_keys=True, ensure_ascii=False))
        items['features'] = [dict(domain=row.domain, input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = items.task.map(lambda task: dict(rule=json.dumps(dict(rule=self.grading['rule'],
            actions=task['actions'], outputs=task['outputs']), sort_keys=True, ensure_ascii=False)))
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))] * len(items)

        # 5. Preserve repeated observations and distinguish published grades from API errors.
        attempts['item_key'] = attempts.response_key
        attempts['task_digest'] = [hashlib.sha256(json.dumps(dict(domain=row.domain, task=row.task),
            sort_keys=True, ensure_ascii=False).encode()).hexdigest() for row in attempts.itertuples()]
        attempts = attempts.sort_values(['source_file', 'source_trial', 'source_row'], kind='stable', na_position='last')
        attempts['trial'] = attempts.groupby(['subject_key', 'task_digest'], sort=False).cumcount() + 1
        attempts['response'] = attempts.reward
        attempts['test_condition'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            original_trial=None if pd.isna(row.source_trial) else int(row.source_trial),
            task_definition_source=row.task_definition_source,
            evaluation_integrity=labels['fewshot_integrity'] if row.collection == 'hal' and
                'few' in row.configuration.get('agent_name', '').lower() else labels['other_integrity']),
            sort_keys=True) for row in attempts.itertuples()]

        # 6. Retain every original result and every associated native logging record.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [dict(source_file=row.source_file, source_row=row.source_row,
            task_id=row.task_id, configuration=row.configuration,
            run_config=row.run_config if isinstance(row.run_config, dict) else None,
            result=row.native_record, logs=row.logs, task_definition_source=row.task_definition_source)
            for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'trial', 'response', 'test_condition']],
            'traces': traces}


if __name__ == '__main__':
    TauBench(__file__).main_from_args()
