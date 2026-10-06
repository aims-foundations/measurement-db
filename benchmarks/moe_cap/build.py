"""Tabulate MoE-CAP's released requests, results and ungraded judgment inputs."""

import json
from pathlib import Path
import sys
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle, native_json_value


class MoECAP(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate all recorded requests, preserving run, task and row coordinates.
        frames, configurations = [], []
        for path in sorted(self.raw_dir.glob(parameters['paths']['results'])):
            data = json.loads(path.read_text())
            source = str(path.relative_to(self.raw_dir))
            config = data['config']
            features = dict(harness=config['inference_framework'], record_origin='harness_export',
                run_configuration=quote(json.dumps(config, sort_keys=True), safe=''),
                evaluator_revision=data['git_hash'], runner_revision=data.get('upper_git_hash', 'unrecorded'))
            configurations.append(dict(subject_key=source, raw_label=config['model_name'], features=features))
            for task, records in data['samples'].items():
                family = task.split('_')[0]
                if family not in parameters['grade_fields']:
                    raise ValueError('A new source task requires grading-protocol review')
                table = pd.json_normalize(records, max_level=0)
                table['native_record'] = records
                table = table.assign(source_file=source, source_task=task, source_row=table.index,
                    subject_key=source, family=family, input_scope='recorded_harness_request')
                table['task_config'] = [data['configs'][task]] * len(table)
                if family == 'arena':
                    if not table.score.eq(-1).all():
                        raise ValueError('A published Arena-Hard grade requires an explicit import mapping')
                    table['response'] = None
                else:
                    table['response'] = table[parameters['grade_fields'][family]].map(float)
                    if not table.response.isin([0., 1.]).all():
                        raise ValueError('Expected a released binary correctness grade')
                frames.append(table)

        # 2. Keep the older ungraded generations with their recorded configuration labels.
        for path in sorted(self.raw_dir.glob(parameters['paths']['judgment_inputs'])):
            records = native_json_value(read_native_pickle(path))
            table = pd.json_normalize(records, max_level=0)
            if set(table) != set(parameters['judgment_fields']):
                raise ValueError('A changed judgment-input format requires source review')
            source = str(path.relative_to(self.raw_dir))
            label = path.name.removesuffix(parameters['labels']['judgment_suffix'])
            model, precision = label.rsplit('-', 1)
            configurations.append(dict(subject_key=source, raw_label=model,
                features=dict(record_origin='judgment_input_export', recorded_precision=precision,
                              published_configuration=label)))
            table['native_record'] = records
            table = table.assign(source_file=source, source_task=parameters['labels']['arena_task'],
                source_row=table.index, subject_key=source, family='arena', input_scope='released_task_text', response=None)
            table['doc'] = table.question
            table['task_config'] = table.configs
            frames.append(table)
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda table: table.index)

        # 3. Project actual inputs and their grading protocol; retain missing judgments as null.
        items = responses.copy()
        items['item_key'] = items.response_key
        multiple_choice = items.family.eq('mmlu')
        generation = items.input_scope.eq('recorded_harness_request') & ~multiple_choice
        question_only = items.input_scope.eq('released_task_text')
        if not items.loc[multiple_choice, 'arguments'].map(lambda args: isinstance(args, list) and len(args) == 4
                and all(isinstance(arg, list) and len(arg) == 2 and all(isinstance(v, str) for v in arg) for arg in args)
                and len({arg[0] for arg in args}) == 1).all():
            raise ValueError('Expected four ordered continuations of the same recorded MMLU prompt')
        if not items.loc[generation, 'arguments'].map(lambda args: isinstance(args, list) and len(args) == 1
                and isinstance(args[0], list) and len(args[0]) == 2 and isinstance(args[0][0], str)
                and isinstance(args[0][1], dict)).all():
            raise ValueError('Expected a recorded generation prompt and its options')
        items['content'] = None
        items.loc[multiple_choice, 'content'] = items.loc[multiple_choice, 'arguments'].map(
            lambda args: dict(prompt=args[0][0], continuations=[arg[1] for arg in args]))
        items.loc[generation, 'content'] = items.loc[generation, 'arguments'].map(lambda args: dict(prompt=args[0][0]))
        items.loc[question_only, 'content'] = items.loc[question_only, 'doc'].map(lambda doc: dict(task=doc['content']))
        items['content'] = items.content.map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
        items['raw_item_id'] = items.source_task + '::' + items.native_record.map(
            lambda record: str(record['doc_id']) if 'doc_id' in record else record['question']['question_id'])
        items['protocol'] = items.family.map(self.grading['verifiers'])
        items['rule'] = items.protocol.map(lambda protocol: protocol['rule'])
        arena = items.family.eq('arena')
        items.loc[arena, 'rule'] = [json.dumps(dict(description=rule, baseline_answer=doc['model_answer']),
            ensure_ascii=False, allow_nan=False) for rule, doc in zip(items.loc[arena, 'rule'], items.loc[arena, 'doc'])]
        references = items.target.map(str).astype(object).where(~arena, None)
        items['grading_criterion'] = [dict(rule=rule, response_scale=protocol['response_scale'],
            **({} if reference is None else dict(reference_answer=reference)))
            for rule, protocol, reference in zip(items.rule, items.protocol, references)]
        items['verifier'] = [(Judge if family == 'arena' else ExactMatcher)(spec=json.dumps(
            dict(protocol=protocol, recorded_task_configuration=native_json_value(config)),
            sort_keys=True, ensure_ascii=False, allow_nan=False))
            for family, protocol, config in zip(items.family, items.protocol, items.task_config)]
        items['features'] = [dict(task=task, input_scope=scope) for task, scope in zip(items.source_task, items.input_scope)]

        # 4. Keep full source records and distinguish exports without claiming independent executions.
        responses['item_key'] = responses.response_key
        responses['test_condition'] = ['source_export=' + quote(row.source_file, safe=' /-._')
            + ';task=' + row.source_task for row in responses.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_task=row.source_task,
            source_row=int(row.source_row), native_record=native_json_value(row.native_record)),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': pd.DataFrame(configurations),
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    MoECAP(__file__).main_from_args()
