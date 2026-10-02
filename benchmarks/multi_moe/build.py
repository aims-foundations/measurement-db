"""Tabulate Multi-MoE's released option predictions and ungraded conversations."""

import json
from pathlib import Path
import sys
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle, native_json_value


class MultiMoE(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read each CSV with its own complete prompt, option order and gold label.
        frames, configurations = [], []
        for path in sorted(self.raw_dir.glob(parameters['paths']['option_results'])):
            source = str(path.relative_to(self.raw_dir))
            task, configuration = path.parts[-5], path.parts[-4]
            if task not in parameters['option_tasks'] or configuration not in parameters['configurations']:
                raise ValueError('Unknown released task or serving configuration')
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            table = table.rename(columns={'Unnamed: 0': ''})
            table['native_record'] = table.to_dict('records')
            keys = ['subject', 'index'] if 'subject' in table else ['index']
            if table.empty or table.duplicated(keys).any():
                raise ValueError('Expected one released row per source question and configuration')
            if not table.string_matching_correctness.isin(['True', 'False']).all():
                raise ValueError('Expected a released Boolean correctness flag')
            if not table[['sample', 'label']].map(lambda value: bool(value.strip())).all().all():
                raise ValueError('Missing released prompt or gold label')
            table = table.assign(source_file=source, source_row=table.index, task=task,
                subject_key='csv:' + configuration, record_origin='option_csv',
                input_scope='recorded_few_shot_prompt', release_status='published_export')
            table['response'] = table.string_matching_correctness.map({'True': 1., 'False': 0.})
            table['content'] = table['sample']
            table['reference'] = table.label
            table['raw_item_id'] = task + ':' + (table.subject + ':' if 'subject' in table else '') + table['index']
            frames.append(table)
        for configuration, label in parameters['configurations'].items():
            configurations.append(dict(subject_key='csv:' + configuration, raw_label=label,
                features=dict(record_origin='option_csv', published_configuration=configuration,
                    **parameters['subject'])))

        # 2. Join released conversations to their original questions without inventing grades.
        bank = pd.read_json(self.raw_dir / parameters['paths']['questions'], lines=True, dtype=False,
                            convert_dates=False, precise_float=True)
        if bank.question_id.duplicated().any():
            raise ValueError('Duplicate released MT-Bench question identifier')
        template = native_json_value(read_native_pickle(self.raw_dir / parameters['paths']['template']))
        for path in sorted(self.raw_dir.glob(parameters['paths']['conversations'])):
            source = str(path.relative_to(self.raw_dir))
            table = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
            table['native_record'] = table.to_dict('records')
            if table.empty or table.answer_id.duplicated().any() or table.model_id.nunique() != 1:
                raise ValueError('Expected unique answer records and one model label per export')
            table = table.assign(source_file=source, source_row=table.index, task='mt_bench',
                record_origin='conversation_jsonl', input_scope='released_user_turns', response=None,
                release_status=parameters['export_status'].get(source, 'published_export'))
            table = table.merge(bank, on='question_id', how='left', validate='many_to_one')
            if table.category.isna().any():
                raise ValueError('A released answer refers to an absent MT-Bench question')
            if not table.apply(lambda row: isinstance(row.turns, list) and len(row.turns) == 2
                    and all(isinstance(turn, str) and bool(turn.strip()) for turn in row.turns)
                    and isinstance(row.choices, list) and len(row.choices) == 1
                    and row.choices[0]['index'] == 0 and len(row.choices[0]['turns']) == len(row.turns)
                    and all(isinstance(turn, str) for turn in row.choices[0]['turns']), axis=1).all():
                raise ValueError('Expected a complete two-turn conversation with one recorded choice')
            table['subject_key'] = 'jsonl:' + table.model_id
            table['content'] = table.turns.map(lambda turns: json.dumps(dict(user_turns=turns), ensure_ascii=False))
            table['reference'] = table.reference.map(lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, list) else None)
            table['raw_item_id'] = 'mt_bench:' + table.question_id.astype(str)
            frames.append(table)
            configurations.append(dict(subject_key=table.subject_key.iloc[0], raw_label=table.model_id.iloc[0],
                features=dict(record_origin='conversation_jsonl', native_model_id=table.model_id.iloc[0],
                    published_generation_template=quote(json.dumps(template, sort_keys=True), safe=''),
                    **parameters['subject'])))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda frame: frame.index)
        subjects = pd.DataFrame(configurations).drop_duplicates('subject_key')

        # 3. Describe the recorded grading protocol and retain each run's actual stimulus.
        items = responses.copy().assign(item_key=lambda frame: frame.response_key)
        items['protocol'] = items.task.map(self.grading['verifiers'])
        items['reference'] = items.reference.astype(object).where(items.reference.notna(), None)
        items['grading_criterion'] = [dict(rule=protocol['rule'], response_scale=protocol['response_scale'],
            **({} if reference is None else dict(reference_answer=reference)))
            for protocol, reference in zip(items.protocol, items.reference)]
        items['verifier'] = [(Judge if task == 'mt_bench' else ExactMatcher)(
            spec=json.dumps(protocol, sort_keys=True, ensure_ascii=False))
            for task, protocol in zip(items.task, items.protocol)]
        items['features'] = [dict(task=row.task, input_scope=row.input_scope,
            **({'category': row.category} if row.task == 'mt_bench' else {'split': row.split}))
            for row in items.itertuples()]

        # 4. Preserve complete native records and distinguish exports, including author-marked files.
        responses['item_key'] = responses.response_key
        responses['test_condition'] = ('source_export=' + responses.source_file.map(lambda value: quote(value, safe='/-._'))
            + ';task=' + responses.task + ';release_status=' + responses.release_status)
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            native_record=row.native_record), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    MultiMoE(__file__).main_from_args()
