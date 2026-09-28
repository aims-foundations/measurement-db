"""Tabulate the recorded RADAR notebook run with independently matched provenance."""

import ast
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class RADAR(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, grading = self.build_parameters, self.grading
        layout, patterns, labels = parameters['layout'], parameters['patterns'], parameters['labels']

        # 1. Flatten saved notebook outputs; do not execute any notebook cell.
        notebook = json.loads((self.raw_dir / layout['notebook']).read_text())
        cells = pd.json_normalize(notebook['cells'], max_level=0).rename_axis('source_cell').reset_index()
        outputs = cells[['source_cell', 'outputs']].explode('outputs', ignore_index=True)
        outputs = outputs.join(pd.json_normalize(outputs.pop('outputs')))
        saved = outputs.loc[outputs['data.text/plain'].notna()].copy()
        saved['result_text'] = saved['data.text/plain'].map(''.join)
        saved = saved.loc[saved.result_text.str.startswith(labels['result_prefix'])].reset_index(drop=True)
        records = pd.json_normalize(saved.result_text.map(ast.literal_eval), max_level=0)
        records['source_cell'] = saved.source_cell
        records['native_record'] = saved.result_text.map(ast.literal_eval)
        if len(records) != 1 or records.is_correct.map(type).ne(bool).any():
            raise ValueError('Expected one original code-agent result with a boolean verdict')

        # 2. Associate the result with its native model log, prompts, replies and usage.
        debug = ''.join(outputs.loc[outputs.output_type.eq('stream'), 'text'].dropna().map(''.join))
        debug = re.sub(patterns['ansi'], '', debug)
        models = pd.Series(re.findall(patterns['model'], debug), dtype='string').drop_duplicates()
        if len(models) != 1:
            raise ValueError('The recorded result lacks an unambiguous logged model')
        prompt_texts = [ast.literal_eval(value) for value in re.findall(patterns['message_text'], debug)]
        record = records.iloc[0]
        messages, usage = record.llm_messages, record.llm_messages_metadata
        if prompt_texts != [message['content'] for message in messages[:2] + messages[:4]]:
            raise ValueError('Saved result and logged prompt history disagree')
        replies = [message['content'] for message in messages if message['role'] == 'assistant']
        if len(replies) != 2 or not all(reply in debug for reply in replies):
            raise ValueError('Saved result and logged model outputs disagree')
        for pattern, field in [('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens')]:
            if list(map(int, re.findall(patterns[pattern], debug))) != [row['usage'][field] for row in usage]:
                raise ValueError('Saved result and logged API usage disagree')
        records['subject_key'] = models.iloc[0]
        records['response_key'] = layout['notebook'] + ':cell=' + records.source_cell.astype(str)

        # 3. Preserve the exact initial messages as the item; keep answers in grading only.
        tasks = pd.json_normalize(records.task, max_level=0)
        items = tasks[['task_id', 'artifact_type', 'num_rows', 'num_cols']].copy()
        items['item_key'], items['raw_item_id'] = records.task_instance_id, records.task_instance_id
        items['content'] = records.llm_messages.map(lambda history: json.dumps(history[:2], ensure_ascii=False))
        items['features'] = tasks[['task_id', 'artifact_type', 'num_rows', 'num_cols']].to_dict('records')
        items['grading_criterion'] = records.ground_truth.map(lambda answer: dict(reference_answer=answer, rule=grading['rule']))
        specification = dict(grading['verifiers']['match_answer'], implementation=(self.raw_dir / layout['grader']).read_text())
        items['verifier'] = [ExactMatcher(spec=json.dumps(specification, sort_keys=True)) for _ in items.index]
        subjects = records[['subject_key']].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects['features'] = [dict(parameters['subject_features']) for _ in subjects.index]

        # 4. Retain the complete result, including tool observations and API usage.
        responses = records.assign(item_key=records.task_instance_id, response=records.is_correct.astype(float),
                                   test_condition=labels['condition'])
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['notebook'], source_cell=int(row.source_cell),
            native_record=row.native_record, model=row.subject_key, model_association=labels['model_association'],
            grade_status=labels['grade_status']), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
                'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    RADAR(__file__).main_from_args()
