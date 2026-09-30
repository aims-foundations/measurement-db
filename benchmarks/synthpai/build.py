#!/usr/bin/env python3
"""Curate SynthPAI's released attribute grades and original recorded prompts."""

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class SynthPAI(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        raw = self.raw_dir / parameters['layout']['release']

        # 1. Melt the merged profiles into model generations and grading entries.
        profiles = pd.read_json(raw / parameters['layout']['merged'], lines=True).rename_axis('source_row').reset_index()
        predictions = pd.DataFrame(profiles.predictions.tolist()).rename_axis('source_row').reset_index().melt(
            id_vars='source_row', var_name='subject_key', value_name='native_prediction').dropna(subset=['native_prediction'])
        evaluations = pd.DataFrame(profiles.evaluations.tolist()).rename_axis('source_row').reset_index().melt(
            id_vars='source_row', var_name='subject_key', value_name='native_evaluation').dropna(subset=['native_evaluation'])
        responses = evaluations.merge(predictions, on=['source_row', 'subject_key'], how='left', validate='one_to_one')
        responses = responses.merge(profiles[['source_row', 'username', 'reviews']], on='source_row', validate='many_to_one')
        responses['full_answer'] = responses.native_prediction.map(lambda value: value['full_answer'])
        responses['entry'] = responses.native_evaluation.map(lambda value: list(value['human_evaluated'].items()))
        responses = responses.explode('entry').dropna(subset=['entry']).reset_index(drop=True)
        responses[['attribute', 'ranked_grades']] = pd.DataFrame(responses.entry.tolist(), index=responses.index)

        # 2. Read original generation rows and their numbered printed prompt blocks.
        inputs, logs = [], []
        for model, filename in parameters['inputs'].items():
            table = pd.read_json(raw / filename, lines=True).rename_axis('prediction_row').reset_index()
            table['full_answer'] = table.predictions.map(lambda value: value.get(model, {}).get('full_answer'))
            table = table.assign(subject_key=model, prediction_file=filename).rename(columns={'username': 'input_username'})
            inputs.append(table[['subject_key', 'prediction_row', 'prediction_file', 'input_username', 'full_answer']].dropna(subset=['full_answer']))
            text = (raw / parameters['logs'][model]).read_text()
            header = text.split('\n', 1)[0]
            if ast.literal_eval(re.search(r'gen_model=ModelConfig\(name=(.*?), tokenizer_name=', header).group(1)) != model:
                raise ValueError('Prediction log model differs from its declared source')
            blocks = pd.Series([text]).str.extractall(r'(?ms)^=+(?P<prediction_row>[0-9]+)=+\n(?P<log_block>.*?)(?=^=+[0-9]+=+\n|\Z)').reset_index(drop=True)
            blocks['prediction_row'] = blocks.prediction_row.astype(int)
            blocks['system_prompt'] = ast.literal_eval(re.search(r'system_prompt=(.*?), individual_prompts=', header).group(1))
            blocks['prompt'] = blocks.log_block.str.split('\nhuman:\n', n=1).str[0]
            logs.append(blocks.assign(subject_key=model, log_file=parameters['logs'][model], log_configuration=header))
        inputs = pd.concat(inputs, ignore_index=True).merge(pd.concat(logs, ignore_index=True),
            on=['subject_key', 'prediction_row'], how='left', validate='one_to_one')
        inputs = inputs.merge(responses[['subject_key', 'full_answer']].drop_duplicates(),
            on=['subject_key', 'full_answer'], how='inner', validate='many_to_one')
        responses = responses.merge(inputs, on=['subject_key', 'full_answer'], how='left', validate='many_to_one', indicator=True)
        if not responses._merge.eq('both').all() or responses.log_block.isna().any():
            raise ValueError('Every released grade must match an original generation and numbered prompt log')

        # 3. Keep each recorded model configuration separate, without guessing versions.
        subjects = responses[['subject_key', 'log_configuration']].drop_duplicates()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(source_model=row.subject_key, harness=parameters['labels']['harness'],
            configuration_sha256=hashlib.sha256(row.log_configuration.encode()).hexdigest()) for row in subjects.itertuples()]

        # 4. Use the recorded joint prompt; keep references and grading axes outside it.
        responses['item_key'] = responses.subject_key + '::' + responses.source_row.astype(str) + '::' + responses.attribute
        items = responses.copy()
        items['content'] = items.system_prompt + '\n\n' + items.prompt
        items['raw_item_id'] = items.username + '::' + items.attribute + '::' + items.content.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:12])
        items['features'] = [dict(source_profile=row.username, attribute=row.attribute) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=json.dumps({row.attribute: row.reviews['human_evaluated'][row.attribute],
            **({'education_category': row.reviews['human_evaluated'].get('education_category')} if row.attribute == 'education' else {})}, ensure_ascii=False, sort_keys=True),
            rule=self.grading['rule'] + ' Attribute: ' + row.attribute + '.') for row in items.itertuples()]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['published'], sort_keys=True))

        # 5. Preserve first-rank grades, nulls, complete answers and source associations.
        responses['response_key'] = responses.index
        responses['response'] = responses.ranked_grades.map(lambda values: float(values[0]) if values else None)
        responses['test_condition'] = 'attribute=' + responses.attribute
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=parameters['layout']['merged'], source_row=int(row.source_row),
            username=row.username, source_model=row.subject_key, attribute=row.attribute,
            native_prediction=row.native_prediction, native_evaluation=row.native_evaluation, native_reviews=row.reviews,
            prediction_file=row.prediction_file, prediction_row=int(row.prediction_row), input_username=row.input_username,
            log_file=row.log_file, log_configuration=row.log_configuration, log_block=row.log_block), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    SynthPAI(__file__).main_from_args()
