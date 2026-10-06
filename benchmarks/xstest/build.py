"""Tabulate XSTest's released completions and five grading protocols."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class XSTest(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        files, protocols = parameters['files'], self.grading['verifiers']

        # 1. Read the original completions and attach their two automated exports.
        panels = []
        for model in parameters['models']:
            paths = {kind: files[kind].format(model=model) for kind in ('native', 'gpt', 'string')}
            native, gpt, string = [pd.read_csv(self.raw_dir / path, dtype=str, keep_default_na=False)
                for path in paths.values()]
            columns = native.columns.drop('agreement')
            for exported in (gpt, string):
                if not native[columns].equals(exported[columns]) or not native.agreement.str.lower().equals(exported.agreement.str.lower()):
                    raise ValueError('Automated labels do not match their original completion records')
            native['native_record'], native['gpt_record'], native['string_record'] = (
                native.to_dict('records'), gpt.to_dict('records'), string.to_dict('records'))
            native['gpt4_label'], native['strmatch_label'] = gpt.gpt4_label, string.strmatch_label
            native['subject_key'], native['source_row'] = model, native.index
            native['source_files'] = [paths] * len(native)
            panels.append(native)
        completions = pd.concat(panels, ignore_index=True)
        historical = pd.read_csv(self.raw_dir / files['historical'], dtype=str, keep_default_na=False)
        historical['historical_record'] = historical.to_dict('records')
        current = pd.read_csv(self.raw_dir / files['current'], dtype=str, keep_default_na=False)
        current['current_record'] = current.to_dict('records')
        current['id_v2'] = 'v2-' + current.id
        bank = historical.merge(current[['id_v2', 'current_record']], on='id_v2', validate='one_to_one')
        completions = completions.merge(bank[['id_v2', 'historical_record', 'current_record']],
            left_on='id', right_on='id_v2', how='left', validate='many_to_one')
        if completions.historical_record.isna().any():
            raise ValueError('A recorded prompt has no source task-bank entry')

        # 2. Unpivot five judgments; preserve unparseable judge outputs as null grades.
        ratings = completions.melt(id_vars=completions.columns.drop(list(protocols)).tolist(),
            value_vars=list(protocols), var_name='grader', value_name='native_label')
        ratings['response'] = ratings.native_label.map(parameters['labels']).astype(float)
        if ratings.loc[ratings.response.isna(), 'grader'].ne('gpt4_label').any():
            raise ValueError('An unexpected non-GPT label falls outside the released category scale')

        # 3. Identify literal model configurations and each prompt's grading protocol.
        subjects = pd.DataFrame(parameters['models'].items(), columns=['subject_key', 'raw_label'])
        subjects['features'] = [dict(source_model_key=model, paper_model=parameters['model_identifiers'][model],
            paper_system_prompt=parameters['system_prompts'][model], paper_collection_date=parameters['collection_dates'][model],
            paper_generation=parameters['generation']) for model in subjects.subject_key]
        definition = ['id', 'prompt', 'type', 'grader']
        items = ratings[definition].drop_duplicates().reset_index(drop=True)
        items['item_key'], items['raw_item_id'], items['content'] = items.index, items.id, items.prompt
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = [ExactMatcher(spec=json.dumps(protocols[grader], sort_keys=True)) if protocols[grader]['kind'] == 'deterministic'
            else Judge(judge=protocols[grader].get('model'), judged_by=protocols[grader]['kind'],
                spec=json.dumps(protocols[grader], sort_keys=True)) for grader in items.grader]
        items['features'] = [dict(source_type=row.type) for row in items.itertuples()]
        ratings = ratings.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')

        # 4. Retain full native exports and task-bank discrepancies for every judgment.
        ratings['response_key'] = ratings.subject_key + ':' + ratings.id + ':' + ratings.grader
        traces = ratings[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_model=row.subject_key, source_row=int(row.source_row),
            source_files=row.source_files, grader=row.grader, native_record=row.native_record,
            gpt_record=row.gpt_record, string_record=row.string_record, historical_record=row.historical_record,
            current_record=row.current_record, task_bank_prompts_match=(row.prompt == row.historical_record['prompt'] == row.current_record['prompt']),
            grade_available=pd.notna(row.response), observation_scope=parameters['descriptions']['observation_scope']),
            ensure_ascii=False, allow_nan=False) for row in ratings.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier', 'features']],
            'responses': ratings[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    XSTest(__file__).main_from_args()
