"""Tabulate Tulu's released human acceptability judgments and model completions."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class TuluHumanEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read literal spreadsheet values and the original comparison-instance table.
        annotations = pd.read_excel(self.raw_dir / layout['annotations'], keep_default_na=False,
            engine_kwargs={'data_only': False}).astype(object).replace('', None)
        annotations['timestamp'] = annotations.timestamp.map(lambda value: value.isoformat())
        annotations['annotation_record'] = annotations.to_dict('records')
        annotations['source_row'] = annotations.index
        latest = annotations.sort_values('timestamp', ascending=False, kind='stable').drop_duplicates(
            ['instance_index', 'evaluator']).index
        annotations['is_latest_annotation'] = annotations.index.isin(latest)
        instances = pd.read_json(self.raw_dir / layout['instances'], lines=True)
        instances['instance_record'] = instances.to_dict('records')
        instances['instance_index'] = instances.index

        # 2. Unpivot the two ratings, then join each model to its released completion.
        ratings = pd.wide_to_long(annotations.rename(columns=parameters['rename']),
            stubnames=['model', 'completion', 'acceptable'], i='source_row', j='side', sep='_', suffix='[ab]'
            ).reset_index().sort_values(['source_row', 'side'], kind='stable').rename(columns={'model': 'subject_key'})
        completions = instances.explode('completions', ignore_index=True)
        completions = completions.join(pd.json_normalize(completions.pop('completions'))).rename(columns={
            'id': 'instance_id', 'model': 'subject_key', 'completion': 'released_completion', 'prompt': 'released_prompt'})
        ratings = ratings.merge(completions, on=['instance_index', 'instance_id', 'subject_key'],
            how='left', validate='many_to_one')
        if ratings.instance_record.isna().any() or not ratings.prompt.eq(ratings.released_prompt).all():
            raise ValueError('An annotation lacks its exact released comparison prompt and model')
        if not ratings.acceptable.isin(parameters['acceptability']).all():
            raise ValueError('An acceptability label is outside the original yes/no scale')
        ratings['response'] = ratings.acceptable.map(parameters['acceptability']).astype(float)
        ratings['completion_exports_match'] = ratings.completion.eq(ratings.released_completion)

        # 3. Retain literal model labels and identify each prompt's human grading protocol.
        subjects = ratings[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=model,
            configuration_status=labels['configuration_status']) for model in subjects.subject_key]
        definition = ['instance_id', 'evaluator']
        items = ratings[definition + ['prompt']].drop_duplicates().reset_index(drop=True)
        if items.duplicated(definition).any():
            raise ValueError('One source prompt identifier has conflicting text')
        items['item_key'] = items.index
        items['raw_item_id'], items['content'] = items.instance_id, items.prompt
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = [Judge(judge=rater, judged_by='human',
            spec=json.dumps(self.grading['verifiers']['human'], sort_keys=True)) for rater in items.evaluator]
        ratings = ratings.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')

        # 4. Preserve every rating and both full exports, including discrepancies and revisions.
        ratings['response_key'] = ratings.id.astype(str) + ':' + ratings.side
        ratings['test_condition'] = [labels['condition'].format(instance_index=int(row.instance_index), side=row.side)
            for row in ratings.itertuples()]
        traces = ratings[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['annotations'], source_row=int(row.source_row),
            side=row.side, annotation_record=row.annotation_record, comparison_instance=row.instance_record,
            completion_exports_match=bool(row.completion_exports_match), is_latest_annotation=bool(row.is_latest_annotation),
            observation_scope=labels['observation_scope']), ensure_ascii=False, allow_nan=False)
            for row in ratings.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier']],
            'responses': ratings[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces}


if __name__ == '__main__':
    TuluHumanEval(__file__).main_from_args()
