"""Tabulate the original LTL predictions and recorded semantic-verifier outcomes."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class InterchangeableTokenEmbeddings(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate complete native result records with their original positions.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            records = pd.Series(json.loads(path.read_text()), dtype=object).rename_axis('source_row').reset_index(name='native_record')
            records = records.join(pd.json_normalize(records.native_record, max_level=0))
            records['subject_key'] = str(path.parents[2].relative_to(self.raw_dir))
            records['split'] = parameters['splits'][path.parent.name]
            records['source_file'] = str(path.relative_to(self.raw_dir))
            parts.append(records)
        responses = pd.concat(parts, ignore_index=True)
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)

        # 2. Preserve the source distinction between a wrong prediction and an ungraded check.
        if not responses.result.isin(parameters['verdicts']).all():
            raise ValueError('Unknown native semantic-verifier result')
        responses['response'] = pd.to_numeric(responses.result.map(parameters['verdicts']).replace('', None))
        responses['test_condition'] = 'split=' + responses.split + ';' + parameters['labels']['condition']

        # 3. Register original formulas and complete reference/grading definitions once.
        items = responses.drop_duplicates(['formula', 'trace']).copy()
        items['item_key'] = range(len(items))
        items['raw_item_id'] = items.split + ':' + items.source_row.astype(str)
        items['content'] = items.formula
        items['features'] = [dict(input_format=parameters['labels']['item_format'])] * len(items)
        items['grading_criterion'] = items.trace.map(lambda value: dict(reference_answer=value, rule=self.grading['rule']))
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['spot'], sort_keys=True))] * len(items)
        responses = responses.merge(items[['formula', 'trace', 'item_key']], on=['formula', 'trace'], validate='many_to_one')

        # 4. Retain the released model labels and complete architecture configuration.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key.map(parameters['model_names'])
        subjects['features'] = [dict(**parameters['subject_features'], model_run=name,
            training_data=parameters['training_data'][name],
            architecture_config=json.dumps(json.loads((self.raw_dir / name / 'config.json').read_text()), sort_keys=True))
            for name in subjects.subject_key]

        # 5. Associate every attempt with its full unmodified source record.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            native_record=row.native_record), allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces}


if __name__ == '__main__':
    InterchangeableTokenEmbeddings(__file__).main_from_args()
