#!/usr/bin/env python3
"""Curate the authors' recipe pairs and released LLM preference predictions."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SustainableFood(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Join the two native CSV tables by their exported row keys.
        pairs = pd.read_csv(self.raw_dir / layout['pairs'], dtype=str, keep_default_na=False).rename(columns={'Unnamed: 0': 'row_key'})
        outputs = pd.read_csv(self.raw_dir / layout['outputs'], dtype=str, keep_default_na=False).rename(columns={'Unnamed: 0': 'row_key'})
        if set(outputs.columns) != {'row_key', *parameters['parsers']}:
            raise ValueError('Released model columns differ from the declared parser configuration')
        pairs['native_pair'] = pairs.rename(columns={'row_key': ''}).to_dict('records')
        pairs['pair_row'] = pairs.index
        outputs['output_row'] = outputs.index
        joined = pairs.merge(outputs, on='row_key', how='outer', validate='one_to_one', indicator=True)
        if not joined._merge.eq('both').all():
            raise ValueError('Recipe pairs and model outputs must have exactly matching row keys')
        pairs['gold'] = pd.to_numeric(pairs.ground_truth, errors='raise')
        if not pairs.gold.isin([1, 2]).all():
            raise ValueError('The released preferred recipe must be 1 or 2')

        # 2. Restore the paper instructions and preserve the full ordered recipe texts.
        items = pairs.assign(item_key=pairs.row_key)
        items['raw_item_id'] = 'foodcom::pair::' + items['index']
        items['content'] = parameters['instructions']['prefix'] + '\n\nRecipe 1:\n' + items.text_1 + '\n\nRecipe 2:\n' + items.text_2 + '\n\nAnswer:'
        items['features'] = [dict(pair_id=row.index, recipe_1_id=row.id_1, recipe_2_id=row.id_2) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer='Recipe ' + str(int(gold)), rule=self.grading['rule']) for gold in items.gold]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['published'], sort_keys=True))

        # 3. Melt every output cell, retaining missing attempts and source positions.
        responses = joined.melt(id_vars=['row_key', 'pair_row', 'output_row', 'native_pair'],
            value_vars=list(parameters['parsers']), var_name='subject_key', value_name='native_output')
        responses = responses.merge(items[['item_key', 'row_key', 'gold']], on='row_key', how='left', validate='many_to_one')
        responses['response_key'] = responses.index
        responses['parser'] = responses.subject_key.map(parameters['parsers'])

        # 4. Apply the published model-specific string operations and compare choices.
        responses['parsed_text'] = responses.native_output
        label = responses.parser.eq('recipe_label')
        recipe_1 = responses.native_output.str.contains('Recipe 1', regex=False)
        recipe_2 = responses.native_output.str.contains('Recipe 2', regex=False)
        responses.loc[label & recipe_2, 'parsed_text'] = '2'
        responses.loc[label & recipe_1, 'parsed_text'] = '1'
        initial = responses.parser.eq('initial')
        responses.loc[initial, 'parsed_text'] = responses.loc[initial, 'native_output'].str[0]
        o1 = responses.parser.eq('o1_initial')
        strip = o1 & responses.native_output.str.contains('Answer: ', regex=False)
        responses.loc[strip, 'parsed_text'] = responses.loc[strip, 'native_output'].str.strip('Answer: ')
        responses.loc[o1, 'parsed_text'] = responses.loc[o1, 'parsed_text'].str[0]
        responses['parsed_choice'] = pd.to_numeric(responses.parsed_text, errors='coerce')
        responses['response'] = responses.parsed_choice.eq(responses.gold).astype(float).where(responses.parsed_choice.notna())
        responses['test_condition'] = 'source_file=' + layout['outputs']
        subjects = responses[['subject_key']].drop_duplicates()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(source_model=model, declared_request_options=parameters['request_options'][model],
            harness=parameters['labels']['harness']) for model in subjects.subject_key]

        # 5. Keep the full native output and pair record, including the original ratings.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['outputs'], source_row=int(row.output_row),
            pair_source_row=int(row.pair_row), row_key=row.row_key, source_column=row.subject_key,
            native_output=row.native_output, native_pair=row.native_pair), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    SustainableFood(__file__).main_from_args()
