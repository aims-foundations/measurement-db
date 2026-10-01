"""Tabulate the two original categorical judgments recorded in each native tensor cell."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Indeterminacy(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Load native tensors, the recorded input order and literal prompt data.
        runs = pd.concat([pd.json_normalize(json.loads(path.read_text()), max_level=0).assign(
            source_file=str(path.relative_to(self.raw_dir)), task=path.parent.name)
            for path in sorted(self.raw_dir.glob(layout['runs']))], ignore_index=True)
        ratings = pd.concat([pd.read_csv(path).rename_axis('item_index').reset_index().assign(
            task=path.parent.name) for path in sorted(self.raw_dir.glob(layout['ratings']))], ignore_index=True)
        tasks = pd.DataFrame.from_dict(json.loads((self.raw_dir / layout['tasks']).read_text()), orient='index')
        prompt_source = ast.parse((self.raw_dir / layout['prompts']).read_text())
        assignment = next(node for node in prompt_source.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'PROMPTS' for target in node.targets))
        prompts = pd.DataFrame.from_dict(ast.literal_eval(assignment.value), orient='index')
        runs['subject_key'] = runs.model_info.map(lambda info: json.dumps(info, sort_keys=True))
        subjects = runs[['subject_key', 'model_info']].drop_duplicates('subject_key').copy()

        # 2. Expand named tensor axes and retain each observed paired trial once.
        observations = runs[['source_file', 'task', 'subject_key', 'resp_table']].rename(columns={'resp_table': 'value'})
        axes = ['item_index', 'fc_index', 'rs_index', 'trial_index']
        groups = ['source_file']
        for axis in axes:
            observations = observations.explode('value', ignore_index=True)
            observations[axis] = observations.groupby(groups, sort=False).cumcount()
            groups.append(axis)
        if not observations.value.isin([0, 1]).all():
            raise ValueError('Native tensor cells must be binary indicators')
        observations = observations.loc[observations.value.eq(1)].drop(columns='value')
        if observations.duplicated(['source_file', 'item_index', 'trial_index']).any():
            raise ValueError('A native trial has more than one joint category')

        # 3. Separate the FC and RS calls without treating category codes as accuracy.
        responses = observations.melt(id_vars=['source_file', 'task', 'subject_key', 'item_index', 'trial_index'],
            value_vars=['fc_index', 'rs_index'], var_name='format_axis', value_name='native_category')
        responses['format'] = responses.format_axis.map(parameters['formats'])
        definitions = responses[['task', 'item_index', 'format']].drop_duplicates()
        items = definitions.merge(ratings, on=['task', 'item_index'], how='left', validate='many_to_one')
        items['item_key'] = items.task + ':' + items.item_index.astype(str) + ':' + items['format']
        items['raw_item_id'] = items.item_key
        items['tokens'] = [tasks.loc[row.task, 'valid_fc_tokens' if row.format == 'FC' else 'valid_rs_tokens']
            for row in items.itertuples()]
        responses = responses.merge(items[['task', 'item_index', 'format', 'item_key', 'tokens']],
            on=['task', 'item_index', 'format'], validate='many_to_one')
        invalid = responses.native_category.eq(responses.tokens.map(len))
        if (responses.native_category > responses.tokens.map(len)).any():
            raise ValueError('A native category index exceeds its released token vocabulary')
        responses['response'] = responses.native_category.astype(float).mask(invalid)
        responses['grade_status'] = invalid.map({False: 'recorded_category', True: 'unparseable_native_output'})
        responses['trial'] = responses.trial_index + 1
        responses['test_condition'] = responses.source_file + ':item=' + responses.item_index.astype(str)
        responses['response_key'] = responses.test_condition + ':' + responses['format'] + ':' + responses.trial_index.astype(str)

        # 4. Preserve the complete prompt and the format-specific unordered scale.
        items['content'] = [prompts.loc[row['task'], row['format']].format_map(
            {field: row[field] for field in tasks.loc[row['task'], 'prompt_fields']}) for row in items.to_dict('records')]
        items['features'] = [dict(task=row.task, elicitation_format=row.format,
            input_scope=parameters['labels']['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'], response_scale=dict(kind='discrete',
            values=list(range(len(row.tokens))), meanings={str(index): token for index, token in enumerate(row.tokens)},
            direction='unordered')) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers'][row.format], sort_keys=True))
            for row in items.itertuples()]

        # 5. Retain exact parsed categories and their original file/row/trial links.
        subjects['raw_label'] = subjects.model_info.map(lambda info: info['provider'] + '/' + info['model'])
        subjects['features'] = [dict(harness=parameters['labels']['harness'], recorded_configuration=info,
            historical_configuration=parameters['labels']['historical_configuration']) for info in subjects.model_info]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, item_index=row.item_index,
            trial_index=row.trial_index, elicitation_format=row.format, native_category=row.native_category,
            grade_status=row.grade_status, output_text_available=False), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'trial', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    Indeterminacy(__file__).main_from_args()
