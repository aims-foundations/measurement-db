"""Tabulate UltraFeedback's complete generations and original aspect ratings."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class UltraFeedback(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout, labels = self.build_parameters['layout'], self.build_parameters['labels']

        # 1. Load original JSONL tables and flatten only recorded completions.
        frames = []
        for path in sorted((self.raw_dir / layout['dataset']).glob('*.jsonl')):
            frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
            frame['instruction_record'] = frame.drop(columns='completions').to_dict('records')
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        instructions = pd.concat(frames, ignore_index=True)
        attempts = instructions.loc[instructions.completions.str.len().gt(0)].explode('completions', ignore_index=True)
        attempts = attempts.rename(columns={'completions': 'native_completion'})
        attempts['completion_index'] = attempts.groupby(['source_file', 'source_row'], sort=False).cumcount()
        attempts = attempts.join(pd.json_normalize(attempts.native_completion, max_level=0))
        attempts['completion_key'] = attempts.index

        # 2. Keep distinct generation configurations, including the full recorded prompt.
        configuration = ['model', 'principle', 'custom_system_prompt']
        subjects = attempts[configuration].drop_duplicates().reset_index(drop=True)
        subjects['subject_key'], subjects['raw_label'] = subjects.index, subjects.model
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=row.model,
            principle=row.principle, configuration_status=labels['configuration_status'],
            system_prompt_json=json.dumps(row.custom_system_prompt, ensure_ascii=True).replace(';', '\\u003b').replace('=', '\\u003d'))
            for row in subjects.itertuples()]
        attempts = attempts.merge(subjects[configuration + ['subject_key']], on=configuration, validate='many_to_one')

        # 3. Unpivot the four aspect annotations; retain missing and invalid grades.
        annotations = pd.json_normalize(attempts.annotations, max_level=0)
        if set(annotations) != set(self.grading['verifiers']):
            raise ValueError('The original annotation aspects differ from the declared grading protocols')
        annotations['completion_key'] = attempts.completion_key
        ratings = annotations.melt(id_vars='completion_key', var_name='aspect', value_name='annotation')
        ratings = ratings.merge(attempts.drop(columns=['annotations', 'response']), on='completion_key', validate='many_to_one')
        ratings['native_rating'] = ratings.annotation.map(lambda annotation: annotation['Rating'])
        if not ratings.native_rating.isin(['1', '2', '3', '4', '5', 'N/A', '0']).all():
            raise ValueError('An original rating needs review against the published scale')
        ratings['response'] = pd.to_numeric(ratings.native_rating, errors='coerce').replace(0, float('nan'))
        ratings['grade_status'] = labels['released']
        ratings.loc[ratings.native_rating.eq('N/A'), 'grade_status'] = labels['missing']
        ratings.loc[ratings.native_rating.eq('0'), 'grade_status'] = labels['invalid']
        definition = ['source_file', 'source_row', 'aspect']
        items = ratings[definition + ['instruction', 'source', 'correct_answers', 'incorrect_answers']].drop_duplicates(definition).reset_index(drop=True)
        items['item_key'], items['content'] = items.index, items.instruction
        items['raw_item_id'] = items.source_file + ':' + items.source_row.astype(str)
        items['features'] = [dict(aspect=row.aspect, upstream_subset=row.source) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=None if row.correct_answers == ['None'] else json.dumps(row.correct_answers, ensure_ascii=False),
            rule=json.dumps(dict(interpretation=self.grading['rule'], aspect=row.aspect, source=row.source,
                correct_answers=row.correct_answers, incorrect_answers=row.incorrect_answers), ensure_ascii=False, sort_keys=True))
            for row in items.itertuples()]
        items['verifier'] = [Judge(judge=labels['judge'], judged_by='llm',
            spec=json.dumps(self.grading['verifiers'][aspect], sort_keys=True)) for aspect in items.aspect]
        ratings = ratings.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')

        # 4. Preserve complete native evidence and each comparison record's provenance.
        ratings['response_key'] = ratings.index
        ratings['test_condition'] = 'source_file=' + ratings.source_file + ';source_row=' + ratings.source_row.astype(str)
        traces = ratings[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            completion_index=int(row.completion_index), aspect=row.aspect, instruction_record=row.instruction_record,
            native_completion=row.native_completion, grade_status=row.grade_status, scope=labels['trace_scope']),
            ensure_ascii=False, allow_nan=False) for row in ratings.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': ratings[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    UltraFeedback(__file__).main_from_args()
