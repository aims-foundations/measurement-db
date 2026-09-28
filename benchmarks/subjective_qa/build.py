"""Tabulate source requests, human labels and complete SubjECTive-QA outputs."""

import ast
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SubjectiveQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']
        release = self.raw_dir / layout['release']

        # 1. Load original CSV rows and join their optional published rating annotation.
        frames = []
        for path in sorted((release / layout['results']).rglob('*.csv')):
            if 'updated' in path.name:
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame['source_record'] = frame.to_dict('records')
            annotations = list(path.parent.glob('*updated*'))
            if len(annotations) > 1:
                raise ValueError('Multiple rating annotations need source review')
            frame['annotation_file'], frame['Rating'] = None, ''
            if annotations:
                annotation = pd.read_csv(annotations[0], dtype=str, keep_default_na=False)
                columns = list(frame.source_record.iloc[0])
                if not frame[columns].equals(annotation[columns]):
                    raise ValueError('The rating export changes an original source record')
                frame['Rating'] = annotation.Rating
                frame['annotation_file'] = str(annotations[0].relative_to(release))
            frames.append(frame.assign(source_file=str(path.relative_to(release)), source_row=frame.index,
                dimension=path.relative_to(release).parts[2], vendor=path.relative_to(release).parts[3]))
        attempts = pd.concat(frames, ignore_index=True)

        # 2. Flatten recorded request headers; never execute or rewrite source traces.
        literal = attempts.complete_responses.str.startswith('{')
        header_text = attempts.loc[literal, 'complete_responses'].str.split(", 'subjobs':", n=1).str[0] + '}'
        headers = pd.json_normalize(header_text.map(ast.literal_eval), max_level=0).set_axis(header_text.index)
        attempts['request_id'] = attempts.complete_responses.str.extract(parameters['patterns']['request_id'], expand=False)
        attempts['model'] = attempts.complete_responses.str.extract(parameters['patterns']['reported_model'], expand=False)
        attempts['settings'] = None
        attempts['prompt'] = None
        attempts.loc[literal, 'request_id'] = headers.id
        attempts.loc[literal, 'model'] = headers.model
        attempts.loc[literal, 'prompt'] = headers.prompt.str[0]
        attempts.loc[literal, 'settings'] = headers.args.map(lambda values: json.dumps(
            {key: value for key, value in values.items() if key not in ['model', 'prompt']}, sort_keys=True))
        if attempts[['request_id', 'model']].isna().any().any():
            raise ValueError('An original response lacks a supported request/model header')
        attempts['input_scope'] = labels['reconstructed']
        attempts.loc[literal, 'input_scope'] = labels['recorded']
        attempts['source_serialization_status'] = 'provider_repr'
        for index, text in attempts.loc[literal, 'complete_responses'].items():
            try:
                ast.literal_eval(text)
                attempts.loc[index, 'source_serialization_status'] = 'valid_python_literal'
            except (SyntaxError, ValueError):
                attempts.loc[index, 'source_serialization_status'] = 'invalid_python_literal_preserved'

        # 3. Match every input and human reference to the released test workbook.
        bank = pd.read_excel(release / layout['bank'], dtype=str, keep_default_na=False)
        bank['bank_row'] = bank.index
        bank = bank.melt(id_vars=['QUESTION', 'ANSWER', 'bank_row'],
            value_vars=list(parameters['definitions']), var_name='dimension', value_name='human_label')
        bank = bank.rename(columns={'QUESTION': 'questions', 'ANSWER': 'answers'})
        attempts = attempts.merge(bank, on=['questions', 'answers', 'dimension'], how='left', validate='many_to_one')
        if attempts.human_label.isna().any() or not attempts.human_label.eq(attempts.actual_labels).all():
            raise ValueError('An input/reference conflicts with the original task workbook')
        reconstructed = attempts.input_scope.eq(labels['reconstructed'])
        attempts.loc[reconstructed, 'prompt'] = [parameters['templates'][
            'openai' if row.vendor == 'gpt-4o' else 'together'].format(feature=row.dimension,
            definition=parameters['definitions'][row.dimension], question=row.questions, answer=row.answers)
            for row in attempts.loc[reconstructed].itertuples()]
        attempts['content'] = [json.dumps(([] if row.input_scope == labels['recorded'] else
            [dict(role='system', content=parameters['templates']['system'])]) + [dict(role='user', content=row.prompt)],
            ensure_ascii=False) for row in attempts.itertuples()]
        annotated = attempts.annotation_file.notna()
        ratings = attempts.Rating.where(annotated,
            attempts.llm_responses.str.extract(parameters['patterns']['rating'], expand=False))
        attempts['rating'] = pd.to_numeric(ratings, errors='coerce')
        if not attempts.rating.dropna().isin([0, 1, 2]).all():
            raise ValueError('A rating lies outside the source classes')
        attempts['grade_status'] = labels['fallback']
        attempts.loc[annotated, 'grade_status'] = labels['annotation']
        attempts.loc[attempts.rating.isna(), 'grade_status'] = labels['missing']
        attempts['response'] = attempts.rating.eq(pd.to_numeric(attempts.human_label)).astype(float).where(attempts.rating.notna())

        # 4. Collapse repeated request exports while retaining every original representation.
        identity = ['model', 'settings', 'content', 'human_label', 'rating', 'llm_responses', 'input_scope', 'grade_status']
        if attempts.groupby('request_id')[identity].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Conflicting observations share an upstream request ID')
        attempts['source_export'] = [dict(source_file=layout['release'] + '/' + row.source_file,
            source_row=int(row.source_row), source_record=row.source_record,
            annotation_file=layout['release'] + '/' + row.annotation_file if row.annotation_file else None,
            annotation_rating=row.Rating if row.annotation_file else None,
            source_serialization_status=row.source_serialization_status) for row in attempts.itertuples()]
        exports = attempts.groupby('request_id', sort=False).source_export.agg(list).rename('source_exports')
        attempts = attempts.drop_duplicates('request_id').merge(exports, on='request_id', validate='one_to_one')
        attempts['response_key'] = attempts.request_id
        attempts['subject_key'] = attempts.model + '::' + attempts.settings.fillna('unknown')

        # 5. Register literal model configurations and complete task/grading definitions.
        subjects = attempts[['subject_key', 'model', 'settings']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=row.model,
            inference_settings=row.settings if row.settings is not None else 'unknown',
            configuration_status=labels['known_settings'] if row.settings is not None else labels['unknown_settings'])
            for row in subjects.itertuples()]
        definition = ['content', 'human_label', 'dimension', 'bank_row', 'input_scope']
        items = attempts[definition].drop_duplicates().reset_index(drop=True)
        items['item_key'] = items.index
        attempts = attempts.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')
        items['raw_item_id'] = items.dimension + '::' + items.bank_row.astype(str)
        items['features'] = [dict(dimension=row.dimension, source_bank_row=int(row.bank_row), input_scope=row.input_scope)
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=value, rule=self.grading['rule']) for value in items.human_label]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['classification'], sort_keys=True))

        # 6. Keep complete source aliases, output strings and explicit grading limitations.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(request_id=row.request_id, source_exports=row.source_exports,
            predicted_rating=None if pd.isna(row.rating) else int(row.rating), grade_status=row.grade_status,
            input_scope=row.input_scope), ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    SubjectiveQA(__file__).main_from_args()
