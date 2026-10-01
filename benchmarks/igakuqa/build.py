"""Tabulate published IgakuQA answers using the original question and grading rules."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class IgakuQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        verifier = self.grading['verifiers']['native']
        profiles = {name.removeprefix('subject_'): profile for name, profile in parameters.items()
            if name.startswith('subject_')}

        # 1. Load the original questions, translations and predictions as tables.
        questions = pd.concat([pd.read_json(path, lines=True, dtype=False).assign(year=path.parent.name)
            for path in sorted(self.raw_dir.glob(parameters['layout']['questions']))], ignore_index=True)
        translations = pd.concat([pd.read_json(path, lines=True, dtype=False)
            for path in sorted(self.raw_dir.glob(parameters['layout']['translations']))], ignore_index=True)
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['predictions'])):
            frame = pd.read_json(path, lines=True, dtype=False)
            parts.append(frame.assign(record=frame.to_dict('records')).rename_axis('source_row').reset_index().assign(
                subject_key=path.stem.split('_', 1)[1], source_file=str(path.relative_to(self.raw_dir))))
        responses = pd.concat(parts, ignore_index=True)

        # 2. Join each prediction to its question ID and retain the text-only scope.
        responses = responses.merge(questions, on='problem_id', how='left', validate='many_to_one')
        if responses.text_only.isna().any():
            raise ValueError('A prediction has no original question')
        responses = responses.loc[responses.text_only].merge(
            translations[['problem_id', 'problem_text_en', 'choices_en']],
            on='problem_id', how='left', validate='many_to_one')
        subjects = pd.DataFrame.from_dict(profiles, orient='index').rename_axis('subject_key').reset_index()
        responses = responses.merge(subjects[['subject_key', 'language']],
            on='subject_key', how='left', validate='many_to_one')
        if responses.language.isna().any():
            raise ValueError('A prediction has an undeclared source configuration')
        responses['item_key'] = responses.problem_id + ':' + responses.language
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)
        responses['test_condition'] = responses.response_key

        # 3. Apply the native comma-split comparison and its two explicit exceptions.
        responses['prediction_parts'] = responses.prediction.str.split(',').map(sorted)
        responses['reference_parts'] = responses.answer.map(sorted)
        responses['response'] = (responses.prediction_parts == responses.reference_parts).astype(float)
        responses.loc[responses.problem_id.isin(verifier['credit_all']), 'response'] = 1.0
        for problem_id, alternatives in verifier['alternative_answers'].items():
            accepted = responses.prediction_parts.map(lambda prediction: prediction in alternatives)
            responses.loc[(responses.problem_id == problem_id) & accepted, 'response'] = 1.0

        # 4. Keep the language-specific stimulus and separate grading information.
        items = responses.drop_duplicates('item_key').copy()
        translated = items.language.eq('en')
        items['question'] = items.problem_text.where(~translated, items.problem_text_en)
        items['options'] = items.choices.where(~translated, items.choices_en)
        items['content'] = items.question + '\n' + items.options.map(
            lambda options: '\n'.join(f'{chr(97 + index)}: {option}' for index, option in enumerate(options)))
        items['raw_item_id'] = items.item_key
        items['features'] = [dict(language=row.language, exam_year=row.year, requested_choices=len(row.answer),
            input_scope=parameters['labels']['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=','.join(row.reference_parts),
            rule=verifier['item_rules'].get(row.problem_id, self.grading['rule'])) for row in items.itertuples()]
        items['verifier'] = ExactMatcher(spec=json.dumps(verifier, sort_keys=True))

        # 5. Retain full native predictions and publish the documented subject context.
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_model=row.subject_key,
            subject_kind=row.kind, source_protocol={key: value for key, value in profiles[row.subject_key].items()
                if key not in ['raw_label', 'language', 'kind', 'demonstrations']},
            demonstrations=(pd.read_json(self.raw_dir / row.demonstrations, lines=True, dtype=False).to_dict('records')
                           if row.demonstrations else []),
            historical_configuration=parameters['labels']['historical_configuration']) for row in subjects.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row, record=row.record),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    IgakuQA(__file__).main_from_args()
