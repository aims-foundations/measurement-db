"""Tabulate released statistical-selection attempts and their task definitions."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class StatQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']
        release = self.raw_dir / layout['release']
        keys = ['dataset', 'refined_question']

        # 1. Load native CSV observations without discarding duplicate attempts.
        frames = []
        with ZipFile(release / layout['answers']) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith('.csv') or Path(member).name.startswith('human_'):
                    continue
                frame = pd.read_csv(archive.open(member), dtype=str, keep_default_na=False)
                frame['source_record'] = frame.to_dict('records')
                frames.append(frame.assign(source_member=member, source_row=frame.index,
                    source_scope=Path(member).parent.name, configuration=Path(member).stem))
        attempts = pd.concat(frames, ignore_index=True)
        identities = attempts.configuration.str.extract(parameters['patterns']['configuration'])
        if identities[['model', 'strategy']].isna().any().any():
            raise ValueError('A source model/strategy needs explicit review')
        attempts = pd.concat([attempts, identities], axis=1)
        attempts['response_key'] = attempts.source_member + '#' + attempts.source_row.astype(str)

        # 2. Join canonical task context while preserving older question wording.
        bank = pd.read_csv(release / layout['bank'], dtype=str, keep_default_na=False).drop_duplicates()
        prompts = pd.read_csv(release / layout['prompts'], dtype=str, keep_default_na=False)
        bank['source_task'] = bank.to_dict('records')
        attempts = attempts.merge(bank, on=keys + ['ground_truth'], how='left', validate='many_to_one', indicator=True)
        general = attempts.source_scope.eq(labels['general_scope'])
        if not attempts.loc[general, '_merge'].eq('both').all():
            raise ValueError('A graded response lacks an unambiguous original task/reference')
        prompt = parameters['prompt']
        prompts['prefix'] = prompts.prompt.str.split(prompt['question_separator'], n=1).str[0]
        prefixes = prompts[['dataset', 'prefix']].drop_duplicates()
        attempts = attempts.merge(prefixes, on='dataset', how='left', validate='many_to_one')
        if attempts.prefix.isna().any():
            raise ValueError('A response lacks original column information')
        attempts['prompt'] = attempts.prefix + prompt['question_separator'] + attempts.refined_question + prompt['response_separator'] + prompt['response_suffix']
        attempts['source_task'] = attempts.source_task.where(attempts.source_task.notna(), None)
        attempts[['task', 'difficulty']] = attempts[['task', 'difficulty']].fillna('unknown')

        # 3. Reproduce the original general-analysis parser and exact-set rule.
        if not attempts.source_scope.isin([labels['general_scope'], labels['qualitative_scope']]).all():
            raise ValueError('An undeclared analysis scope needs source review')
        for column, text in [('answer', attempts.extracted_answer),
                             ('reference', attempts.ground_truth.str.replace("'", '"', regex=False))]:
            decoded = []
            for value in text:
                try:
                    decoded.append(json.loads(value))
                except (ValueError, TypeError):
                    decoded.append(None)
            attempts[column] = pd.Series(decoded, dtype=object)
        valid_answer = attempts.answer.map(lambda value: isinstance(value, dict))
        valid_reference = attempts.reference.map(lambda value: isinstance(value, dict))
        exact = pd.Series(True, index=attempts.index)
        for target in ['columns', 'methods']:
            for column in ['answer', 'reference']:
                values = attempts[column].map(lambda value: value.get(target, []) if isinstance(value, dict) else None)
                valid = values.map(lambda value: isinstance(value, list) and all(isinstance(word, str) for word in value))
                sets = values.where(valid, None).map(lambda value: frozenset(word.lower().strip() for word in value) if value is not None else None)
                attempts[column + '_set'] = sets
                if column == 'answer':
                    valid_answer &= valid
                else:
                    valid_reference &= valid
            exact &= attempts.answer_set.eq(attempts.reference_set) & attempts.reference_set.map(bool)
        attempts['response'] = (exact & valid_answer & valid_reference).astype(float).where(general)
        attempts['grade_status'] = labels['valid_grade']
        attempts.loc[~valid_answer, 'grade_status'] = labels['invalid_answer']
        attempts.loc[~valid_reference, 'grade_status'] = labels['invalid_reference']
        attempts.loc[~general, 'grade_status'] = labels['qualitative']

        # 4. Treat each recorded model/strategy as a distinct subject configuration.
        subjects = attempts[['configuration', 'model', 'strategy', 'source_scope']].drop_duplicates().copy()
        subjects['subject_key'] = subjects.configuration
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['harness'], source_model_label=row.model,
            strategy=row.strategy, analysis_scope=row.source_scope, configuration_status=labels['configuration_status'])
            for row in subjects.itertuples()]
        attempts['subject_key'] = attempts.configuration

        # 5. Restore the shared task definition without claiming unknown historical prompts.
        attempts['item_key'] = attempts.dataset + '::' + attempts.refined_question
        items = attempts.drop_duplicates('item_key')[['item_key', 'dataset', 'prompt', 'task', 'difficulty', 'ground_truth']].copy()
        items['raw_item_id'] = items.item_key
        items['content'] = items.prompt
        items['features'] = [dict(dataset=row.dataset, task=row.task, difficulty=row.difficulty,
            input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=value, rule=self.grading['rule']) for value in items.ground_truth]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['selection'], sort_keys=True))

        # 6. Keep full source records and explicit parser/prompt limitations in the traces.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['release'] + '/' + layout['answers'],
            source_member=row.source_member, source_row=int(row.source_row), source_record=row.source_record,
            source_task=row.source_task, grade_status=row.grade_status, grade_scope=labels['grade_scope']),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    StatQA(__file__).main_from_args()
