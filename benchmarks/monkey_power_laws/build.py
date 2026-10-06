"""Tabulate Monkey Business's original prompts, samples and published verdicts."""

import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MonkeyPowerLaws(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the original JSON files as tables, preserving their row order.
        frames = []
        for path in sorted(self.raw_dir.glob(parameters['paths']['results'])):
            task, model = path.stem.split('_', 1)
            if task not in self.grading['verifiers']:
                raise ValueError('Unknown released task configuration')
            frame = pd.read_json(path, orient='records', dtype=False, convert_dates=False)
            if frame.empty or frame.duplicated(['orig_dset_split', 'orig_dset_idx']).any():
                raise ValueError('Expected distinct original problems in each configuration')
            frames.append(frame.assign(task=task, model=model, source_row=frame.index,
                source_file=str(path.relative_to(self.raw_dir))))
        items = pd.concat(frames, ignore_index=True).assign(item_key=lambda frame: frame.index)
        if not items.orig_dset_split.eq('test').all():
            raise ValueError('Unexpected original split; review its task-bank mapping')
        for column in ['question', 'prompt']:
            if not items[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError('Missing original problem or prompt')
        if not items.orig_dset_idx.map(lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0).all():
            raise ValueError('Invalid original task-bank index')
        lengths = items[['samples', 'is_corrects']].map(len)
        if not (lengths.samples.gt(0) & lengths.samples.eq(lengths.is_corrects)).all():
            raise ValueError('Samples and verdicts must have equal nonzero lengths')
        items['reference_answer'] = items.get('gt_answer', pd.Series(None, index=items.index)).astype(object)
        items['reference_answer'] = items.reference_answer.where(items.reference_answer.notna(), None)
        math = items.task.isin(['GSM8K', 'MATH'])
        if not items.loc[math, 'reference_answer'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Missing released mathematical reference answer')

        # 2. Join CodeContests to all three test sets; retain an immutable grading pointer.
        bank = pd.DataFrame(pq.read_table(self.raw_dir / parameters['paths']['code_contests'], columns=[
            'name', 'description', 'public_tests', 'private_tests', 'generated_tests', 'time_limit']).to_pylist())
        bank = bank.assign(orig_dset_idx=bank.index)
        coding = items.loc[items.task.eq('CodeContests')].merge(bank, on='orig_dset_idx',
            how='left', validate='many_to_one')
        if coding.name.isna().any() or not coding.question.eq(coding.description).all():
            raise ValueError('CodeContests source index and original problem text disagree')
        cases = coding[['public_tests', 'private_tests', 'generated_tests', 'time_limit']].to_dict('records')
        for record in cases:
            for name in ['public_tests', 'private_tests', 'generated_tests']:
                if len(record[name]['input']) != len(record[name]['output']):
                    raise ValueError('CodeContests test inputs and expected outputs are misaligned')
        coding['tests_sha256'] = [hashlib.sha256(json.dumps(record, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() for record in cases]
        coding['rule'] = [json.dumps(dict(description=self.grading['verifiers']['CodeContests']['rule'],
            source=self.grading['verifiers']['CodeContests']['test_source'], source_row=int(row.orig_dset_idx),
            problem_name=row.name, test_sets=['public_tests', 'private_tests', 'generated_tests'],
            tests_and_time_limit_sha256=row.tests_sha256), ensure_ascii=False, sort_keys=True)
            for row in coding.itertuples()]

        # 3. Join Lean statements to their named theorems in the grader's pinned repository.
        theorems = pd.read_json(self.raw_dir / parameters['paths']['minif2f'], lines=True, dtype=False)
        theorems = theorems.assign(orig_dset_idx=theorems.index)
        proving = items.loc[items.task.eq('MiniF2F-MATH')].merge(theorems[
            ['orig_dset_idx', 'id', 'formal_statement']], on='orig_dset_idx', how='left', validate='many_to_one')
        if proving.id.isna().any() or not proving.question.eq(proving.formal_statement).all():
            raise ValueError('MiniF2F source index and formal statement disagree')
        lean = (self.raw_dir / parameters['paths']['lean_test']).read_text()
        for name in proving.id.unique():
            if not re.search(r'\btheorem\s+' + re.escape(name) + r'\b', lean):
                raise ValueError('Named theorem is absent from the captured grading repository')
        proving['rule'] = [json.dumps(dict(description=self.grading['verifiers']['MiniF2F-MATH']['rule'],
            theorem=name, source=self.grading['verifiers']['MiniF2F-MATH']['theorem_source']), sort_keys=True)
            for name in proving.id]
        rules = pd.concat([coding[['item_key', 'rule']], proving[['item_key', 'rule']]]).set_index('item_key').rule
        items['rule'] = items.item_key.map(rules).fillna(items.task.map(
            {task: protocol['rule'] for task, protocol in self.grading['verifiers'].items()}))

        # 4. Preserve the actual few-shot prompt and each component's grading protocol.
        items['content'] = items.prompt
        items['raw_item_id'] = items.task + ':' + items.orig_dset_split + ':' + items.orig_dset_idx.astype(str)
        items['features'] = [dict(task=row.task, split=row.orig_dset_split,
            original_question=quote(row.question, safe='')) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=reference, rule=rule)
            for reference, rule in zip(items.reference_answer, items.rule)]
        items['verifier'] = items.task.map({task: ExactMatcher(spec=json.dumps(protocol, sort_keys=True))
            for task, protocol in self.grading['verifiers'].items()})
        subjects = items[['model']].drop_duplicates().rename(columns={'model': 'raw_label'})
        subjects['subject_key'] = subjects.raw_label
        subjects['features'] = [dict(source_model_label=model, **parameters['subject']) for model in subjects.raw_label]

        # 5. Explode samples and verdicts together; trial n is the original sample at n-1.
        responses = items[['item_key', 'model', 'task', 'source_file', 'source_row', 'samples', 'is_corrects']].explode(
            ['samples', 'is_corrects'], ignore_index=True)
        if not responses.is_corrects.map(lambda value: isinstance(value, bool)).all():
            raise ValueError('Released correctness flags must be booleans')
        if not responses.samples.map(lambda value: isinstance(value, str)).all():
            raise ValueError('Released model samples must be strings, including empty failed completions')
        responses['trial'] = responses.groupby('item_key', sort=False).cumcount() + 1
        responses['response_key'] = responses.index
        responses['subject_key'] = responses.model
        responses['response'] = responses.is_corrects.astype(float)
        responses['test_condition'] = ('source_export=' + responses.source_file + ';' +
            responses.task.map(parameters['sampling']))
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            sample_index=int(row.trial) - 1, sample=row.samples, is_correct=row.is_corrects),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'trial', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    MonkeyPowerLaws(__file__).main_from_args()
