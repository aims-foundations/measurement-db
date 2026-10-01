"""Tabulate original Tengu answers and ratings with their separate judge protocols."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Tengu(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate complete native judgment records into one table.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['judgments'])):
            with path.open() as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            frame = pd.json_normalize(records, max_level=0).assign(record=records)
            parts.append(frame.rename_axis('source_row').reset_index().assign(
                source_file=str(path.relative_to(self.raw_dir)), subject_key=path.stem, judge_group=path.parts[-3]))
        responses = pd.concat(parts, ignore_index=True)
        responses['subject_key'] = responses.subject_key.replace(parameters['filename_aliases'])
        questions = pd.read_parquet(self.raw_dir / parameters['layout']['questions'])
        questions['source_question_id'] = questions.index + 1

        # 2. Join by the full original question, rather than its position in a file.
        responses = responses.merge(questions[['Question', 'source_question_id']],
            on='Question', how='left', validate='many_to_one')
        if responses.source_question_id.isna().any():
            raise ValueError('An original judgment has no matching source question')
        responses = responses.astype(object).where(responses.notna(), None)
        responses['definition'] = [json.dumps([row.Question, row.Answer, row.Criteria, row.judge_group],
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_model_filename=model,
            historical_configuration=parameters['labels']['historical_configuration']) for model in subjects.subject_key]

        # 3. Keep native ratings; explicitly flag missing and invalid source grades.
        responses['response'] = pd.to_numeric(responses.score, errors='raise')
        if responses.response.isin([float('inf'), float('-inf')]).any():
            raise ValueError('An infinite source rating is not an unavailable grade')
        valid = responses.response.isin(self.INFO['response_scale']['values'])
        responses['grade_status'] = 'recorded'
        responses.loc[responses.response.isna(), 'grade_status'] = 'unavailable_upstream_score'
        responses.loc[responses.response.notna() & ~valid, 'grade_status'] = 'invalid_upstream_score'
        responses.loc[~valid, 'response'] = None
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)
        responses['item_key'] = responses.definition
        responses['test_condition'] = responses.response_key

        # 4. Retain the original reference and rubric under each recorded judge.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.source_question_id.astype(str) + ':' + items.judge_group
        items['content'] = items.Question
        items['features'] = [dict(category=row.Category, judge_group=row.judge_group,
            input_scope=parameters['labels']['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + '\nOriginal criteria:\n' + row.Criteria,
            **(dict(reference_answer=row.Answer) if row.Answer is not None else {})) for row in items.itertuples()]
        items['verifier'] = [Judge(judge=profile['judge'], judged_by='llm', spec=json.dumps(profile, sort_keys=True))
            for profile in items.judge_group.map(self.grading['verifiers'])]

        # 5. Preserve full answers, judge explanations and the original invalid scores.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            grade_status=row.grade_status, record=row.record), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    Tengu(__file__).main_from_args()
