#!/usr/bin/env python3
"""Tabulate original Rakuda ratings with full answers and explicit judge protocols."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class Rakuda(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters['labels']

        # 1. Concatenate original judgments, preserving physical JSONL records exactly.
        files = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['judgments'])):
            with path.open() as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            table = pd.json_normalize(records, max_level=0)
            table['record'] = records
            table['subject_key'], table['judge_group'] = path.stem, path.parts[-3]
            table['source_file'] = str(path.relative_to(self.raw_dir))
            files.append(table.rename_axis('source_row').reset_index())
        responses = pd.concat(files, ignore_index=True)
        responses['subject_key'] = responses.subject_key.replace(parameters['filename_aliases'])
        questions = pd.read_json(self.raw_dir / parameters['layout']['questions'], lines=True, convert_dates=False)

        # 2. Retain generating-model aliases without guessing unavailable runtime settings.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=labels['harness'], source_model_filename=model) for model in subjects.subject_key]

        # 3. Join complete questions to the separately identified judge protocols.
        items = responses[['question_id', 'judge_group']].drop_duplicates().merge(questions, on='question_id', validate='many_to_one')
        items['item_key'] = items.question_id + ':' + items.judge_group
        items['raw_item_id'] = items.item_key
        items['features'] = [dict(category=row.category, judge_group=row.judge_group, input_scope=labels['input_scope'])
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        protocols = items.judge_group.map(self.grading['verifiers'])
        items['verifier'] = [Judge(judge=protocol['judge'], judged_by='llm', spec=json.dumps(protocol, sort_keys=True)) for protocol in protocols]
        items = items.rename(columns={'text': 'content'})

        # 4. Keep native scores and source occasions, including unavailable judgments.
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)
        responses['item_key'] = responses.question_id + ':' + responses.judge_group
        responses['response'] = responses.score
        responses['test_condition'] = responses.response_key

        # 5. Preserve the complete judged answer and any original judge explanation.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            record=row.record), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects,
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    Rakuda(__file__).main_from_args()
