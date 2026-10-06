"""Tabulate the released WikiHowAgent completion flags and complete conversations."""

import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class WikiHowAgent(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']
        grading = self.grading['verifiers']['completion']

        # 1. Normalize the original conversation records, keeping their file and row locations.
        paths = sorted(self.raw_dir.glob(layout['conversations']))
        sources = pd.DataFrame(dict(source_file=[str(path.relative_to(self.raw_dir)) for path in paths],
            document=[json.loads(path.read_text()) for path in paths]))
        sources['config_files'] = sources.document.map(lambda value: value['config_files'])
        sources['subject_key'] = sources.config_files.map(json.dumps)
        sources['record'] = sources.document.map(lambda value: value['total_conversations'])
        runs = sources[['source_file', 'subject_key', 'config_files', 'record']].explode('record', ignore_index=True)
        runs['source_row'] = runs.groupby('source_file', sort=False).cumcount()
        runs = runs.join(pd.json_normalize(runs.record, max_level=0))
        runs['response_key'] = runs.source_file + '#' + runs.source_row.astype(str)

        # 2. Resolve the recorded configuration paths and retain the complete reference settings.
        paths = sorted(self.raw_dir.glob(layout['configurations']))
        configurations = pd.DataFrame(dict(config_file=[str(path.relative_to(self.raw_dir / 'protocol')) for path in paths],
            configuration=[yaml.safe_load(path.read_text()) for path in paths])).set_index('config_file')
        subjects = sources[['subject_key', 'config_files']].drop_duplicates('subject_key').copy()
        subjects['configuration'] = subjects.config_files.map(lambda names: dict(zip(
            ['teacher', 'learner', 'evaluator'], configurations.loc[names, 'configuration'])))
        subjects['raw_label'] = subjects.configuration.map(lambda value: value['learner']['llm']['model'])
        subjects['features'] = [dict(harness=self.build_parameters['labels']['harness'],
            source_configuration=json.dumps(value, sort_keys=True).replace(';', r'\u003b').replace('=', r'\u003d'))
            for value in subjects.configuration]

        # 3. Keep each upstream tutorial identity and all supplied title, summary and step text.
        runs['item_key'] = runs.doc_id.astype(str) + ':' + runs.method_id.astype(str)
        runs['content'] = [json.dumps(value, ensure_ascii=True, allow_nan=False)
            for value in runs[['title', 'summary', 'tutorial']].to_dict('records')]
        if runs.groupby('item_key').content.nunique().gt(1).any():
            raise ValueError('An upstream tutorial identity has conflicting instructions')
        items = runs[['item_key', 'doc_id', 'method_id', 'content']].drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.item_key
        items['features'] = items[['doc_id', 'method_id']].to_dict('records')
        items['grading_criterion'] = [dict(rule=grading['rule']) for _ in range(len(items))]
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['spec'], sort_keys=True)) for _ in range(len(items))]

        # 4. Preserve the released binary flags without filtering failed conversations or regrading.
        runs['response'] = pd.json_normalize(runs.evaluation, max_level=0)[grading['field']]
        if not runs.response.isin([0, 1]).all():
            raise ValueError('Each recorded conversation requires its released binary completion flag')
        runs['response'] = runs.response.astype(float)

        # 5. Keep the entire original record, including long conversations and auxiliary rubric scores.
        runs['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            config_files=row.config_files, record=row.record), ensure_ascii=True, allow_nan=False)
            for row in runs.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=runs[['response_key', 'subject_key', 'item_key', 'response']],
            traces=runs[['response_key', 'trace']])


if __name__ == '__main__':
    WikiHowAgent(__file__).main_from_args()
