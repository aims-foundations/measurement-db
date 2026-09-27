"""Tabulate NatureBench's published outcomes and complete released run evidence."""

import json
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class NatureBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout = parameters['layout']

        # 1. Read the two JSON-bearing page snapshots without executing JavaScript.
        pages = {}
        for version in ['current', 'historical']:
            text = (self.raw_dir / layout[version + '_page']).read_text().strip()
            pages[version] = json.loads(text.split('=', 1)[1].strip().removesuffix(';'))
        page = pages['current']
        models = pd.json_normalize(page['models'], max_level=0).rename(columns={'id': 'subject_key'})
        models['source_model'] = page['models']
        cases = pd.json_normalize(page['cases'], max_level=0).rename(columns={'caseId': 'case_id'})
        board = pd.json_normalize(page['leaderboard'], max_level=0)
        historical = pd.DataFrame(pages['historical']['cases']).set_index('caseId').scores.to_dict()

        # 2. Unpivot both published score layouts, retaining each configuration once.
        cells = pd.DataFrame.from_dict(cases.set_index('case_id').scores.to_dict(), orient='index')
        cells = cells.rename_axis('case_id').reset_index().melt(id_vars='case_id', var_name='name', value_name='cell')
        embedded = pd.DataFrame.from_dict(board.dropna(subset=['scores']).set_index('name').scores.to_dict(), orient='index')
        embedded = embedded.rename_axis('name').reset_index().melt(id_vars='name', var_name='case_id', value_name='cell')
        responses = pd.concat([cells, embedded], ignore_index=True).dropna(subset=['cell']).reset_index(drop=True)
        if responses.duplicated(['name', 'case_id']).any():
            raise ValueError('Overlapping published score layouts need explicit reconciliation')
        responses = responses.join(pd.json_normalize(responses.cell, max_level=0))
        responses = responses.merge(models, on='name', how='left', validate='many_to_one')
        if responses.subject_key.isna().any() or not responses.state.isin(['valid', 'invalid', 'none']).all():
            raise ValueError('Unknown published configuration or score state')

        # 3. Join full-precision native scores by both model and harness.
        index = pd.read_parquet(self.raw_dir / layout['trace_index'])
        index['source_index'] = index.astype(object).where(index.notna(), None).to_dict('records')
        index['trace_configuration'] = index.harness + '__' + index.model
        index['subject_key'] = index.trace_configuration.map(parameters['trace_configurations'])
        if index.subject_key.isna().any():
            raise ValueError('Unmapped original trace configuration')
        responses = responses.merge(index[['subject_key', 'case_id', 'trace_configuration', 'source_index',
            'effective_improvement']], on=['subject_key', 'case_id'], how='left', validate='one_to_one')
        responses['score_used'] = responses.effective_improvement.where(responses.source_index.notna(), responses.value)
        if (responses.state.eq('valid') & ~np.isfinite(responses.score_used)).any():
            raise ValueError('A published valid result must have a finite score')
        responses['response'] = (responses.state.eq('valid') & responses.score_used.ge(0)).astype(float)
        responses['response_key'] = responses.subject_key + ':' + responses.case_id

        # 4. Link the published task briefs and their grading definitions, preserving the known revision.
        documents = []
        for version, directory in parameters['task_roots'].items():
            for path in sorted((self.raw_dir / directory).glob('*/metadata.json')):
                documents.append(dict(case_id=path.parent.name, brief_version=version,
                    content=(path.parent / 'problem/README.md').read_bytes().decode('utf-8'),
                    task_metadata=json.loads(path.read_bytes()), metadata_file=str(path.relative_to(self.raw_dir))))
        documents = pd.DataFrame(documents)
        responses['brief_version'] = 'current'
        older = responses.case_id.isin(parameters['revised_briefs']) & responses.subject_key.isin(parameters['legacy_brief_configurations'])
        responses.loc[older, 'brief_version'] = 'historical'
        responses['item_key'] = responses.case_id + ':' + responses.brief_version + ':' + responses.validityJudge
        items = responses[['item_key', 'case_id', 'brief_version', 'validityJudge']].drop_duplicates()
        items = items.merge(documents, on=['case_id', 'brief_version'], how='left', validate='many_to_one')
        items = items.merge(cases[['case_id', *parameters['item_columns']]], on='case_id', how='left', validate='many_to_one')
        if items.content.isna().any():
            raise ValueError('A published result is missing its task description')
        items['raw_item_id'] = items.case_id
        features = items[list(parameters['item_columns'])].rename(columns=parameters['item_columns'])
        features['input_scope'] = parameters['labels']['input_scope']
        items['features'] = features.to_dict('records')
        items['grading_criterion'] = [dict(rule=json.dumps(dict(rule=grading['rule'],
            performance_entries=row.task_metadata['performance_entries']), sort_keys=True)) for row in items.itertuples()]
        items['verifier'] = [Judge(judge=row.validityJudge, judged_by='llm',
            spec=json.dumps(grading['verifiers']['published_match_sota'], sort_keys=True)) for row in items.itertuples()]

        # 5. Preserve every original trace file as text, including malformed JSON lines.
        files = []
        for path in sorted((self.raw_dir / layout['trajectories']).glob('*/*/*')):
            if path.is_file():
                files.append(dict(trace_configuration=path.parent.parent.name, case_id=path.parent.name,
                    file_record=(str(path.relative_to(self.raw_dir)), path.read_bytes().decode('utf-8'))))
        bundles = pd.DataFrame(files).groupby(['trace_configuration', 'case_id']).file_record.agg(list).map(dict)
        bundles = bundles.rename('native_files').reset_index()
        responses = responses.merge(bundles, on=['trace_configuration', 'case_id'], how='left', validate='many_to_one')
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(leaderboard_file=layout['current_page'], case_id=row.case_id,
            configuration=row.source_model, cell=row.cell,
            native_index=row.source_index if isinstance(row.source_index, dict) else None,
            score_used=None if pd.isna(row.score_used) else row.score_used,
            native_files=row.native_files if isinstance(row.native_files, dict) else {},
            historical_leaderboard_cell=historical.get(row.case_id, {}).get(row.name),
            task_brief_version=row.brief_version), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]

        # 6. Retain the published harness, budget and access conditions for each subject.
        subjects = models[['subject_key']].copy()
        subjects['raw_label'] = models.displayName.fillna(models.name)
        # Escape the feature format's separators; the original strings remain in each trace.
        subjects['features'] = models.reindex(columns=parameters['subject_columns']).rename(
            columns=parameters['subject_columns']).apply(
                lambda row: row.dropna().map(lambda value: quote(str(value), safe=' /-._(),:')).to_dict(), axis=1)
        return {'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    NatureBench(__file__).main_from_args()
