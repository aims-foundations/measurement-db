"""Tabulate all published research attempts, their native scores and released evidence."""

import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class ResearchClawBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Use the complete native run index, before best-run leaderboard selection.
        original_index = json.loads((self.raw_dir / layout['index']).read_bytes())
        responses = pd.json_normalize(original_index)
        responses['native_index'] = original_index
        responses['source_index'] = layout['index']
        historical_index = json.loads((self.raw_dir / layout['historical_index']).read_bytes())
        historical = pd.json_normalize(historical_index)
        historical['native_index'] = historical_index
        historical['source_index'] = layout['historical_index']
        selected = historical[list(parameters['historical_configuration'])].eq(parameters['historical_configuration']).all(axis=1)
        responses = pd.concat([responses, historical.loc[selected]], ignore_index=True)
        if responses.run_id.duplicated().any():
            raise ValueError('A historical attempt duplicates a current native run ID')
        responses = responses.rename(columns={'run_id': 'response_key', 'task_id': 'item_key', 'total_score': 'response'})
        responses = responses.sort_values(['timestamp', 'response_key']).reset_index(drop=True)
        configurations = responses[['agent_name', 'model', 'model_display']].drop_duplicates()
        configurations['subject_key'] = configurations.apply(lambda row: json.dumps(row.to_list()), axis=1)
        responses = responses.merge(configurations, on=['agent_name', 'model', 'model_display'], validate='many_to_one')
        subjects = configurations.rename(columns={'agent_name': 'raw_label'}).copy()
        subjects['features'] = [dict(harness=labels['harness'], recorded_agent=row.raw_label, recorded_model=row.model,
            recorded_model_display=row.model_display, historical_settings=labels['historical_settings'])
            for row in subjects.itertuples()]

        # 2. Read the original instruction templates and complete published rubrics.
        documents = []
        for path in sorted((self.raw_dir / layout['tasks']).glob('*/INSTRUCTIONS.md')):
            documents.append(dict(item_key=path.parent.name, content=path.read_bytes().decode('utf-8'),
                checklist=json.loads(path.with_name('checklist.json').read_bytes())))
        items = pd.DataFrame(documents)
        items = responses[['item_key']].drop_duplicates().merge(items, on='item_key', how='left', validate='one_to_one')
        if items.content.isna().any():
            raise ValueError('A recorded attempt lacks its published task instruction')
        items['raw_item_id'] = items.item_key
        items['features'] = [dict(domain=key.split('_')[0], input_scope=labels['input_scope']) for key in items.item_key]
        items['grading_criterion'] = items.checklist.map(lambda checklist: dict(rule=json.dumps(
            dict(rule=grading['rule'], published_checklist=checklist), ensure_ascii=False)))
        items['verifier'] = [Judge(judged_by='llm', spec=json.dumps(grading['verifiers']['reported'], sort_keys=True))] * len(items)

        # 3. Attach the original task inputs; hidden reference papers and rubrics stay separate.
        assets = []
        for path in sorted((self.raw_dir / layout['original_tasks']).glob('*/*/**/*')):
            if not path.is_file():
                continue
            relative = path.relative_to(self.raw_dir / layout['original_tasks'])
            if relative.parts[1] not in ['data', 'related_work']:
                continue
            logical_path = re.sub(parameters['patterns']['filename_escape'], lambda m: chr(int(m[1], 16)),
                                  str(Path(*relative.parts[1:])))
            assets.append(dict(item_key=relative.parts[0], attachment=dict(source_path=path, path=logical_path,
                media_type=parameters['media_types'].get(Path(logical_path).suffix.lower(), labels['default_media_type']),
                role='input')))
        assets = pd.DataFrame(assets).groupby('item_key', sort=False).attachment.agg(list).rename('attachments').reset_index()
        items = items.merge(assets, on='item_key', how='left', validate='one_to_one')
        if items.attachments.isna().any():
            raise ValueError('A scientific task lacks its published input assets')

        # 4. Join the original reports, grading records, exact instructions and exported log lines.
        bundles = []
        for path in sorted((self.raw_dir / layout['runs']).glob('*/data.json')):
            record = json.loads(path.read_bytes())
            instructions = (path.parent / 'workspace/INSTRUCTIONS.md').read_bytes().decode('utf-8')
            bundles.append(dict(response_key=path.parent.name, native_detail=record, instructions=instructions,
                exported_output=json.loads(path.with_name('output.json').read_bytes()),
                exported_files=json.loads(path.with_name('files.json').read_bytes())))
        responses = responses.merge(pd.DataFrame(bundles), on='response_key', how='left', validate='one_to_one')
        templates = items.set_index('item_key').content.str.split().str.join(' ')
        detailed = responses.loc[responses.native_detail.notna()]
        normalized = detailed.instructions.str.replace(parameters['patterns']['workspace'], labels['workspace'], regex=True).str.split().str.join(' ')
        if not normalized.eq(detailed.item_key.map(templates)).all():
            raise ValueError('A recorded instruction differs from its published task template')
        for field, expected in [('run_id', detailed.response_key), ('task_id', detailed.item_key),
                                ('agent_name', detailed.agent_name), ('model', detailed.model)]:
            if not detailed.native_detail.map(lambda record: record[field]).eq(expected).all():
                raise ValueError('A native run record disagrees with its index')
        if not detailed.native_detail.map(lambda record: record['score']['total_score']).eq(detailed.response).all():
            raise ValueError('Native grades differ between the run record and index')
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_index=row.source_index, native_index=row.native_index,
            native_detail=row.native_detail if isinstance(row.native_detail, dict) else None,
            recorded_instructions=row.instructions if isinstance(row.instructions, str) else None,
            exported_output=row.exported_output if isinstance(row.exported_output, list) else None,
            exported_files=row.exported_files if isinstance(row.exported_files, list) else None),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    ResearchClawBench(__file__).main_from_args()
