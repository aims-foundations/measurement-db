#!/usr/bin/env python3
"""Tabulate QATCH's original scores, full predictions and verified table contexts."""

import ast
from fnmatch import fnmatch
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class QATCH(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']
        metrics = list(self.grading['verifiers'])

        # 1. Read original CSV strings, then reshape the merged model columns.
        frames = []
        with ZipFile(self.raw_dir / layout['results']) as archive:
            for name in sorted(archive.namelist()):
                if name != layout['merged'] and not fnmatch(name, layout['result_pattern']):
                    continue
                with archive.open(name) as stream:
                    table = pd.read_csv(stream, header=None, dtype=str, keep_default_na=False)
                # Preserve the source's blank index-column header instead of an inferred name.
                table.columns = table.iloc[0]
                table = table.iloc[1:].reset_index(drop=True)
                table['record'] = table.to_dict('records')
                table = table.rename_axis('source_row').reset_index()
                table['source_file'] = name
                table['task'], table['dataset'] = Path(name).parts[1:3]
                if name == layout['merged']:
                    table = table.rename(columns={model + '_predictions': 'predictions_' + model
                        for model in parameters['merged_models']})
                    table = pd.wide_to_long(table, stubnames=['predictions', *metrics], i='source_row',
                        j='source_model', sep='_', suffix='(' + '|'.join(parameters['merged_models']) + ')').reset_index()
                else:
                    table['source_model'] = Path(name).parts[3]
                frames.append(table)
        attempts = pd.concat(frames, ignore_index=True)
        attempts['context_key'] = attempts.source_file + ':' + attempts.db_id + ':' + attempts.tbl_name

        # 2. Recover custom-table headers from complete, unfiltered projections.
        # Cell attachments are exact released CSV field strings, not newly sampled data.
        contexts = []
        custom = attempts.loc[attempts.dataset.eq('custom-data')].copy()
        custom['projection'] = custom['query'].str.extract(r'(?i)^\s*select\s+(.*?)\s+from\s+[^\s;]+\s*;?\s*$', expand=False)
        for key, group in custom.groupby('context_key', sort=True):
            complete = group.loc[group.projection.eq('*')]
            if len(complete) != 1:
                raise ValueError('Each custom table needs one original complete-table result')
            full = complete.iloc[0]
            vectors = list(zip(*ast.literal_eval(full.query_result)))
            projected = group.loc[group.projection.notna() & group.projection.ne('*') &
                ~group.projection.str.lower().str.startswith('distinct ', na=False)]
            columns = []
            for row in projected.itertuples():
                names = [name.strip().strip('"`[]') for name in row.projection.split(',')]
                values = list(zip(*ast.literal_eval(row.query_result)))
                if len(names) != len(values):
                    raise ValueError('A source column projection has inconsistent width')
                columns.append(pd.DataFrame({'name': names, 'vector': values}))
            names = pd.concat(columns, ignore_index=True).drop_duplicates(['name', 'vector'])
            ordered = pd.DataFrame({'ordinal': range(len(vectors)), 'vector': vectors}).merge(
                names, on='vector', how='left', validate='one_to_one').sort_values('ordinal')
            if ordered.name.isna().any():
                raise ValueError('A custom-table column has no unambiguous source projection')
            contexts.append(dict(context_key=key, columns=ordered.name.tolist(),
                attachment=dict(data=full.query_result.encode('utf8'), path=labels['table_attachment'],
                    media_type='text/plain', role='input'), context_kind='released_complete_table_cells'))

        # Keep official Spider files intact; the source audit checks their associations.
        spider = attempts.loc[attempts.dataset.eq('spider'), ['context_key', 'db_id', 'tbl_name']].drop_duplicates()
        with ZipFile(self.raw_dir / layout['spider']) as archive:
            schemas = pd.json_normalize(json.loads(archive.read(layout['spider_prefix'] + 'tables.json')), max_level=0)
            spider = spider.merge(schemas, on='db_id', how='left', validate='many_to_one')
            files = {db: archive.read(layout['spider_prefix'] + f'database/{db}/{db}.sqlite') for db in spider.db_id.unique()}
        for row in spider.itertuples():
            indices = [index for index, name in enumerate(row.table_names_original) if name.lower() == row.tbl_name.lower()]
            if len(indices) != 1:
                raise ValueError('A source question does not identify one Spider table')
            columns = [name for index, name in row.column_names_original if index == indices[0]]
            contexts.append(dict(context_key=row.context_key, columns=columns,
                attachment=dict(data=files[row.db_id], path=labels['database_attachment'],
                    media_type='application/vnd.sqlite3', role='input'), context_kind='official_spider_database'))
        attempts = attempts.merge(pd.DataFrame(contexts), on='context_key', how='left', validate='many_to_one')
        if attempts.context_kind.isna().any():
            raise ValueError('An original result has no table context')

        # 3. Separate model/task configurations, preserving full source records once.
        attempts['model'] = attempts.source_model.map(parameters['models'])
        if attempts.model.isna().any():
            raise ValueError('An original model alias has no documented mapping')
        attempts['subject_key'] = attempts.task + ':' + attempts.model
        subjects = attempts[['subject_key', 'model', 'task']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['harness'], prediction_task=row.task) for row in subjects.itertuples()]
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_model=row.source_model, record=row.record), ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        attempts['content'] = [json.dumps(dict(question=row.question, task=row.task, dataset=row.dataset,
            database=row.db_id, table=row.tbl_name, columns=row.columns, context_kind=row.context_kind,
            table_resource=row.attachment['path']), ensure_ascii=False, sort_keys=True) for row in attempts.itertuples()]
        attempts['reference'] = [json.dumps(dict(sql=row.query, released_target=row.query_result), ensure_ascii=False, sort_keys=True)
            for row in attempts.itertuples()]

        # 4. Unpivot recorded metrics; blanks outside ORDERBY are inapplicable.
        identifiers = ['source_file', 'source_row', 'source_model', 'subject_key', 'task', 'dataset', 'db_id',
            'tbl_name', 'sql_tags', 'content', 'reference', 'attachment', 'trace']
        responses = attempts.melt(id_vars=identifiers, value_vars=metrics, var_name='metric', value_name='score_text')
        inapplicable = responses.metric.eq('tuple_order') & ~responses.sql_tags.str.contains('orderby', case=False, regex=False)
        if responses.loc[inapplicable, 'score_text'].ne('').any():
            raise ValueError('The source reports an order score for an inapplicable task')
        responses = responses.loc[~inapplicable].copy()
        responses['response'] = responses.score_text.map(lambda value: None if value == '' else float(value))
        if not (responses.score_text.eq('') | responses.response.between(0, 1)).all():
            raise ValueError('A recorded QATCH grade is not finite within its [0, 1] scale')
        responses['item_key'] = responses.source_file + ':' + responses.source_row.astype(str) + ':' + responses.metric
        responses['response_key'] = responses.item_key + ':' + responses.source_model
        responses['test_condition'] = responses.source_file + ':' + responses.source_row.astype(str) + ';model=' + responses.source_model

        # 5. Keep distinct grading protocols and link every metric to its original output.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.item_key
        items['features'] = [dict(task=row.task, dataset=row.dataset, database=row.db_id, table=row.tbl_name)
            for row in items.itertuples()]
        items['attachments'] = items.attachment.map(lambda attachment: [attachment])
        items['grading_criterion'] = [dict(reference_answer=row.reference,
            rule=self.grading['verifiers'][row.metric]['rule']) for row in items.itertuples()]
        items['verifier'] = items.metric.map(lambda metric: ExactMatcher(spec=json.dumps(self.grading['verifiers'][metric], sort_keys=True)))
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            traces=responses[['response_key', 'trace']])


if __name__ == '__main__':
    QATCH(__file__).main_from_args()
