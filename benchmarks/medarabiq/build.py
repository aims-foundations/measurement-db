"""Tabulate recorded QIMMA evaluations of the curated MedArabiQ task release."""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MedArabiQ(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, protocols = self.build_parameters, self.grading['verifiers']
        layout = parameters['layout']

        # 1. Load native per-sample tables and retain each complete source record.
        frames = []
        for path in sorted((self.raw_dir / layout['details']).rglob('*.parquet')):
            frame = pd.DataFrame(pq.read_table(path).to_pylist()).reset_index(names='source_row')
            frame['source_record'] = frame[['doc', 'metric', 'model_response']].to_dict('records')
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir / layout['details']))))
        attempts = pd.concat(frames, ignore_index=True)
        attempts = attempts.join(attempts.source_file.str.extract(layout['detail_pattern']))
        if attempts[['model', 'subset', 'run']].isna().any().any():
            raise ValueError('Unrecognized source result path')
        attempts = attempts.join(pd.json_normalize(attempts.doc, max_level=0).add_prefix('doc_'))
        attempts['source_item'] = attempts.doc_id.map(str)
        attempts['item_key'] = attempts.subset + '/' + attempts.source_item
        attempts['subject_key'] = attempts.model + '/' + attempts.run
        attempts['response_key'] = attempts.source_file + '#' + attempts.source_row.astype(str)

        # 2. Join each run to its recorded model and historical harness settings.
        frames = []
        for path in sorted((self.raw_dir / layout['configurations']).rglob('*.json')):
            record = json.loads(path.read_text())
            relative = path.relative_to(self.raw_dir / layout['configurations'])
            run = path.stem.removeprefix(layout['configuration_prefix'])
            frames.append(dict(model=str(relative.parent), run=run,
                configuration_file=str(relative), source_configuration=record['config_general']))
        configurations = pd.DataFrame(frames)
        attempts = attempts.merge(configurations, on=['model', 'run'], how='left', validate='many_to_one', indicator=True)
        if not attempts._merge.eq('both').all():
            raise ValueError('Missing recorded configuration for a result file')
        attempts = attempts.drop(columns='_merge')
        if not attempts.source_configuration.map(lambda value: value['model_name']).eq(attempts.model).all():
            raise ValueError('Result path and recorded model name disagree')

        # 3. Verify exact prompt/reference correspondence with the released task bank.
        frames = []
        for path in sorted((self.raw_dir / layout['bank']).glob('*/data.parquet')):
            frame = pd.DataFrame(pq.read_table(path).to_pylist())
            frame['bank_record'] = frame.to_dict('records')
            frame = frame.reset_index(names='source_item')
            frame['source_item'] = frame.source_item.astype(str)
            frames.append(frame.assign(subset=path.parent.name))
        bank = pd.concat(frames, ignore_index=True).rename(columns={'index': 'bank_gold_index'})
        attempts = attempts.merge(bank, on=['subset', 'source_item'],
            how='left', validate='many_to_one', indicator=True)
        if not attempts._merge.eq('both').all() or not attempts.doc_query.eq(attempts.prompt).all():
            raise ValueError('A recorded task identifier or prompt differs from the frozen task bank')
        attempts['protocol'] = attempts.subset.map(parameters['task_protocols'])
        if attempts.protocol.isna().any():
            raise ValueError('Unknown MedArabiQ task family')
        expected_choices = [list(parameters['choice_letters'].values())[:len(row.choices)]
            if row.protocol == 'multiple_choice' else [row.choices] for row in attempts.itertuples()]
        if any(left != right for left, right in zip(attempts.doc_choices, expected_choices)):
            raise ValueError('The recorded response options differ from the native task definition')
        expected_gold = attempts.bank_gold_index.where(attempts.protocol.eq('multiple_choice'), 0)
        if not attempts.doc_gold_index.eq(expected_gold).all():
            raise ValueError('The recorded reference index differs from the native task definition')

        # 4. Copy primary native grades; preserve auxiliary and corpus-level fields in traces.
        metric_names = attempts.protocol.map(parameters['primary_metrics'])
        attempts['response'] = [metrics.get(name) for metrics, name in zip(attempts.metric, metric_names)]
        attempts['reference'] = [json.dumps(choices[index], ensure_ascii=False)
            for choices, index in zip(attempts.doc_choices, attempts.doc_gold_index)]
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            configuration_file=row.configuration_file, source_record=row.source_record,
            source_configuration=row.source_configuration, reference_record=row.bank_record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]

        # 5. Emit one subject per recorded run and one item per exact stimulus/grading definition.
        items = attempts[['item_key', 'subset', 'source_item', 'doc_query', 'doc_choices',
            'doc_gold_index', 'reference', 'protocol']].copy()
        items['definition'] = items[['doc_query', 'doc_choices', 'doc_gold_index']].apply(
            lambda row: json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False), axis=1)
        if items.groupby('item_key').definition.nunique().gt(1).any():
            raise ValueError('Conflicting definitions share a native task identifier')
        items = items.drop_duplicates('item_key')
        items['raw_item_id'], items['content'] = items.item_key, items.doc_query
        items['features'] = [dict(task_family=row.subset,
            **parameters['item_features']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference, rule=protocols[row.protocol]['rule'],
            response_scale=protocols[row.protocol]['response_scale']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(protocols[key], sort_keys=True)) for key in items.protocol]
        subjects = attempts[['subject_key', 'model', 'run', 'source_configuration']].drop_duplicates('subject_key').copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.model
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=row.model,
            source_run=row.run, recorded_configuration=row.source_configuration) for row in subjects.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': attempts[['response_key', 'trace']]}


if __name__ == '__main__':
    MedArabiQ(__file__).main_from_args()
