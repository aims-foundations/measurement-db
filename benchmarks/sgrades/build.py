"""Tabulate original S-GRADES predictions, inputs and verified human references."""

from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class SGrades(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        paths, labels = parameters['paths'], parameters['labels']
        release = self.raw_dir / paths['release']

        # 1. Load the author's target inputs and complete three-call prediction rows.
        inputs, csv_frames = [], []
        family_names = sorted([*parameters['id_columns'], *parameters['dataset_aliases']], key=len, reverse=True)
        for path in sorted(release.rglob('*.csv')):
            relative = str(path.relative_to(release))
            if relative in parameters['corrupt_input_files'] or any(word in relative for word in parameters['training_input_markers']):
                continue
            family = next((name for name in family_names if name in path.name), None)
            if family is None:
                continue
            dataset = parameters['dataset_aliases'].get(family, family)
            id_column = parameters['id_columns'][dataset]
            text_column = parameters['text_columns'][dataset]
            question_column = parameters['question_columns'][dataset]
            header = pd.read_csv(path, nrows=0).columns
            if id_column not in header or text_column not in header:
                continue
            source = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame = pd.DataFrame(dict(dataset=dataset, source_id=source[id_column], answer=source[text_column],
                question=source[question_column] if question_column in source else '',
                question_id=source['question_id'] if 'question_id' in source else '',
                question_column_available=question_column in source,
                export_prediction=source.get(parameters['score_columns'][dataset], ''),
                input_file=str(path.relative_to(self.raw_dir)), input_row=source.index))
            inputs.append(frame)
            if path.name.endswith('_3call_FULL.csv'):
                model = parameters['model_codes'][str(path.parent.parent.relative_to(release))]
                expected = parameters['model_names'][model] + '_D_' + dataset + '_3call_FULL.csv'
                if path.name != expected:
                    raise ValueError('Model, dataset and native filename disagree: ' + str(path))
                frame = frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=source.index,
                    model=model, strategy=path.parent.name.removesuffix('_3call_predictions'),
                    export_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                frame['source_record'] = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    for row in source.to_dict('records')]
                csv_frames.append(pd.concat([frame, source[['prediction_1', 'prediction_2', 'prediction_3']]], axis=1))
        for source_path, dataset in parameters['author_input_tables'].items():
            source = pd.read_csv(self.raw_dir / source_path, dtype=str, keep_default_na=False)
            question_column = parameters['question_columns'][dataset]
            inputs.append(pd.DataFrame(dict(dataset=dataset, source_id=source[parameters['id_columns'][dataset]],
                question=source.get(question_column, ''), answer=source[parameters['text_columns'][dataset]],
                question_id=source.get('question_id', ''), question_column_available=question_column in source,
                export_prediction='', input_file=source_path, input_row=source.index)))
        inputs = pd.concat(inputs, ignore_index=True)
        input_columns = ['dataset', 'source_id', 'question_id', 'question', 'answer', 'question_column_available']
        inputs['input_key'] = inputs.groupby(input_columns, sort=False, dropna=False).ngroup().astype(str)
        definitions = inputs.drop_duplicates('input_key').copy()
        csv_records = pd.concat(csv_frames, ignore_index=True).merge(definitions[input_columns + ['input_key']],
            on=input_columns, how='left', validate='many_to_one')
        copies = csv_records.drop_duplicates('source_file').groupby('export_sha256', sort=False).source_file.agg(list)
        csv_records['identical_exports'] = csv_records.export_sha256.map(copies)

        # 2. Recover original references by exact question/answer text and retained IDs.
        frames = []
        with zipfile.ZipFile(self.raw_dir / paths['references']) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith('.xml'):
                    continue
                _, granularity, corpus = member.split('/')[:3]
                data = archive.read(member)
                root = ET.fromstring(data)
                frame = pd.read_xml(io.BytesIO(data), xpath='./studentAnswers/studentAnswer',
                    parser='etree', dtype=str).rename(columns={
                        'id': 'reference_id', 'accuracy': 'gold', 'studentAnswer': 'answer'})
                frames.append(frame.assign(dataset=parameters['corpus_names'][corpus] + '_' + granularity,
                    question_id=root.attrib['id'], question=root.findtext('questionText'),
                    reference_file=paths['references'], reference_member=member))
        references = pd.concat(frames, ignore_index=True)
        references[['question', 'answer']] = references[['question', 'answer']].fillna('').apply(lambda col: col.str.strip())
        definitions[['question', 'answer']] = definitions[['question', 'answer']].apply(lambda col: col.str.strip())
        join = ['dataset', 'question_id', 'question', 'answer']
        references['reference_record'] = references[['reference_file', 'reference_member', 'reference_id', 'gold']].to_dict('records')
        bank = references.groupby(join, dropna=False, sort=False).agg(gold=('gold', 'first'),
            label_count=('gold', 'nunique'), reference_records=('reference_record', list)).reset_index()
        bank.loc[bank.label_count.ne(1), 'gold'] = None
        exact = references.loc[references.dataset.str.startswith('SciEntSBank')].rename(columns={'reference_id': 'source_id'})
        exact = exact.groupby(join + ['source_id'], dropna=False, sort=False).agg(
            exact_gold=('gold', 'first'), exact_count=('gold', 'nunique'),
            exact_records=('reference_record', list)).reset_index()
        definitions = definitions.merge(bank, how='left', on=join, validate='many_to_one').merge(
            exact, how='left', on=join + ['source_id'], validate='many_to_one')
        preserved = definitions.exact_count.eq(1)
        definitions.loc[preserved, 'gold'] = definitions.loc[preserved, 'exact_gold']
        definitions.loc[preserved, 'reference_records'] = definitions.loc[preserved, 'exact_records']
        definitions['reference_status'] = 'reference_unavailable'
        definitions.loc[definitions.label_count.gt(1), 'reference_status'] = 'conflicting_reference'
        definitions.loc[definitions.gold.notna(), 'reference_status'] = 'verified_reference'

        # 3. Flatten native JSON runs; retain original records and all snapshot locations.
        json_frames = []
        for path in sorted(release.rglob('*.json')):
            data = json.loads(path.read_text(), parse_constant=lambda value: {'source_nonfinite': value})
            if not isinstance(data, dict) or not isinstance(data.get('datasets'), list):
                continue
            datasets = pd.json_normalize(data['datasets'], max_level=0)
            if datasets.empty or not any(kind in datasets for kind in ['predictions', 'failed_predictions']):
                continue
            identity = {name: data.get(name) for name in parameters['json_run_identity_fields']}
            run_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            run_metadata = json.dumps({k: v for k, v in data.items() if k != 'datasets'}, ensure_ascii=False, allow_nan=False)
            for kind in ['predictions', 'failed_predictions']:
                if kind not in datasets:
                    continue
                frame = datasets.loc[datasets[kind].map(lambda value: isinstance(value, list) and bool(value))].copy()
                frame['dataset_metadata'] = [json.dumps({k: v for k, v in data['datasets'][i].items()
                    if k not in ['predictions', 'failed_predictions']}, ensure_ascii=False, allow_nan=False) for i in frame.index]
                frame = frame.explode(kind)
                if frame.empty:
                    continue
                frame['source_row'] = frame.groupby(level=0).cumcount()
                frame['dataset_position'] = frame.index
                frame['source_id'] = frame[kind].map(lambda row: str(row.get('id', row.get('essay_id'))))
                frame['prediction'] = frame[kind].map(lambda row: row.get('prediction'))
                frame['source_record'] = frame[kind].map(lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
                frame['dataset'] = frame['dataset_name' if 'dataset_name' in frame else 'name'].str.removeprefix('D_').replace(parameters['dataset_aliases'])
                frame['linked_csv'] = [str((path.parent / entry).resolve().relative_to(self.raw_dir.resolve()))
                    if isinstance(entry, str) and (path.parent / entry).resolve().is_relative_to(self.raw_dir.resolve())
                    and (path.parent / entry).is_file() else ''
                    for entry in frame.get('csv_output', pd.Series('', index=frame.index))]
                frame = frame.assign(source_file=str(path.relative_to(self.raw_dir)), run_key=run_key,
                    model=data['model_code'], strategy=data['reasoning_approach'], kind=kind,
                    run_metadata=run_metadata, experiment='native_json')
                json_frames.append(frame[['dataset', 'source_id', 'prediction', 'source_record', 'source_file',
                    'source_row', 'dataset_position', 'linked_csv', 'run_key', 'model', 'strategy', 'kind',
                    'run_metadata', 'dataset_metadata', 'experiment']])
        native = pd.concat(json_frames, ignore_index=True)
        if not native.dataset.isin(parameters['id_columns']).all():
            raise ValueError('An undeclared JSON dataset needs source review')

        # Export filenames can be overwritten. Check all recorded predictions before
        # using a declared export to choose between conflicting input definitions.
        exported = inputs[['input_file', 'dataset', 'source_id', 'export_prediction']].rename(
            columns={'input_file': 'linked_csv', 'export_prediction': 'prediction'})
        recorded = native.loc[native.kind.eq('predictions')].drop_duplicates(
            ['source_file', 'dataset_position', 'source_id'], keep='last').copy()
        for table in [exported, recorded]:
            table['comparison'] = table.prediction.map(lambda value: '' if value is None else
                str(value).strip() if isinstance(value, (str, int, float)) else json.dumps(value, sort_keys=True)).astype(object)
            numeric = table.comparison.str.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
            table.loc[numeric, 'comparison'] = table.loc[numeric, 'comparison'].map(Decimal)
        exported = exported.groupby(['linked_csv', 'dataset', 'source_id'], sort=False).agg(
            exported_value=('comparison', 'first'), exported_count=('comparison', 'nunique')).reset_index()
        recorded = recorded.merge(exported, on=['linked_csv', 'dataset', 'source_id'],
            how='left', validate='many_to_one')
        recorded['matches'] = recorded.exported_count.eq(1) & recorded.comparison.eq(recorded.exported_value)
        corroboration = recorded.groupby(['source_file', 'dataset_position'], sort=False).matches.all().rename(
            'declared_csv_verified').reset_index()
        native = native.merge(corroboration, on=['source_file', 'dataset_position'],
            how='left', validate='many_to_one')
        native['declared_csv_verified'] = native.declared_csv_verified.eq(True)
        event = ['run_key', 'dataset', 'kind', 'source_id', 'source_record']
        native['occurrence'] = native.groupby(['source_file'] + event, sort=False, dropna=False).cumcount()
        native['source_location'] = native[['source_file', 'dataset_position', 'kind', 'source_row']].to_dict('records')
        locations = native.groupby(event + ['occurrence'], sort=False, dropna=False).source_location.agg(list).rename('source_locations')
        native = native.drop_duplicates(event + ['occurrence']).merge(locations, on=event + ['occurrence'], validate='one_to_one')
        native['event_key'] = [hashlib.sha256(json.dumps([row.run_key, row.dataset, row.kind, row.source_id,
            row.source_record, int(row.occurrence)], ensure_ascii=False).encode()).hexdigest() for row in native.itertuples()]

        # 4. Associate JSON results with unambiguous original input definitions.
        lookup = inputs.groupby(['dataset', 'source_id'], sort=False).agg(
            input_key=('input_key', 'first'), candidate_count=('input_key', 'nunique')).reset_index()
        native = native.merge(lookup, on=['dataset', 'source_id'], how='left', validate='many_to_one')
        native['input_method'] = 'exact_id'
        numeric_inputs = inputs.loc[inputs.source_id.str.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')].copy()
        numeric_inputs['numeric_id'] = numeric_inputs.source_id.map(Decimal)
        numeric_lookup = numeric_inputs.groupby(['dataset', 'numeric_id'], sort=False).agg(
            numeric_key=('input_key', 'first'), numeric_count=('input_key', 'nunique')).reset_index()
        native['numeric_id'] = native.source_id.where(native.source_id.str.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')).map(
            lambda value: Decimal(value) if isinstance(value, str) else None)
        native = native.merge(numeric_lookup, on=['dataset', 'numeric_id'], how='left', validate='many_to_one')
        fallback = native.candidate_count.isna() & native.numeric_count.notna()
        native.loc[fallback, 'input_key'] = native.loc[fallback, 'numeric_key']
        native.loc[fallback, 'candidate_count'] = native.loc[fallback, 'numeric_count']
        native.loc[fallback, 'input_method'] = 'numeric_id_representation'
        declared = inputs.groupby(['input_file', 'dataset', 'source_id'], sort=False).agg(
            declared_key=('input_key', 'first'), declared_count=('input_key', 'nunique')).reset_index().rename(columns={'input_file': 'linked_csv'})
        native = native.merge(declared, on=['linked_csv', 'dataset', 'source_id'], how='left', validate='many_to_one')
        selected = native.candidate_count.gt(1) & native.declared_count.eq(1) & native.declared_csv_verified
        native.loc[selected, 'input_key'] = native.loc[selected, 'declared_key']
        native.loc[selected, 'candidate_count'] = 1
        native.loc[selected, 'input_method'] = 'declared_native_csv'
        unresolved = ~native.candidate_count.eq(1)
        print(f'Native JSON records: {len(native)} distinct; {int(unresolved.sum())} unresolved inputs remain in raw.', flush=True)
        native = native.loc[~unresolved].copy()
        native['response_key'] = 'json/' + native.event_key

        # 5. Expand retained CSV outputs and combine both native record formats.
        csv_records['record_key'] = csv_records.source_file + '#' + csv_records.source_row.astype(str)
        columns = ['prediction_1', 'prediction_2', 'prediction_3']
        csv_attempts = csv_records.melt(id_vars=[column for column in csv_records if column not in columns],
            value_vars=columns, var_name='source_column', value_name='prediction')
        csv_attempts['retained_position'] = csv_attempts.source_column.str[-1].astype(int)
        csv_attempts['response_key'] = 'csv/' + csv_attempts.record_key + '/' + csv_attempts.source_column
        csv_attempts['experiment'] = 'three_call_csv'
        csv_attempts['kind'] = 'predictions'
        shared = ['response_key', 'input_key', 'model', 'strategy', 'experiment', 'kind', 'prediction']
        attempts = pd.concat([csv_attempts[shared], native[shared]], ignore_index=True).merge(
            definitions, on='input_key', how='left', validate='many_to_one')
        attempts['grading_kind'] = attempts.dataset.str.extract(r'_(2way|3way)$')[0].fillna('numeric')
        attempts['grading_status'] = attempts.reference_status
        normalized = attempts.prediction.map(lambda value: value.strip().lower() if isinstance(value, str) else None)
        attempts['response'] = None
        numeric = attempts.grading_kind.eq('numeric')
        parsed = normalized.isin(parameters['category_labels'])
        graded = ~numeric & attempts.gold.notna() & parsed
        attempts.loc[graded, 'response'] = normalized[graded].eq(attempts.loc[graded, 'gold']).astype(float)
        attempts.loc[graded, 'grading_status'] = 'verified_category_match'
        attempts.loc[~numeric & attempts.gold.notna() & ~parsed, 'grading_status'] = 'unparsed_reply'
        attempts.loc[numeric, 'grading_status'] = 'numeric_reference_unavailable'
        missing = attempts.kind.eq('failed_predictions') | attempts.prediction.isna() | normalized.eq('')
        attempts.loc[missing, 'grading_status'] = 'prediction_unavailable'
        attempts.loc[missing, 'response'] = None

        # 6. Preserve recorded run configurations rather than inferring missing settings.
        csv_attempts['subject_configuration'] = [json.dumps(dict(harness=labels['csv_harness'],
            provider_model_id=row.model, reasoning_strategy=row.strategy, trial_order=labels['trial_order']), sort_keys=True)
            for row in csv_attempts.itertuples()]
        native['subject_configuration'] = [json.dumps(dict(harness=labels['json_harness'],
            provider_model_id=row.model, reasoning_strategy=row.strategy,
            recorded_run={key: json.loads(row.run_metadata).get(key) for key in parameters['json_run_identity_fields']},
            configuration_status=labels['json_configuration']), sort_keys=True, ensure_ascii=False, allow_nan=False)
            for row in native.itertuples()]
        configurations = pd.concat([csv_attempts[['response_key', 'model', 'subject_configuration']],
            native[['response_key', 'model', 'subject_configuration']]], ignore_index=True)
        configurations['subject_key'] = configurations.subject_configuration
        subjects = configurations.drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.model.map(parameters['model_names'])
        if subjects.raw_label.isna().any():
            raise ValueError('An undeclared model identifier needs source review')
        subjects['features'] = subjects.subject_configuration.map(json.loads)
        attempts = attempts.merge(configurations[['response_key', 'subject_key']], on='response_key', validate='one_to_one')

        # 7. Keep complete known input text and explicit reference availability.
        attempts['item_key'] = attempts.input_key
        items = attempts.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.dataset + '::' + items.source_id
        items['content'] = 'Question: ' + items.question + '\n\nStudent answer: ' + items.answer
        items['features'] = [dict(dataset=row.dataset, input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.gold if pd.notna(row.gold) else None,
            rule=self.grading['verifiers'][row.grading_kind]['rule']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(self.grading['verifiers'][row.grading_kind],
            dataset=row.dataset), sort_keys=True)) if row.grading_kind != 'numeric' else
            Judge(spec=json.dumps(dict(self.grading['verifiers'][row.grading_kind], dataset=row.dataset), sort_keys=True))
            for row in items.itertuples()]

        # 8. Preserve full source text, versioned input links and every snapshot location.
        csv_traces = csv_attempts[['response_key']].copy()
        csv_traces['source_trace'] = [dict(source_file=row.source_file, source_row=row.source_row,
            source_record=json.loads(row.source_record), source_column=row.source_column,
            retained_position=row.retained_position, original_call_position=None,
            export_sha256=row.export_sha256,
            identical_export_files=[path for path in row.identical_exports if path != row.source_file],
            identical_export_run_identity='not_recorded' if len(row.identical_exports) > 1 else None,
            input_file=row.input_file, input_row=row.input_row) for row in csv_attempts.itertuples()]
        json_traces = native[['response_key']].copy()
        json_traces['source_trace'] = [dict(source_file=row.source_file, source_row=row.source_row,
            source_record=json.loads(row.source_record), dataset_position=row.dataset_position, kind=row.kind,
            event_key=row.event_key, source_locations=row.source_locations,
            native_run_metadata=json.loads(row.run_metadata), native_dataset_metadata=json.loads(row.dataset_metadata),
            input_method=row.input_method, declared_input_csv=row.linked_csv or None,
            declared_csv_verified=bool(row.declared_csv_verified)) for row in native.itertuples()]
        trace_data = pd.concat([csv_traces, json_traces], ignore_index=True).merge(attempts[['response_key',
            'grading_status', 'question_column_available', 'reference_records', 'input_file', 'input_row']],
            on='response_key', validate='one_to_one')
        traces = trace_data[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(**row.source_trace, grading_status=row.grading_status,
            question_column_available=row.question_column_available,
            representative_input=dict(file=row.input_file, row=row.input_row),
            reference_records=row.reference_records if isinstance(row.reference_records, list) else []),
            ensure_ascii=False, allow_nan=False) for row in trace_data.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    SGrades(__file__).main_from_args()
