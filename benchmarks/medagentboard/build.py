"""Tabulate released MedAgentBoard attempts and their available native assessments."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class MedAgentBoard(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, protocols = self.build_parameters, self.grading['verifiers']
        layout = parameters['layout']

        # 1. Read original result objects, retaining incomplete logs without repairing their bytes.
        records = []
        with ZipFile(self.raw_dir / layout['archive']) as archive:
            for path in sorted(archive.namelist()):
                if not path.startswith(layout['prefix'] + 'logs/') or not path.endswith('.json'):
                    continue
                text = archive.read(path).decode('utf-8')
                partial = False
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    if layout['incomplete_logs'] not in path:
                        raise
                    # Upstream stopped while serializing case_history. Earlier complete
                    # values remain usable; the full original text stays in the trace.
                    decoder, position, record = json.JSONDecoder(), 1, {}
                    while position < len(text):
                        try:
                            while text[position].isspace() or text[position] == ',':
                                position += 1
                            key, position = decoder.raw_decode(text, position)
                            while text[position].isspace() or text[position] == ':':
                                position += 1
                            value, position = decoder.raw_decode(text, position)
                        except json.JSONDecodeError:
                            break
                        record[key] = value
                    if set(record) != set(parameters['incomplete_prefix_fields']):
                        raise ValueError('Unexpected incomplete source record')
                    partial = True
                parts = path.removeprefix(layout['prefix'] + 'logs/').split('/')
                family, dataset = parts[:2]
                setting, method = ('summary', parts[2]) if family == 'laysummary' else parts[2:4]
                records.append(dict(record, family=family, dataset=dataset, setting=setting, method=method,
                    source_item=str(record['id'] if family == 'laysummary' else record['qid']),
                    source_file=path, native_json=text, source_prefix_decoded=partial))
            attempts = pd.json_normalize(records, max_level=0).astype(object)
            attempts = attempts.where(attempts.notna(), None)

            # 2. Recover the documented image-directory relocation, preserving every original image.
            attempts['image_file'] = attempts.image_path.fillna('').str.removeprefix('./')
            relocated = attempts.image_file.str.startswith(layout['old_image_prefix']) & ~attempts.image_file.str.startswith(layout['image_prefix'])
            attempts.loc[relocated, 'image_file'] = attempts.loc[relocated, 'image_file'].str.replace(
                layout['old_image_prefix'], layout['image_prefix'], n=1, regex=False)
            images = {path: archive.read(layout['prefix'] + path) for path in sorted(set(attempts.image_file) - {''})}

        # 3. Define actual questions, options, references and source configurations.
        summary = attempts.family.eq('laysummary')
        attempts['stimulus'] = attempts.question
        attempts.loc[summary, 'stimulus'] = attempts.loc[summary, 'source']
        attempts['reference'] = attempts.ground_truth.map(lambda value: str(value) if value is not None else None)
        attempts.loc[summary, 'reference'] = attempts.loc[summary, 'target']
        attempts['protocol'] = attempts.family
        mc = attempts.family.eq('medqa') & attempts.setting.eq('multiple_choice')
        attempts.loc[mc, 'protocol'] = 'multiple_choice'
        ff = attempts.family.eq('medqa') & attempts.setting.eq('free-form')
        attempts.loc[ff, 'protocol'] = attempts.loc[ff, 'dataset'].map(parameters['free_form_protocols'])
        if attempts.protocol.isna().any():
            raise ValueError('Unknown source grading protocol')
        attempts['item_key'] = attempts[['family', 'dataset', 'setting', 'source_item']].agg('/'.join, axis=1)
        attempts['subject_key'] = attempts[['family', 'dataset', 'setting', 'method']].agg('/'.join, axis=1)
        attempts['configuration'] = [
            {key: value for key, value in (row.metadata or {}).items() if key != 'processing_time'}
            if row.family == 'laysummary' else
            {key: value for key, value in (row.case_history or {}).items() if key in parameters['configuration_fields']}
            for row in attempts.itertuples()]
        attempts['response'] = None
        attempts.loc[mc, 'response'] = attempts.loc[mc, 'predicted_answer'].eq(attempts.loc[mc, 'ground_truth']).astype(float)
        attempts['response_key'] = attempts.source_file
        attempts['grade_status'] = attempts.protocol.map(parameters['grade_status'])
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, native_json=row.native_json,
            source_prefix_decoded=row.source_prefix_decoded, grade_status=row.grade_status), ensure_ascii=False)
            for row in attempts.itertuples()]

        # 4. Join the consolidated human categories to their tasks, preserving all six individual reviews.
        workflow = self.raw_dir / layout['workflow']
        bank = pd.read_json(workflow / 'task100.json').rename(columns={'ID': 'source_item'})
        judgments = pd.read_json(workflow / 'evaluation/English_version/Merged.json').rename(columns={'ID': 'source_item'})
        joined = judgments.merge(bank, on='source_item', how='left', suffixes=('_assessment', ''), validate='one_to_one')
        if (not joined.task.map(lambda text: ' '.join(text.split())).eq(joined.task_assessment.map(lambda text: ' '.join(text.split()))).all()
            or not joined.dataset.eq(joined.dataset_assessment).all() or not joined.task_type.eq(joined.task_type_assessment).all()):
            raise ValueError('Workflow assessments disagree with the released task bank')
        panel = pd.concat([pd.read_json(workflow / f'evaluation/English_version/{rater}.json')
            .rename(columns={'ID': 'source_item', 'SmolAgent': 'SmolAgents'}).assign(rater=rater)
            for rater in parameters['workflow_raters']], ignore_index=True)
        reviews = panel.melt(id_vars=['source_item', 'rater'], value_vars=list(parameters['workflow_methods']),
            var_name='method', value_name='assessment')
        reviews['review'] = reviews[['rater', 'assessment']].to_dict('records')
        reviews = reviews.groupby(['source_item', 'method'], sort=False).review.agg(list).reset_index()
        results = joined.melt(id_vars=bank.columns.tolist(), value_vars=list(parameters['workflow_methods']),
            var_name='method', value_name='assessment').merge(reviews, on=['source_item', 'method'], validate='one_to_one')
        code = pd.read_json(workflow / 'results/Single_LLM_code.json').rename(columns={'ID': 'source_item'})
        if not code.merge(bank[['source_item', 'task']], on='source_item', validate='one_to_one', suffixes=('', '_bank')).eval('task == task_bank').all():
            raise ValueError('Workflow code refers to different task instructions')
        results = results.merge(code[['source_item', 'code']].assign(method=parameters['labels']['workflow_code_method']),
            on=['source_item', 'method'], how='left', validate='one_to_one')
        results['family'], results['setting'], results['protocol'] = 'workflow', 'clinical_workflow', 'workflow'
        results['stimulus'], results['reference'] = results.task, None
        results['options'], results['image_file'], results['configuration'] = None, '', [{} for _ in range(len(results))]
        results['subject_key'] = 'workflow/' + results.method
        results['source_item'] = results.source_item.map(str)
        results['item_key'] = 'workflow/' + results.source_item
        results['response_key'] = results.subject_key + '/' + results.source_item
        results['response'] = results.assessment.map(parameters['workflow_category_codes']).astype(float)
        if results.response.isna().any():
            raise ValueError('Unrecognized human assessment category')
        results['trace'] = [json.dumps(dict(source_file='workflow/evaluation/English_version/Merged.json',
            source_item=row.source_item, method=row.method, assessment=row.assessment, individual_reviews=row.review,
            code=row.code if isinstance(row.code, str) else None, grade_status=parameters['grade_status']['workflow']), ensure_ascii=False)
            for row in results.itertuples()]
        observations = pd.concat([attempts, results], ignore_index=True).astype(object)
        observations = observations.where(observations.notna(), None)

        # 5. Emit one item per stimulus/protocol and one subject per recorded source configuration.
        fields = ['item_key', 'source_item', 'stimulus', 'reference', 'options', 'image_file', 'protocol', 'family', 'dataset', 'setting']
        items = observations[fields].copy()
        items['identity'] = items[fields].apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
        if items.groupby('item_key').identity.nunique().gt(1).any():
            raise ValueError('Conflicting task definitions share a source key')
        items = items.drop_duplicates('item_key')
        items['raw_item_id'] = items.item_key
        items['content'] = [json.dumps(dict(multimedia_elements=[dict(content_type='text/plain', text=row.stimulus)]
            + ([dict(content_type='application/json', text=json.dumps(row.options, ensure_ascii=False))] if row.options else [])
            + ([dict(content_type='image/jpeg', location=row.image_file)] if row.image_file else [])), ensure_ascii=False)
            for row in items.itertuples()]
        items['attachments'] = [[dict(data=images[row.image_file], path=row.image_file, media_type='image/jpeg', role='input')]
            if row.image_file else [] for row in items.itertuples()]
        items['features'] = [dict(family=row.family, dataset=row.dataset, setting=row.setting,
            source_item=row.source_item, input_scope=parameters['input_scope'].get(row.family, parameters['input_scope']['default']))
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference, rule=protocols[row.protocol]['rule'],
            response_scale=protocols[row.protocol]['response_scale']) for row in items.itertuples()]
        items['verifier'] = [Judge(spec=json.dumps(protocols[key], sort_keys=True), judged_by=protocols[key]['judged_by'])
            if 'judged_by' in protocols[key] else ExactMatcher(spec=json.dumps(protocols[key], sort_keys=True)) for key in items.protocol]
        subjects = observations[['subject_key', 'configuration']].copy()
        subjects['configuration_json'] = subjects.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        if subjects.groupby('subject_key').configuration_json.nunique().gt(1).any():
            raise ValueError('A released subject label contains different recorded model settings')
        subjects = subjects.drop_duplicates('subject_key')
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(**parameters['subject_features'], source_configuration=row.subject_key,
            recorded_configuration=row.configuration) for row in subjects.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': observations[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': observations[['response_key', 'trace']]}


if __name__ == '__main__':
    MedAgentBoard(__file__).main_from_args()
