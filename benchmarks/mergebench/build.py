"""Tabulate native MergeBench safety assessments with their original task attribution."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MergeBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, protocols = self.build_parameters, self.grading['verifiers']
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Load the complete native records into tables, retaining source coordinates.
        frames = []
        with ZipFile(self.raw_dir / layout['archive']) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith('/safety_generation.json'):
                    continue
                for task, records in json.loads(archive.read(name)).items():
                    frame = pd.json_normalize(records, max_level=0).reset_index(names='source_row')
                    frame['source_record_json'] = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
                    frames.append(frame.assign(source_file=name.removeprefix(layout['prefix']), source_task=task))
        attempts = pd.concat(frames, ignore_index=True).rename(columns={'response': 'source_output'})
        attempts = attempts.join(attempts.source_file.str.extract(layout['path_pattern']))
        if attempts[['base_model', 'merge_configuration']].isna().any().any():
            raise ValueError('Unrecognized native result path')
        attempts['subject_key'] = attempts.source_file
        attempts['response_key'] = attempts.source_file + '#' + attempts.source_task + '#' + attempts.source_row.astype(str)
        attempts['raw_item_id'] = attempts.source_task + '#' + attempts.id.astype(str)

        # 2. Select the actual task and judge inputs, and decode each task's cached grade.
        attempts['content'], attempts['judge_prompt'] = None, None
        for task, field in parameters['task_inputs'].items():
            selected = attempts.source_task.eq(task)
            attempts.loc[selected, 'content'] = attempts.loc[selected, field]
            attempts.loc[selected, 'judge_prompt'] = attempts.loc[selected, parameters['judge_inputs'][task]]
        if attempts[['content', 'judge_prompt']].isna().any().any():
            raise ValueError('A source task has no supported generator or classifier input')
        attempts['protocol'] = attempts.source_task
        xstest = attempts.source_task.eq('xstest')
        attempts.loc[xstest, 'protocol'] = labels['safe_protocol']
        attempts.loc[xstest & attempts.type.str.contains('contrast', na=False), 'protocol'] = labels['unsafe_protocol']
        attempts['response'] = None
        for protocol, spec in protocols.items():
            selected = attempts.protocol.eq(protocol)
            native_labels = attempts.loc[selected, spec['field']]
            if not native_labels.dropna().isin(spec['values']).all():
                raise ValueError('Unrecognized cached safety category')
            attempts.loc[selected, 'response'] = native_labels.map(spec['values'])
        attempts.loc[attempts.is_parsing_error, 'response'] = None
        if not attempts.protocol.isin(protocols).all():
            raise ValueError('Unknown source grading protocol')

        # 3. Identify source model configurations and exact stimulus/grading definitions.
        subjects = attempts[['subject_key', 'base_model', 'merge_configuration']].drop_duplicates().copy()
        subjects['raw_label'] = labels['subject_prefix'] + subjects.base_model + ' / ' + subjects.merge_configuration
        subjects['features'] = [dict(**parameters['subject_features'], source_base_model=row.base_model,
            source_merge_configuration=row.merge_configuration, source_result_file=row.subject_key) for row in subjects.itertuples()]
        identity = ['source_task', 'content', 'judge_prompt', 'protocol']
        items = attempts.drop_duplicates(identity).copy()
        items['item_key'] = range(len(items))
        items['features'] = [dict(source_benchmark=row.source_task, grading_protocol=row.protocol) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=protocols[row.protocol]['rule'] +
            (labels['context_prefix'] + row.judge_prompt if row.source_task == 'do_anything_now' else ''),
            response_scale=protocols[row.protocol]['response_scale']) for row in items.itertuples()]
        items['verifier'] = [Judge(spec=json.dumps(protocols[key], sort_keys=True), judged_by='llm') for key in items.protocol]
        attempts = attempts.merge(items[identity + ['item_key']], on=identity, how='left', validate='many_to_one')

        # 4. Retain full native records without misattributing overwritten reference responses.
        attempts = attempts.astype(object).where(attempts.notna(), None)
        template = parameters['model_input']['template']
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, source_task=row.source_task,
            source_row=int(row.source_row), source_record_json=row.source_record_json,
            source_record_encoding=layout['source_record_encoding'],
            generated_output=None if row.source_task == 'wildguardtest' else row.source_output,
            output_status=labels['wildguard_output_status'] if row.source_task == 'wildguardtest' else labels['recorded_output_status'],
            judge_prompt=row.judge_prompt, declared_model_input=template.format(instruction=row.content),
            grade_status='unavailable_cached_label' if row.response is None else 'available_cached_label'),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': attempts[['response_key', 'trace']]}


if __name__ == '__main__':
    MergeBench(__file__).main_from_args()
