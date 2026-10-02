"""Build tables from complete original Robust Reasoning Benchmark records."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class RobustReasoningBenchmark(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        labels = parameters['labels']
        grading = self.grading

        # 1. Read the original result tables and identify aliases of the same named run.
        files = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            raw = path.read_bytes()
            records = json.loads(raw)
            if not isinstance(records, list) or set(records[-1]) != {'summary'}:
                raise ValueError(f'Unexpected native result format: {path}')
            relative = path.relative_to(self.raw_dir)
            files.append(dict(source_file=relative.as_posix(), run_id=path.name,
                digest=hashlib.sha256(raw).hexdigest(), experiment=relative.parts[1],
                model=relative.parts[3], dataset=relative.parts[4],
                records=records[:-1], native_summary=records[-1]['summary']))
        files = pd.DataFrame(files)
        if files.groupby('run_id').digest.nunique().gt(1).any():
            raise ValueError('Copies of a named run have different native records')
        aliases = files.groupby('run_id', sort=False).source_file.agg(list).rename('source_files')
        files = files.drop_duplicates('run_id').merge(aliases, on='run_id', validate='one_to_one')
        records = files.explode('records', ignore_index=True).rename(columns={'records': 'native_record'})
        records['source_row'] = records.groupby('run_id', sort=False).cumcount()
        rows = pd.concat([records, pd.json_normalize(records.native_record, max_level=0)], axis=1)
        for field in ['system_prompt', 'original', 'output']:
            if not rows[field].map(lambda value: isinstance(value, str)).all():
                raise ValueError(f'Missing recorded {field}')
        rows['response_key'] = rows.run_id + ':' + rows.source_row.astype(str)

        # 2. Preserve the model label and recorded inference settings as the subject configuration.
        files['features'] = [dict(harness=labels['harness'], recorded_model=row.model,
            inference_settings={name: row.native_summary.get(name) for name in parameters['inference_fields']},
            historical_api_revision=labels['historical_settings']) for row in files.itertuples()]
        files['subject_key'] = files.features.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = files[['subject_key', 'model', 'features']].drop_duplicates('subject_key').rename(columns={'model': 'raw_label'})
        rows = rows.merge(files[['run_id', 'subject_key']], on='run_id', validate='many_to_one')

        # 3. Form complete input/grading definitions and join each observation to its exact stimulus.
        recovery = rows.experiment.eq(labels['recovery'])
        if not rows.loc[recovery, 'native_summary'].map(lambda value: value['char_error_threshold']).eq(grading['verifiers']['recovery']['char_error_threshold']).all():
            raise ValueError('A recorded text-recovery threshold differs from its grading protocol')
        rows['task'] = recovery.map({False: 'math', True: 'recovery'})
        rows['content'] = labels['system'] + rows.system_prompt + labels['user'] + rows.original
        reference = rows.ground_truth.astype(str).where(~recovery, rows.canonical_original)
        rows['grading_criterion'] = [json.dumps(dict(reference_answer=value, rule=grading['verifiers'][task]['rule']), sort_keys=True)
                                    for value, task in zip(reference, rows.task)]
        rows['verifier'] = rows.task.map({name: json.dumps(value, sort_keys=True) for name, value in grading['verifiers'].items()})
        rows['raw_item_id'] = rows.dataset + ':' + rows.id.astype(str)
        rows['features'] = [dict(task=task, source_dataset=dataset) for task, dataset in zip(rows.task, rows.dataset)]
        definition = ['content', 'grading_criterion', 'verifier']
        items = rows.drop_duplicates(definition)[definition + ['raw_item_id', 'features', 'response_key']].rename(columns={'response_key': 'item_key'})
        rows = rows.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')
        items['verifier'] = items.verifier.map(lambda value: Judge(spec=value))

        # 4. Retain native grades and full provenance; API failures remain observed, ungraded attempts.
        grade = rows.correct.where(rows.task.eq('math'), rows.recovered)
        if not grade.map(lambda value: type(value) is bool).all():
            raise ValueError('A native task grade is not boolean')
        api_error = rows.output.str.startswith(labels['api_error_prefix']) & ~rows.refusal.eq(True)
        rows['response'] = grade.astype(float).mask(api_error)
        traces = rows[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_files=row.source_files, source_row=row.source_row,
            native_summary=row.native_summary, native_record=row.native_record), ensure_ascii=False, allow_nan=False)
            for row in rows.itertuples()]
        return {'subjects': subjects, 'items': items,
            'responses': rows[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    RobustReasoningBenchmark(__file__).main_from_args()
