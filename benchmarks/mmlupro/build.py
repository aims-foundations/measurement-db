"""Tabulate MMLU-Pro's original question-level predictions and complete outputs."""

import json
from pathlib import Path
import sys
from urllib.parse import quote
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MMLUPro(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate every question file, preserving archive/member/row coordinates.
        frames = []
        for path in sorted(self.raw_dir.glob(parameters['paths']['results'])):
            with ZipFile(path) as archive:
                members = sorted(name for name in archive.namelist()
                    if name.endswith('.json') and not name.startswith('__MACOSX/'))
                for member in members:
                    records = json.loads(archive.read(member))
                    if Path(member).name == 'summary.json' and isinstance(records, dict):
                        continue  # Published category aggregates are not individual observations.
                    if not isinstance(records, list):
                        raise ValueError('Expected an original array of question records')
                    entries = pd.Series(records, dtype=object, name='native_record')
                    is_record = entries.map(lambda value: isinstance(value, dict))
                    if not entries.loc[~is_record].isin(parameters['nonrecord_labels']).all():
                        raise ValueError('An unrecognized non-record entry needs source review')
                    table = pd.json_normalize(entries.loc[is_record], max_level=0)
                    frames.append(table.assign(native_record=entries.loc[is_record].tolist(),
                        source_row=entries.loc[is_record].index,
                        source_file=str(path.relative_to(self.raw_dir)), source_member=member))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda table: table.index)
        required = ['question_id', 'question', 'options', 'answer', 'answer_index',
                    'category', 'src', 'cot_content', 'pred']
        if not responses.native_record.map(lambda row: set(row) >= set(required)).all():
            raise ValueError('An original question record is missing required fields')
        if not responses.pred.dropna().isin(list(parameters['labels']['answer_letters'])).all():
            raise ValueError('Use released extracted answer letters, without a replacement parser')
        if not responses.answer.isin(list(parameters['labels']['answer_letters'])).all():
            raise ValueError('An original reference answer is invalid')
        if not responses.options.map(lambda values: isinstance(values, list) and
                1 <= len(values) <= 10 and all(isinstance(value, str) for value in values)).all():
            raise ValueError('An original option list is invalid')
        output_fields = set(parameters['output_fields'])
        if not responses.native_record.map(lambda row:
                len(set(row) & output_fields) == 1 and
                all(isinstance(row[key], str) for key in set(row) & output_fields)).all():
            raise ValueError('Expected exactly one complete original output field')
        if responses.native_record.map(lambda row: bool(set(row) & set(parameters['grade_fields']))).any():
            raise ValueError('A new native grade field requires an explicit import decision')

        # 2. Keep each recorded model/run configuration, including nominal shot counts.
        subjects = responses[['source_file']].drop_duplicates().rename(columns={'source_file': 'subject_key'})
        names = subjects.subject_key.map(lambda name: Path(name).name)
        names = names.replace(parameters['source_filenames'])
        settings = names.str.extract(parameters['parsing']['run_filename'])
        if settings.isna().any().any():
            raise ValueError('An archive filename has an unrecognized run configuration')
        subjects['raw_label'] = settings.raw_label
        subjects['features'] = [dict(harness=parameters['labels']['harness'],
            source_run=quote(filename, safe=' /-._'), nominal_shots=shots)
            for filename, shots in zip(names, settings.nominal_shots)]
        responses['subject_key'] = responses.source_file

        # 3. Join actual question/option/reference definitions; upstream IDs can be reused.
        responses['options_json'] = responses.options.map(lambda values: json.dumps(values, ensure_ascii=False))
        identity = ['question', 'options_json', 'answer', 'category', 'src']
        items = responses.drop_duplicates(identity).reset_index(drop=True).copy()
        items['item_key'] = items.index
        responses = responses.merge(items[identity + ['item_key']], on=identity, validate='many_to_one')
        items['content'] = [json.dumps(dict(question=row.question, options=row.options),
            ensure_ascii=False) for row in items.itertuples()]
        items['raw_item_id'] = items.question_id.astype(str)
        items['features'] = [dict(category=quote(row.category, safe=' /-._'),
            source=quote(row.src, safe=' /-._')) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=value, rule=self.grading['rule']) for value in items.answer]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))

        # 4. Compare released letters directly; unrecorded random fallbacks remain unknown.
        responses['response'] = responses.pred.eq(responses.answer).astype(float).where(responses.pred.notna())
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_member=row.source_member,
            source_row=int(row.source_row), native_record=row.native_record),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': traces,
        }


if __name__ == '__main__':
    MMLUPro(__file__).main_from_args()
