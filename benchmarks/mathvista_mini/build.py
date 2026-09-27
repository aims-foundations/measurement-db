"""Tabulate released MathVista predictions, cached extractions and original images."""

import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from openpyxl.utils.escape import unescape
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MathVistaMini(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, parsing = parameters['paths'], parameters['parsing']

        # 1. Concatenate native workbooks; identical whole exports are aliases, not new runs.
        frames = []
        for path in sorted((self.raw_dir / paths['results']).rglob('*.xlsx')):
            frame = pd.read_excel(path, dtype=object, keep_default_na=False, na_values=[''])
            frame = frame.where(frame.notna(), None)
            prediction_columns = frame.columns.drop(['res', 'log'], errors='ignore').tolist()
            predictions = frame.sort_values('index')[prediction_columns].to_dict('records')
            fingerprint = hashlib.sha256(json.dumps(predictions, sort_keys=True,
                ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            frames.append(frame.assign(native_record=frame.to_dict('records'),
                source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index,
                model_label=path.name.split(parsing['filename_marker'])[0], export_hash=fingerprint,
                has_extraction='res' in frame))
        records = pd.concat(frames, ignore_index=True).astype(object).where(lambda table: table.notna(), None)
        records['source_ref'] = records[['source_file', 'source_row']].to_dict('records')
        records['response_key'] = records.model_label + '::' + records.export_hash + '::' + records['index'].astype(str)
        sources = records.groupby('response_key', sort=False).source_ref.agg(list)
        responses = records.drop_duplicates('response_key').copy()
        responses['prediction_record'] = [{key: value for key, value in row.items() if key not in ('res', 'log')}
            for row in responses.native_record]
        responses['source_aliases'] = responses.response_key.map(sources)
        responses['subject_key'] = responses.model_label
        responses['item_key'] = responses['index'].astype(int)

        # 2. Verify task joins and restore the exact question/image pair, including source MIME types.
        bank = pd.json_normalize(pq.read_table(self.raw_dir / paths['tasks']).to_pylist(), max_level=0)
        bank = bank.astype(object).where(bank.notna(), None)
        bank['item_key'] = bank.pid.astype(int)
        expected = bank[['item_key', 'query', 'answer', 'choices', 'question_type', 'answer_type']]
        joined = responses.merge(expected, on='item_key', how='left', suffixes=('', '_bank'), validate='many_to_one')
        if (joined['query'].isna().any() or not joined.question.map(unescape).eq(joined['query']).all()
            or not joined.answer.astype(str).eq(joined.answer_bank.astype(str)).all()
            or not joined.question_type.eq(joined.question_type_bank).all()
            or not joined.answer_type.eq(joined.answer_type_bank).all()
            or not all((ast.literal_eval(value) if isinstance(value, str) else value) == original
                for value, original in zip(joined.choices, joined.choices_bank))):
            raise ValueError('A released question or reference differs from the pinned task bank')
        items = bank.copy()
        items['image_bytes'] = items.decoded_image.map(lambda image: image['bytes'])
        items['media_type'] = items.image_bytes.map(lambda data:
            parameters['image_types'][Image.open(io.BytesIO(data)).format])
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type=row.media_type, location=row.image),
            dict(content_type='text/plain', text=row.query)]), ensure_ascii=False) for row in items.itertuples()]
        items['attachments'] = [[dict(data=row.image_bytes, path=row.image,
            media_type=row.media_type, role='input')] for row in items.itertuples()]
        items['raw_item_id'] = items.pid
        items['features'] = [dict(**row.metadata, source_image_path=row.image,
            precision=row.precision, unit=row.unit) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=answer, rule=self.grading['rule']) for answer in items.answer]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['released_extraction'], sort_keys=True))

        # 3. Deduplicate cached extractions, retaining every alternative and its original file coordinates.
        extractions = records.loc[records.has_extraction].copy()
        extractions['extraction_record'] = extractions[['res', 'log']].to_dict('records')
        extractions['extraction_key'] = extractions.extraction_record.map(
            lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
        keys = ['response_key', 'extraction_key']
        aliases = extractions.groupby(keys, sort=False).source_ref.agg(list).rename('extraction_sources')
        extractions = extractions.drop_duplicates(keys).merge(aliases, on=keys, how='left', validate='one_to_one')
        extractions['extraction_id'] = range(len(extractions))
        succeeded = extractions.log.fillna('').str.endswith(('Succeed', 'Prefetch succeed'))
        unavailable = extractions.prediction.isna() | extractions.prediction.astype(str).str.contains(parsing['api_failure'], regex=False)
        usable = succeeded & ~unavailable & extractions.res.notna()
        extractions['grade'] = float('nan')

        # 4. Apply the historical deterministic comparator to released extractions; never call a judge.
        choices = bank[['item_key', 'choices']].explode('choices').dropna(subset=['choices'])
        choices['letter'] = choices.groupby('item_key').cumcount().map(lambda index: chr(65 + index))
        multiple = extractions.loc[usable & extractions.question_type.eq('multi_choice')].copy()
        multiple['item_key'] = multiple['index'].astype(int)
        words = multiple[['extraction_id', 'item_key']].assign(word=multiple.res.astype(str)
            .str.translate(str.maketrans(parsing['punctuation'], ' ' * len(parsing['punctuation']))).str.split()).explode('word')
        words = words.drop_duplicates(['extraction_id', 'word'])
        matches = words.merge(choices[['item_key', 'letter']], left_on=['item_key', 'word'],
            right_on=['item_key', 'letter'], how='inner', validate='many_to_one')
        matches = matches.groupby('extraction_id').letter.agg(['size', 'first'])
        option = multiple.extraction_id.map(matches['first'].where(matches['size'].eq(1)))
        option = option.mask(~multiple.extraction_id.isin(matches.index)
            & multiple.extraction_id.isin(words.loc[words.word.eq('Z'), 'extraction_id']), 'Z')
        refused = multiple.res.astype(str).str.contains('|'.join(re.escape(value)
            for value in parameters['refusals'].values()), regex=True)
        option = option.mask(refused, 'Z')
        api_failure = multiple.res.astype(str).str.contains(parsing['api_failure'], regex=False)
        option = option.mask(api_failure)
        remaining = multiple.loc[option.isna(), ['extraction_id', 'item_key', 'res']].merge(
            choices, on='item_key', how='inner', validate='many_to_many')
        remaining = remaining.loc[[str(choice).lower() in str(answer).lower()
            for choice, answer in zip(remaining.choices, remaining.res)]]
        fallback = remaining.groupby('extraction_id').letter.agg(['size', 'first'])
        option = option.fillna(multiple.extraction_id.map(fallback['first'].where(fallback['size'].eq(1))))
        extractions.loc[multiple.index, 'grade'] = option.eq(multiple.answer_option).astype(float)

        numeric = extractions.loc[usable & extractions.question_type.ne('multi_choice')
            & extractions.answer_type.isin(['integer', 'float'])]
        # Python's native int/float conversions differ from permissive numeric coercion.
        grades = []
        for row in numeric.itertuples():
            converter = int if row.answer_type == 'integer' else float
            try:
                grades.append(float(converter(row.res) == converter(row.answer)))
            except (ValueError, TypeError, OverflowError):
                grades.append(0.)
        extractions.loc[numeric.index, 'grade'] = grades
        textual = usable & extractions.question_type.ne('multi_choice') & ~extractions.answer_type.isin(['integer', 'float'])
        extractions.loc[textual, 'grade'] = extractions.loc[textual, 'answer'].astype(str).eq(parsing['historical_text_result']).astype(float)

        # 5. Conflicting grading histories remain visible; unavailable grades are never failures.
        agreement = extractions.groupby('response_key').grade.agg(['nunique', 'first'])
        responses['response'] = responses.response_key.map(agreement['first'].where(agreement['nunique'].eq(1)))
        responses['grade_status'] = responses.response.notna().map({True: 'derived_historical_comparator', False: 'unavailable_extraction'})
        conflict = responses.response_key.isin(agreement.index[agreement['nunique'].gt(1)])
        responses.loc[conflict, 'grade_status'] = 'conflicting_extraction_grades'
        extractions['comparison_grade'] = extractions.grade.astype(object).where(extractions.grade.notna(), None)
        extractions['extraction_evidence'] = extractions[['extraction_record', 'extraction_sources', 'comparison_grade']].to_dict('records')
        evidence = extractions.groupby('response_key', sort=False).extraction_evidence.agg(list)
        responses['extractions'] = responses.response_key.map(evidence).map(lambda value: value if isinstance(value, list) else [])
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=value) for value in subjects.subject_key]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(export_hash=row.export_hash, prediction_record=row.prediction_record,
            source_aliases=row.source_aliases, extractions=row.extractions, grade_status=row.grade_status),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    MathVistaMini(__file__).main_from_args()
