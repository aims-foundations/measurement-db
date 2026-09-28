"""Tabulate the original joint classifications and their two scored axes."""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class RelianceScope(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, patterns = parameters['layout'], parameters['patterns']

        # 1. Read the native input and classification files directly into tables.
        segments = pd.read_json(self.raw_dir / layout['segments'], lines=True).rename(columns={'input': 'target_input'})
        references = pd.json_normalize(segments.output).set_index(segments.index)
        segments = pd.concat([segments[['id', 'target_input']], references], axis=1)
        outputs = []
        for path in sorted(self.raw_dir.glob(layout['outputs'])):
            frame = pd.read_json(path, lines=True)
            frame['native_output'] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            frame['source_file'], frame['source_row'] = str(path.relative_to(self.raw_dir)), frame.index
            filename = path.name
            for escaped, original in parameters['filename_escapes'].items():
                filename = filename.replace(escaped, original)
            frame['source_filename'] = filename
            outputs.append(frame)
        calls = pd.concat(outputs, ignore_index=True)
        calls = pd.concat([calls, calls.source_filename.str.extract(patterns['filename'])], axis=1)
        calls['call_key'] = calls.source_file + '#' + calls.source_row.astype(str)
        calls = calls.merge(segments, on='id', validate='many_to_one', how='left', suffixes=('_recorded', ''))
        if calls.target_input.isna().any():
            raise ValueError('A recorded classifier call has no matching input segment')
        for field in parameters['reference_fields'].values():
            if not calls[field].eq(calls[field + '_recorded']).all():
                raise ValueError('Recorded reference labels disagree with the source segment')

        # 2. Associate the documented prompt context without inventing API requests.
        report = '\n'.join(page.extract_text() or '' for page in PdfReader(self.raw_dir / layout['report']).pages)
        system = re.search(patterns['system'], report, re.DOTALL).group(0)
        examples = {'none': None, **{name: re.search(patterns[name], report, re.DOTALL).group(1).strip()
                                   for name in ['three_shot', 'nine_shot']}}
        calls['examples'] = calls.prompting.map(parameters['strategies']).map(examples)
        calls['examples'] = calls.examples.astype(object).where(calls.examples.notna(), None)
        if not calls.prompting.isin(parameters['strategies']).all():
            raise ValueError('An original prompting strategy lacks its documented context')
        calls['content'] = [json.dumps(dict(documented_system_prompt=system, documented_examples=row.examples,
            documented_response_schema=parameters['cot_schema' if row.prompting == '9shot+cot' else 'plain_schema'],
            target_input=row.target_input), ensure_ascii=False, allow_nan=False) for row in calls.itertuples()]
        configurations = pd.DataFrame({'model_identifier': parameters['models'], 'temperature': parameters['temperatures']}).rename_axis('model').reset_index()
        calls = calls.merge(configurations, on='model', how='left', validate='many_to_one')
        if calls.model_identifier.isna().any():
            raise ValueError('A classifier model lacks its report-declared identifier')

        # 3. Keep each model and prompting strategy as a separate configuration.
        calls['subject_key'] = calls.model + ':' + calls.prompting
        subjects = calls.drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.model_identifier + parameters['labels']['subject_separator'] + subjects.prompting
        subjects['features'] = [dict(harness=parameters['labels']['harness'], prompting_strategy=row.prompting,
            reported_model_identifier=row.model_identifier, reported_temperature=row.temperature)
            for row in subjects.itertuples()]

        # 4. Unpivot the two measured axes, preserving their common classifier call.
        axes = pd.DataFrame({'reference_field': parameters['reference_fields'], 'prediction_field': parameters['prediction_fields']}).rename_axis('axis').reset_index()
        measurements = calls.merge(axes, how='cross')
        measurements['reference'] = [record[field] for record, field in zip(measurements.native_output, measurements.reference_field)]
        measurements['prediction'] = [record.get(field) for record, field in zip(measurements.native_output, measurements.prediction_field)]
        predicted, reference = measurements.prediction.astype('string'), measurements.reference.astype('string')
        measurements['response'] = predicted.str.strip().str.lower().eq(reference.str.strip().str.lower()).astype('Float64')
        measurements['item_key'] = measurements.id.astype(str) + ':' + measurements.prompting + ':' + measurements.axis
        measurements['response_key'] = measurements.call_key + ':' + measurements.axis
        measurements['test_condition'] = 'task=' + measurements.axis + ';prompting=' + measurements.prompting
        items = measurements.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.item_key
        items['features'] = [dict(prompting_strategy=row.prompting,
            prompt_origin=parameters['labels']['prompt_origin']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference, rule=grading['rule'] + ' Axis: ' + row.axis)
                                    for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(**grading['verifiers']['reported'], prediction_field=row.prediction_field), sort_keys=True))
                             for row in items.itertuples()]

        # 5. Preserve the complete native prediction and any recorded reasoning.
        traces = measurements[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_segment_id=row.id, source_call_key=row.call_key, axis=row.axis, native_output=row.native_output),
            ensure_ascii=False, allow_nan=False) for row in measurements.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
                'responses': measurements[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    RelianceScope(__file__).main_from_args()
