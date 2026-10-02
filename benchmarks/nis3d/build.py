"""Tabulate published per-volume segmentation scores and their original inputs."""

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class NIS3D(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read the published table into columns; preserve its literal score strings.
        with pymupdf.open(self.raw_dir / layout['paper']) as document:
            lines = pd.Series(document[int(layout['page']) - 1].get_text(sort=True).splitlines(), name='source_line')
        matrix = lines.str.extract(parameters['patterns']['paper_row'])
        matrix['source_line'] = lines
        matrix = matrix.loc[matrix.source_method.notna()].copy()
        matrix['volume'] = matrix.volume.ffill()
        matrix['method'] = matrix.source_method.replace(parameters['method_aliases'])
        matrix = matrix.loc[matrix.method.ne('Human')].copy()
        if matrix.empty or matrix.duplicated(['volume', 'method']).any():
            raise ValueError('The published table must have unique, nonempty method-volume rows')
        matrix['published_record'] = matrix.drop(columns='method').to_dict('records')
        subjects = pd.DataFrame({'subject_key': parameters['methods'], 'protocol': parameters['method_protocols']}).rename_axis('method').reset_index()
        matrix = matrix.merge(subjects, on='method', how='left', validate='many_to_one')
        if matrix.subject_key.isna().any():
            raise ValueError('A published method lacks its source configuration')

        # 2. Keep original image bytes separate from the ground-truth and confidence maps.
        volumes = pd.DataFrame({'folder': parameters['volumes']}).rename_axis('volume').reset_index()
        volumes['ground_truth'] = volumes.folder.map(parameters['ground_truth_files'])
        volumes['confidence'] = volumes.folder.map(parameters['confidence_files'])
        with ZipFile(self.raw_dir / layout['volumes_archive']) as archive:
            volumes['input_member'] = layout['volumes_prefix'] + volumes.folder + '/data.tif'
            volumes['input_bytes'] = volumes.input_member.map(archive.read)
            volumes['source_info'] = (layout['volumes_prefix'] + volumes.folder + '/Info.txt').map(
                lambda member: archive.read(member).decode('utf-8'))
            references = []
            for row in volumes.itertuples():
                references.append(dict(archive=layout['volumes_archive'], **{
                    role: dict(member=layout['volumes_prefix'] + row.folder + '/' + filename,
                               sha256=hashlib.sha256(archive.read(layout['volumes_prefix'] + row.folder + '/' + filename)).hexdigest())
                    for role, filename in [('ground_truth', row.ground_truth), ('confidence', row.confidence)]}))
            volumes['reference'] = references
        matrix = matrix.merge(volumes.drop(columns='input_bytes'), on='volume', how='left', validate='many_to_one')
        if matrix.folder.isna().any():
            raise ValueError('A published volume lacks its original input')

        # 3. Locate unchanged binary predictions and preserve the released grading implementation.
        with ZipFile(self.raw_dir / layout['results_archive']) as archive:
            predictions = pd.DataFrame({'member': archive.namelist()})
            predictions = predictions.loc[predictions.member.str.startswith(layout['results_prefix'])
                                          & predictions.member.str.endswith('.mat')].copy()
            paths = predictions.member.str.removeprefix(layout['results_prefix']).str.split('/')
            predictions['folder'], predictions['subject_key'] = paths.str[0], paths.str[1]
            if predictions.duplicated(['folder', 'subject_key']).any():
                raise ValueError('The source has more than one output for a method-volume pair')
            predictions['prediction_artifact'] = [dict(archive=layout['results_archive'], member=member,
                bytes=archive.getinfo(member).file_size, sha256=hashlib.sha256(archive.read(member)).hexdigest())
                for member in predictions.member]
            implementation = {name.removeprefix(layout['evaluator_prefix']): archive.read(name).decode('utf-8')
                for name in sorted(archive.namelist()) if name.startswith(layout['evaluator_prefix']) and name.endswith('.m')}
        matrix = matrix.merge(predictions[['folder', 'subject_key', 'prediction_artifact']],
                              on=['folder', 'subject_key'], how='left', validate='one_to_one')
        matrix['prediction_artifact'] = matrix.prediction_artifact.where(matrix.prediction_artifact.notna(), None)

        # 4. Unpivot quality metrics; runtime remains in the original record, not the grade column.
        metrics = pd.DataFrame.from_dict(grading['verifiers'], orient='index').rename_axis('metric').reset_index()
        responses = matrix.melt(id_vars=['folder', 'subject_key', 'published_record', 'source_info', 'prediction_artifact'],
                                value_vars=list(grading['verifiers']), var_name='metric', value_name='response')
        responses['response'] = pd.to_numeric(responses.response, errors='raise')
        responses = responses.merge(metrics, on='metric', how='left', validate='many_to_one')
        responses['item_key'] = responses.folder + ':' + responses.name
        responses['response_key'] = responses.item_key + ':' + responses.subject_key
        responses['test_condition'] = labels['condition']
        items = volumes.merge(metrics, how='cross')
        items['item_key'] = items.folder + ':' + items.name
        items['raw_item_id'], items['content'] = items.item_key, None
        items['features'] = [dict(source_volume=row.folder) for row in items.itertuples()]
        items['attachments'] = [[dict(data=row.input_bytes, path=row.folder + '/data.tif',
            media_type=labels['media_type'], role=labels['input_role'])] for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=json.dumps(row.reference, sort_keys=True),
            rule=grading['rule'] + ' ' + row.rule) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(**grading['verifiers'][row.metric], implementation=implementation), sort_keys=True)) for row in items.itertuples()]

        # 5. Retain every published record and link binary outputs without turning them into inputs.
        subjects['raw_label'] = subjects.method
        subjects['features'] = [dict(**parameters['subject_features'], source_method=row.method,
            protocol=row.protocol) for row in subjects.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['paper'], source_page=int(layout['page']),
            published_record=row.published_record, source_info=row.source_info, metric=row.name,
            prediction_artifact=row.prediction_artifact, prediction_scope=labels['prediction_scope'],
            grade_status=labels['grade_status']), allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
                'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    NIS3D(__file__).main_from_args()
