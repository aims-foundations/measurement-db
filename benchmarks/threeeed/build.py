"""Tabulate both released 3EED prediction collections with original scene inputs."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ThreeEED(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Load every released prediction and retain its original run directory.
        with ZipFile(self.raw_dir / layout['predictions_archive']) as archive:
            files = pd.DataFrame({'member': sorted(archive.namelist())})
            files = files.join(files.member.str.extract(layout['prediction_pattern'])).dropna().reset_index(drop=True)
            records = files.member.map(lambda name: json.loads(archive.read(name)))
            responses = pd.concat([files, pd.json_normalize(records, max_level=0)], axis=1).assign(record=records)
            subjects = files[['run']].drop_duplicates().copy()
            subjects['configuration_member'] = subjects.run + '/config.json'
            subjects['configuration'] = subjects.configuration_member.map(lambda name: json.loads(archive.read(name)))
        if responses.empty or not responses.id.eq(responses.frame_id).all() or not responses.ious.str.len().eq(1).all():
            raise ValueError('Each released prediction must identify its frame and one native IoU')
        responses['response'] = pd.to_numeric(responses.ious.str[0], errors='raise')
        responses['response_key'] = layout['predictions_archive'] + '#' + responses.member
        responses['subject_key'] = responses.run
        responses['reference'] = responses.gt_box.map(lambda value: json.dumps(value, allow_nan=False))

        # 2. Keep the same scene/utterance together across the distinct evaluations.
        items = responses[['frame_id', 'utterance', 'reference']].drop_duplicates().copy()
        if items.frame_id.duplicated().any():
            raise ValueError('A frame has conflicting released utterances or reference boxes')
        items['item_key'], items['raw_item_id'], items['content'] = items.frame_id, items.frame_id, items.utterance
        items['platform'] = items.frame_id.str.split('/').str[0]
        items['folder'] = layout['inputs_prefix'] + items.frame_id
        items['point_cloud_file'] = items.platform.map(parameters['point_cloud_files'])
        if items.point_cloud_file.isna().any():
            raise ValueError('An original platform has no declared LiDAR format')

        # 3. Attach unchanged scene inputs; target annotations stay in the reference.
        with ZipFile(self.raw_dir / layout['inputs_archive']) as archive:
            items['attachments'] = [[
                dict(data=archive.read(row.folder + '/' + row.point_cloud_file),
                     path=row.frame_id + '/' + row.point_cloud_file, media_type='application/octet-stream', role='point_cloud'),
                dict(data=archive.read(row.folder + '/image.jpg'), path=row.frame_id + '/image.jpg',
                     media_type='image/jpeg', role='context_image')]
                for row in items.itertuples()]
        items['features'] = [dict(source_platform=row.platform, source_frame_id=row.frame_id,
            input_scope=labels['input_scope'], input_preprocessing=json.dumps(parameters['input_preprocessing'], sort_keys=True)) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=reference, rule=self.grading['rule']) for reference in items.reference]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))] * len(items)

        # 4. Retain all recorded configurations rather than merging by model name.
        subjects['subject_key'] = subjects.run
        subjects['raw_label'] = labels['subject'] + ' [' + subjects.run + ']'
        subjects['features'] = [dict(harness=labels['harness'], source_run=row.run,
            configuration_member=row.configuration_member, configuration=json.dumps(row.configuration, sort_keys=True),
            checkpoint_scope=labels['checkpoint_scope']) for row in subjects.itertuples()]
        responses['item_key'] = responses.frame_id
        responses['test_condition'] = responses.run + ':validation'

        # 5. Preserve complete predicted boxes, native scores and source locations.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['predictions_archive'], source_member=row.member,
            configuration_member=row.run + '/config.json', input_archive=layout['inputs_archive'],
            input_metadata_member=layout['inputs_prefix'] + row.frame_id + '/meta_info.json',
            grade_status='recorded_native_iou', record=row.record), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    ThreeEED(__file__).main_from_args()
