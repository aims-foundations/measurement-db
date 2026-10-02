"""Tabulate released SCReasoner observations and their original spatial inputs."""

import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class Situat3DChange(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Join released response records to the original task annotations.
        results, annotations = [], []
        with ZipFile(self.raw_dir / layout['annotations']) as archive:
            for task, directory in parameters['tasks'].items():
                source_file = layout['release'] + '/results/SCReasoner/' + directory + '/results_wscore.json'
                records = json.loads((self.raw_dir / source_file).read_text())
                unscored_file = source_file.replace('results_wscore.json', 'results.json')
                unscored = json.loads((self.raw_dir / unscored_file).read_text())
                if [{key: value for key, value in row.items() if key != 'score'} for row in records] != [
                        {key: value for key, value in row.items() if key != 'score'} for row in unscored]:
                    raise ValueError('Scored and unscored exports disagree on their original observations')
                frame = pd.json_normalize(records, max_level=0).rename(columns={'scene_id': 'scan_id', 'index': 'source_index'})
                frame['source_record'] = records
                frame = frame.assign(task=task, source_file=source_file, unscored_file=unscored_file, source_row=frame.index)
                results.append(frame)
                member = layout['annotation_prefix'] + task + '_val_v2.json'
                groups = pd.DataFrame.from_dict(json.loads(archive.read(member)), orient='index').rename_axis('reference_scan').reset_index()
                groups = groups.explode('response', ignore_index=True).dropna(subset=['response']).reset_index(drop=True)
                frame = pd.json_normalize(groups.response.tolist(), max_level=0).rename(columns={
                    'index': 'annotation_index', parameters['question_fields'][task]: 'question',
                    parameters['reference_fields'][task]: 'reference', parameters['type_fields'][task]: 'task_type'})
                frame['source_annotation'] = groups.response.tolist()
                annotations.append(frame.assign(task=task, reference_scan=groups.reference_scan.to_numpy(),
                    annotation_member=member, annotation_row=groups.groupby('reference_scan', sort=False).cumcount()))
            alignment = pd.Series(json.loads(archive.read(layout['alignment'])), name='alignment').rename_axis('scan_id').reset_index()
        results, annotations = pd.concat(results, ignore_index=True), pd.concat(annotations, ignore_index=True)
        annotations['source_index'] = annotations.annotation_index.astype(str)
        observations = results.merge(annotations, on=['task', 'scan_id', 'source_index'],
            how='left', validate='one_to_one', indicator=True)
        if not observations['_merge'].eq('both').all():
            raise ValueError('A recorded response has no unique original task annotation')
        expected_reference = observations.reference.copy()
        direction = observations.task.eq('qa') & observations.task_type.str.contains('Direction', regex=False)
        expected_reference.loc[direction] = 'At your ' + expected_reference[direction]
        if not observations.response_gt.eq(expected_reference).all() or not observations.instruction.eq(
                'USER: ' + observations.question + ' ASSISTANT:').all():
            raise ValueError('Recorded questions or references differ from the original task loader')
        observations = observations.drop(columns='_merge').merge(alignment, on='scan_id', how='left', validate='many_to_one')

        # 2. Resolve the long-form camera poses from the released situation tables.
        poses = []
        with ZipFile(self.raw_dir / layout['annotations']) as archive:
            for member in sorted(archive.namelist()):
                if not member.startswith(layout['situations']) or not member.endswith('.json'):
                    continue
                groups = pd.DataFrame.from_dict(json.loads(archive.read(member)), orient='index').rename_axis('pose_scene').reset_index()
                groups = groups.explode('response', ignore_index=True).dropna(subset=['response']).reset_index(drop=True)
                frame = pd.json_normalize(groups.response.tolist(), max_level=0)
                frame['pose_record'] = groups.response.tolist()
                poses.append(frame.assign(pose_member=member, pose_scene=groups.pose_scene.to_numpy(),
                    pose_row=groups.groupby('pose_scene', sort=False).cumcount()))
        poses = pd.concat(poses, ignore_index=True).rename(columns={
            'situation': 'brief_situation', 'location': 'lookup_location', 'orientation': 'lookup_orientation'})
        standing = poses.brief_situation.str.extract(parameters['patterns']['standing'])
        matched = standing[0].notna()
        poses.loc[matched, 'brief_situation'] = standing.loc[matched, 0] + standing.loc[matched, 1].str.split().str.join('_') + standing.loc[matched, 2]
        sitting = poses.brief_situation.str.startswith('sitting')
        poses.loc[sitting, 'brief_situation'] = 'sitting on ' + poses.loc[sitting, 'brief_situation'].str.slice(len('sitting on ')).str.split().str.join('_')
        keys = ['scan_id', 'brief_situation']
        needed = observations.loc[observations.task.ne('qa'), keys].drop_duplicates()
        poses = poses.merge(needed, on=keys, how='inner', validate='many_to_one')
        observations = observations.merge(poses, on=keys, how='left', validate='many_to_one')
        qa = observations.task.eq('qa')
        observations['pose_location'] = observations.location.where(qa, observations.lookup_location)
        observations['pose_orientation'] = observations.orientation.where(qa, observations.lookup_orientation)
        if observations.pose_location.isna().any() or observations.pose_orientation.isna().any() or observations.alignment.isna().any():
            raise ValueError('An observation lacks its original pose or scene alignment')

        # 3. Apply only the published score-reporting conversion; preserve null grades.
        observations['protocol'] = 'longform'
        observations.loc[qa, 'protocol'] = 'qa'
        distance = qa & observations.task_type.str.lower().str.contains('distance', regex=False)
        observations.loc[distance, 'protocol'] = 'distance'
        observations['response'] = pd.to_numeric(observations.score.where(distance), errors='raise')
        rating = ~distance & observations.score.notna()
        numbers = observations.loc[rating, 'score'].str.findall(parameters['patterns']['number'])
        if not numbers.map(len).eq(1).all():
            raise ValueError('An original judge reply does not contain exactly one rating')
        values = pd.to_numeric(numbers.str[0], errors='raise')
        if not values.isin([1, 2, 3, 4, 5]).all():
            raise ValueError('An original judge rating is outside its declared scale')
        observations.loc[rating, 'response'] = (values - 1) / 4
        available = observations.response.notna()
        if not observations.loc[available, 'response'].between(0, 1).all():
            raise ValueError('A source score is outside the original reporting scale')

        # 4. Attach original scene arrays and images; keep semantic label names in raw.
        scans = pd.DataFrame({'scan_id': pd.unique(pd.concat([observations.scan_id, observations.reference_scan]))})
        scans['point_member'] = layout['scene_prefix'] + scans.scan_id + '/pcd-align.pth'
        scans['instance_member'] = layout['scene_prefix'] + scans.scan_id + '/inst_to_label.pth'
        with ZipFile(self.raw_dir / layout['pointclouds']) as archive:
            scans['point_data'] = scans.point_member.map(archive.read)
            instances = scans.instance_member.map(archive.read)
            scans['instance_order'] = instances.map(lambda data: list(read_native_pickle(io.BytesIO(data))))
            scans['instance_sha256'] = instances.map(lambda data: hashlib.sha256(data).hexdigest())
        observations = observations.merge(scans, on='scan_id', how='left', validate='many_to_one')
        previous = scans.rename(columns={name: 'reference_' + name for name in scans if name != 'scan_id'}).rename(columns={'scan_id': 'reference_scan'})
        observations = observations.merge(previous, on='reference_scan', how='left', validate='many_to_one')
        observations['image_member'] = 'ego_view/' + observations.scan_id + '/' + observations.brief_situation.str.replace('/', '', regex=False).str.replace(' ', '_', regex=False) + '.png'
        images = observations[['image_member']].drop_duplicates().copy()
        with ZipFile(self.raw_dir / layout['images']) as archive:
            images['image_data'] = images.image_member.map(archive.read)
        observations = observations.merge(images, on='image_member', how='left', validate='many_to_one')

        # 5. Construct complete known stimuli and keep reference answers in grading.
        observations['item_key'] = observations.task + ':' + observations.scan_id + ':' + observations.source_index
        observations['response_key'] = observations.task + ':' + observations.source_row.astype(str)
        observations['subject_key'] = labels['subject']
        items = observations.copy()
        items['raw_item_id'] = items.item_key
        items['content'] = [labels['content_template'].format(situation=row.situation, instruction=row.instruction) for row in items.itertuples()]
        items['features'] = [dict(task=row.task, input_scope=labels['input_scope'], spatial_context=json.dumps(dict(
            current_scan=row.scan_id, reference_scan=row.reference_scan, brief_situation=row.brief_situation,
            recorded_anchor=row.anchor, source_pose=dict(location=row.pose_location, orientation=row.pose_orientation),
            alignment=row.alignment, current_instance_order=row.instance_order,
            reference_instance_order=row.reference_instance_order), sort_keys=True, allow_nan=False)) for row in items.itertuples()]
        items['attachments'] = [[
            dict(data=row.point_data, path='current_scene/pcd-align.pth', media_type=parameters['media_types']['pointcloud'], role=labels['current_role']),
            dict(data=row.reference_point_data, path='reference_scene/pcd-align.pth', media_type=parameters['media_types']['pointcloud'], role=labels['reference_role']),
            dict(data=row.image_data, path='egocentric_view.png', media_type=parameters['media_types']['image'], role=labels['image_role'])] for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.response_gt, rule=grading['verifiers'][row.protocol]['rule']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['verifiers'][row.protocol], sort_keys=True))
            if row.protocol == 'distance' else Judge(spec=json.dumps(grading['verifiers'][row.protocol], sort_keys=True)) for row in items.itertuples()]
        subjects = pd.DataFrame([dict(subject_key=labels['subject'], raw_label=labels['subject'],
            features=dict(harness=labels['harness'], **parameters['subject_features']))])

        # 6. Preserve full source records and the original input/pose associations.
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_record=row.source_record, unscored_file=row.unscored_file,
            annotation=dict(archive=layout['annotations'], member=row.annotation_member,
                scene=row.reference_scan, row=int(row.annotation_row), record=row.source_annotation),
            pose=dict(archive=layout['annotations'], member=row.annotation_member if row.task == 'qa' else row.pose_member,
                scene=row.reference_scan if row.task == 'qa' else row.pose_scene,
                row=int(row.annotation_row if row.task == 'qa' else row.pose_row),
                record=row.source_annotation if row.task == 'qa' else row.pose_record),
            alignment=dict(archive=layout['annotations'], member=layout['alignment'], scan_id=row.scan_id, record=row.alignment),
            scene_inputs=dict(archive=layout['pointclouds'], current_pointcloud=row.point_member,
                current_instances=row.instance_member, current_instances_sha256=row.instance_sha256,
                reference_pointcloud=row.reference_point_member, reference_instances=row.reference_instance_member,
                reference_instances_sha256=row.reference_instance_sha256),
            image=dict(archive=layout['images'], member=row.image_member), protocol=row.protocol,
            grade_status='source_grade_unavailable' if pd.isna(row.response) else 'recorded_score',
            grade_scope=labels['grade_scope']), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'features',
            'attachments', 'grading_criterion', 'verifier']],
            'responses': observations[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    Situat3DChange(__file__).main_from_args()
