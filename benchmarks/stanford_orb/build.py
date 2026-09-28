"""Tabulate the original Stanford-ORB capture-level assessments and source assets."""

import ast
import json
from pathlib import Path, PurePosixPath
import posixpath
import sys
import tarfile
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class StanfordORB(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, layout = self.build_parameters, self.build_parameters['layout']

        # 1. Read the complete score fields and their original evaluation records.
        score_frames, input_frames = [], []
        for path in sorted((self.raw_dir / layout['methods']).glob('*.json')):
            document = json.loads(path.read_text())
            values = pd.json_normalize(document['scores'], sep='|').T.rename(columns={0: 'response'})
            values = values.loc[~values.response.map(lambda value: isinstance(value, dict))]
            fields = values.index.to_series().str.split('|', expand=True)
            if fields.shape[1] != 3:
                raise ValueError('A native score lacks its task, capture or metric')
            fields.columns = ['task', 'native_capture', 'metric']
            absent = fields.metric.isna()
            if values.loc[absent, 'response'].notna().any():
                raise ValueError('Unexpected scalar outside a native metric record')
            values = values.loc[~absent]
            values = values.join(fields)
            values['task'] = values.task.str.removesuffix('_all')
            score_frames.append(values.assign(method=path.stem, source_file=str(path.relative_to(self.raw_dir))))
            inputs = pd.DataFrame(document['info']).T.rename_axis('native_capture').stack(future_stack=True)
            inputs.index.names = ['native_capture', 'task']
            input_frames.append(inputs.rename('native_info').reset_index().assign(method=path.stem))
        scores = pd.concat(score_frames, ignore_index=True)
        scores['response'] = pd.to_numeric(scores.response, errors='raise')
        inputs = pd.concat(input_frames, ignore_index=True)
        identity = ['method', 'native_capture', 'task']
        scores = scores.merge(inputs, on=identity, validate='many_to_one', indicator=True)
        if not scores._merge.eq('both').all() or not np.isfinite(scores.response).all():
            raise ValueError('Every finite native score must have its original evaluation record')
        unsupported = scores.task.eq('shape') & scores.native_info.map(
            lambda value: isinstance(value, dict) and value.get('output_mesh') is None and value.get('target_mesh') is None)
        if not scores.loc[unsupported, 'response'].eq(0).all():
            raise ValueError('Unexpected score for an unavailable shape assessment')
        scores = scores.loc[~unsupported].drop(columns='_merge').reset_index(drop=True)

        # 2. Use the authors' literal capture-name mapping, without executing their notebook.
        notebook = json.loads((self.raw_dir / layout['notebook']).read_text())
        mappings = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code' or parameters['labels']['capture_mapping'] not in ''.join(cell['source']):
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                        target.id == parameters['labels']['capture_mapping'] for target in node.targets):
                    mappings.append(ast.literal_eval(node.value))
        if len(mappings) != 1:
            raise ValueError('Expected one original capture-name mapping')
        captures = pd.Series(mappings[0], name='capture').rename_axis('native_capture').reset_index()
        if captures.capture.duplicated().any():
            raise ValueError('Original capture mapping is not one-to-one')
        scores = scores.merge(captures, on='native_capture', validate='many_to_one', how='left')
        if scores.capture.isna().any():
            raise ValueError('A scored capture lacks its original released name')

        # 3. Read unmodified camera metadata from the original image archive.
        camera_records = []
        with tarfile.open(self.raw_dir / layout['ldr_archive'], 'r|gz') as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name.startswith('transforms_') and member.name.endswith('.json'):
                    camera_records.append(dict(member=member.name, camera=json.load(archive.extractfile(member))))
        cameras = pd.DataFrame(camera_records)
        cameras['capture'] = cameras.member.str.split('/').str[1]
        cameras['partition'] = cameras.member.str.extract(r'/transforms_([^/]+)\.json$', expand=False)
        cameras['source'] = cameras[['member', 'camera']].to_dict('records')
        camera_sets = cameras.pivot(index='capture', columns='partition', values='source')
        if camera_sets.isna().any().any() or set(camera_sets) != set(parameters['camera_partitions']):
            raise ValueError('Every capture needs its complete original camera partitions')
        camera_sets = camera_sets.apply(dict, axis=1).rename('camera_metadata').reset_index()
        scores = scores.merge(camera_sets, on='capture', validate='many_to_one', how='left')
        if scores.camera_metadata.isna().any():
            raise ValueError('A scored capture has no released camera metadata')

        # 4. Join each metric to its exact reference files, keeping repeated targets.
        entries = inputs.loc[inputs.set_index(identity).index.isin(scores.set_index(identity).index)].copy()
        entries['native_info'] = entries.native_info.map(lambda value: value if isinstance(value, list) else [value])
        entries = entries.explode('native_info').reset_index(drop=True)
        entries = pd.concat([entries[identity], pd.json_normalize(entries.native_info)], axis=1)
        if 'target_mask' not in entries:
            entries['target_mask'] = None
        for task, field in parameters['implicit_mask_inputs'].items():
            missing = entries.task.eq(task) & entries.target_mask.isna()
            filenames = entries.loc[missing, field].map(lambda value: Path(value).stem + '.png')
            entries.loc[missing, 'target_mask'] = (parameters['labels']['historical_mask_prefix'] +
                entries.loc[missing, 'native_capture'] + '/final_output/blender_format_LDR/test_mask/' + filenames)
        targets = entries.melt(id_vars=identity, value_vars=[key for key in entries if key.startswith('target_')],
            var_name='target_field', value_name='original_target').dropna(subset=['original_target'])
        coordinates = targets.original_target.map(posixpath.normpath).str.extract(parameters['patterns']['target'])
        if coordinates.isna().any().any():
            raise ValueError('An original reference file has no supported capture association')
        targets = targets.join(coordinates).merge(captures.rename(columns={
            'native_capture': 'target_native_capture', 'capture': 'target_capture'}), on='target_native_capture', validate='many_to_one')
        targets['container'] = targets.asset.str.split('/').str[0]
        targets['member'] = targets.asset.str.split('/', n=1).str[1]
        for original, released in parameters['ground_truth_names'].items():
            targets['member'] = targets.member.str.replace('^' + original + '/', released + '/', regex=True)
        targets['archive'] = targets.container.map(parameters['target_archives'])
        targets['member'] = targets.container.map(parameters['target_roots']) + '/' + targets.target_capture + '/' + targets.member
        if targets[['archive', 'member']].isna().any().any():
            raise ValueError('An original reference has no released archive location')
        targets['reference'] = targets[['target_field', 'archive', 'member']].to_dict('records')
        scores['verifier_key'] = scores.task + '.' + scores.metric
        if not scores.verifier_key.isin(self.grading['verifiers']).all():
            raise ValueError('A native metric has no declared grading interpretation')
        required = scores[identity + ['metric', 'verifier_key']].copy()
        required['target_field'] = required.verifier_key.map(lambda key: self.grading['verifiers'][key]['target_fields'])
        references = required.explode('target_field').merge(targets, on=identity + ['target_field'], validate='many_to_many')
        references = references.sort_values(identity + ['metric', 'target_field', 'member'])
        references = references.groupby(identity + ['metric'], sort=False).reference.agg(list).rename('references').reset_index()
        scores = scores.merge(references, on=identity + ['metric'], validate='one_to_one', how='left')
        if scores.references.isna().any():
            raise ValueError('A measured score lacks its native reference files')
        scores['reference_json'] = scores.references.map(lambda value: json.dumps(value, sort_keys=True))

        # 5. Join complete image captures and exact metric references to their unchanged bytes.
        item_identity = ['capture', 'task', 'metric', 'reference_json']
        items = scores.drop_duplicates(item_identity).reset_index(drop=True).copy()
        items['item_key'] = items.index
        reference_links = pd.json_normalize(items[['item_key', 'references']].to_dict('records'),
            record_path='references', meta='item_key')
        required_assets = set(zip(reference_links.archive, reference_links.member))
        payload_records = []
        input_archives = set(parameters['input_archives'].values())
        for source in sorted(input_archives | set(reference_links.archive)):
            with tarfile.open(self.raw_dir / source, 'r|gz') as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('An archived source path is not a safe relative path')
                    captured_image = source in input_archives and path.suffix in {'.png', '.exr'}
                    if not captured_image and (source, member.name) not in required_assets:
                        continue
                    if not member.isfile() or path.suffix not in parameters['asset_media_types']:
                        raise ValueError('A required asset has an unsupported file type')
                    payload_records.append(dict(archive=source, member=member.name,
                        capture=path.parts[1], data=archive.extractfile(member).read(),
                        media_type=parameters['asset_media_types'][path.suffix],
                        role='capture_mask' if path.parent.name.endswith('_mask') else 'capture_observation'))
        payloads = pd.DataFrame(payload_records)
        if payloads.duplicated(['archive', 'member']).any():
            raise ValueError('The source archives contain duplicate asset coordinates')
        capture_links = items[['item_key', 'capture']].merge(
            payloads.loc[payloads.archive.isin(input_archives), ['capture', 'archive', 'member', 'role']],
            on='capture', validate='many_to_many').drop(columns='capture')
        reference_links = reference_links[['item_key', 'archive', 'member']].assign(role='reference')
        links = pd.concat([reference_links, capture_links], ignore_index=True)
        links = links.drop_duplicates(['item_key', 'archive', 'member'])
        links = links.merge(payloads.drop(columns=['capture', 'role']), on=['archive', 'member'],
            how='left', validate='many_to_one', indicator=True)
        if not links._merge.eq('both').all():
            raise ValueError('A metric reference is absent from its original archive')
        links = links.sort_values(['item_key', 'member'])
        links['attachment'] = [dict(path=row.member, media_type=row.media_type, role=row.role, data=row.data)
            for row in links.itertuples()]
        items = items.merge(links.groupby('item_key', sort=False).attachment.agg(list).rename('attachments'),
            on='item_key', validate='one_to_one')

        # 6. Register the capture/metric definitions and literal method workflows.
        items['raw_item_id'] = items.capture + '/' + items.task + '/' + items.metric
        items['content'] = [json.dumps(dict(capture=row.capture, cameras=row.camera_metadata,
            input_archives=parameters['input_archives'], input_scope=parameters['labels']['input_scope']), sort_keys=True)
            for row in items.itertuples()]
        items['features'] = [dict(capture=row.capture, task=row.task,
            aggregation_unit=parameters['labels']['aggregation_unit']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference_json,
            rule=self.grading['verifiers'][row.verifier_key]['rule'],
            response_scale=self.grading['verifiers'][row.verifier_key]['response_scale']) for row in items.itertuples()]
        items['verifier'] = items.verifier_key.map(lambda key: ExactMatcher(spec=json.dumps(self.grading['verifiers'][key], sort_keys=True)))
        subjects = scores[['method']].drop_duplicates().rename(columns={'method': 'subject_key'})
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(source_method=method, configuration_scope=parameters['labels']['configuration_scope'],
            protocol=parameters['method_protocols'][method], configuration_sources=parameters['method_configs'][method])
            for method in subjects.subject_key]

        # 7. Link released rendered outputs at their verified method/capture level.
        with ZipFile(self.raw_dir / layout['render_archive']) as archive:
            rendered = pd.DataFrame({'member': [member.filename for member in archive.infolist() if not member.is_dir()]})
        rendered = rendered.join(rendered.member.str.extract(parameters['patterns']['render']))
        if rendered[['method', 'native_capture', 'output_index', 'format']].isna().any().any():
            raise ValueError('A released rendered image has no explicit method/capture association')
        rendered['asset'] = [dict(archive=layout['render_archive'], member=member) for member in rendered.member]
        rendered = rendered.groupby(['method', 'native_capture'], sort=False).asset.agg(list).rename('rendered_assets').reset_index()
        scores = scores.merge(rendered, on=['method', 'native_capture'], validate='many_to_one', how='left')
        scores = scores.merge(items[item_identity + ['item_key']], on=item_identity, validate='many_to_one')
        scores['response_key'] = scores[identity + ['metric']].agg('/'.join, axis=1)
        scores['subject_key'] = scores.method
        traces = scores[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_capture=row.native_capture,
            source_task=row.task, source_metric=row.metric, source_score=row.response, native_info=row.native_info,
            references=row.references, rendered_assets=row.rendered_assets if row.task == 'light' and isinstance(row.rendered_assets, list) else [],
            rendered_asset_scope=parameters['labels']['rendered_asset_scope'], evaluator_scope=parameters['labels']['evaluator_scope']),
            ensure_ascii=False, allow_nan=False) for row in scores.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            'responses': scores[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    StanfordORB(__file__).main_from_args()
