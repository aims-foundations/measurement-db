"""Tabulate MJ-Bench's original image inputs and released judge observations."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import tarfile
from zipfile import ZipFile

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MJBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the original task tables and resolve repeated image-pair exports.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['task_glob'])):
            table = pd.read_parquet(path).rename_axis('source_row').reset_index()
            table['split'] = path.stem
            table['source_file'] = str(path.relative_to(self.raw_dir))
            parts.append(table)
        bank = pd.concat(parts, ignore_index=True)
        # Earlier author annotations preserve captions, image order and references
        # that changed in the later HF packaging. Resolve their files by unique
        # image identity, retaining the original annotation as the grading source.
        images = pd.concat([bank[['source_file', 'source_row', role]].rename(columns={role: 'image'}).assign(role=role)
            for role in ['image0', 'image1']], ignore_index=True)
        images['image_name'] = images.image.map(lambda image: image['path'])
        images['image_hash'] = images.image.map(lambda image: hashlib.sha256(image['bytes']).hexdigest())
        unique = images.groupby('image_name').image_hash.nunique().eq(1)
        images = images.loc[images.image_name.isin(unique.index[unique])].drop_duplicates('image_name')
        images['coordinate'] = [dict(source_file=row.source_file, source_row=row.source_row, role=row.role)
            for row in images.itertuples()]

        # Restore earlier alignment images from the authors' identified sources.
        # Pick-a-Pic used an RGB/default-JPEG cache; the other two sources used
        # original bytes. Every overlap must match the published MJ-Bench bytes.
        path = self.raw_dir / parameters['layout']['pickapic']
        source = pd.read_parquet(path).rename_axis('source_row').reset_index()
        pickapic = pd.concat([source[['source_row', f'image_{side}_uid', f'jpg_{side}']].rename(
            columns={f'image_{side}_uid': 'uid', f'jpg_{side}': 'bytes'}).assign(role=f'jpg_{side}')
            for side in [0, 1]], ignore_index=True).drop_duplicates('uid')
        recovered = []
        for row in pickapic.itertuples():
            encoded = BytesIO()
            with Image.open(BytesIO(row.bytes)) as opened:
                opened.convert('RGB').save(encoded, format='JPEG')
            recovered.append(dict(image_name=row.uid + '.jpg', image=dict(path=row.uid + '.jpg', bytes=encoded.getvalue()),
                coordinate=dict(source_file=str(path.relative_to(self.raw_dir)), source_row=row.source_row,
                    role=row.role, transformation='RGB conversion and default JPEG serialization',
                    method_source='author/get_rm_score.py', source_sha256=hashlib.sha256(row.bytes).hexdigest())))
        path = self.raw_dir / parameters['layout']['hpdv2']
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile() and Path(member.name).suffix.lower() == '.jpg':
                    recovered.append(dict(image_name=Path(member.name).name,
                        image=dict(path=Path(member.name).name, bytes=archive.extractfile(member).read()),
                        coordinate=dict(source_file=str(path.relative_to(self.raw_dir)), member=member.name)))
        for path in sorted(self.raw_dir.glob(parameters['layout']['imagereward'])):
            with ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir() and Path(member.filename).suffix.lower() == '.webp':
                        recovered.append(dict(image_name=Path(member.filename).name,
                            image=dict(path=Path(member.filename).name, bytes=archive.read(member)),
                            coordinate=dict(source_file=str(path.relative_to(self.raw_dir)), member=member.filename)))
        recovered = pd.DataFrame(recovered)
        recovered['image_hash'] = recovered.image.map(lambda image: hashlib.sha256(image['bytes']).hexdigest())
        combined = pd.concat([images, recovered], ignore_index=True)
        if combined.groupby('image_name').image_hash.nunique().gt(1).any():
            raise ValueError('Recovered image bytes disagree with a published input of the same name')
        images = combined.drop_duplicates('image_name')
        original = []
        for split, pattern in parameters['annotation_globs'].items():
            for path in sorted(self.raw_dir.glob(pattern)):
                table = pd.json_normalize(json.loads(path.read_text()), max_level=0).rename_axis('source_row').reset_index()
                table['source_file'], table['split'], table['info'] = str(path.relative_to(self.raw_dir)), 'historical_' + split, ''
                if 'image0' in table:
                    table = table.rename(columns={'image0': 'image0_name', 'image1': 'image1_name'})
                else:
                    table['label'] = table.label_0.map({1: 0, 0: 1, .5: 'tie'})
                    if 'sharp_image' in table:
                        table = table.melt(id_vars=[column for column in table if column not in ['motion_blur_image', 'defocused_blur_image']],
                            value_vars=['motion_blur_image', 'defocused_blur_image'], var_name='annotation_variant', value_name='image1_name')
                        table = table.rename(columns={'sharp_image': 'image0_name'})
                    else:
                        table = table.rename(columns={'image_0': 'image0_name', 'image_1': 'image1_name'})
                for role in ['image0', 'image1']:
                    table[role + '_name'] = table[role + '_name'].map(lambda name: str(Path(name).with_suffix(
                        parameters['image_suffix_aliases'].get(Path(name).suffix, Path(name).suffix))))
                    lookup = images[['image_name', 'image', 'coordinate']].rename(columns={
                        'image_name': role + '_name', 'image': role, 'coordinate': role + '_source'})
                    table = table.merge(lookup, on=role + '_name', how='left', validate='many_to_one')
                original.append(table.loc[table.image0.notna() & table.image1.notna()])
        bank = pd.concat([bank, *original], ignore_index=True)
        for role in ['image0', 'image1']:
            bank[role + '_name'] = bank[role].map(lambda image: image['path'])
            bank[role + '_hash'] = bank[role].map(lambda image: hashlib.sha256(image['bytes']).hexdigest())
        bank['input_key'] = [json.dumps([row.caption, row.info, row.image0_name] if row.split == 'bias'
            else [row.caption, row.image0_name, row.image1_name, str(row.label)], ensure_ascii=False)
            for row in bank.itertuples()]
        identities = bank.groupby('input_key')[['image0_hash', 'image1_hash']].nunique()
        if identities.gt(1).any().any():
            raise ValueError('A repeated task identifier refers to conflicting image bytes')
        bank['task_coordinate'] = [dict(source_file=row.source_file, source_row=row.source_row,
            images={role: row._asdict()[role + '_source'] if isinstance(row._asdict().get(role + '_source'), dict)
                else dict(source_file=row.source_file, source_row=row.source_row, role=role)
                for role in ['image0', 'image1']})
            for row in bank.itertuples()]
        coordinates = bank.groupby('input_key', sort=False).task_coordinate.agg(list).rename('task_coordinates')
        bank = bank.drop_duplicates('input_key').merge(coordinates, on='input_key', validate='one_to_one')
        bank = bank[['input_key', 'split', 'image0', 'image1', 'image0_hash', 'image1_hash', 'task_coordinates']]

        # 2. Normalize native result tables; unpivot wide-form bias model ratings.
        paths = sorted({path for pattern in parameters['result_globs'].values() for path in self.raw_dir.glob(pattern)})
        parts = []
        for path in paths:
            # Workspace names must never become part of the source protocol.
            source_parts = path.relative_to(self.raw_dir).parts
            rows = json.loads(path.read_text())
            if not rows:
                continue
            table = pd.json_normalize(rows, max_level=0).rename_axis('source_row').reset_index()
            table['native_record'] = rows
            table['source_file'] = str(path.relative_to(self.raw_dir))
            if 'ranking_id' in table:
                # These dated files are the separate Pick-a-Pic development split.
                continue
            basename = path.stem
            standard = re.fullmatch(r'(.+)_(number|narrative)_scale(\d+)', basename)
            closed = re.fullmatch(r'(.+)_alignment_number10', basename)
            wide = 'images_dir' in table and 'ImageReward' in table
            if wide:
                target = basename.split('bias_dataset_', 1)[1] if 'bias_dataset_' in basename else None
                models = list(parameters['score_models']) + ([target] if target else [])
                table = table.melt(id_vars=[column for column in table if column not in models],
                    value_vars=models, var_name='model', value_name='native_rating')
                table = table.loc[[model in record for model, record in zip(table.model, table.native_record)]].copy()
                table['dimension'], table['mode'] = 'bias', 'single_image'
                table['style'] = 'narrative' if basename.startswith('narrative_') else 'number'
                table['scale'] = '10' if basename.startswith('scale_10_') else 'not_recorded'
                table['rating_column'] = table.model
                table['model_output'] = [dict(rating=record[model], **({'analysis': record.get('analysis')}
                    if model == target else {})) for record, model in zip(table.native_record, table.model)]
                table['preference_encoding'] = 'single'
            else:
                table['model'] = standard[1] if standard else closed[1] if closed else path.parent.name
                table['style'] = standard[2] if standard else 'number' if closed else 'not_recorded'
                table['scale'] = standard[3] if standard else '10' if closed else 'not_recorded'
                dimension = next((name for name in ['alignment', 'artifacts', 'safety', 'bias'] if name in source_parts), None)
                if 'images_dir' in table:
                    dimension = 'bias'
                if dimension is None and closed:
                    dimension = 'alignment'
                if dimension is None:
                    dimension = parameters['legacy_dimensions'][re.sub(r'_?0\.0$', '', basename).lower()]
                table['dimension'] = dimension
                table['mode'] = ('multi_image' if 'vlm_pred' in table else 'single_image'
                    if 'output_0' in table or 'vlm_output' in table else 'not_recorded')
                table['rating_column'] = 'score' if dimension == 'bias' else None
                if 'vlm_output' in table:
                    table['model_output'] = table.native_record.map(lambda row: dict(vlm_output=row['vlm_output']))
                elif 'output_0' in table:
                    table['model_output'] = table.native_record.map(lambda row: dict(output_0=row['output_0'], output_1=row['output_1']))
                elif 'output' in table:
                    table['model_output'] = table.native_record.map(lambda row: dict(output=row['output']))
                elif 'scores' in table:
                    table['model_output'] = table.native_record.map(lambda row: dict(scores=row['scores']))
                else:
                    table['model_output'] = table.native_record.map(lambda row: dict(score_0=row.get('score_0'), score_1=row.get('score_1')))
                table['preference_encoding'] = 'online' if 'online_result' in source_parts else (
                    'multi' if 'vlm_pred' in table else 'single')
            score_models = table.model.isin(parameters['score_models'])
            table.loc[score_models, ['style', 'scale', 'mode']] = ['scalar', 'not_applicable', 'single_image']
            table['caption_text'] = table.native_record.map(lambda row: row.get('caption', row.get('prompt')))
            table['input_key'] = [json.dumps([row['prompt'], row['demographic'], Path(row['images_dir']).name], ensure_ascii=False)
                if 'images_dir' in row else json.dumps([row['caption'],
                    *next(([str(Path(str(row[field])).with_suffix(parameters['image_suffix_aliases'].get(
                            Path(str(row[field])).suffix, Path(str(row[field])).suffix))).split('/')[-1] for field in [left, right]]
                        for left, right in parameters['image_fields'].items() if left in row and right in row)), str(row['label'])],
                    ensure_ascii=False) for row in table.native_record]
            table['source_preference'] = [parameters['preference_' + encoding].get(
                str(record.get('vlm_pred' if 'vlm_pred' in record else 'pred')).strip())
                for record, encoding in zip(table.native_record, table.preference_encoding)]
            parts.append(table[['source_file', 'source_row', 'native_record', 'model', 'dimension', 'mode', 'style', 'scale',
                'rating_column', 'model_output', 'caption_text', 'input_key', 'source_preference']])
        native = pd.concat(parts, ignore_index=True)
        native['subject_key'] = [json.dumps([row.model, row.mode, row.style, row.scale], ensure_ascii=False)
            for row in native.itertuples()]
        joined = native.merge(bank, on='input_key', how='left', validate='many_to_one')
        unmatched = joined.split.isna()
        print(f'MJ-Bench: {len(native)} native model-record occurrences; {int(unmatched.sum())} lack a matched original input and remain in raw.', flush=True)
        observations = joined.loc[~unmatched].copy()

        # 3. Collapse repeated exports while retaining every complete source record.
        observations['item_key'] = [json.dumps([row.input_key, row.dimension, row.mode, row.style, row.scale], ensure_ascii=False)
            for row in observations.itertuples()]
        observations['response_key'] = [hashlib.sha256(json.dumps([row.subject_key, row.item_key, row.model_output],
            ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest() for row in observations.itertuples()]
        observations['source_assessment'] = [dict(source_file=row.source_file, source_row=row.source_row,
            rating_column=row.rating_column, native_record=row.native_record) for row in observations.itertuples()]
        aliases = observations.groupby('response_key', sort=False).source_assessment.agg(list).rename('source_assessments')
        preferences = observations.groupby('response_key', sort=False).source_preference.agg(
            lambda values: sorted({value for value in values if value is not None and not (isinstance(value, float) and pd.isna(value))}, key=str)).rename('reported_preferences')
        observations = observations.drop_duplicates('response_key').merge(aliases, on='response_key', validate='one_to_one')
        observations = observations.merge(preferences, on='response_key', validate='one_to_one')
        observations['grade_kind'] = observations.dimension.map(lambda value: 'bias' if value == 'bias' else 'preference')
        observations['reference'] = observations.native_record.map(lambda row: str(row['label']) if 'label' in row else None)
        observations['response'] = pd.Series([None if row.grade_kind == 'bias' or len(row.reported_preferences) != 1
            else float(str(row.reported_preferences[0]) == row.reference) for row in observations.itertuples()], dtype=object)
        observations['grade_status'] = ['population_metric_only' if row.grade_kind == 'bias' else
            'unavailable_preference' if not row.reported_preferences else 'conflicting_released_preferences'
            if len(row.reported_preferences) > 1 else 'recorded_preference_agreement' for row in observations.itertuples()]

        # 4. Keep model configurations, full image bytes and grading definitions.
        subjects = observations[['subject_key', 'model', 'mode', 'style', 'scale']].drop_duplicates('subject_key').copy()
        subjects['raw_label'] = [parameters['labels']['subject_prefix'] + ' / '.join([row.model, row.mode, row.style, row.scale])
            for row in subjects.itertuples()]
        subjects['features'] = [dict(**parameters['subject_features'], source_model=row.model,
            input_mode=row.mode, response_style=row.style, declared_scale=row.scale) for row in subjects.itertuples()]
        items = observations.drop_duplicates('item_key').copy()
        attachments, contents, features, criteria, verifiers = [], [], [], [], []
        for row in items.itertuples():
            images = [row.image0] if row.grade_kind == 'bias' else [row.image0, row.image1]
            assets = []
            for index, image in enumerate(images):
                with Image.open(BytesIO(image['bytes'])) as opened:
                    media_type = parameters['image_types'][opened.format]
                assets.append(dict(data=image['bytes'], path=f'image{index}/' + image['path'], role='input', media_type=media_type))
            attachments.append(assets)
            prompt = self.raw_dir / parameters['layout']['prompt_root'] / ('prompts_' + row.mode) / (
                f'{row.dimension}_{row.mode.removesuffix("_image")}_{row.style}_scale{row.scale}.txt')
            declared = dict(source_file=str(prompt.relative_to(self.raw_dir)), template=prompt.read_text(),
                historical_execution_revision='not_recorded') if prompt.is_file() else None
            contents.append(json.dumps(dict(multimedia_elements=[dict(content_type='text/plain', text=row.caption_text)]
                + [dict(content_type=asset['media_type'], location=asset['path']) for asset in assets],
                published_prompt_template=declared), ensure_ascii=False))
            features.append(dict(dimension=row.dimension, input_mode=row.mode, response_style=row.style,
                declared_scale=row.scale, input_scope=parameters['labels']['input_scope']))
            criteria.append(dict(rule=self.grading['verifiers']['bias']['rule']) if row.grade_kind == 'bias' else
                dict(reference_answer=row.reference, rule=self.grading['rule']))
            verifiers.append(ExactMatcher(spec=json.dumps(self.grading['verifiers'][row.grade_kind], sort_keys=True)))
        items['raw_item_id'] = items.input_key.map(lambda key: hashlib.sha256(key.encode()).hexdigest())
        items['attachments'], items['content'], items['features'] = attachments, contents, features
        items['grading_criterion'], items['verifier'] = criteria, verifiers

        # 5. Preserve full native outputs, assessment aliases and input coordinates.
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_assessments=row.source_assessments,
            model_output=row.model_output, task_coordinates=row.task_coordinates,
            input_key=row.input_key, subject_key=row.subject_key, image0_sha256=row.image0_hash,
            image1_sha256=None if row.grade_kind == 'bias' else row.image1_hash,
            reported_preferences=row.reported_preferences, grade_status=row.grade_status),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': observations[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    MJBench(__file__).main_from_args()
