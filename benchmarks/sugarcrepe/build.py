#!/usr/bin/env python3
"""Tabulate SugarCrepe's published GPT-4V answers with their ordered visual inputs."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SugarCrepe(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Concatenate the two presentation orders, retaining each complete record.
        frames = []
        for path in sorted((self.raw_dir / layout['results']).glob('*/*.json')):
            records = json.loads(path.read_text())
            records.pop(parameters['parsing']['summary_key'], None)
            frame = pd.DataFrame.from_dict(records, orient='index').rename_axis('source_key').reset_index()
            frame['native_record'] = list(records.values())
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)),
                caption_order=path.parent.name, category=path.stem.removeprefix(parameters['parsing']['prefix'])))
        responses = pd.concat(frames, ignore_index=True)
        if not responses.correct.dropna().map(lambda value: type(value) is bool).all():
            raise ValueError('Released correctness flags must be boolean or null')
        if not responses.caption_order.isin(parameters['first_caption']).all():
            raise ValueError('Unknown caption presentation order')
        responses['response_key'] = responses.index
        responses['response'] = responses.correct.map({True: 1., False: 0.})
        responses['test_condition'] = 'caption_order=' + responses.caption_order

        # 2. Join original COCO image records and their per-image license declarations.
        with ZipFile(self.raw_dir / layout['annotations']) as archive:
            annotation = json.loads(archive.read(layout['annotation_member']))
        images = pd.json_normalize(annotation['images'], max_level=0).rename(columns={'file_name': 'filename'})
        images['coco_image'] = annotation['images']
        licenses = pd.json_normalize(annotation['licenses'], max_level=0).rename(columns={'id': 'license'})
        licenses['coco_license'] = annotation['licenses']
        images = images.merge(licenses[['license', 'coco_license']], on='license', how='left', validate='many_to_one')
        responses = responses.merge(images[['filename', 'coco_image', 'coco_license']],
            on='filename', how='left', validate='many_to_one', indicator=True)
        if not responses._merge.eq('both').all() or responses.coco_license.isna().any():
            raise ValueError('Every observed image must have its original COCO attribution')

        # 3. Reconstruct the documented ordered prompt without marking its correct choice.
        items = responses.copy()
        positive_first = items.caption_order.eq(parameters['labels']['positive_first'])
        items['first'] = items.caption.where(positive_first, items.negative_caption)
        items['second'] = items.negative_caption.where(positive_first, items.caption)
        items['content'] = [parameters['prompt']['template'].format(caption1=row.first, caption2=row.second)
            for row in items.itertuples()]
        items['item_key'] = items.response_key
        items['raw_item_id'] = items.caption_order + ':' + items.category + ':' + items.source_key
        items['features'] = items[['category', 'caption_order', 'filename']].to_dict('records')
        items['grading_criterion'] = [dict(reference_answer=caption, rule=self.grading['rule']) for caption in items.caption]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['published'], sort_keys=True))

        # 4. Attach unchanged original image bytes, reading each shared image only once.
        with ZipFile(self.raw_dir / layout['images']) as archive:
            assets = {name: archive.read(layout['image_prefix'] + name) for name in items.filename.unique()}
        items['attachments'] = [[dict(data=assets[name], path=layout['image_prefix'] + name,
            media_type='image/jpeg', role='input')] for name in items.filename]

        # 5. Preserve literal model identity, native answers, source keys and attribution.
        subjects = pd.DataFrame([dict(subject_key=parameters['labels']['model'],
            raw_label=parameters['labels']['subject_prefix'] + parameters['labels']['model'],
            features=dict(source_model=parameters['labels']['model'], harness=parameters['labels']['harness']))])
        responses['subject_key'] = parameters['labels']['model']
        responses['item_key'] = responses.response_key
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_key=row.source_key,
            native_record=row.native_record, coco_image=row.coco_image, coco_license=row.coco_license),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features',
            'grading_criterion', 'verifier', 'attachments']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    SugarCrepe(__file__).main_from_args()
