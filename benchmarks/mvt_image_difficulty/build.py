"""Tabulate the released MVT classification matrix with its original image stimuli."""

import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MVTImageDifficulty(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read the original wide prediction matrix and its class-index dictionary.
        with ZipFile(self.raw_dir / layout['results_archive']) as archive:
            matrix = pd.read_csv(archive.open(layout['prediction_matrix'])).rename(columns={'Unnamed: 0': 'source_index'})
            classes = pd.Series(json.loads(archive.read(layout['reference_classes'])), name='label').rename_axis('class_name').reset_index()
        matrix['source_row'] = matrix.index
        if matrix.image.duplicated().any() or classes.label.duplicated().any():
            raise ValueError('The source image identifiers and class indices must be unique')

        # 2. Melt model columns into observations and compare their recorded class indices.
        responses = matrix.melt(id_vars=['source_index', 'image', 'label', 'source_row'],
                                var_name='subject_key', value_name='prediction')
        if not responses.prediction.isin(classes.label).all() or not responses.label.isin(classes.label).all():
            raise ValueError('A recorded prediction or reference is outside the released class dictionary')
        responses['item_key'] = responses.image
        responses['response_key'] = responses.source_row.astype(str) + ':' + responses.subject_key
        responses['response'] = responses.prediction.eq(responses.label).astype(float)
        responses['test_condition'] = labels['condition']
        items = matrix[['image', 'label']].merge(classes, on='label', how='left', validate='many_to_one')

        # 3. Join the exact cropped image bytes without copying gold classes into item inputs.
        with ZipFile(self.raw_dir / layout['images_archive']) as outer:
            with ZipFile(io.BytesIO(outer.read(layout['nested_images']))) as archive:
                images = pd.DataFrame({'member': archive.namelist()})
                images['suffix'] = images.member.str.extract(r'(\.[^.\/]+)$', expand=False).str.lower()
                images = images[images.suffix.isin(parameters['media_types'])].copy()
                images['image'] = images.member.str.rsplit('/', n=1).str[-1]
                if images.image.duplicated().any():
                    raise ValueError('An image filename occurs more than once in the original archive')
                items = items.merge(images, on='image', how='left', validate='one_to_one', indicator=True)
                if not items['_merge'].eq('both').all():
                    raise ValueError('A recorded observation lacks its original image stimulus')
                items['data'] = items.member.map(archive.read)
        items['item_key'], items['raw_item_id'] = items.image, items.image
        items['content'] = None
        items['attachments'] = [[dict(data=row.data, path='images/' + row.image,
            media_type=parameters['media_types'][row.suffix], role=labels['image_role'])] for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.class_name, rule=grading['rule']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['verifiers']['classification'], sort_keys=True)) for _ in items.index]

        # 4. Preserve model variants and every recorded prediction with its original source location.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=value) for value in subjects.subject_key]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_archive=layout['results_archive'], source_file=layout['prediction_matrix'],
            source_row=row.source_row, source_index=row.source_index, image=row.image, model=row.subject_key,
            reference_index=row.label, predicted_index=row.prediction), allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'grading_criterion', 'verifier']],
                'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    MVTImageDifficulty(__file__).main_from_args()
