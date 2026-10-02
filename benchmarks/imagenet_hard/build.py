"""Tabulate the fixed center-crop ImageNet-A cohort of the original zoom study."""

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle


class ImageNetHard(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, selection = parameters['layout'], parameters['selection']

        # 1. Load original images in the source ImageFolder traversal order.
        with tarfile.open(self.raw_dir / paths['images']) as archive:
            names = sorted(member.name for member in archive if member.isfile()
                and Path(member.name).suffix.lower() in selection['image_suffixes'].split(','))
            items = pd.DataFrame({'source_image': names})
            items['image_bytes'] = [archive.extractfile(name).read() for name in names]
        items['image_sha256'] = items.image_bytes.map(lambda data: hashlib.sha256(data).hexdigest())
        items['logical_path'] = 'images/' + items.image_sha256 + '.jpg'
        items['attachments'] = [[dict(data=row.image_bytes, path=row.logical_path,
            media_type='image/jpeg', role='input')] for row in items.itertuples()]
        items['source_image_index'] = range(len(items))
        items['reference_class'] = items.source_image.map(lambda name: Path(name).parent.name)
        items['item_key'] = items.image_sha256 + ':' + items.reference_class
        classes = sorted(items.reference_class.unique())
        items['reference_index'] = items.reference_class.map({name: index for index, name in enumerate(classes)})

        # 2. Select the documented baseline from the original correctness tables.
        matrices = read_native_pickle(self.raw_dir / paths['correctness'])[selection['dataset']]
        main = pd.DataFrame({model: frame.loc[selection['transform']] for model, frame in matrices.items()})
        main = main.rename_axis('source_image_index').reset_index().melt(
            id_vars='source_image_index', var_name='source_model', value_name='response')
        main['source_image_index'] = main.source_image_index.astype(int)
        main['source_release'] = 'main'
        main['source_file'] = paths['correctness']
        main['predicted_class_index'] = None

        # 3. Load the independently released baseline predictions and join images.
        additional = []
        with ZipFile(self.raw_dir / paths['additional']) as archive:
            for name in sorted(archive.namelist()):
                match = re.fullmatch(selection['prediction_pattern'], name)
                if not match:
                    continue
                predictions = read_native_pickle(io.BytesIO(archive.read(name)))[int(selection['scale'])]
                frame = pd.DataFrame({'predicted_class_index': predictions})
                frame['source_image_index'] = range(len(frame))
                additional.append(frame.assign(source_release='additional', source_model=match[1], source_file=name))
        additional = pd.concat(additional, ignore_index=True).merge(
            items[['source_image_index', 'reference_index']], on='source_image_index', validate='many_to_one')
        additional['response'] = additional.predicted_class_index.eq(additional.reference_index)
        responses = pd.concat([main, additional], ignore_index=True).merge(
            items[['source_image_index', 'source_image', 'item_key']], on='source_image_index', how='left', validate='many_to_one')
        if responses.item_key.isna().any():
            raise ValueError('A native observation has no original image')
        responses['response'] = responses.response.astype(float)
        responses['subject_key'] = responses.source_release + ':' + responses.source_model
        responses['response_key'] = responses.subject_key + ':' + responses.source_image_index.astype(str)
        responses['test_condition'] = selection['transform'] + ':' + responses.source_image_index.astype(str)

        # 4. Preserve the original stimulus, reference class and masked grading rule.
        items = items.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.source_image
        items['content'] = items.logical_path.map(lambda name: json.dumps(dict(multimedia_elements=[
            dict(content_type='image/jpeg', location=name)])))
        items['features'] = [dict(source_cohort=selection['dataset'])] * len(items)
        items['grading_criterion'] = [dict(reference_answer=label, rule=json.dumps(dict(
            instruction=self.grading['rule'], class_order=classes))) for label in items.reference_class]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['classification'], sort_keys=True))] * len(items)

        # 5. Keep release-specific subjects and unchanged grade/prediction evidence.
        subjects = responses[['subject_key', 'source_release', 'source_model']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.source_model.map(parameters['model_names'])
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_release=row.source_release,
            model_identifier=row.source_model, crop=parameters['labels']['transform'],
            historical_configuration=parameters['labels']['unavailable']) for row in subjects.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_release=row.source_release, source_file=row.source_file,
            source_model=row.source_model, source_image_index=row.source_image_index, source_image=row.source_image,
            transform=selection['transform'], native_correctness=bool(row.response) if row.source_release == 'main' else None,
            predicted_class_index=None if row.source_release == 'main' else int(row.predicted_class_index)), allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces[['response_key', 'trace']]}


if __name__ == '__main__':
    ImageNetHard(__file__).main_from_args()
