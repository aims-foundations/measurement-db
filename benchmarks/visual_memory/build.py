"""Tabulate original visual-memory retrievals with their actual query images."""

from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
from urllib.parse import quote

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class VisualMemory(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate the released retrieval tables, keeping all neighbor fields.
        frames = []
        for path in sorted(self.raw_dir.glob(parameters['paths']['results'])):
            table = pd.read_json(path, orient='index', dtype=False, convert_dates=False, precise_float=True)
            table['native_record'] = table.to_dict('records')
            table = table.rename_axis('source_image').reset_index()
            task = path.name.split('_query-', 1)[1].split('_qsplit-', 1)[0]
            model = path.name.split('_qsplit-test_', 1)[1].removesuffix('_full_neighbor_info.json')
            if task not in parameters['archives'] or model not in parameters['models']:
                raise ValueError('Unknown original query dataset or featurizer')
            if not table.source_image.eq(table.image_id).all() or not table.featurizer.eq(model).all():
                raise ValueError('Source filename, image key and recorded configuration disagree')
            lengths = table[['neighbor_image_ids', 'neighbor_classes', 'neighbor_distances']].map(len)
            if not lengths.eq(int(parameters['protocol']['neighbors_saved'])).all().all():
                raise ValueError('An original neighbor array is missing or misaligned')
            if not table.image_class.map(lambda value: isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value < 1000).all():
                raise ValueError('Expected original ImageNet class indices')
            frames.append(table.assign(task=task, source_file=str(path.relative_to(self.raw_dir))))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda table: table.index)
        if responses.duplicated(['task', 'image_id', 'featurizer']).any():
            raise ValueError('Duplicate original model/image observation')

        # 2. Read the original images from their archives, without modifying raw files.
        image_rows = []
        for task, filename in parameters['archives'].items():
            with tarfile.open(self.raw_dir / filename) as archive:
                for member in archive.getmembers():
                    if not member.isfile() or member.name.startswith('__MACOSX/'):
                        continue
                    if Path(member.name).suffix.lower() not in parameters['image_suffixes']:
                        continue
                    parts = Path(member.name).parts
                    key = '/'.join(parts[-int(parameters['image_key_parts'][task]):])
                    data = archive.extractfile(member).read()
                    with Image.open(BytesIO(data)) as image:
                        media_type = Image.MIME[image.format]
                        image.verify()
                    image_rows.append(dict(task=task, image_id=key, image_bytes=data,
                        media_type=media_type, source_archive=filename, source_member=member.name))
        images = pd.DataFrame(image_rows)
        if images.duplicated(['task', 'image_id']).any():
            raise ValueError('Ambiguous original image identifier within a dataset')

        # 3. Join query definitions to their exact image bytes and original reference class.
        if responses.groupby(['task', 'image_id']).image_class.nunique().gt(1).any():
            raise ValueError('Original model exports disagree about a query reference class')
        items = responses[['task', 'image_id', 'image_class']].drop_duplicates().merge(
            images, on=['task', 'image_id'], how='outer', validate='one_to_one', indicator=True)
        if not items['_merge'].eq('both').all():
            raise ValueError('Query-image archives and original retrieval records do not correspond exactly')
        items['item_key'] = items.index
        items['raw_item_id'] = items.task + '||' + items.image_id
        items['content'] = parameters['protocol']['instruction']
        items['attachments'] = [[dict(path='images/' + row.task + '/' + quote(row.image_id, safe='/'),
            data=row.image_bytes, media_type=row.media_type, role='query_image')] for row in items.itertuples()]
        items['features'] = [dict(query_dataset=row.task, source_archive=row.source_archive,
            source_member=quote(row.source_member, safe='/')) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=str(int(value)), rule=self.grading['rule'])
            for value in items.image_class]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))
        responses = responses.merge(items[['task', 'image_id', 'item_key']], on=['task', 'image_id'], validate='many_to_one')

        # 4. Keep recorded featurizers distinct and apply the authors' explicit k=1 comparison.
        subjects = responses[['featurizer']].drop_duplicates().rename(columns={'featurizer': 'subject_key'})
        subjects['raw_label'] = subjects.subject_key.map(parameters['models'])
        subjects['features'] = [dict(source_featurizer=model, **parameters['subject']) for model in subjects.subject_key]
        responses['subject_key'] = responses.featurizer
        responses['response'] = responses.neighbor_classes.str[0].eq(responses.image_class).astype(float)
        responses['test_condition'] = 'k=1;memory=imagenet2012:train;query=' + responses.task + ';source_export=' + responses.source_file
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_image=row.source_image,
            native_record=row.native_record), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    VisualMemory(__file__).main_from_args()
