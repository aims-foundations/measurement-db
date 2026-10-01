"""Tabulate MOCHI's published per-image-set accuracy estimates and original images."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Mochi(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the two result tables and melt their model/readout score columns.
        item_frames, response_frames = [], []
        for family, filename in parameters['result_files'].items():
            table = pd.read_csv(self.raw_dir / filename, header=None, dtype=str, keep_default_na=False)
            table.columns = table.iloc[0]  # Preserve the original empty index-column name too.
            table = table.iloc[1:].reset_index(drop=True)
            if not table.columns.is_unique:
                raise ValueError('Duplicate original CSV column names')
            if table.empty or table.duplicated(['dataset', 'trial']).any():
                raise ValueError('Expected unique original image-set identifiers in each result file')
            table['native_record'] = table.to_dict('records')
            table = table.assign(source_file=filename, source_row=table.index, family=family)
            table['item_key'] = filename + ':' + table.source_row.astype(str)
            columns = table.columns[table.columns.str.fullmatch(parameters['score_patterns'][family])].tolist()
            if not columns:
                raise ValueError('No published model/readout estimates found')
            item_frames.append(table)
            response_frames.append(table.melt(id_vars='item_key', value_vars=columns,
                var_name='subject_key', value_name='response'))
        items = pd.concat(item_frames, ignore_index=True)
        responses = pd.concat(response_frames, ignore_index=True).assign(response_key=lambda table: table.index)
        responses['response'] = responses.response.map(float)

        # 2. Join the released image bank, checking task, image-set and reference correspondence.
        images = pd.read_parquet(self.raw_dir / parameters['paths']['images'])
        images['image_source_row'] = images.index
        images['dataset'] = images.dataset.replace(parameters['dataset_aliases'])
        if images.duplicated(['dataset', 'trial']).any():
            raise ValueError('Duplicate image-bank task identifiers')
        images['image_names'] = images.images.map(lambda values: [image['path'] for image in values])
        images['reference_image'] = [names[index] for names, index in zip(images.image_names, images.oddity_index)]
        items['image_names'] = items.images.map(ast.literal_eval)
        items['oddity_index'] = items.oddity_index.astype(int)
        if not items.image_names.map(lambda names: isinstance(names, list) and len(names) in (3, 4)
                and len(set(names)) == len(names) and all(isinstance(name, str) for name in names)).all():
            raise ValueError('Expected three or four distinct image filenames per task')
        if not ((items.oddity_index >= 0) & (items.oddity_index < items.image_names.map(len))).all():
            raise ValueError('Reference index is outside the recorded image order')
        items['reference_image'] = [names[index] for names, index in zip(items.image_names, items.oddity_index)]
        items = items.merge(images[['dataset', 'trial', 'condition', 'image_names', 'reference_image',
            'DINOv2G_avg', 'image_source_row']], on=['dataset', 'trial'], how='left',
            validate='many_to_one', suffixes=('', '_bank'))
        if items.image_source_row.isna().any():
            raise ValueError('A recorded image set is missing from the image bank')
        if not (items.image_names.map(set).eq(items.image_names_bank.map(set))
                & items.reference_image.eq(items.reference_image_bank)
                & items.condition.replace(parameters['condition_aliases']).eq(items.condition_bank)).all():
            raise ValueError('Image set, reference or condition disagrees with the image bank')
        svm = items.loc[items.family.eq('svm')]
        if not (svm.image_names.eq(svm.image_names_bank)
                & (svm['dinov2-giant_svm_avg'].map(float) - svm.DINOv2G_avg).abs().le(1e-12)).all():
            raise ValueError('SVM image order or duplicated DINOv2-G estimate differs from the bank')
        pooled = items.pivot(index=['dataset', 'trial'], columns='family', values='dino_distance_avg')
        if pooled.isna().any().any() or not pooled.svm.map(float).eq(pooled.distance.map(float)).all():
            raise ValueError('The duplicated pooled-distance estimate disagrees across result files')
        image_rows = pd.json_normalize(images.images.explode().tolist(), max_level=0)
        image_rows['sha256'] = image_rows['bytes'].map(lambda value: hashlib.sha256(value).hexdigest())
        if image_rows.groupby('path').sha256.nunique().gt(1).any():
            raise ValueError('One original image filename refers to different byte payloads')
        image_bytes = image_rows.drop_duplicates('path').set_index('path')['bytes']

        # 3. Attach exact image bytes in each source order; answers stay in grading fields.
        items['attachments'] = [[dict(data=image_bytes[name], path=f'image_{index}.png',
            media_type='image/png', role='input') for index, name in enumerate(names)] for names in items.image_names]
        items['content'] = [json.dumps(dict(instruction=parameters['labels']['instruction'],
            images=[asset['path'] for asset in attachments])) for attachments in items.attachments]
        items['raw_item_id'] = items.dataset + '/' + items.trial
        items['features'] = [dict(dataset=row.dataset, condition=row.condition) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=str(index), rule=self.grading['rule']) for index in items.oddity_index]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))

        # 4. Keep literal model/readout identities and full assessment rows; do not expand averages into trials.
        subjects = responses[['subject_key']].drop_duplicates().reset_index(drop=True)
        settings = subjects.subject_key.str.rsplit('_', n=2, expand=True)
        subjects['raw_label'] = settings[0]
        subjects['features'] = [dict(harness=parameters['labels']['harness'], readout=readout,
            statistic=parameters['labels']['statistic'], source_column=column)
            for readout, column in zip(settings[1], subjects.subject_key)]
        traces = responses[['response_key', 'item_key', 'subject_key']].merge(items[
            ['item_key', 'source_file', 'source_row', 'native_record', 'image_source_row']],
            on='item_key', validate='many_to_one')
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            score_column=row.subject_key, native_record=row.native_record,
            image_source_file=parameters['paths']['images'], image_source_row=int(row.image_source_row)),
            ensure_ascii=False, allow_nan=False) for row in traces.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': traces[['response_key', 'trace']],
        }


if __name__ == '__main__':
    Mochi(__file__).main_from_args()
