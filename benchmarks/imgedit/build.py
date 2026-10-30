"""Tabulate ImgEdit's released BAGEL attempts, images and original judge scores."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge
from measurement_db.benchmarks.imgedit.ratings import read_ratings, rubric_dimensions
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


class ImgEdit(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, files, labels = parameters['layout'], parameters['files'], parameters['labels']

        # 1. Read original instructions and attach their unmodified input images.
        with tarfile.open(self.raw_dir / paths['inputs']) as archive:
            tasks = json.load(archive.extractfile(paths['tasks']))
            rubrics = json.load(archive.extractfile(paths['rubrics']))
            items = pd.DataFrame.from_dict(tasks, orient='index').rename_axis('item_key').reset_index()
            images = items[['id']].drop_duplicates().copy()
            images['data'] = images.id.map(lambda name: archive.extractfile(paths['input_prefix'] + name).read())
        images['sha256'] = images.data.map(lambda data: hashlib.sha256(data).hexdigest())
        images['path'] = 'images/' + images.sha256 + '.jpg'
        images['attachments'] = [[dict(data=row.data, path=row.path, media_type='image/jpeg', role='input')]
            for row in images.itertuples()]
        items = items.merge(images[['id', 'path', 'attachments']], on='id', validate='many_to_one')

        # 2. Read every saved judgment and left-join the grades the publisher kept.
        with ZipFile(self.raw_dir / paths['results']) as archive:
            responses = pd.Series(json.loads(archive.read(paths['output_prefix'] + files['judgments'])), dtype=object
                ).rename_axis('item_key').reset_index(name='judge_text')
            grades = pd.Series(json.loads(archive.read(paths['output_prefix'] + files['grades'])), dtype=float
                ).rename_axis('item_key').reset_index(name='response')
            responses = responses.merge(grades, on='item_key', how='left', validate='one_to_one')
            responses['output_member'] = paths['output_prefix'] + responses.item_key + '.png'
            responses['output_sha256'] = responses.output_member.map(lambda name: hashlib.sha256(archive.read(name)).hexdigest())
            responses['output_bytes'] = responses.output_member.map(lambda name: archive.getinfo(name).file_size)
        responses = responses.merge(items[['item_key', 'id', 'prompt', 'edit_type']], on='item_key', how='left', validate='one_to_one')
        if responses.id.isna().any():
            raise ValueError('A released judgment has no original editing task')

        # 3. Keep the editing stimulus separate from the rubric and generated image.
        items['raw_item_id'] = items.item_key
        items['testlet_id'] = items.raw_item_id
        items['content'] = [json.dumps(dict(text=row.prompt, multimedia_elements=[
            dict(content_type='image/jpeg', location=row.path)])) for row in items.itertuples()]
        items['features'] = items.edit_type.map(lambda value: dict(edit_type=value))
        items['grading_criterion'] = [dict(rule=json.dumps(dict(instruction=self.grading['verifiers']['native_judge']['aggregate_rule'],
            judge_prompt=rubrics[row.edit_type].replace('<edit_prompt>', row.prompt)))) for row in items.itertuples()]
        items['verifier'] = [Judge(judge=labels['judge'], judged_by='llm',
            spec=json.dumps(self.grading['verifiers']['native_judge'], sort_keys=True))] * len(items)

        # 4. Describe only the released BAGEL configuration and preserve null grades.
        subjects = pd.DataFrame([dict(subject_key=labels['subject_key'], raw_label=labels['model'],
            features=parameters['subject_features'])])
        responses['subject_key'] = labels['subject_key']
        responses['response_key'] = responses.item_key
        responses['test_condition'] = labels['condition']

        # 5. Link full judge text and unchanged output images to every attempt.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=paths['results'],
            source_key=row.item_key, source_task=dict(id=row.id, prompt=row.prompt, edit_type=row.edit_type),
            judge_text=row.judge_text, native_score=None if pd.isna(row.response) else row.response,
            grade_status=labels['omitted'] if pd.isna(row.response) else labels['saved'],
            output=dict(member=row.output_member, sha256=row.output_sha256, bytes=row.output_bytes)), allow_nan=False)
            for row in responses.itertuples()]
        tables = {'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'testlet_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces}
        dimensions = {category: rubric_dimensions(prompt) for category, prompt in rubrics.items()}
        protocol = self.grading['verifiers']['native_judge']
        components, outcomes = [], []
        for row in items.itertuples():
            for dimension in dimensions[row.edit_type]:
                rule = dict(instruction=self.grading['rule'], dimension=dimension,
                    judge_prompt=rubrics[row.edit_type].replace('<edit_prompt>', row.prompt))
                components.append(dict(parent_item_key=row.item_key, component_id=dimension,
                    grading_criterion=dict(rule=json.dumps(rule)),
                    verifier=Judge(judge=labels['judge'], judged_by='llm',
                        spec=json.dumps(dict(protocol, selected_dimension=dimension), sort_keys=True))))
        for row in responses.itertuples():
            ratings = read_ratings(row.judge_text, dimensions[row.edit_type], protocol['dimension_aliases'])
            outcomes.extend(dict(parent_response_key=row.response_key, component_id=dimension, response=score)
                for dimension, score in ratings.items())
        return component_tables(tables, pd.DataFrame(components), pd.DataFrame(outcomes))


if __name__ == '__main__':
    ImgEdit(__file__).main_from_args()
