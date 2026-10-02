"""Tabulate original VisualWebArena trajectories, images and separate grading protocols."""

import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import urlsplit
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class VisualWebArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']
        grading = self.grading['verifiers']

        # 1. Load native trajectories, original annotations and the recorded package's task bank.
        paths = sorted(self.raw_dir.glob(layout['trajectories']))
        attempts = pd.DataFrame(dict(source_file=[str(path.relative_to(self.raw_dir)) for path in paths],
            task_id=[path.stem for path in paths], record=[json.loads(path.read_text()) for path in paths]))
        attempts = attempts.join(pd.DataFrame.from_records(attempts.record))
        annotations = pd.read_csv(self.raw_dir / layout['annotations'], keep_default_na=False)
        annotations['annotation_row'] = annotations.index
        annotations = annotations.loc[annotations.benchmark.eq(layout['benchmark_filter'])].copy()
        annotations['annotation'] = annotations.drop(columns='annotation_row').to_dict('records')
        with ZipFile(self.raw_dir / layout['task_archive']) as archive:
            tasks = pd.DataFrame(dict(task_configuration=json.loads(archive.read(layout['task_member']))))
        tasks = tasks.join(pd.DataFrame.from_records(tasks.task_configuration)).rename(columns={'task_id': 'task_number'})

        # 2. Retain complete recorded agent configurations and require exact task/image correspondence.
        attempts['configuration'] = attempts[['agent', 'model', 'model_args', 'flags', 'package_version']].to_dict('records')
        attempts['subject_key'] = attempts.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = attempts[['subject_key', 'model', 'configuration']].drop_duplicates('subject_key').copy()
        subjects['features'] = [dict(harness=self.build_parameters['labels']['harness'],
            source_configuration=json.dumps(value, sort_keys=True).replace(';', r'\u003b').replace('=', r'\u003d'))
            for value in subjects.configuration]
        subjects = subjects.rename(columns={'model': 'raw_label'})
        attempts['task_number'] = attempts.task_id.str.rsplit('.', n=1).str[-1].astype(int)
        attempts = attempts.merge(tasks[['task_number', 'intent', 'image']], on='task_number', how='left', validate='many_to_one')
        recorded_text = attempts.goal.str.split(layout['recorded_goal_pattern'], n=1, regex=True).str[0].str.strip()
        if attempts.intent.isna().any() or not recorded_text.eq(attempts.intent.str.strip()).all():
            raise ValueError('Every recorded instruction must match its task in the historical package')
        tasks['input_images'] = tasks.image.map(lambda value: value if isinstance(value, list) else [value] if isinstance(value, str) else [])
        tasks['image_paths'] = tasks.input_images.map(lambda values: [value.removeprefix(layout['input_url_prefix']) for value in values])
        recorded_images = attempts.goal.str.findall(layout['recorded_image_pattern']).map(
            lambda values: [urlsplit(value).path.removeprefix(layout['image_path_prefix']) for value in values])
        expected_images = attempts.task_number.map(tasks.set_index('task_number').image_paths)
        if any(left != right for left, right in zip(recorded_images, expected_images)):
            raise ValueError('Recorded input images must match the task bank in their original order')

        # 3. Represent automatic and human judgments as distinct item grading protocols.
        protocols = pd.DataFrame([dict(protocol=name, **definition) for name, definition in grading.items()])
        items = tasks.loc[tasks.task_number.isin(attempts.task_number)].merge(protocols, how='cross')
        items['item_key'] = items.task_number.astype(str) + ':' + items.protocol
        items['raw_item_id'] = items.task_number.astype(str)
        items['features'] = items[['task_number', 'protocol']].rename(columns={'protocol': 'grading_channel'}).to_dict('records')
        items['content'] = [json.dumps(dict(multimedia_elements=[dict(content_type='text/plain', text=row.intent)] +
            [dict(content_type=mimetypes.guess_type(name)[0], location='input_images/' + name) for name in row.image_paths]))
            for row in items.itertuples()]
        items['attachments'] = [[dict(source_path=str(Path(layout['input_images']) / name), path='input_images/' + name,
            media_type=mimetypes.guess_type(name)[0], role='input') for name in paths] for paths in items.image_paths]
        items['grading_criterion'] = items.rule.map(lambda rule: dict(rule=rule))
        items['verifier'] = [Judge(judge=grading['human']['judge'], judged_by='human', spec=json.dumps(row.spec, sort_keys=True))
            if row.protocol == 'human' else ExactMatcher(spec=json.dumps(dict(**row.spec, task_configuration=row.task_configuration), sort_keys=True))
            for row in items.itertuples()]

        # 4. Keep both original grades, including disagreements, joined by exact source keys.
        human = annotations.merge(attempts, left_on=['model_name', 'exp_name', 'task_id'],
            right_on=['agent', 'experiment', 'task_id'], how='outer', validate='many_to_one', indicator=True)
        if not human['_merge'].eq('both').all():
            raise ValueError('Every human rating and selected trajectory must have an exact counterpart')
        human['response'] = human.trajectory_success.map(grading['human']['values'])
        human['protocol'] = 'human'
        human['response_key'] = layout['annotations'] + '#' + human.annotation_row.astype(str)
        automatic = attempts.assign(protocol='automatic', annotation=None, annotation_row=None)
        automatic['response'] = automatic.summary_info.map(lambda value: value['cum_reward'])
        automatic['response_key'] = automatic.source_file + '#automatic'
        responses = pd.concat([human, automatic], ignore_index=True)
        if not responses.response.isin([0, 1]).all():
            raise ValueError('Each grading channel requires its original binary verdict')
        responses['response'] = responses.response.astype(float)
        responses['item_key'] = responses.task_number.astype(str) + ':' + responses.protocol

        # 5. Preserve all released trajectory fields; omitted image data remain explicitly omitted.
        responses['trace'] = [json.dumps(dict(source_file=row.source_file, record=row.record, grading_protocol=row.protocol,
            annotation_file=layout['annotations'] if row.protocol == 'human' else None,
            annotation_row=int(row.annotation_row) if row.protocol == 'human' else None, annotation=row.annotation),
            ensure_ascii=True, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response']],
            traces=responses[['response_key', 'trace']])


if __name__ == '__main__':
    VisualWebArena(__file__).main_from_args()
