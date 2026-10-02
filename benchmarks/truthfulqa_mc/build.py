"""Tabulate the original TruthfulQA-MC1 leaderboard observations."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TruthfulQAMC(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters

        # 1. Read each original result table and its accompanying run configuration.
        frames, configurations = [], []
        for path in sorted(self.raw_dir.glob(parameters['layout']['runs'])):
            settings = json.loads(path.read_text())
            config = settings.get('config_general', settings.get('config'))
            if not isinstance(config, dict) or not config.get('model_name'):
                raise ValueError('A recorded evaluation lacks its model configuration')
            source = path.with_name(parameters['layout']['observations'])
            frame = pd.read_parquet(source)
            frame['source_record'] = frame.to_dict('records')
            frame['source_file'] = str(source.relative_to(self.raw_dir))
            frame['source_row'] = range(len(frame))
            frame['subject_key'] = path.parent.name
            frame['run_configuration'] = [config] * len(frame)
            frames.append(frame)
            configurations.append(dict(subject_key=path.parent.name,
                raw_label=config['model_name'], configuration=config))
        attempts = pd.concat(frames, ignore_index=True)
        if not attempts.mc1.isin([True, False]).all():
            raise ValueError('A native MC1 grade is missing or not binary')

        # 2. Associate complete prompts and candidate continuations across models.
        targets = pd.json_normalize(attempts.mc1_targets)
        attempts['choices_json'] = targets.choices.map(
            lambda value: json.dumps(list(value), ensure_ascii=False))
        attempts['labels_json'] = targets.labels.map(lambda value: json.dumps(list(map(int, value))))
        identity = ['full_prompt', 'choices_json', 'labels_json']
        items = attempts.drop_duplicates(identity).reset_index(drop=True).copy()
        items['item_key'] = items.index
        items['raw_item_id'] = items.source_row.astype(str)
        if items.raw_item_id.duplicated().any() or not attempts.groupby('source_row').question.nunique().eq(1).all():
            raise ValueError('Original row positions do not consistently identify the same questions')
        attempts = attempts.merge(items[identity + ['item_key']], on=identity, validate='many_to_one')
        items['content'] = [json.dumps(dict(prompt=row.full_prompt,
            candidate_continuations=[' ' + choice for choice in json.loads(row.choices_json)],
            request_protocol=parameters['labels']['request_protocol']), ensure_ascii=False)
            for row in items.itertuples()]
        references = []
        for row in items.itertuples():
            labels = json.loads(row.labels_json)
            choices = json.loads(row.choices_json)
            if labels != [1] + [0] * (len(choices) - 1):
                raise ValueError('The native MC1 reference must be the first and only true option')
            references.append(choices[0])
        items['grading_criterion'] = [dict(reference_answer=value, rule=self.grading['rule'])
            for value in references]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))

        # 3. Retain recorded model snapshots and evaluation settings as subject attributes.
        subjects = pd.DataFrame(configurations)
        subjects['features'] = [dict(source_configuration={key: value for key, value in config.items()
            if key not in parameters['non_identity_configuration_fields']},
            configuration_scope=parameters['labels']['configuration_scope'])
            for config in subjects.configuration]

        # 4. Preserve original grades and full native records, including token likelihoods.
        attempts['response_key'] = attempts.source_file + '#' + attempts.source_row.astype(str)
        attempts['response'] = attempts.mc1.astype(float)
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            run_configuration=row.run_configuration, source_record=row.source_record), ensure_ascii=False,
            allow_nan=False, default=lambda value: value.tolist()) for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': traces}


if __name__ == '__main__':
    TruthfulQAMC(__file__).main_from_args()
