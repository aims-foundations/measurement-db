"""Tabulate the original MMLU leaderboard observations and complete native traces."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MMLU(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters

        # 1. Read each task table with its exact original run configuration.
        frames, configurations, trace_frames = [], [], []
        for run_path in sorted(self.raw_dir.glob(parameters['layout']['runs'])):
            settings = json.loads(run_path.read_text())
            config = settings.get('config_general', settings.get('config'))
            if not isinstance(config, dict) or not config.get('model_name'):
                raise ValueError('A recorded evaluation lacks its model configuration')
            configurations.append(dict(subject_key=run_path.parent.name,
                raw_label=config['model_name'], configuration=config))
            for path in sorted(run_path.parent.glob(parameters['layout']['observations'])):
                native = pd.read_parquet(path)
                source_file = str(path.relative_to(self.raw_dir))
                response_keys = source_file + '#' + native.index.astype(str)
                # Serialize one file at a time so token arrays need not all remain in memory.
                trace_frames.append(pd.DataFrame(dict(response_key=response_keys, trace=[
                    json.dumps(dict(source_file=source_file, source_row=index,
                        run_configuration=config, source_record=record), ensure_ascii=False,
                        allow_nan=False, default=lambda value: value.tolist())
                    for index, record in enumerate(native.to_dict('records'))])))
                frames.append(native[['full_prompt', 'choices', 'gold', 'acc']].assign(
                    response_key=response_keys, source_row=range(len(native)), task=path.stem,
                    subject_key=run_path.parent.name))
        attempts = pd.concat(frames, ignore_index=True)
        if not attempts.acc.isin([0.0, 1.0]).all():
            raise ValueError('A native accuracy grade is missing or not binary')

        # 2. Join identical full prompts and grading criteria, retaining every response occurrence.
        attempts['choices_json'] = attempts.choices.map(lambda value: json.dumps(list(value)))
        identity = ['task', 'full_prompt', 'choices_json', 'gold']
        items = attempts.drop_duplicates(identity).reset_index(drop=True).copy()
        items['item_key'] = items.index
        items['raw_item_id'] = items.task + ':' + items.source_row.astype(str)
        attempts = attempts.merge(items[identity + ['item_key']], on=identity, validate='many_to_one')
        if not items.gold.map(lambda value: int(value) == value and 0 <= value < 4).all():
            raise ValueError('An original reference index is invalid')
        if not items.choices_json.eq(json.dumps(['A', 'B', 'C', 'D'])).all():
            raise ValueError('An original MMLU item does not have the four answer-letter continuations')

        # 3. Describe the rendered prompt, separately scored continuations and native criterion.
        items['content'] = [json.dumps(dict(prompt=row.full_prompt,
            candidate_continuations=[' ' + value for value in json.loads(row.choices_json)],
            request_protocol=parameters['labels']['request_protocol']), ensure_ascii=False)
            for row in items.itertuples()]
        items['features'] = items.task.map(lambda task: dict(task=task))
        items['grading_criterion'] = [dict(reference_answer=json.loads(row.choices_json)[int(row.gold)],
            rule=self.grading['rule']) for row in items.itertuples()]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))

        # 4. Preserve literal model snapshots and settings rather than guessing missing configuration.
        subjects = pd.DataFrame(configurations)
        subjects['features'] = [dict(source_configuration={key: value for key, value in config.items()
            if key not in parameters['non_identity_configuration_fields']},
            configuration_scope=parameters['labels']['configuration_scope'])
            for config in subjects.configuration]

        # 5. Return unchanged grades and full traces; the shared writer numbers repeated occurrences.
        attempts['response'] = attempts.acc.astype(float)
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': pd.concat(trace_frames, ignore_index=True)}


if __name__ == '__main__':
    MMLU(__file__).main_from_args()
