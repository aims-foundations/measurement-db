"""Tabulate original Reasoning Gym completions without replacing them by averages."""

import json
from dataclasses import asdict
import unicodedata
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.benchmarks.reasoning_gym.component_checks import SOURCE_FILES, reference_checks, comparison_outcomes, root_configuration, component_scale
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables
from measurement_db.scripts.build_measurement_tables import subject_id_for
from measurement_db.scripts.build_measurement_tables.hash_measurement_ids import item_id_from_content
from measurement_db.scripts.build_measurement_tables.normalize_evaluation_settings import canonicalize_features, features_string


class ReasoningGym(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        root = self.raw_dir / parameters['layout']['results']

        # 1. Read run configurations; keep the documented identity conflict raw-only.
        run_paths = sorted(root.glob('*/*/summary.json'))
        runs = pd.json_normalize([json.loads(path.read_text()) for path in run_paths], max_level=0)
        runs['subject_key'] = [str(path.parent.relative_to(root)) for path in run_paths]
        excluded = runs.subject_key.eq(parameters['exclusions']['ambiguous_identity_run'])
        if excluded.sum() != 1:
            raise ValueError('The declared unresolved run no longer matches the original release')
        runs = runs.loc[~excluded].copy()
        names = runs.subject_key.str.split('/').str[-1].str.replace(r'_\d{8}_\d{6}$', '', regex=True)
        if not names.eq(runs.model.str.replace('/', '_', regex=False)).all():
            raise ValueError('Run folder and recorded model identity disagree')
        runs['condition'] = [json.dumps(row, sort_keys=True) for row in
            runs[['subject_key', *parameters['condition_fields']]].to_dict('records')]
        settings = runs.set_index('subject_key').to_dict('index')

        # 2. Flatten question records and join their original run-level settings.
        frames = []
        for run, summary in settings.items():
            for path in sorted((root / run).glob('*/*.json')):
                data = json.loads(path.read_text())
                if (data['name'] != path.stem or data['category'] != path.parent.name
                        or data['system_prompt'] != summary['system_prompt']
                        or data['completions_per_prompt'] != summary['completions_per_prompt']):
                    raise ValueError('Dataset path or generation settings disagree with the native record')
                frame = pd.json_normalize(data['results'], max_level=0)
                if len(frame) != data['total_examples'] or not frame.completions.map(
                        lambda value: isinstance(value, list) and bool(value)).all():
                    raise ValueError('Every question needs its recorded completions and matching count')
                frames.append(frame.assign(subject_key=run, source_file=str(path.relative_to(self.raw_dir)),
                    source_row=frame.index, dataset=data['name'], category=data['category'],
                    difficulty=run.split('/')[0], generator_config=json.dumps(data['config'], sort_keys=True)))
        records = pd.concat(frames, ignore_index=True)
        records = records.merge(runs[['subject_key', 'system_prompt', 'git_hash', 'condition']],
                                on='subject_key', how='left', validate='many_to_one')
        if not records.question.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Every item must retain its complete original question')
        records['item_key'] = records.source_file + ':' + records.source_row.astype(str)

        # 3. Define subjects and items with the complete stimulus and grading identity.
        subjects = runs[['subject_key', 'model']].rename(columns={'model': 'raw_label'})
        subjects['features'] = runs[list(parameters['subject_fields'])].astype(str).assign(
            source_model=runs.model, harness=parameters['labels']['harness']).to_dict('records')
        stimulus = records[['system_prompt', 'question']].to_dict('records')
        criteria = records[['expected_answer', 'dataset', 'generator_config', 'git_hash']].to_dict('records')
        items = records[['item_key']].assign(raw_item_id=records.item_key,
            content=[json.dumps(row, ensure_ascii=False, sort_keys=True) for row in stimulus],
            grading_criterion=[dict(reference_answer=str(row['expected_answer']), rule=self.grading['rule'],
                response_scale=self.grading['verifiers']['components']['aggregate_scale']) for row in criteria],
            verifier=[ExactMatcher(spec=json.dumps({**self.grading['verifiers']['algorithmic'],
                'dataset': row['dataset'], 'configuration': json.loads(row['generator_config']),
                'revision': row['git_hash']}, sort_keys=True)) for row in criteria],
            features=records[['dataset', 'category', 'difficulty']].to_dict('records'))

        items["testlet_id"] = items.content.map(lambda text: hashlib.sha256(unicodedata.normalize("NFC", text).encode()).hexdigest())
        # 4. Expand individual completions, preserving fractional scores and exceptions.
        expanded = records.explode('completions', ignore_index=True)
        completions = pd.json_normalize(expanded.completions, max_level=0).reindex(
            columns=['model_answer', 'full_model_response', 'score', 'error'])
        numeric = pd.to_numeric(completions.score, errors='raise')
        if not np.isfinite(numeric).all() or numeric.lt(0).any() or numeric.gt(np.nextafter(1., np.inf)).any():
            raise ValueError('A native reward is non-finite or exceeds its declared scale beyond boundary roundoff')
        scores = numeric.clip(upper=1).astype(object)
        failed_grading = completions.error.notna()
        scores.loc[failed_grading] = None
        expanded['completion_index'] = expanded.groupby('item_key', sort=False).cumcount()
        expanded['response_key'] = expanded.item_key + ':' + expanded.completion_index.astype(str)
        responses = expanded[['response_key', 'subject_key', 'item_key']].assign(response=scores,
            test_condition=expanded.condition)

        # 5. Preserve every original completion and its source position, without clipping text.
        evidence = expanded[['source_file', 'source_row', 'completion_index', 'completions',
                             'mean_score', 'best_score']].rename(columns={'completions': 'original_completion'})
        traces = expanded[['response_key']].assign(
            trace=[json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                   for row in evidence.to_dict('records')])
        # 6. Number original attempts after canonical stimulus/configuration merges.
        identities = {row.item_key: item_id_from_content(self.slug, row.content,
            features_string(canonicalize_features(row.features)),
            verifier=json.dumps(dict(**{'class': 'exact_matcher'}, **asdict(row.verifier))),
            grading_criterion=row.grading_criterion, response_scale={'kind': 'mixed'}) for row in items.itertuples()}
        models = {row.subject_key: subject_id_for(row.raw_label, features=row.features) for row in subjects.itertuples()}
        responses['trial'] = responses.groupby([responses.subject_key.map(models), responses.item_key.map(identities),
            responses.test_condition], sort=False, dropna=False).cumcount() + 1

        # 7. Expose comparisons explicitly performed by the captured native scorers.
        protocol = self.grading['verifiers']['components']
        definitions, checklists, source_hashes, root_configs = {}, {}, {}, {}
        for row in records.loc[records.dataset.isin(SOURCE_FILES)].itertuples():
            checks = reference_checks(row.dataset, row.expected_answer)
            checklists[row.item_key] = checks
            source_path = 'protocol/' + row.git_hash + '/reasoning_gym/' + SOURCE_FILES[row.dataset]
            if source_path not in source_hashes:
                source_hashes[source_path] = hashlib.sha256((self.raw_dir / source_path).read_bytes()).hexdigest()
            if row.dataset == 'polynomial_equations':
                root_configs[row.item_key] = dict(root_configuration((self.raw_dir / source_path).read_text()), **json.loads(row.generator_config))
            for component, reference in checks.items():
                definitions[row.item_key, component] = dict(parent_item_key=row.item_key, component_id=component,
                    grading_criterion=dict(reference_answer=json.dumps(reference, ensure_ascii=False, sort_keys=True),
                        rule=protocol['root_rule'] if row.dataset == 'polynomial_equations' else protocol['rule'],
                        response_scale=component_scale(protocol, row.dataset, component)),
                    verifier=ExactMatcher(spec=json.dumps(dict(kind='reconstructed_native_comparison',
                        dataset=row.dataset, configuration=json.loads(row.generator_config), revision=row.git_hash,
                        source_file=source_path, source_sha256=source_hashes[source_path], component_id=component), sort_keys=True)))
        outcomes = []
        for row in expanded.itertuples():
            checks = checklists.get(row.item_key, {})
            failed = row.completions.get('error') is not None
            flags = (dict.fromkeys(checks, None) if failed else comparison_outcomes(row.dataset,
                row.completions['model_answer'], row.expected_answer, root_configs.get(row.item_key))) if checks else None
            for component, grade in (flags or {}).items():
                outcomes.append(dict(parent_response_key=row.response_key, component_id=component, response=grade))
        return component_tables(dict(subjects=subjects, items=items, responses=responses, traces=traces),
            pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))


if __name__ == '__main__':
    ReasoningGym(__file__).main_from_args()
