"""Tabulate original InfiBench attempts, grading details and associated completions."""

import json
import re
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class InfiBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']
        reports, questions, profiles, completions = [], [], [], []

        # 1. Load the published reports and their version-specific question banks.
        with tarfile.open(self.raw_dir / layout['legacy']) as archive:
            root = archive.getmembers()[0].name.split('/')[0] + '/'
            suite = yaml.safe_load(archive.extractfile(root + layout['legacy_suite']).read())
            for case in suite['cases']:
                config = yaml.safe_load(archive.extractfile(root + case).read())
                prompt = archive.extractfile(root + str(Path(case).parent / config['prompt_path'])).read().decode('utf-8')
                questions.append(dict(version='legacy_dev', case_file=case, config=config, prompt=prompt, dependencies=None))
            for member in sorted(archive.getmembers(), key=lambda member: member.name):
                name = member.name.removeprefix(root)
                if not member.isfile() or not re.fullmatch(layout['legacy_results'], name):
                    continue
                run = Path(name).stem.removeprefix(Path(layout['legacy_suite']).stem + '_')
                subject = 'legacy_dev:' + run
                frame = pd.DataFrame.from_dict(yaml.load(archive.extractfile(member).read(), Loader=yaml.CSafeLoader), orient='index')
                reports.append(frame.rename_axis('case_file').reset_index().assign(
                    subject_key=subject, version='legacy_dev', source_file=name))
                directory = 'responses/' + run + '/'
                config = yaml.safe_load(archive.extractfile(root + directory + 'params.yaml').read())
                profiles.append(dict(subject_key=subject, raw_label=config['model_name'], version='legacy_dev',
                    source_report=name, subject_kind='model', configuration={k: v for k, v in config.items() if k != 'answer_paths'}))
                for case, paths in config['answer_paths'].items():
                    completions.extend(dict(subject_key=subject, case_file=case, trial=index + 1,
                        completion=archive.extractfile(root + directory + path).read().decode('utf-8'),
                        completion_file=directory + path, completion_row=None) for index, path in enumerate(paths))
        with tarfile.open(self.raw_dir / layout['harness']) as archive:
            root = archive.getmembers()[0].name.split('/')[0] + '/'
            bank = pd.read_csv(archive.extractfile(root + layout['modern_cases']), keep_default_na=False)
            questions.extend(dict(version='v2_1', case_file=row.case_path, config=yaml.safe_load(row.eval_spec),
                prompt=row.prompt, dependencies=json.loads(row.dependencies)) for row in bank.itertuples())
        with ZipFile(self.raw_dir / layout['results']) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith('.yaml'):
                    continue
                path = Path(name)
                subject = 'v2_1:' + path.stem
                frame = pd.DataFrame.from_dict(yaml.load(archive.read(name), Loader=yaml.CSafeLoader), orient='index')
                reports.append(frame.rename_axis('case_file').reset_index().assign(
                    subject_key=subject, version='v2_1', source_file=name))
                label_file = (path.with_name('evaltable_' + path.name[5:]).with_suffix('.txt')
                    if path.name.startswith('eval_') else path.with_name(path.stem + '_table.txt'))
                labels = [line.split('|')[1].strip() for line in archive.read(str(label_file)).decode('utf-8').splitlines()
                    if '|' in line and not line.split('|')[0].strip() and line.split('|')[1].strip()]
                if len(labels) != 1:
                    raise ValueError('An original result table must identify exactly one source configuration')
                alias = re.sub(parameters['parsing']['result_prefix_pattern'], '', path.stem).removesuffix('_parallel')
                profiles.append(dict(subject_key=subject, raw_label=labels[0], version='v2_1', source_report=name,
                    subject_kind='human_answer_collection' if path.name.startswith('human-') else 'model',
                    configuration={}, completion_alias=alias if alias.endswith('_output') else alias + '_output'))
        subjects = pd.DataFrame(profiles)
        cases = pd.DataFrame(questions)
        reports = pd.concat(reports, ignore_index=True)

        # 2. Keep each native attempt and its complete grading detail in order.
        legacy = reports.version.eq('legacy_dev')
        reports.loc[legacy, 'all_scores'] = reports.loc[legacy, 'detail'].map(lambda detail: [None] * len(detail))
        if not reports.apply(lambda row: len(row.detail) == len(row.all_scores), axis=1).all():
            raise ValueError('The native score and grading-detail lists have different lengths')
        responses = reports.loc[reports.detail.map(len).gt(0)].explode(['detail', 'all_scores'], ignore_index=True)
        responses['trial'] = responses.groupby(['subject_key', 'case_file'], sort=False).cumcount() + 1
        detail = pd.json_normalize(responses.detail, max_level=0)
        parts = self.grading['verifiers']['legacy_dev']['score_components']
        numerator = detail.reindex(columns=[part + '_score' for part in parts]).sum(axis=1)
        denominator = detail.reindex(columns=[part + '_totscore' for part in parts]).sum(axis=1)
        maximum = detail.reindex(columns=['max_score']).max_score
        minimum = detail.reindex(columns=['min_score']).min_score
        denominator = denominator.where(maximum.isna(), maximum)
        numerator = numerator.clip(upper=maximum, lower=minimum)
        if denominator.le(0).any():
            raise ValueError('An original grading detail has no positive denominator')
        reconstructed = numerator / denominator * responses.full_score
        responses['response'] = responses.all_scores.where(responses.version.ne('legacy_dev'), reconstructed).astype(float)
        if (responses.response - reconstructed).abs().gt(1e-12).any():
            raise ValueError('An original score differs from its recorded component totals')
        responses = responses.merge(cases, on=['version', 'case_file'], how='left', validate='many_to_one')
        if responses.prompt.isna().any():
            raise ValueError('A released attempt has no source question and grading definition')

        # 3. Attach original outputs only through exact source aliases and row order.
        outputs = []
        aliases = set(subjects.completion_alias.dropna())
        with tarfile.open(self.raw_dir / layout['responses']) as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).stem not in aliases:
                    continue
                frame = pd.read_csv(archive.extractfile(member), keep_default_na=False)
                frame['completion_row'] = range(len(frame))
                frame['trial'] = frame.groupby('filename', sort=False).cumcount() + 1
                outputs.append(frame.rename(columns={'filename': 'case_file'}).assign(
                    completion_alias=Path(member.name).stem, completion_file=member.name))
        modern = pd.concat(outputs, ignore_index=True).merge(
            subjects[['subject_key', 'completion_alias']].dropna(), on='completion_alias', validate='many_to_one')
        output_columns = ['subject_key', 'case_file', 'trial', 'completion', 'completion_file', 'completion_row']
        outputs = pd.concat([pd.DataFrame(completions), modern[output_columns]], ignore_index=True)
        responses = responses.merge(outputs, on=['subject_key', 'case_file', 'trial'], how='left', validate='one_to_one')
        responses['item_key'] = responses.version + ':' + responses.case_file
        responses['test_condition'] = responses.version + ':' + responses.source_file + ':' + responses.case_file
        responses['response_key'] = responses.test_condition + ':' + responses.trial.astype(str)

        # 4. Preserve full task text, original grading rules and recorded subject context.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.item_key
        items['content'] = items.prompt
        items['features'] = [dict(language=row.lang, question_type=row.type, source_version=row.version,
            input_scope=parameters['labels']['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=json.dumps(dict(instruction=self.grading['rule'],
            source_definition=row.config, dependencies=row.dependencies, full_score=row.full_score),
            ensure_ascii=False, sort_keys=True)) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers'][version], sort_keys=True))
            for version in items.version]
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_version=row.version,
            source_report=row.source_report, subject_kind=row.subject_kind, recorded_configuration=row.configuration,
            historical_configuration=parameters['labels']['historical_configuration']) for row in subjects.itertuples()]

        # 5. Keep grading logs and available outputs without clipping or invented links.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_version=row.version, source_file=row.source_file,
            case_file=row.case_file, trial_index=row.trial - 1, native_detail=row.detail,
            native_score=None if row.version == 'legacy_dev' else row.all_scores,
            completion=None if pd.isna(row.completion) else row.completion,
            completion_file=None if pd.isna(row.completion_file) else row.completion_file,
            completion_row=None if pd.isna(row.completion_row) else int(row.completion_row)),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'trial', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    InfiBench(__file__).main_from_args()
