"""Tabulate SciVisAgentBench's published case reports and original task resources."""

import hashlib
import html
import json
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class SciVisAgentBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Read the original HTML case records; retain their exact source text.
        frames = []
        for pattern in parameters['report_patterns'].values():
            for path in sorted(self.raw_dir.glob(pattern)):
                document = path.read_bytes().decode('utf-8')
                page = BeautifulSoup(document, 'html.parser')
                model = page.find('label', string='Model').find_next('value').get_text(strip=True)
                agent = page.select_one('.agent-name').get_text(strip=True)
                generated = page.select_one('.timestamp').get_text(strip=True).removeprefix('Generated: ')
                native = [document[match.start():document.index('</section>', match.end()) + len('</section>')]
                    for match in re.finditer(r'<section class="case-section[^"]*" id="[^"]+"', document)]
                cases = pd.DataFrame({'native_html': native})
                cases['node'] = cases.native_html.map(lambda value: BeautifulSoup(value, 'html.parser'))
                cases['case_id'] = cases.node.map(lambda node: node.section['id'])
                cases['content'] = cases.native_html.str.extract(
                    r'<div class="task-description">(.*?)</div>', flags=re.S, expand=False).map(self._text)
                scores = cases.node.map(lambda node: node.select_one('.case-score').get_text(strip=True)).str.extract(
                    r'^([0-9.]+)/([0-9.]+)\s*\(([0-9.]+)%\)$').astype(float)
                cases[['score', 'maximum', 'percent']] = scores
                cases['rubrics'] = cases.node.map(lambda node: [self._text(str(value))
                    for value in node.select('.rubric-criterion')])
                cases['answer_keys'] = cases.node.map(lambda node: [self._text(str(value.find_next('div')))
                    for value in node.find_all('h4') if value.get_text(strip=True) == 'Questions & Correct Answers'])
                cases['rule_checks'] = cases.native_html.map(lambda text: [self._text(value) for value in
                    re.findall(r'<span style="font-weight: 600; color: #333;">(.*?)</span>', text, re.S)])
                cases['reference_images'] = cases.node.map(lambda node: [image['src'] for image in node.find_all('img')
                    if 'ground truth' in image.get('alt', '').lower()])
                cases['report_directory'] = str(path.parent.relative_to(self.raw_dir))
                cases = cases.assign(source_file=str(path.relative_to(self.raw_dir)),
                    source_suite=path.relative_to(self.raw_dir).parts[2], model=model, agent=agent, generated=generated)
                frames.append(cases.drop(columns='node'))
        observations = pd.concat(frames, ignore_index=True)
        if observations[['case_id', 'content', 'percent']].isna().any().any():
            raise ValueError('A native case lacks its identity, task description or reported percentage')
        observations['suite'] = observations.source_suite.map(parameters['suites'])
        observations['harness'] = observations.agent.map(parameters['harnesses'])
        if observations[['suite', 'harness']].isna().any().any():
            raise ValueError('An original report uses an undeclared suite or harness')

        # 2. Index original data files and recover the released anonymous path aliases.
        resources = []
        for path in sorted((self.raw_dir / layout['tasks']).rglob('*')):
            relative = path.relative_to(self.raw_dir / layout['tasks']).as_posix()
            if not path.is_file() or '/data/' not in relative:
                continue
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            resources.append(dict(source_path=str(path.relative_to(self.raw_dir)), sha256=digest,
                size=path.stat().st_size, data_suite=relative.split('/')[0],
                logical_path=relative.split('/', 1)[1]))
        resources = pd.DataFrame(resources)
        resources['suite'] = resources.data_suite.map(parameters['data_suites'])
        original = pd.json_normalize(yaml.safe_load((self.raw_dir / layout['named_tasks']).read_text()))
        anonymous = pd.json_normalize(yaml.safe_load((self.raw_dir / layout['anonymous_tasks']).read_text()))
        if len(original) != len(anonymous) or original['assert'].tolist() != anonymous['assert'].tolist():
            raise ValueError('The named and anonymous task definitions disagree on their grading rules')
        aliases = pd.DataFrame({
            'logical_path': original['vars.question'].str.extract(r'sci_volume_data/([^"\s]+\.raw)', expand=False),
            'anonymous_path': anonymous['vars.question'].str.extract(r'"([^"\s]+\.raw)"', expand=False)})
        resources = resources.merge(aliases.assign(suite='object_identification'),
            on=['suite', 'logical_path'], how='left', validate='many_to_one')
        resources['logical_path'] = resources.anonymous_path.fillna(resources.logical_path)
        anonymous_resources = resources.loc[resources.suite.eq('object_identification')]
        if set(anonymous_resources.anonymous_path.dropna()) != set(aliases.anonymous_path):
            raise ValueError('An administered anonymous task lacks its original volume')
        resources = resources.loc[~resources.suite.eq('object_identification') | resources.anonymous_path.notna()]

        # 3. Join each administered task to the data paths it names.
        tasks = observations[['suite', 'case_id', 'content']].drop_duplicates().reset_index(drop=True)
        tasks['task_key'] = tasks.index.astype(str)
        matches = tasks.merge(resources, on='suite', how='inner', validate='many_to_many')
        named = [row.logical_path in row.content
            or (row.logical_path.startswith(row.case_id + '/data/') and Path(row.logical_path).name in row.content)
            or any(re.fullmatch(re.escape(value).replace(r'\{timestep\}', r'\d+'), row.logical_path)
                for value in re.findall(r'"([^"\n]+)"', row.content) if '{timestep}' in value)
            for row in matches.itertuples()]
        matches = matches.loc[named].copy()
        matches['resource'] = matches[['logical_path', 'source_path', 'sha256', 'size']].to_dict('records')
        bundles = matches.groupby('task_key').resource.agg(list).rename('resources').reset_index()
        tasks = tasks.merge(bundles, on='task_key', how='left', validate='one_to_one')
        tasks['resources'] = tasks.resources.map(lambda value: value if isinstance(value, list) else [])
        observations = observations.merge(tasks, on=['suite', 'case_id', 'content'], validate='many_to_one')

        # 4. Preserve each report's grading definition, without borrowing another run's rules.
        observations['response_key'] = observations.source_file + '#' + observations.case_id
        observations['item_key'] = observations.response_key
        observations['subject_key'] = observations.harness + '/' + observations.model
        observations['response'] = observations.percent.ge(self.grading['verifiers']['reported_case']['pass_percent']).astype(float)
        items = observations[['item_key', 'case_id', 'suite', 'content', 'resources', 'rubrics',
            'answer_keys', 'rule_checks', 'reference_images', 'report_directory']].copy()
        items['raw_item_id'] = items.suite + '/' + items.case_id
        items['attachments'] = [[dict(source_path=entry['source_path'], path=entry['logical_path'],
            role='input', media_type=parameters['resources']['media_type']) for entry in row.resources
            if entry['size'] <= int(parameters['resources']['maximum_inline_bytes'])] for row in items.itertuples()]
        items['features'] = [dict(suite=row.suite, input_scope=parameters['labels']['input_scope'],
            external_input_files=json.dumps([entry for entry in row.resources
                if entry['size'] > int(parameters['resources']['maximum_inline_bytes'])], sort_keys=True))
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=json.dumps(dict(rule=self.grading['rule'],
            vision_rubrics=row.rubrics, deterministic_checks=row.rule_checks), ensure_ascii=False),
            reference_answer=json.dumps(dict(text_answers=row.answer_keys, images=[
                dict(path=value, sha256=(hashlib.sha256((self.raw_dir / row.report_directory / value).read_bytes()).hexdigest()
                     if (self.raw_dir / row.report_directory / value).is_file() else None))
                for value in row.reference_images]), ensure_ascii=False)) for row in items.itertuples()]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['reported_case'], sort_keys=True))

        # 5. Retain model/harness attribution and complete report evidence for every outcome.
        subjects = observations[['subject_key', 'model', 'harness']].drop_duplicates().rename(columns={'model': 'raw_label'})
        subjects['features'] = [dict(harness=row.harness, model_identifier=row.raw_label,
            historical_settings=parameters['labels']['historical_settings']) for row in subjects.itertuples()]
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, case_id=row.case_id,
            generated=row.generated, native_agent=row.agent, model=row.model, score=row.score,
            maximum=row.maximum, displayed_percent=row.percent, case_html=row.native_html),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier', 'attachments']],
            'responses': observations[['response_key', 'subject_key', 'item_key', 'response']],
            'traces': traces[['response_key', 'trace']]}

    @staticmethod
    def _text(value):
        value = re.sub(r'<br\s*/?>|</p\s*>', '\n', value, flags=re.I)
        return html.unescape(re.sub(r'<[^>]+>', '', value)).replace('\r\n', '\n').replace('\r', '\n').strip()


if __name__ == '__main__':
    SciVisAgentBench(__file__).main_from_args()
