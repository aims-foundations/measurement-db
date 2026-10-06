"""Join WONDERBREAD's released grades to original workflow inputs and outputs."""

import ast
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict
from urllib.parse import quote
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class Wonderbread(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        p = self.build_parameters
        layout, protocols = p['layout'], self.grading['verifiers']

        # 1. Read unchanged demonstrations, SOPs and screenshot assets. Load only
        # the original pure action-to-text formatter, never the API harness.
        path = self.raw_dir / layout['helpers']
        source = ast.parse(path.read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == p['labels']['action_formatter']]
        if len(functions) != 1:
            raise ValueError('Missing original action formatter')
        native = {'Dict': Dict, 'Any': Any}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), native)
        action_text = native[p['labels']['action_formatter']]
        assets, demos = {}, []

        def read_demo(name, payload, sop, image_reader):
            original = json.loads(payload)
            steps = []
            for event in original['trace']:
                if event['type'] == 'state':
                    filename = Path(event['data']['path_to_screenshot']).name
                    key = name + '/screenshots/' + filename
                    if key not in assets:
                        data = image_reader(filename)
                        location = 'screenshots/' + hashlib.sha256(data).hexdigest() + '.png'
                        assets[key] = dict(data=data, path=location, media_type='image/png', role='input')
                    steps.append(dict(type='image', location=assets[key]['path']))
                elif event['type'] == 'action':
                    steps.append(dict(type='text', text='Action: ' + action_text(event)['action']))
                else:
                    raise ValueError('Unexpected original workflow event')
            task = original['webarena']
            return dict(demo_name=name, task_id=int(task['task_id']), intent=task['intent'],
                        sop=sop, steps=steps, images=[s for s in steps if s['type'] == 'image'])

        with ZipFile(self.raw_dir / layout['gold_archive']) as archive:
            names = [n for n in archive.namelist() if n.startswith(p['labels']['gold_prefix']) and n.endswith('.json')]
            for name in sorted(names):
                folder = str(Path(name).parent)
                sop_names = [n for n in archive.namelist() if str(Path(n).parent) == folder and Path(n).name.startswith('SOP') and n.endswith('.txt')]
                if len(sop_names) != 1:
                    raise ValueError('A gold demonstration has no unique original SOP')
                sop = archive.read(sop_names[0]).decode().replace('\r\n', '\n').replace('\r', '\n')
                demos.append(read_demo(Path(folder).name, archive.read(name), sop,
                                       lambda filename: archive.read(folder + '/screenshots/' + filename)))
        for path in sorted((self.raw_dir / layout['additional_demos']).glob('*/*.json')):
            name = re.sub(r'_x([0-9a-f]{2,6})_', lambda m: chr(int(m[1], 16)), path.parent.name)
            sop_paths = list(path.parent.glob('SOP*.txt'))
            if len(sop_paths) != 1:
                raise ValueError('An additional demonstration has no unique original SOP')
            demos.append(read_demo(name, path.read_bytes(), sop_paths[0].read_text(),
                                   lambda filename: (path.parent / 'screenshots' / re.sub(
                                       r'[^A-Za-z0-9._/-]', lambda m: f'_x{ord(m[0]):02x}_', filename)).read_bytes()))
        demos = pd.DataFrame(demos)
        if demos.demo_name.duplicated().any():
            raise ValueError('Ambiguous original demonstration names')
        demo = demos.set_index('demo_name').to_dict('index')
        attachments = {value['path']: value for value in assets.values()}

        # 2. Read native result tables; keep source positions before any filtering.
        tables = {}
        for family, filename in p['results'].items():
            frame = pd.read_csv(self.raw_dir / filename, keep_default_na=False, float_precision='round_trip')
            frame['source_record'] = frame.to_dict('records')
            frame = frame.assign(source_file=filename, source_row=frame.index, family=family)
            frame['source_rows'] = [[index] for index in frame.index]
            frame['origin'] = family + '/' + frame.source_row.astype(str)
            frame['model'] = frame['ablation--model']
            tables[family] = frame
        observations = []

        # 3. Construct the original task context for each graded response. Each
        # family has a different observation unit; metric columns are melted later.
        qa = tables['question_answering'].loc[lambda t: t.model.ne('Human')].copy()
        qa['names'] = qa['Task ID(s)'].str.split(',').map(lambda names: [name.strip() for name in names])
        contexts = []
        for row in qa.to_dict('records'):
            if row['Evidence'].startswith('SOP'):
                evidence = [demo[name]['sop'][demo[name]['sop'].index('\n'):] for name in row['names']]
            else:
                names = row['names']
                if len(names) == 3:
                    logs = pd.DataFrame({'task_id': [demo[name]['task_id'] for name in names],
                        'steps': [demo[name]['steps'] + ([dict(type='text', text=p['labels']['transition'])]
                                  if name != names[-1] else []) for name in names]})
                    logs = logs.groupby('task_id', sort=True).steps.agg(lambda values: sum(values, []))
                    order = list(logs.index)
                    random.Random(1).shuffle(order)
                    evidence = [step for task in order for step in logs.loc[task]]
                else:
                    evidence = [demo[name]['steps'] for name in names]
            contexts.append(dict(text=p['instructions']['question_answering'], question=row['Question Instantiation'], evidence=evidence))
        qa['context'] = contexts
        qa['reference'] = qa['Human Label']
        qa['configuration'] = qa.Evidence
        observations.append(qa)

        validation = tables['demo_validation'].copy()
        if validation['ablation--is_act'].any() or not validation['ablation--is_kf'].all():
            raise ValueError('This release uses screenshot-only validation inputs')
        validation['context'] = [dict(text=p['instructions'][row['ablation--version']],
            intent=demo[row['demo_name']]['intent'] if row['ablation--is_td'] else None,
            sop=demo[row['demo_name']]['sop'] if row['ablation--is_include_sop'] else None,
            screenshots=[dict(type='image', location=assets[row['demo_name'] + '/screenshots/' + Path(name).name]['path'])
                         for name in ast.literal_eval(row['paths_to_screenshots'])]) for row in validation.to_dict('records')]
        validation['reference'] = validation.gt_is_met.map(lambda value: json.dumps(bool(value)))
        validation['is_correct'] = validation.is_correct.map({True: 1.0, False: 0.0})
        validation['configuration'] = validation.ablation
        observations.append(validation)

        ranking = tables['sop_ranking']
        keys = ['task_id', 'model', 'ablation', 'demo_name']
        if ranking.groupby(keys)[['spearman_corr', 'kendall_corr']].nunique(dropna=False).ne(1).any().any():
            raise ValueError('A ranking group has conflicting whole-ranking grades')
        ranking = ranking.groupby(keys, sort=False, as_index=False).agg(
            folder_name=('folder_name', list), sop=('sop', list), gt_ranking=('gt_ranking', list),
            pred_ranking=('pred_ranking', list), spearman_corr=('spearman_corr', 'first'),
            kendall_corr=('kendall_corr', 'first'), source_rows=('source_row', list),
            source_record=('source_record', list), source_file=('source_file', 'first'))
        ranking['family'] = 'sop_ranking'
        ranking['origin'] = 'sop_ranking/' + ranking.source_rows.map(lambda rows: str(rows[0]))
        ranking['context'] = [dict(text=p['instructions']['sop_ranking'], intent=demo[row.folder_name[0]]['intent'],
                                   sops=row.sop) for row in ranking.itertuples()]
        ranking['reference'] = ranking.gt_ranking.map(json.dumps)
        ranking['configuration'] = ranking.ablation
        observations.append(ranking)

        generation = tables['sop_generation'].copy()
        task_definitions = {int(path.stem): json.loads(path.read_text())
            for path in (self.raw_dir / layout['webarena_tasks']).glob('*.json') if path.stem.isdigit()}
        generation['context'] = [dict(text=p['instructions']['sop_generation'],
            intent=task_definitions[int(row['task_id'])]['intent'],
            interface=p['interfaces'][task_definitions[int(row['task_id'])]['sites'][0]],
            workflow=[step for step in demo[row['demo_name']]['steps']
                      if (step['type'] == 'image' and row['ablation--is_kf']) or
                         (step['type'] == 'text' and row['ablation--is_act'])]) for row in generation.to_dict('records')]
        generation['reference'] = generation.gold_sop
        generation['configuration'] = generation.ablation
        observations.append(generation)

        segmentation = tables['demo_segmentation'].loc[lambda t: t.is_correct.ne('')].copy()
        if (not segmentation['ablation--is_td'].all() or segmentation['ablation--is_act'].any()
                or not segmentation['ablation--is_kf'].all() or not segmentation['ablation--is_concatenate'].all()):
            raise ValueError('Unexpected graded segmentation condition')
        contexts, references = {}, {}
        for key, group in segmentation.groupby(['demos', 'trial', 'ablation--is_include_sop'], sort=False):
            names = ast.literal_eval(key[0])
            logs = demos.set_index('demo_name').loc[names].groupby('task_id', sort=True).images.agg(lambda values: sum(values, []))
            order = list(logs.index)
            random.Random(int(key[1])).shuffle(order)
            sequence = [step for task in order for step in logs.loc[task]]
            targets = [int(task) for task in order for _ in logs.loc[task]]
            workflow_definitions = [dict(label=chr(65 + index), intent=demo[name]['intent'],
                                         sop=demo[name]['sop'] if key[2] else None) for index, name in enumerate(names)]
            for row in group.itertuples():
                if int(row.gt_task_id) != targets[int(row.uuid)] or row.item_type != 'state':
                    raise ValueError('A released segmentation UUID does not match the original input sequence')
                contexts[row.origin] = dict(text=p['instructions']['demo_segmentation'],
                    workflows=workflow_definitions, screenshots=sequence, target_uuid=int(row.uuid))
                references[row.origin] = str(int(row.gt_task_id))
        segmentation['context'] = segmentation.origin.map(contexts)
        segmentation['reference'] = segmentation.origin.map(references)
        segmentation['is_correct'] = segmentation.is_correct.map({'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0})
        segmentation['configuration'] = segmentation.ablation
        observations.append(segmentation)

        # 4. Melt original grades, with a distinct grading rule and scale per axis.
        attempts = pd.concat(observations, ignore_index=True)
        attempts['context_json'] = attempts.context.map(lambda value: json.dumps(value, sort_keys=True, allow_nan=False))
        attempts['modality'] = attempts.context_json.str.contains('"type": "image"', regex=False).map({True: 'image', False: 'text'})
        attempts['subject_key'] = attempts.family + '/' + attempts.model + '/' + attempts.configuration
        long = pd.concat([frame.melt(id_vars=['origin'], value_vars=[key.split('/')[1] for key in protocols if key.startswith(family + '/')],
            var_name='grade_column', value_name='response').assign(family=family)
            for family, frame in attempts.groupby('family', sort=False)], ignore_index=True)
        long['protocol'] = long.family + '/' + long.grade_column
        long = long.merge(attempts[['origin', 'subject_key', 'context_json', 'reference']], on='origin', validate='many_to_one')
        long['response'] = pd.to_numeric(long.response)
        long['item_key'] = [hashlib.sha256(json.dumps([r.context_json, r.reference, r.protocol]).encode()).hexdigest()
                            for r in long.itertuples()]
        long['response_key'] = long.origin + '/' + long.grade_column
        items = long[['item_key', 'origin', 'family', 'context_json', 'reference', 'protocol']].drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.origin + '/' + items.protocol.str.split('/').str[-1]
        items['content'] = items.context_json
        items['features'] = items.family.map(lambda family: dict(task_family=family))
        items['grading_criterion'] = [dict(rule=protocols[row.protocol]['rule'], reference_answer=row.reference,
            response_scale=protocols[row.protocol]['response_scale']) for row in items.itertuples()]
        items['verifier'] = [Judge(spec=json.dumps(protocols[key]['implementation'], sort_keys=True), judged_by='llm')
            if protocols[key]['kind'] == 'judge' else ExactMatcher(spec=json.dumps(protocols[key]['implementation'], sort_keys=True))
            for key in items.protocol]
        items['attachments'] = items.content.map(lambda text: [attachments[path] for path in
            dict.fromkeys(re.findall(r'"location": "(screenshots/[0-9a-f]{64}[.]png)"', text))])

        # 5. Preserve complete native output records and explicit model settings.
        subjects = attempts[['subject_key', 'model', 'family', 'configuration', 'modality']].drop_duplicates('subject_key').copy()
        subjects['raw_label'] = 'WONDERBREAD ' + subjects.model
        subjects['features'] = [dict(native_model=row.model, task_family=row.family,
            configuration=quote(row.configuration, safe=' _-,+'), declared_backend=p['models'][row.model + '/' + row.modality],
            harness='WONDERBREAD', harness_version=p['labels']['harness_revision']) for row in subjects.itertuples()]
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, source_rows=row.source_rows, source_records=row.source_record),
            sort_keys=True, allow_nan=False) for row in attempts.itertuples()]
        traces = long[['response_key', 'origin']].merge(attempts[['origin', 'trace']], on='origin', validate='many_to_one')
        responses = long[['response_key', 'subject_key', 'item_key', 'response']].assign(test_condition=p['labels']['condition'])
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            responses=responses, traces=traces[['response_key', 'trace']])


if __name__ == '__main__':
    Wonderbread(__file__).main_from_args()
