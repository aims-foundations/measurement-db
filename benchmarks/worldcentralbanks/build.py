"""Tabulate released WorldCentralBanks predictions and their original prompt inputs."""

import ast
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class WorldCentralBanks(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, templates, protocol = (parameters[name] for name in ['paths', 'prompts', 'protocol'])

        # 1. Read the original CSV tables, preserving every row and all native fields.
        frames = []
        for path in sorted(self.raw_dir.glob(paths['outputs'])):
            source = str(path.relative_to(self.raw_dir))
            coordinates = re.fullmatch(paths['output_pattern'], source)
            if coordinates is None:
                raise ValueError('An unknown output path requires an explicit source mapping')
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if not {'documents', 'llm_responses', 'actual_labels'}.issubset(frame.columns):
                raise ValueError('An original output table is missing required columns')
            if frame.empty or not frame.documents.str.strip().astype(bool).all():
                raise ValueError('An original prediction has no input sentence')
            frame['native_record'] = frame.to_dict('records')
            frames.append(frame.assign(source_file=source, source_row=frame.index, **coordinates.groupdict()))
        responses = pd.concat(frames, ignore_index=True)
        responses['model'] = responses.source_model.str.replace(r'^(stance|time|certain)_', '', regex=True)
        responses['response_key'] = responses.index
        responses['item_key'] = responses.response_key
        responses['subject_key'] = responses.provider + '/' + responses.model

        # 2. Join each task/regime/split to its exact original guide or few-shot examples.
        context_keys = ['regime', 'bank', 'task', 'seed']
        contexts = responses[context_keys].drop_duplicates().copy()
        contexts['bank_name'] = contexts.bank.map(parameters['banks'])
        contexts['labels'] = contexts.task.map(parameters['labels'])
        contexts['label_format'] = contexts.task.map(parameters['label_formats'])
        if contexts[['bank_name', 'labels', 'label_format']].isna().any().any():
            raise ValueError('An original bank or task has no documented prompt definition')
        if not contexts.regime.isin(['no_guide', 'with_guide', 'few_shot']).all():
            raise ValueError('An unknown prompt regime requires review')
        contexts['guide_text'] = [
            (self.raw_dir / paths['guides'].format(**row)).read_text(encoding='utf-8')
            if row['regime'] == 'with_guide' else '' for row in contexts.to_dict('records')]
        contexts['examples'] = [
            (self.raw_dir / paths['examples'].format(**row)).read_text(encoding='utf-8').strip()
            if row['regime'] == 'few_shot' else '' for row in contexts.to_dict('records')]
        responses = responses.merge(contexts, on=context_keys, how='left', validate='many_to_one')

        # 3. Expand source-derived templates without changing whitespace or message roles.
        prompts = responses[['bank_name', 'labels', 'label_format', 'guide_text', 'examples', 'task']].copy()
        prompts = prompts.rename(columns={'task': 'feature'})
        prompts['sentence'] = responses.documents
        prompts['bank_phrase'] = prompts.bank_name + "'s monetary‑policy meeting"
        philippines = prompts.bank_name.eq(protocol['philippines'])
        prompts.loc[philippines, 'bank_phrase'] = 'the ' + prompts.loc[philippines, 'bank_name'] + "' monetary‑policy meeting"
        choices = responses.task + philippines.map({True: '_philippines', False: ''})
        with_guide = responses.regime.eq('with_guide')
        few_shot = responses.regime.eq('few_shot')
        finma = responses.model.eq(protocol['finma_model'])
        choices.loc[with_guide] = responses.loc[with_guide, 'task'] + '_prompt_with_guide_user'
        choices.loc[few_shot] = 'few_shot_user'
        choices.loc[finma] = 'finma'
        prompts['bank'] = prompts.bank_name
        prompts['apostrophe'] = philippines.map({True: "'", False: "'s"})
        prompts['task'] = responses.task.map(parameters['finma_tasks'])
        prompts['sent'] = responses.documents
        prompts['choices'] = responses.task.map(parameters['finma_choices'])
        if not responses.loc[finma, 'regime'].eq('no_guide').all():
            raise ValueError('A new FinMA prompting regime requires review')
        inputs = [templates[key].format(**row) for key, row in zip(choices, prompts.to_dict('records'))]
        system = pd.Series('', index=responses.index)
        system_keys = philippines.map({True: 'guide_system_philippines', False: 'guide_system'})
        system_keys.loc[few_shot] = 'few_shot_system'
        adaptive = with_guide | few_shot
        system.loc[adaptive] = [templates[key].format(**row) for key, row in zip(
            system_keys.loc[adaptive], prompts.loc[adaptive].to_dict('records'))]
        request_inputs = [
            dict(kind='plain_text', prompt=header + '\n' + text)
            if header and provider == protocol['gemini_provider'] else
            dict(kind='chat_messages', messages=[dict(role='system', content=header), dict(role='user', content=text)])
            if header else dict(kind='plain_text', prompt=text)
            for header, text, provider in zip(system, inputs, responses.provider)]

        # 4. Derive the declared label readout; retain failed requests as ungraded attempts.
        outputs = responses.llm_responses.str.strip()
        cleaned = outputs.str.replace(r'^```(?:json)?\s*|```', '', flags=re.I, regex=True).str.strip()
        cleaned = cleaned.str.replace(r'^[^{]*(?=\{)', '', regex=True)
        cleaned = cleaned.str.replace(r'""([^"]+)""', r'"\1"', regex=True)
        cleaned = cleaned.str.replace('“', '"', regex=False).str.replace('”', '"', regex=False)
        cleaned = cleaned.str.extract(r'(\{[\s\S]*?\})', expand=False).fillna(cleaned)
        cleaned = cleaned.str.replace('""', '"', regex=False).str.rstrip(',')
        labels = pd.Series('error', index=responses.index)
        for index, text in cleaned.loc[~finma].items():
            try:
                value = json.loads(text)
                label = value.get('label', '').strip().lower()
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(text)
                    label = str(value.get('label', '')).strip().lower() if isinstance(value, dict) else ''
                except (ValueError, SyntaxError, TypeError):
                    match = re.search(r'"?label"?\s*:\s*"?([a-zA-Z]+)"?', text, re.I)
                    label = match.group(1).lower() if match else 'error'
            labels.at[index] = label or 'error'
        for index, text in outputs.loc[finma].items():
            try:
                value = json.loads(text) if text.startswith('{') else text
                label = (value.get('label') if isinstance(value, dict) else str(value)).strip().lower()
            except (ValueError, TypeError, AttributeError):
                label = 'error'
            labels.at[index] = label
        gold = responses.actual_labels.str.strip().str.lower()
        allowed = responses.assign(gold=gold).groupby('source_file').gold.agg(set)
        valid_finma = [label in allowed[source] for source, label in zip(
            responses.loc[finma, 'source_file'], labels.loc[finma])]
        labels.loc[finma] = labels.loc[finma].where(valid_finma, 'error')
        responses['response'] = labels.eq(gold).astype(float)
        unavailable = outputs.isin(['', 'error']) | gold.eq('')
        responses.loc[unavailable, 'response'] = None

        # 5. Return linked tables and complete native observations for independent review.
        subjects = responses[['subject_key', 'model', 'provider']].drop_duplicates().rename(columns={'model': 'raw_label'})
        subjects['features'] = [dict(source_model_label=row.raw_label, source_provider=row.provider,
            harness=protocol['harness'], executed_inference_settings='not fully established by the released configs')
            for row in subjects.itertuples()]
        items = responses[['item_key', 'source_file', 'source_row', 'bank', 'task', 'actual_labels']].copy()
        items['raw_item_id'] = items.source_file + '/row/' + items.source_row.astype(str)
        items['content'] = [json.dumps(value, ensure_ascii=True) for value in request_inputs]
        items['features'] = [dict(bank=row.bank, task=row.task) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=label, rule=self.grading['rule']) for label in items.actual_labels]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))
        responses['test_condition'] = 'task=' + responses.task + ';regime=' + responses.regime + ';split_seed=' + responses.seed
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            original_record=row.native_record, derived_label=label), ensure_ascii=False)
            for row, label in zip(responses.itertuples(), labels)]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    WorldCentralBanks(__file__).main_from_args()
