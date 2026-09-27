"""Tabulate MMOral's original tasks and released OPTG choice records."""

import base64
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class MMOralBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']
        parsing, labels = parameters['parsing'], parameters['labels']
        choices = parameters['choices']
        grading = self.grading['verifiers']

        # 1. Read the complete task bank without treating the option text None as missing.
        banks = []
        for split, filename in parameters['tasks'].items():
            frame = pd.read_csv(self.raw_dir / filename, sep='\t', dtype=str, keep_default_na=False)
            frame = frame.rename(columns={'image_name': 'file_name'})
            frame['source_file'], frame['source_row'] = filename, frame.index
            frame['split'] = split
            frame['item_key'] = split + ':' + frame['index']
            banks.append(frame)
        items = pd.concat(banks, ignore_index=True)
        if items.item_key.duplicated().any():
            raise ValueError('A task split repeats an upstream item index')
        items['raw_item_id'] = items.item_key
        items['image_sha256'] = items.image.map(lambda value: hashlib.sha256(base64.b64decode(value, validate=True)).hexdigest())

        # 2. Verify the analysis study's task mapping, preserving its lossy option export.
        compact = pd.json_normalize(json.loads((self.raw_dir / layout['compact_tasks']).read_text()), max_level=0)
        compact['index'] = compact['index'].astype(str)
        closed = items.loc[items.split.eq('closed')].set_index('index')
        if compact['index'].duplicated().any() or set(compact['index']) != set(closed.index):
            raise ValueError('The compact task export does not cover the original closed questions exactly')
        compact = compact.set_index('index').reindex(closed.index)
        for field in parsing['aligned_fields'].split():
            if not compact[field].eq(closed[field]).all():
                raise ValueError('The compact export changes a task field: ' + field)
        for field in choices:
            restored = compact[field].fillna(parsing['missing_option_literal'])
            if not restored.eq(closed[field]).all():
                raise ValueError('An option discrepancy is not the documented literal-None conversion')
        missing_options = compact[list(choices)].isna().apply(
            lambda row: [choices[column] for column in row.index[row]], axis=1)
        items['compact_missing_options'] = items['index'].map(missing_options).where(items.split.eq('closed'))
        items['compact_missing_options'] = items.compact_missing_options.map(lambda value: value if isinstance(value, list) else [])

        # 3. Concatenate complete and partial choice maps; merge only documented copied exports.
        exports = []
        for pattern in parameters['prediction_globs'].values():
            for path in sorted(self.raw_dir.glob(pattern)):
                values = json.loads(path.read_text())
                if not isinstance(values, dict) or not all(isinstance(value, str) for value in values.values()):
                    raise ValueError('Expected a released item-index to choice map')
                frame = pd.Series(values, name='prediction', dtype=object).rename_axis('upstream_index').reset_index()
                source_file = str(path.relative_to(self.raw_dir))
                frame['source_file'] = source_file
                frame['subject_key'] = parameters['copied_exports'].get(source_file, source_file)
                frame['item_key'] = 'closed:' + frame.upstream_index
                exports.append(frame)
        occurrences = pd.concat(exports, ignore_index=True)
        if not occurrences.item_key.isin(closed.item_key).all():
            raise ValueError('A released choice references an unknown closed question')
        occurrences['response_key'] = occurrences.subject_key + '::' + occurrences.upstream_index
        if occurrences.groupby('response_key').prediction.nunique(dropna=False).gt(1).any():
            raise ValueError('Documented copied exports disagree')
        occurrences['source_coordinate'] = occurrences[['source_file', 'upstream_index']].to_dict('records')
        aliases = occurrences.groupby('response_key', sort=False).source_coordinate.agg(list)
        responses = occurrences.drop_duplicates('response_key').copy()
        responses['source_aliases'] = responses.response_key.map(aliases)
        responses = responses.merge(items[['item_key', 'answer', 'compact_missing_options']],
            on='item_key', how='left', validate='many_to_one')
        normalized = responses.prediction.str.strip().str.upper().str[:1]
        responses['normalized_choice'] = normalized
        responses['response'] = normalized.eq(responses.answer).astype(float).where(normalized.isin(choices.values()))

        # 4. Describe the original benchmark inputs and each item's own grading scale.
        prompts = parameters['prompt']
        items['prompt'] = items.question.map(lambda value: prompts['question'].format(question=value))
        option_rows = items.loc[items.split.eq('closed'), ['item_key'] + list(choices)].melt(
            id_vars='item_key', var_name='field', value_name='value')
        option_rows['text'] = [prompts['choice'].format(letter=choices[row.field], value=row.value)
            for row in option_rows.itertuples()]
        option_text = option_rows.groupby('item_key', sort=False).text.agg(''.join)
        closed_mask = items.split.eq('closed')
        items.loc[closed_mask, 'prompt'] += prompts['options'] + items.loc[closed_mask, 'item_key'].map(option_text)
        # Closed/open banks reuse filenames for different original JPEG bytes.
        items['image_path'] = 'images/' + items.image_sha256 + '.jpg'
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type=labels['image_media_type'], location=row.image_path),
            dict(content_type='text/plain', text=row.prompt)]), ensure_ascii=False)
            for row in items.itertuples()]
        items['attachments'] = [[dict(path=row.image_path, data=base64.b64decode(row.image, validate=True),
            media_type=labels['image_media_type'], role='input')] for row in items.itertuples()]
        items['features'] = [dict(split=row.split, upstream_index=row.index, category=row.category,
            source_file=row.source_file, source_row=int(row.source_row), image_name=row.file_name,
            image_sha256=row.image_sha256, compact_export_missing_options=json.dumps(row.compact_missing_options),
            input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.answer,
            rule=grading[row.split]['rule'], response_scale=grading[row.split]['response_scale'])
            for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(grading['closed']['verifier'], sort_keys=True))
            if split == 'closed' else Judge(spec=json.dumps(grading['open']['verifier'], sort_keys=True), judged_by='llm')
            for split in items.split]

        # 5. Keep policy labels and full recorded choices without inventing model configurations.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = labels['subject_prefix'] + subjects.subject_key.str.removeprefix(layout['policy_prefix'])
        subjects['features'] = [dict(**parameters['subject_features'], source_export=value,
            policy_kind='ensemble' if value.startswith(layout['ensemble_prefix']) else 'voter')
            for value in subjects.subject_key]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(prediction=row.prediction, normalized_choice=row.normalized_choice,
            upstream_index=row.upstream_index, source_aliases=row.source_aliases,
            compact_export_missing_options=row.compact_missing_options), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    MMOralBench(__file__).main_from_args()
