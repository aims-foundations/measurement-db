"""Tabulate TUMLU's complete recorded prompts, outputs and native parser grades."""

import ast
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TUMLU(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        labels = parameters['labels']

        # 1. Concatenate native records; preserve their positions and model paths.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            records = json.loads(path.read_text())
            if not records:
                continue
            source_file = str(path.relative_to(self.raw_dir))
            native_path = Path(re.sub(r'_x([0-9a-f]{2,6})_', lambda m: chr(int(m[1], 16)), source_file))
            frame = pd.json_normalize(records, max_level=0).assign(record=records)
            parts.append(frame.rename_axis('source_row').reset_index().assign(source_file=source_file,
                language=native_path.parts[1], variant=native_path.parts[3],
                model='/'.join(native_path.parts[4:-1]), course=native_path.stem))
        responses = pd.concat(parts, ignore_index=True)
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)
        responses['subject_key'] = responses.model + ':' + responses.variant
        if not responses.variant.isin(parameters['prompting_variants']).all():
            raise ValueError('An original result uses an undeclared prompting mode')

        # 2. Load only the original pure answer parser and its literal dictionary.
        path = self.raw_dir / parameters['layout']['parser']
        source = ast.parse(path.read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == labels['parser_function']]
        dictionaries = [ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == labels['answer_dictionary'] for target in node.targets)]
        if len(functions) != 1 or len(dictionaries) != 1:
            raise ValueError('The original parser or language dictionary is ambiguous')
        namespace = {'re': re, labels['answer_dictionary']: dictionaries[0]}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        supported = responses.language.isin(dictionaries[0])
        responses['prediction'] = None
        responses.loc[supported, 'prediction'] = responses.loc[supported, 'output'].combine(
            responses.loc[supported, 'language'], namespace[labels['parser_function']])
        responses['prediction'] = responses.prediction.astype(object).where(responses.prediction.notna(), None)
        responses['response'] = (responses.prediction == responses.answer).astype(float).astype(object)
        responses.loc[~supported, 'response'] = None
        responses['grade_status'] = 'native_parser_comparison'
        responses.loc[~supported, 'grade_status'] = 'unavailable_native_language_keyword'

        # 3. Define the stimulus from its recorded input, not a shared row index.
        responses['content'] = [json.dumps(dict(system=row.system, user=row.input), ensure_ascii=False)
                                for row in responses.itertuples()]
        responses['item_key'] = [json.dumps([row.content, row.language, row.answer], ensure_ascii=False)
                                 for row in responses.itertuples()]
        responses['test_condition'] = responses.response_key
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.response_key
        items['features'] = [dict(language=row.language, input_scope=labels['input_scope'])
                             for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=answer, rule=self.grading['rule']) for answer in items.answer]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(**self.grading['verifiers']['native'],
            language=language, answer_keyword=dictionaries[0].get(language)), sort_keys=True)) for language in items.language]

        # 4. Keep original model identifiers and distinguish prompting conditions.
        subjects = responses[['subject_key', 'model', 'variant']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.model + ' [' + subjects.variant + ']'
        subjects['features'] = [dict(harness=labels['harness'], source_model=row.model,
            prompting_variant=row.variant, prompting=parameters['prompting_variants'][row.variant],
            historical_configuration=labels['historical_configuration']) for row in subjects.itertuples()]

        # 5. Retain the complete original attempt and its grade availability.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            grade_status=row.grade_status, extracted_answer=row.prediction, record=row.record),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], 'traces': traces}


if __name__ == '__main__':
    TUMLU(__file__).main_from_args()
