"""Tabulate the released RNA-modification predictions and their separate references."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class NanoBaseLib(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, grading = self.build_parameters, self.grading
        labels = parameters['labels']

        # 1. Read each complete CSV, retaining literal cells and the separate assay labels.
        matrices, method_columns = [], {}
        for modification, path in parameters['matrices'].items():
            frame = pd.read_csv(self.raw_dir / path, dtype=str, keep_default_na=False)
            references = grading['verifiers'][modification]['reference_assays']
            attributes = frame.columns.intersection(parameters['site_columns'])
            method_columns[modification] = frame.columns.difference([*references, *attributes], sort=False)
            if frame.pos.eq('').any() or frame.pos.duplicated().any():
                raise ValueError('Source genomic sites must have unique, nonempty identifiers')
            if not frame[references].isin(['0', '1']).all().all():
                raise ValueError('A reference assay has an undeclared label')
            frame['source_record'] = frame.to_dict('records')
            frame['source_row'] = frame.index
            frame['source_file'] = path
            frame['modification'] = modification
            frame['item_key'] = modification + ':' + frame.pos
            frame['attributes'] = frame[attributes].to_dict('records')
            frame['reference'] = frame[references].to_dict('records')
            matrices.append(frame)

        # 2. Keep one item per site, with reference assays confined to grading information.
        items = pd.concat(matrices, ignore_index=True)
        items['raw_item_id'] = items.item_key
        items['content'] = items.attributes.map(lambda row: json.dumps(row, sort_keys=True))
        items['features'] = [dict(modification=row.modification, input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=json.dumps(row.reference, sort_keys=True),
            rule=grading['rule']) for row in items.itertuples()]
        items['verifier'] = items.modification.map(lambda modification:
            ExactMatcher(spec=json.dumps(grading['verifiers'][modification], sort_keys=True)))

        # 3. Unpivot method columns; empty source cells are not recorded attempts.
        observations = []
        for frame in matrices:
            methods = method_columns[frame.modification.iloc[0]]
            observations.append(frame.melt(id_vars=['item_key', 'modification', 'source_file',
                'source_row', 'source_record'], value_vars=methods, var_name='method', value_name='prediction'))
        responses = pd.concat(observations, ignore_index=True)
        responses = responses.loc[responses.prediction.ne('')].copy()
        if not np.isfinite(pd.to_numeric(responses.prediction, errors='raise')).all():
            raise ValueError('A recorded method prediction is not finite')
        responses['subject_key'] = responses.modification + ':' + responses.method
        responses['response_key'] = responses.item_key + ':' + responses.method
        responses['test_condition'] = labels['condition_prefix'] + responses.modification
        responses['response'] = None

        # 4. Preserve method identities and full native rows without inventing individual grades.
        subjects = responses[['subject_key', 'modification', 'method']].drop_duplicates().copy()
        subjects['raw_label'] = labels['subject_prefix'] + subjects.modification + ' / ' + subjects.method
        subjects['features'] = [dict(**parameters['subject_features'], modification=row.modification,
            source_method=row.method) for row in subjects.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            modification=row.modification, method=row.method, prediction=row.prediction,
            record=row.source_record, grade_status=labels['grade_status']), allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
                'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
                'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
                'traces': traces}


if __name__ == '__main__':
    NanoBaseLib(__file__).main_from_args()
