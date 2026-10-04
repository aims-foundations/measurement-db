"""Tabulate released MLIP Arena simulations without executing interatomic models."""

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

from ase.symbols import Symbols
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle, native_json_value


class MLIPArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        grading = self.grading['verifiers']['native']
        github = self.raw_dir / parameters['layout']['github']
        huggingface = self.raw_dir / parameters['layout']['huggingface']

        # 1. Read the original atomic structures as tables. Reference properties
        # remain separate from the positions, species, cell and periodicity.
        banks = {}
        for family, path, key in [
            ('wbm', github / 'benchmarks/wbm_structures.db', 'wbm_id'),
            ('c2db', github / 'benchmarks/c2db/c2db.db', 'uid'),
            ('stability', huggingface / 'stability/random-mixture.db', None),
        ]:
            with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                table = pd.read_sql_query('SELECT id, numbers, positions, cell, pbc, key_value_pairs, '
                    'initial_magmoms, initial_charges, masses, tags, momenta, constraints FROM systems', connection)
            table['properties'] = table.key_value_pairs.map(json.loads)
            table['structure'] = [dict(atomic_numbers=np.frombuffer(row.numbers, dtype='<i4').tolist(),
                positions_angstrom=np.frombuffer(row.positions, dtype='<f8').reshape(-1, 3).tolist(),
                cell_angstrom=np.frombuffer(row.cell, dtype='<f8').reshape(3, 3).tolist(),
                periodic=[bool(row.pbc & flag) for flag in [1, 2, 4]]) for row in table.itertuples()]
            if table.constraints.notna().any():
                raise ValueError('Review newly supplied atomic constraints before importing this source')
            for field in ['initial_magmoms', 'initial_charges', 'masses', 'tags', 'momenta']:
                present = table[field].notna()
                values = table.loc[present, field].map(lambda data: np.frombuffer(data, dtype='<i4' if field == 'tags' else '<f8'))
                if field == 'momenta':
                    values = values.map(lambda value: value.reshape(-1, 3))
                for structure, value in zip(table.loc[present, 'structure'], values):
                    structure.setdefault('atom_arrays_ase_units', {})[field] = value.tolist()
            if key:
                table['task_name'] = table.properties.map(lambda values: values[key])
            else:
                # Match the native Hill-formula label to the full database cell,
                # requiring a unique formula; no structure is inferred from text.
                table['task_name'] = table.structure.map(lambda row: Symbols(row['atomic_numbers']).get_chemical_formula(mode='hill'))
            if table.task_name.duplicated().any():
                raise ValueError(f'Ambiguous {family} structure identifier')
            table['input_source'] = [dict(source_file=str(path.relative_to(self.raw_dir)), database_row=int(row)) for row in table.id]
            banks[family] = table[['task_name', 'structure', 'properties', 'input_source']]

        # 2. Join native energy-volume outputs to their published per-system
        # assessments. Missing aggregate placeholders are not recorded attempts.
        parts = []
        for family in ['eos_bulk', 'ev']:
            for path in sorted((github / 'benchmarks' / family).glob('*_processed.parquet')):
                score_data = pq.read_table(path)
                scores = score_data.to_pandas().rename_axis('assessment_row').reset_index()
                native_path = path.with_name(path.name.replace('_processed', ''))
                native_data = pq.read_table(native_path)
                native = native_data.to_pandas().rename_axis('source_row').reset_index()
                native['native_output'] = native_data.to_pylist()
                scores['native_assessment'] = score_data.to_pylist()
                table = native.merge(scores, left_on='id', right_on='structure', validate='one_to_one')
                if len(table) != len(native) or table['missing'].any():
                    raise ValueError('Recorded curve and processed assessment do not agree')
                table = table.rename(columns={'structure': 'task_name'}).merge(banks['wbm'], on='task_name', validate='many_to_one')
                if len(table) != len(native):
                    raise ValueError('Unmatched WBM input')
                table['family'], table['source_file'] = family, str(native_path.relative_to(self.raw_dir))
                table['stimulus'] = table.structure
                table['reference'] = None
                table['native_record'] = [dict(output=output, assessment=assessment,
                    assessment_source=dict(source_file=str(path.relative_to(self.raw_dir)), source_row=int(index)))
                    for output, assessment, index in zip(table.native_output, table.native_assessment, table.assessment_row)]
                parts.append(table)

        # 3. Diatomic and combustion exports already contain individual metrics.
        # Retain every full curve; identical JSON/JSONL exports are aliases below.
        for family in ['diatomics', 'combustion']:
            for path in sorted((github / 'benchmarks' / family).rglob('*.json')):
                text = path.read_text()
                records = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
                table = pd.json_normalize(records, max_level=0).rename_axis('source_row').reset_index()
                table['native_record'] = records
                table['family'], table['source_file'] = family, str(path.relative_to(self.raw_dir))
                table['model'] = table['method'] if 'method' in table else path.stem
                table['reference'] = None
                table['input_source'] = [dict(source_file=str(path.relative_to(self.raw_dir)), source_row=int(row)) for row in table.source_row]
                if family == 'diatomics':
                    table['task_name'] = table['name']
                    table['stimulus'] = [dict(homonuclear_pair=name, separations_angstrom=distances)
                        for name, distances in zip(table['name'], table.R)]
                else:
                    input_path = huggingface / 'combustion/H256O128.extxyz'
                    table['task_name'] = table.formula
                    table['stimulus'] = [dict(extended_xyz=input_path.read_text()) for _ in table.index]
                    table['input_source'] = [dict(source_file=str(input_path.relative_to(self.raw_dir))) for _ in table.index]
                    rule = grading['combustion']
                    table['enthalpy_difference'] = table.energies.map(lambda values:
                        (values[-1] - values[0]) / rule['water_count'] * rule['energy_conversion'] - rule['reference_enthalpy'])
                    # Keep the published norm's rounding independent of NumPy's BLAS kernel.
                    table['com_drift'] = table.com_drifts.map(lambda values: math.hypot(*values[-1]))
                parts.append(table)

        # 4. A stability file contains many frames per simulation. Group them
        # before melting metrics, preserving their original order and every field.
        for path in sorted((github / 'benchmarks/stability').rglob('*.parquet')):
            frame_data = pq.read_table(path)
            frames = frame_data.to_pandas().rename_axis('source_row').reset_index()
            frames['native_frame'] = frame_data.to_pylist()
            metrics = grading['families']['stability']['metrics']
            grouped = frames.groupby('formula', sort=False)
            if grouped[metrics].nunique(dropna=False).gt(1).any().any():
                raise ValueError('Conflicting per-simulation stability summary')
            table = grouped[metrics].first().reset_index().rename(columns={'formula': 'task_name'})
            table['source_row'] = grouped.source_row.first().to_numpy()
            table['native_record'] = [dict(frames=group.native_frame.tolist(),
                source_rows=group.source_row.tolist()) for _, group in grouped]
            table = table.merge(banks['stability'], on='task_name', validate='many_to_one')
            if len(table) != grouped.ngroups:
                raise ValueError('Unmatched stability input')
            table['protocol'] = path.stem.rsplit('-', 1)[1]
            table['stimulus'] = [dict(structure=structure, protocol=protocol) for structure, protocol in zip(table.structure, table.protocol)]
            table['model'] = re.sub(r'_x([0-9a-f]+)_', lambda match: chr(int(match[1], 16)), path.stem.rsplit('-', 1)[0])
            table['family'] = 'stability'
            table['source_file'], table['reference'] = str(path.relative_to(self.raw_dir)), None
            parts.append(table)

        # 5. Reconstruct the released classification rules from existing outputs,
        # keeping references out of input structures and unknown labels ungraded.
        for path in sorted((github / 'benchmarks/c2db').glob('*.parquet')):
            native_data = pq.read_table(path)
            table = native_data.to_pandas().rename_axis('source_row').reset_index().rename(columns={'uid': 'task_name'})
            table['native_record'] = native_data.to_pylist()
            before = len(table)
            table = table.merge(banks['c2db'], on='task_name', validate='many_to_one')
            if len(table) != before:
                raise ValueError('Unmatched C2DB input')
            threshold = grading['c2db_threshold']
            eigenvalues = table.eigenvalues.map(lambda values: np.min(values) if np.isreal(values).all() else threshold)
            frequencies = table.frequencies.map(lambda values: np.min(values) if np.isreal(values).all() else threshold)
            predicted = ~(eigenvalues.lt(threshold) | frequencies.lt(threshold))
            table['reference'] = table.properties.map(lambda values: values['dyn_stab'])
            valid = table.reference.isin(['Yes', 'No']) & eigenvalues.map(np.isfinite) & frequencies.map(np.isfinite)
            table['classification_agreement'] = predicted.eq(table.reference.eq('Yes')).astype(float).where(valid)
            table['stimulus'] = table.structure
            table['family'], table['source_file'] = 'c2db', str(path.relative_to(self.raw_dir))
            parts.append(table)

        mof_path = github / 'benchmarks/mof/classification/input.pkl'
        bank = read_native_pickle(mof_path).rename_axis('input_row').reset_index().rename(columns={'structure': 'input_structure'})
        for path in sorted(mof_path.parent.glob('*.pkl')):
            if path == mof_path:
                continue
            # Retain the original DataFrame index as the source row identifier.
            table = read_native_pickle(path).rename_axis('source_row').reset_index()
            table['native_record'] = table.drop(columns='source_row').to_dict('records')
            before = len(table)
            table = table.merge(bank, on=['name', 'class'], validate='many_to_one')
            if len(table) != before:
                raise ValueError('Unmatched MOF input')
            source_states = table.structure.map(native_json_value)
            input_states = table.input_structure.map(native_json_value)
            if any(a != b for a, b in zip(source_states, input_states)):
                raise ValueError('MOF output refers to a different input structure')
            heat = -table.heat_of_adsorption.map(lambda values: float(np.mean(values)))
            bounds = table['class'].map(grading['mof_intervals'])
            table['classification_agreement'] = [float((lower is None or value >= lower) and (upper is None or value < upper))
                if np.isfinite(value) and value > 0 and name not in grading['mof_excluded_names'] else None
                for value, (lower, upper), name in zip(heat, bounds, table['name'])]
            table['stimulus'] = [dict(atomic_numbers=record.state['arrays']['numbers'].tolist(),
                positions_angstrom=record.state['arrays']['positions'].tolist(),
                cell_angstrom=record.state['_cellobj'].state['array'].tolist(), periodic=record.state['_pbc'].tolist(),
                atom_arrays_ase_units=native_json_value({key: value for key, value in record.state['arrays'].items()
                    if key not in ['numbers', 'positions']}), info=native_json_value(record.state.get('info', {})))
                for record in table.input_structure]
            table['input_source'] = [dict(source_file=str(mof_path.relative_to(self.raw_dir)), source_row=int(index)) for index in table.input_row]
            table['reference'], table['task_name'] = table['class'], table['name']
            table['family'], table['source_file'] = 'mof', str(path.relative_to(self.raw_dir))
            parts.append(table)

        # 6. Vacancy pickles are decoded as inert data. The post-relaxation
        # geometry stays in the output, never masquerading as the initial cell.
        records = []
        for path in sorted((huggingface / 'vacancy_migration').rglob('*.pkl')):
            record = read_native_pickle(path)
            original_name = re.sub(r'_x([0-9a-f]+)_', lambda match: chr(int(match[1], 16)), path.stem)
            model, task = original_name.rsplit('-', 2)[0], '-'.join(original_name.rsplit('-', 2)[1:])
            lattice, element, count = re.fullmatch(r'(fcc|hcp)-([A-Z][a-z]?)(\d+)', task).groups()
            records.append(dict(model=model, family='vacancy_migration', source_file=str(path.relative_to(self.raw_dir)),
                source_row=0, task_name=task, stimulus=dict(lattice=lattice, element=element, supercell_atoms=int(count),
                    initial_geometry_available=False), input_source=dict(source_file='github/mlip_arena/tasks/vacancy_migration/input.py'),
                reference=None, native_record=record, asymmetry=record['asymmetry']))
        parts.append(pd.DataFrame(records))

        # 7. Melt native scalar assessments and link them to full source records.
        # Multiple metrics share a run; copied exports retain all source aliases.
        observations = []
        for family, definition in grading['families'].items():
            native = pd.concat([table for table in parts if table.family.iloc[0] == family], ignore_index=True)
            native['native_record'] = native.native_record.map(native_json_value)
            native['stimulus'] = [dict(instruction=definition['instruction'], input=native_json_value(value)) for value in native.stimulus]
            native['record_key'] = [hashlib.sha256(json.dumps([family, row.model, row.task_name, row.stimulus['input'], row.native_record],
                sort_keys=True, allow_nan=False).encode()).hexdigest() for row in native.itertuples()]
            native['source_coordinate'] = [dict(source_file=row.source_file, source_row=int(row.source_row)) for row in native.itertuples()]
            aliases = native.groupby('record_key', sort=False).source_coordinate.agg(list).rename('source_coordinates')
            native = native.drop_duplicates('record_key').merge(aliases, on='record_key', validate='one_to_one')
            for metric in definition['metrics']:
                if metric not in native:
                    native[metric] = None
            id_columns = ['record_key', 'model', 'task_name', 'family', 'stimulus', 'input_source', 'reference', 'native_record', 'source_coordinates']
            table = native.melt(id_vars=id_columns, value_vars=definition['metrics'], var_name='metric', value_name='native_metric')
            values = pd.to_numeric(table.native_metric, errors='raise')
            table['response'] = values.where(np.isfinite(values))
            table['grade_status'] = np.where(values.isna(), 'unavailable_native_grade',
                np.where(np.isfinite(values), 'graded', 'nonfinite_native_grade'))
            observations.append(table)
        observations = pd.concat(observations, ignore_index=True)
        observations['subject_key'] = observations.family + '/' + observations.model
        observations['response_key'] = observations.record_key + '/' + observations.metric
        observations['item_key'] = [hashlib.sha256(json.dumps([row.family, row.task_name, row.stimulus, row.metric, row.reference],
            sort_keys=True, allow_nan=False).encode()).hexdigest() for row in observations.itertuples()]
        subjects = observations[['subject_key', 'model', 'family']].drop_duplicates('subject_key')
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.model + ' / ' + subjects.family
        subjects['features'] = [dict(parameters['subject_features'], native_model=model, task_family=family)
            for model, family in zip(subjects.model, subjects.family)]
        items = observations.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.family + '/' + items.task_name + '/' + items.metric
        items['content'] = items.stimulus.map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
        items['features'] = [dict(task_family=row.family, native_task=row.task_name,
            input_source=json.dumps(row.input_source, sort_keys=True)) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=grading['metrics'][row.metric]['rule'],
            response_scale=grading['metrics'][row.metric]['response_scale'],
            **({'reference_answer': row.reference} if row.reference is not None else {})) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(dict(source=grading['families'][row.family]['source'],
            metric=row.metric, unit=grading['metrics'][row.metric]['unit']), sort_keys=True)) for row in items.itertuples()]
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(record_key=row.record_key, family=row.family, model=row.model, task_name=row.task_name,
            source_coordinates=row.source_coordinates, input_source=row.input_source, reference=row.reference,
            native_record=row.native_record, metric=row.metric, native_metric=native_json_value(row.native_metric),
            grade_status=row.grade_status), ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier', 'features']],
            responses=observations[['response_key', 'subject_key', 'item_key', 'response']], traces=traces)


if __name__ == '__main__':
    MLIPArena(__file__).main_from_args()
