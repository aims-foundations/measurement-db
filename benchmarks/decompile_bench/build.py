"""Tabulate recorded decompiler outputs using the original edit-similarity rule."""

import json
from pathlib import Path
import sys

import editdistance
import pandas as pd
import pyarrow.ipc as ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class DecompileBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        p = self.build_parameters

        # 1. Read the original Arrow tables with explicit split and row provenance.
        frames = []
        for path in sorted((self.raw_dir / p['layout']['release']).glob('*/*.arrow')):
            with path.open('rb') as stream:
                frame = ipc.open_stream(stream).read_all().to_pandas()
            if list(frame.columns) != list(p['columns']) or path.parent.name not in p['splits']:
                raise ValueError('Unexpected native columns or split')
            frames.append(frame.assign(split=p['splits'][path.parent.name], source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        records = pd.concat(frames, ignore_index=True)
        if records.isna().any().any() or records.duplicated(['split', 'index']).any():
            raise ValueError('Missing native fields or duplicate split/index identifiers')
        if not records.language.isin(['c', 'cpp']).all() or not records.opt.isin(['O0', 'O1', 'O2', 'O3']).all():
            raise ValueError('Unknown native language or optimization level')
        if records.func.str.strip().eq('').any():
            raise ValueError('Missing reference source')

        # 2. Collapse byte-identical native records, retaining every original alias.
        identity = ['split', *[name for name in p['columns'] if name != 'index']]
        records['record_key'] = records.groupby(identity, sort=False, dropna=False).ngroup()
        records['origin'] = records[['source_file', 'source_row', 'index']].to_dict('records')
        origins = records.groupby('record_key', sort=False).origin.agg(list)
        unique = records.drop_duplicates('record_key').set_index('record_key').join(origins.rename('source_records')).reset_index()
        has_input = unique[['asm', 'ida_asm', 'ghidra_asm']].apply(lambda column: column.str.strip().ne('')).any(axis=1)
        unique = unique.loc[has_input].copy()

        # 3. Melt the recorded outputs and apply the authors' deterministic metric.
        long = unique.melt(id_vars=['record_key', 'func'], value_vars=list(p['systems']),
                           var_name='subject_key', value_name='prediction')
        long = long.loc[long.prediction.str.strip().ne('')].copy()
        normalized = long[['func', 'prediction']].map(lambda text: '\n'.join(line.strip() for line in text.splitlines() if line.strip()))
        long['response'] = [1 - editdistance.eval(row.func, row.prediction) / max(len(row.func), len(row.prediction))
                            for row in normalized.itertuples()]
        long['item_key'] = long.record_key
        long['response_key'] = long.record_key.astype(str) + '/' + long.subject_key

        # 4. Separate the assembly stimulus from the reference source and tests.
        items = unique.loc[unique.record_key.isin(long.record_key)].copy()
        fields = ['func_name', 'asm', 'ida_asm', 'ghidra_asm', 'opt', 'language']
        items['content'] = [json.dumps(dict(text=p['labels']['instruction'], **row), sort_keys=True)
                            for row in items[fields].to_dict('records')]
        items['item_key'] = items.record_key
        items['raw_item_id'] = items.split + '/' + items['index'].astype(str)
        items['features'] = items[['split', 'opt', 'language']].to_dict('records')
        references = items[['func', 'func_dep', 'test']].to_dict('records')
        items['grading_criterion'] = [dict(rule=self.grading['rule'], reference_answer=json.dumps(row, sort_keys=True)) for row in references]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['edit_similarity'], sort_keys=True))
        subjects = pd.DataFrame({'subject_key': list(p['systems']), 'raw_label': list(p['systems'].values())})
        subjects['features'] = subjects.subject_key.map(lambda name: dict(native_output_column=name, configuration=p['labels']['configuration']))
        responses = long[['response_key', 'subject_key', 'item_key', 'response']].assign(test_condition=p['labels']['condition'])

        # 5. Keep complete original outputs and their source aliases as trace evidence.
        evidence = long[['response_key', 'record_key', 'subject_key', 'prediction']].merge(
            unique[['record_key', 'source_records']], on='record_key', validate='many_to_one')
        evidence['trace'] = [json.dumps(row, sort_keys=True) for row in evidence[['subject_key', 'prediction', 'source_records']].to_dict('records')]
        return dict(subjects=subjects,
                    items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
                    responses=responses, traces=evidence[['response_key', 'trace']])


if __name__ == '__main__':
    DecompileBench(__file__).main_from_args()
