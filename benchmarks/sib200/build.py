"""Tabulate SIB-200's original classifier predictions and complete LLM replies."""

import csv
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class SIB200(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        paths, labels = parameters['paths'], parameters['labels']

        # 1. Load each original test table, retaining its row position and reference.
        banks = []
        for path in sorted((self.raw_dir / paths['bank']).glob('*/test.tsv')):
            frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
            banks.append(frame.assign(language=path.parent.name, source_row=frame.index,
                bank_file=str(path.relative_to(self.raw_dir))))
        bank = pd.concat(banks, ignore_index=True).rename(columns={
            'text': 'bank_text', 'category': 'reference', 'index_id': 'bank_index_id'})
        if bank.duplicated(['language', 'bank_index_id']).any():
            raise ValueError('A test item ID is repeated within its language')

        # 2. Read saved category tokens and their associated native run summaries.
        frames = []
        with zipfile.ZipFile(self.raw_dir / paths['archive']) as archive:
            names = sorted(name for name in archive.namelist() if name.startswith('outputs/')
                and Path(name).name.startswith('test_predictions_') and name.endswith('.txt'))
            for name in names:
                match = re.fullmatch(r'outputs/([a-z]{3}_[A-Za-z]{4})_(\w+)/test_predictions_([a-z]{3}_[A-Za-z]{4})_(\d+)\.txt', name)
                if match is None or match[1] != match[3] or match[2] not in parameters['classifier_models']:
                    raise ValueError('An undeclared classifier run needs source review: ' + name)
                frame = pd.read_csv(io.BytesIO(archive.read(name)), sep='\t', quoting=csv.QUOTE_NONE,
                    header=None, names=['text', 'prediction'], dtype=str, keep_default_na=False)
                frame['source_record'] = frame.to_dict('records')
                summary_text = archive.read(name.replace('test_predictions_', 'test_result_')).decode('utf-8')
                summary = dict(line.split(' = ', 1) for line in summary_text.splitlines())
                frames.append(frame.assign(source_file=paths['archive'], source_member=name,
                    source_row=frame.index, language=match[3], model=match[2], source_run=match[4],
                    protocol='classifier', summary_text=summary_text, reported_accuracy=float(summary['acc'])))

        # 3. Retain every original LLM row, including empty and unparsed replies.
        for folder, column in parameters['llm_columns'].items():
            for path in sorted((self.raw_dir / paths['llm'] / folder).glob('*.tsv')):
                frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
                frame['source_record'] = frame.to_dict('records')
                frame = frame.rename(columns={column: 'prediction', 'category': 'reported_reference'})
                frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_member='',
                    source_row=frame.index, language=path.stem, model=parameters['llm_models'][folder],
                    source_run='', protocol='llm', summary_text='', reported_accuracy=None))
        attempts = pd.concat(frames, ignore_index=True).merge(bank,
            on=['language', 'source_row'], how='left', validate='many_to_one', indicator=True)
        if not attempts._merge.eq('both').all() or not attempts.text.eq(attempts.bank_text).all():
            raise ValueError('A saved output does not match its original test sentence and position')
        llm = attempts.protocol.eq('llm')
        if not (attempts.loc[llm, 'index_id'].eq(attempts.loc[llm, 'bank_index_id']).all()
                and attempts.loc[llm, 'reported_reference'].eq(attempts.loc[llm, 'reference']).all()):
            raise ValueError('An LLM record disagrees with its original item ID or reference')
        if not attempts.loc[~llm, 'prediction'].isin(bank.reference.unique()).all():
            raise ValueError('A classifier output is not a released topic category')
        attempts['response'] = attempts.prediction.eq(attempts.reference).astype(float).astype(object)
        attempts.loc[llm, 'response'] = None
        attempts['reconstructed_accuracy'] = attempts.groupby(['source_file', 'source_member']).response.transform('mean')
        attempts['response_key'] = attempts.source_file + '#' + attempts.source_member + '#' + attempts.source_row.astype(str)

        # 4. Keep each language/run classifier distinct; LLMs share their recorded model label.
        attempts['subject_key'] = attempts.protocol + '/' + attempts.model
        attempts.loc[~llm, 'subject_key'] += '/' + attempts.loc[~llm, 'language'] + '/' + attempts.loc[~llm, 'source_run']
        subjects = attempts.drop_duplicates('subject_key')[['subject_key', 'model', 'protocol', 'language', 'source_run']].copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['classifier_harness'], source_model_label=row.model,
            finetuning_language=row.language, source_run=row.source_run,
            configuration_status=labels['classifier_configuration']) if row.protocol == 'classifier'
            else dict(harness=labels['llm_harness'], source_model_label=row.model,
                paper_model_version=parameters['paper_model_versions'][row.model],
                configuration_status=labels['llm_configuration']) for row in subjects.itertuples()]

        # 5. Preserve the original sentence, reference, language and grading availability.
        attempts['item_key'] = attempts.protocol + '/' + attempts.language + '/' + attempts.bank_index_id
        items = attempts.drop_duplicates('item_key')[['item_key', 'protocol', 'language', 'bank_index_id', 'text', 'reference']].copy()
        items['raw_item_id'] = 'sib200_' + items.language + '_' + items.bank_index_id
        items['content'] = items.text
        items['features'] = [dict(language=row.language, protocol=row.protocol,
            input_scope=labels[row.protocol + '_scope'],
            documented_prompt=parameters['prompts']['llm'] if row.protocol == 'llm' else 'Original test sentence')
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference, rule=self.grading['verifiers'][row.protocol]['rule'])
            for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['classifier'], sort_keys=True))
            if kind == 'classifier' else Judge(spec=json.dumps(self.grading['verifiers']['llm'], sort_keys=True))
            for kind in items.protocol]

        # 6. Keep complete source rows and summary disagreements without changing either source.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_member=row.source_member,
            source_row=row.source_row, source_record=row.source_record, bank_file=row.bank_file,
            reference=row.reference, native_run_summary=row.summary_text or None,
            grading_status='reconstructed_category_match' if row.protocol == 'classifier' else 'original_grade_unavailable',
            reconstructed_accuracy=float(row.reconstructed_accuracy) if row.protocol == 'classifier' else None,
            summary_agrees=(abs(row.reported_accuracy - row.reconstructed_accuracy) < 1e-12)
                if row.protocol == 'classifier' else None), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    SIB200(__file__).main_from_args()
