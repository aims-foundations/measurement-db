"""Tabulate released MasakhaNER predictions against their historical references."""

import csv
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Afrieval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        protocol = parameters['protocol']

        # 1. Read native token tables and group their blank-line-delimited sentences.
        frames = []
        for group in parameters.values():
            if 'glob' not in group:
                continue
            for path in sorted(self.raw_dir.glob(group['glob'])):
                source = str(path.relative_to(self.raw_dir))
                match = re.fullmatch(group['pattern'], source)
                if match is None:
                    raise ValueError('An unknown source filename requires an explicit mapping')
                columns = group['columns'].split()
                tokens = pd.read_csv(path, sep=r'\s+', header=None, names=columns,
                    dtype=str, keep_default_na=False, skip_blank_lines=False, quoting=csv.QUOTE_NONE)
                blank = tokens.token.eq('')
                if not tokens.index.equals(pd.RangeIndex(len(tokens))) or tokens.empty:
                    raise ValueError('Expected an unindexed, nonempty CoNLL token table')
                if not tokens.loc[~blank].map(lambda value: isinstance(value, str) and bool(value)).all().all():
                    raise ValueError('A token or label is missing from the native record')
                if not tokens.loc[blank].eq('').all().all():
                    raise ValueError('A sentence boundary contains unexpected data')
                sentences = tokens.assign(sentence=blank.cumsum()).loc[~blank].groupby(
                    'sentence', sort=False)[columns].agg(list).reset_index(drop=True)
                sentences['source_sentence'] = sentences.index
                coordinates = dict(kind=group['kind'], version=group['version'], run=group.get('run', ''))
                coordinates.update(match.groupdict())
                frames.append(sentences.assign(source_file=source, **coordinates))
        records = pd.concat(frames, ignore_index=True)
        if records.empty:
            raise ValueError('No native reference or prediction records were found')

        # 2. Join every prediction to its original reference, checking the entire token sequence.
        keys = ['version', 'language', 'source_sentence']
        references = records.loc[records.kind.eq('reference'), keys + ['source_file', 'token', 'gold']].rename(
            columns={'source_file': 'reference_file', 'token': 'reference_tokens'})
        predictions = records.loc[records.kind.eq('prediction')].copy()
        responses = predictions.merge(references, on=keys, how='left', validate='many_to_one',
            suffixes=('', '_reference'), indicator=True)
        if not responses._merge.eq('both').all():
            raise ValueError('A released prediction has no original reference sentence')
        expected_tokens = responses.reference_tokens.copy()
        normalized = responses.model.eq(protocol['digit_normalized_model']) & responses.version.eq('v1')
        expected_tokens.loc[normalized] = expected_tokens.loc[normalized].map(
            lambda sentence: [re.sub(r'\d', '0', token) for token in sentence])
        if not responses.token.eq(expected_tokens).all():
            raise ValueError('Native tokens do not align with the historical reference')
        if not responses.loc[normalized, 'embedded_gold'].eq(responses.loc[normalized, 'gold_reference']).all():
            raise ValueError('An embedded reference disagrees with the original annotation')
        if not responses.predicted.map(len).eq(responses.gold_reference.map(len)).all():
            raise ValueError('A native prediction omits part of the reference sentence')
        expected_counts = references.groupby(['version', 'language']).size().rename('reference_count')
        counts = responses.groupby(['source_file', 'version', 'language']).size().rename('prediction_count').reset_index()
        counts = counts.merge(expected_counts, on=['version', 'language'], validate='many_to_one')
        if not counts.prediction_count.eq(counts.reference_count).all():
            raise ValueError('A prediction export does not cover its complete original test split')
        responses['response'] = responses.predicted.eq(responses.gold_reference).astype(float)
        responses['response_key'] = responses.index
        responses['subject_key'] = responses.source_file
        responses['item_key'] = responses.response_key

        # 3. Keep each released model/language/run configuration and its actual tokenized input.
        subjects = responses[['subject_key', 'version', 'language', 'model', 'run']].drop_duplicates()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(source_model_label=row.model, dataset_version=row.version,
            evaluation_language=row.language, released_run=row.run, source_export=row.subject_key,
            harness=protocol['harness']) for row in subjects.itertuples()]
        items = responses[['item_key', 'version', 'language', 'source_sentence', 'token', 'gold_reference']].copy()
        items['raw_item_id'] = items.version + '/' + items.language + '/sentence/' + items.source_sentence.astype(str)
        # Escapes preserve literal token code points through shared text normalization.
        items['content'] = [json.dumps(dict(task=protocol['instruction'], language=language, tokens=tokens),
            ensure_ascii=True) for language, tokens in zip(items.language, items.token)]
        items['features'] = [dict(language=language, split='test') for language in items.language]
        items['grading_criterion'] = [dict(reference_answer=json.dumps(tags, ensure_ascii=False),
            rule=self.grading['rule']) for tags in items.gold_reference]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))

        # 4. Preserve all native output labels and original coordinates without clipping.
        responses['test_condition'] = 'source_export=' + responses.source_file
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_sentence=int(row.source_sentence),
            reference_file=row.reference_file, tokens=row.token, predicted=row.predicted,
            embedded_gold=row.embedded_gold if isinstance(row.embedded_gold, list) else None,
            reference_tokens=row.reference_tokens, reference_tags=row.gold_reference), ensure_ascii=False)
            for row in responses.itertuples()]
        return {
            'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces,
        }


if __name__ == '__main__':
    Afrieval(__file__).main_from_args()
