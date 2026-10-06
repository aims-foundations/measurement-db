"""Tabulate original MQM annotations with the existing binary error-indicator readout."""

import csv
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class WmtMqm(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('release')

    def build_tables(self):
        parameters = self.build_parameters
        protocol = parameters['protocol']

        # 1. Read full native tables, including unnamed trailing comment fields.
        frames = []
        for source_file in parameters['campaigns']:
            path = self.raw_dir / source_file
            with path.open(newline='') as stream:
                reader = csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)
                header, first = next(reader), next(reader)
            names = header + [f'unlabeled_column_{index + 1}' for index in range(len(first) - len(header))]
            frame = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                header=None, skiprows=1, names=names)
            frame['native_record'] = frame.to_dict('records')
            frame['restored_target'] = None
            if source_file in parameters['corrected_files']:
                corrected = pd.read_csv(self.raw_dir / parameters['corrected_files'][source_file], sep='\t', quoting=csv.QUOTE_NONE,
                    dtype=str, keep_default_na=False)
                unchanged = [name for name in names if name != 'target']
                if len(frame) != len(corrected) or not frame[unchanged].equals(corrected[unchanged]):
                    raise ValueError('The author correction changes more than the annotated target text')
                frame['restored_target'] = corrected.target.str.replace(r'</?v>', '', regex=True)
            frame = frame.rename(columns=parameters['column_aliases'])
            required = ['system', 'doc', 'seg_id', 'rater', 'source', 'target', 'category', 'severity']
            if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
                raise ValueError('An annotation table has incomplete required fields')
            frames.append(frame.assign(source_file=source_file, source_row=frame.index))
        annotations = pd.concat(frames, ignore_index=True)

        # 2. Apply the declared readout while retaining all original annotations.
        if not annotations.category.isin(parameters['category_mapping']).all() or not annotations.severity.isin(parameters['severity_mapping']).all():
            raise ValueError('An unreviewed annotation category or severity requires an explicit mapping')
        annotations['readout_category'] = annotations.category.map(parameters['category_mapping']).replace('', None)
        annotations['readout_severity'] = annotations.severity.map(parameters['severity_mapping']).replace('', None)
        annotations['reviewed'] = annotations.category.isin(parameters['no_error_categories']) | (
            annotations.readout_category.notna() & annotations.readout_severity.notna())
        annotations['source_text'] = annotations.source.str.replace(r'</?v>', '', regex=True).str.strip()
        nominal_item = ['source_file', 'doc', 'seg_id']
        variation = annotations.groupby(nominal_item).source_text.nunique().rename('source_variants').reset_index()
        annotations = annotations.merge(variation, on=nominal_item, validate='many_to_one')
        excluded = annotations.source_variants.ne(1) | annotations.source_text.eq('')
        print('Withheld ambiguous/empty source segments:', annotations.loc[excluded, nominal_item].drop_duplicates().shape[0])
        eligible = annotations.loc[~excluded].copy()
        rating_key = nominal_item + ['system', 'rater']
        eligible['annotation'] = [dict(source_row=int(row.source_row), original_record=row.native_record,
            author_restored_translation=row.restored_target) for row in eligible.itertuples()]
        ratings = eligible.groupby(rating_key, sort=False).agg(
            reviewed=('reviewed', 'any'), annotations=('annotation', list)).reset_index()
        ratings = ratings.loc[ratings.reviewed].reset_index(drop=True)
        ratings['rating_key'] = ratings.index
        eligible = eligible.merge(ratings[rating_key + ['rating_key']], on=rating_key, validate='many_to_one')

        # 3. Expand only reviewed rating units over the original 18 grading buckets.
        dimensions = pd.MultiIndex.from_product([parameters['categories'], parameters['severities']],
            names=['readout_category', 'readout_severity']).to_frame(index=False)
        present = eligible.dropna(subset=['readout_category', 'readout_severity'])[
            ['rating_key', 'readout_category', 'readout_severity']].drop_duplicates().assign(response=1.0)
        responses = ratings.drop(columns=['annotations', 'reviewed']).merge(dimensions, how='cross')
        responses = responses.merge(present, on=['rating_key', 'readout_category', 'readout_severity'],
            how='left', validate='one_to_one')
        responses['response'] = responses.response.fillna(0.0)
        scope = pd.DataFrame({'campaign': parameters['campaigns'], 'language_pair': parameters['language_pairs']})
        scope[['source_language', 'target_language']] = scope.language_pair.str.split('-', expand=True)
        scope = scope.rename_axis('source_file').reset_index()
        responses = responses.merge(scope, on='source_file', validate='many_to_one')

        # 4. Keep source configurations and grading protocols in their identities.
        subject_key = ['source_file', 'system']
        subjects = responses[subject_key + ['campaign', 'language_pair']].drop_duplicates().reset_index(drop=True)
        subjects['subject_key'] = subjects.index
        subjects['raw_label'] = 'WMT ' + subjects.campaign + ' ' + subjects.language_pair + ': ' + subjects.system
        subjects['features'] = [dict(harness=protocol['harness'], source_system_label=row.system,
            campaign=row.campaign, language_pair=row.language_pair, source_file=row.source_file,
            configuration_status=protocol['configuration_status']) for row in subjects.itertuples()]
        responses = responses.merge(subjects[subject_key + ['subject_key']], on=subject_key, validate='many_to_one')
        item_key = nominal_item + ['readout_category', 'readout_severity']
        source_items = eligible[nominal_item + ['source_text']].drop_duplicates()
        items = responses[item_key + ['source_language', 'target_language']].drop_duplicates().merge(
            source_items, on=nominal_item, validate='many_to_one').reset_index(drop=True)
        items['item_key'] = items.index
        items['raw_item_id'] = (items.source_file + '::' + items.doc + '::' + items.seg_id + '::'
            + items.readout_category + '::' + items.readout_severity)
        items['content'] = [json.dumps(dict(source_text=row.source_text, source_language=row.source_language,
            target_language=row.target_language), ensure_ascii=True) for row in items.itertuples()]
        items['features'] = [dict(source_file=row.source_file, source_document=row.doc, source_segment=row.seg_id,
            category=row.readout_category, severity=row.readout_severity) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + ' Selected category: ' + row.readout_category
            + '; severity bucket: ' + row.readout_severity + '.') for row in items.itertuples()]
        items['verifier'] = [Judge(judged_by='human', spec=json.dumps(dict(self.grading['verifiers']['human'],
            category=row.readout_category, severity_bucket=row.readout_severity), sort_keys=True)) for row in items.itertuples()]
        responses = responses.merge(items[item_key + ['item_key']], on=item_key, validate='many_to_one')
        responses['response_key'] = responses.index
        responses['test_condition'] = [json.dumps(dict(source_file=row.source_file, rater=row.rater,
            source_document=row.doc, source_segment=row.seg_id, category=row.readout_category,
            severity_bucket=row.readout_severity), sort_keys=True) for row in responses.itertuples()]

        # 5. Attach complete rating evidence; never choose or clip a target variant.
        ratings['trace'] = [json.dumps(dict(source_file=row.source_file, source_document=row.doc,
            source_segment=row.seg_id, source_system=row.system, rater=row.rater, annotations=row.annotations,
            readout=protocol['readout'], trial_scope=protocol['trial_scope']), ensure_ascii=False)
            for row in ratings.itertuples()]
        traces = responses[['response_key', 'rating_key']].merge(ratings[['rating_key', 'trace']],
            on='rating_key', validate='many_to_one')
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            'traces': traces[['response_key', 'trace']]}


if __name__ == '__main__':
    WmtMqm(__file__).main_from_args()
