"""Tabulate SummEval summaries, source articles and individual human ratings."""

import json
from pathlib import Path
import sys
import tarfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class SummEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Load original summary records and their source articles without extraction.
        summaries = pd.read_json(self.raw_dir / layout['annotations'], lines=True)
        summaries['source_record'] = summaries.to_dict('records')
        summaries['source_row'] = summaries.index
        summaries['story_path'] = summaries.filepath.str.removeprefix('cnndm/')
        wanted, stories = set(summaries.story_path), []
        for filename in parameters['archives'].values():
            with tarfile.open(self.raw_dir / filename, mode='r|gz') as archive:
                for member in archive:
                    path = member.name.removeprefix('./')
                    if member.isfile() and path in wanted:
                        stories.append(dict(story_path=path, story_text=archive.extractfile(member).read().decode('utf-8')))
        articles = pd.DataFrame(stories)
        articles['content'] = articles.story_text.str.split('@highlight', n=1).str[0].map(
            lambda text: ' '.join(line.strip() for line in text.split('\n') if line.strip()))
        summaries = summaries.merge(articles[['story_path', 'content']], on='story_path', how='left', validate='many_to_one')
        if summaries.content.isna().any() or summaries.content.eq('').any():
            raise ValueError('A summary lacks its original source article')
        summaries['reference_json'] = summaries.references.map(lambda values: json.dumps(values, ensure_ascii=False))
        if summaries.groupby('id')[['story_path', 'content', 'reference_json']].nunique().gt(1).any().any():
            raise ValueError('A source article has inconsistent task or reference definitions')

        # 2. Unpivot the two rater groups and four dimensions, retaining rater positions.
        ratings = summaries.melt(id_vars=['source_row', 'id'], value_vars=list(parameters['groups']),
            var_name='rater_group', value_name='annotation').explode('annotation', ignore_index=True)
        ratings['trial'] = ratings.groupby(['source_row', 'rater_group']).cumcount() + 1
        values = pd.json_normalize(ratings.pop('annotation')).set_axis(ratings.index)
        ratings = ratings.join(values).melt(id_vars=['source_row', 'id', 'rater_group', 'trial'],
            value_vars=list(parameters['dimensions']), var_name='dimension', value_name='response')
        ratings = ratings.merge(summaries.drop(columns=['id']), on='source_row', validate='many_to_one')
        ratings['response_key'] = (ratings.source_row.astype(str) + '::' + ratings.rater_group + '::'
            + ratings.trial.astype(str) + '::' + ratings.dimension)

        # 3. Identify source model aliases without guessing historical checkpoints.
        subjects = pd.DataFrame({name: parameters[key] for name, key in [('model_name', 'model_names'),
            ('model_paper', 'model_papers'), ('source_title', 'model_titles'), ('model_type', 'model_types')]
            }).rename_axis('subject_key').reset_index()
        subjects = subjects.loc[subjects.subject_key.isin(ratings.model_id)].copy()
        subjects['raw_label'] = 'SummEval ' + subjects.subject_key + ' - ' + subjects.model_name
        subjects['features'] = [dict(harness=labels['harness'], source_model_alias=row.subject_key,
            source_model_name=row.model_name, model_paper=row.model_paper, source_title=row.source_title,
            model_type=row.model_type, configuration_status=labels['configuration_status']) for row in subjects.itertuples()]
        ratings['subject_key'] = ratings.model_id

        # 4. Separate article/dimension/rater-group grading protocols in item identity.
        definition = ['id', 'rater_group', 'dimension']
        items = ratings[definition + ['content', 'story_path', 'reference_json']].drop_duplicates().reset_index(drop=True)
        items['item_key'] = items.index
        items['raw_item_id'] = items.id + '::' + items.rater_group + '::' + items.dimension
        items['features'] = [dict(source_article_id=row.id, article_file=row.story_path, dimension=row.dimension,
            rater_group=parameters['groups'][row.rater_group], input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=row.reference_json,
            rule=parameters['dimensions'][row.dimension] + ' ' + self.grading['rule']) for row in items.itertuples()]
        items['verifier'] = [Judge(judged_by='human', spec=json.dumps(dict(self.grading['verifiers']['human'],
            dimension=row.dimension, rater_group=parameters['groups'][row.rater_group],
            rater_procedure=parameters['group_procedures'][row.rater_group]), sort_keys=True)) for row in items.itertuples()]
        ratings = ratings.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')

        # 5. Keep complete native records, including the shared generated summary.
        traces = ratings[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['annotations'], source_row=int(row.source_row),
            source_record=row.source_record, rater_group=row.rater_group, rater_index=int(row.trial) - 1,
            dimension=row.dimension, trial_scope=labels['trial_scope']), ensure_ascii=False, allow_nan=False)
            for row in ratings.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': ratings[['response_key', 'subject_key', 'item_key', 'response', 'trial']], 'traces': traces}


if __name__ == '__main__':
    SummEval(__file__).main_from_args()
