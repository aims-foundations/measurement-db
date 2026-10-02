"""Tabulate RClicks' original clickability metrics and complete visual stimuli."""

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class RClicks(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters = self.build_parameters
        layout, metrics = parameters['layout'], self.grading['verifiers']

        # 1. Read every released per-stimulus metric row and its original position.
        frames = []
        for dataset in parameters['archives']:
            source_file = layout['results'] + '/' + dataset + '_per_image.csv'
            frame = pd.read_csv(self.raw_dir / source_file, keep_default_na=False, float_precision='round_trip')
            if list(frame.columns) != ['full_stem', 'model_name', 'click_type', *metrics]:
                raise ValueError('The native per-stimulus columns differ from the declared release')
            frames.append(frame.assign(dataset=dataset, source_file=source_file, source_row=frame.index))
        records = pd.concat(frames, ignore_index=True)
        if records.duplicated(['dataset', 'full_stem', 'model_name']).any():
            raise ValueError('Duplicate native model/stimulus observations')
        if not records.model_name.isin(parameters['models']).all() or not np.isfinite(records[list(metrics)]).all().all():
            raise ValueError('Unknown model identifier or non-finite native metric')
        records['stimulus_key'] = records.dataset + '/' + records.full_stem

        # 2. Join the original click-state descriptors and human reference clicks.
        clicks = pd.read_csv(self.raw_dir / layout['clicks'], keep_default_na=False)
        state_fields = ['dataset', 'full_stem', *parameters['stimulus_fields']]
        states = clicks[state_fields].drop_duplicates()
        stimuli = records[['dataset', 'full_stem', 'click_type', 'stimulus_key']].drop_duplicates()
        stimuli = stimuli.merge(states, on=['dataset', 'full_stem', 'click_type'], how='left', validate='one_to_one')
        if stimuli.image_stem.isna().any():
            raise ValueError('A published score has no unambiguous original click state')
        validation = (self.raw_dir / layout['validation']).read_text().splitlines()
        if not stimuli.loc[stimuli.dataset.eq('TETRIS'), 'image_stem'].isin(validation).all():
            raise ValueError('A TETRIS result falls outside the original validation split')
        references = clicks.merge(stimuli[['dataset', 'full_stem', 'stimulus_key']], on=['dataset', 'full_stem'], validate='many_to_one')
        references['click'] = references[['x', 'y', 'w', 'h', 'device']].to_dict('records')
        stimuli = stimuli.join(references.groupby('stimulus_key', sort=False).click.agg(list).rename('human_clicks'), on='stimulus_key')

        # 3. Attach unchanged images and masks by explicit, validated archive joins.
        asset_frames = []
        for dataset, archive_path in parameters['archives'].items():
            selected = stimuli.loc[stimuli.dataset.eq(dataset)]
            with ZipFile(self.raw_dir / archive_path) as archive:
                members = pd.DataFrame({'member': [name for name in archive.namelist() if not name.endswith('/')]})
                members['image_stem'] = members.member.map(lambda value: Path(value).stem)
                for role in ['image', 'mask']:
                    available = members.loc[members.member.str.startswith(parameters[role + '_prefixes'][dataset])]
                    matched = selected[['stimulus_key', 'image_stem']].merge(available, on='image_stem', how='left', validate='many_to_one')
                    if matched.member.isna().any():
                        raise ValueError('Missing original image or object mask')
                    payloads = {name: archive.read(name) for name in matched.member.unique()}
                    asset_frames.append(matched.assign(data=matched.member.map(payloads), role=role, archive=archive_path))
            previous = selected.loc[selected.click_type.ne('first'), ['stimulus_key', 'full_stem']].copy()
            previous['member'] = parameters['previous_mask_prefixes'][dataset] + previous.full_stem + '.png'
            with ZipFile(self.raw_dir / layout['previous_masks']) as archive:
                previous['data'] = previous.member.map(archive.read)
            asset_frames.append(previous.assign(role='previous_mask', archive=layout['previous_masks']))
        assets = pd.concat(asset_frames, ignore_index=True)
        assets['sha256'] = assets.data.map(lambda value: hashlib.sha256(value).hexdigest())
        assets['extension'] = assets.member.map(lambda value: Path(value).suffix.lower())
        assets['path'] = 'images/' + assets.sha256 + assets.extension
        assets['media_type'] = assets.extension.map(parameters['mime_types'])
        if assets.media_type.isna().any():
            raise ValueError('Unknown original image encoding')
        assets['attachment'] = assets[['data', 'path', 'media_type']].assign(role='input').to_dict('records')
        assets['element'] = assets[['path', 'media_type', 'role']].rename(columns={'path': 'location', 'media_type': 'content_type'}).to_dict('records')
        stimuli = stimuli.join(assets.groupby('stimulus_key', sort=False).attachment.agg(list).rename('attachments'), on='stimulus_key')
        stimuli = stimuli.join(assets.groupby('stimulus_key', sort=False).element.agg(list).rename('multimedia_elements'), on='stimulus_key')
        descriptions = stimuli[['dataset', 'full_stem', *parameters['stimulus_fields'], 'multimedia_elements']].to_dict('records')
        stimuli['content'] = [json.dumps(dict(text=parameters['labels']['instruction'], **row), sort_keys=True) for row in descriptions]

        # 4. Melt the metric columns; each metric has its own grading rule and scale.
        long = records.melt(id_vars=['dataset', 'full_stem', 'click_type', 'model_name', 'stimulus_key', 'source_file', 'source_row'],
                            value_vars=list(metrics), var_name='metric', value_name='response')
        long['item_key'] = long.stimulus_key + ':' + long.metric
        long['subject_key'] = long.model_name
        long['response_key'] = long.item_key + ':' + long.model_name
        items = long[['item_key', 'stimulus_key', 'metric']].drop_duplicates().merge(stimuli, on='stimulus_key', validate='many_to_one')
        items['raw_item_id'] = items.item_key
        items['features'] = items[['dataset', 'full_stem', 'click_type']].to_dict('records')
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + ' ' + metrics[row.metric]['rule'],
            reference_answer=json.dumps(dict(human_clicks=row.human_clicks), sort_keys=True),
            response_scale=metrics[row.metric]['response_scale']) for row in items.itertuples()]
        items['verifier'] = [ExactMatcher(spec=json.dumps(metrics[name]['implementation'], sort_keys=True)) for name in items.metric]
        subjects = pd.DataFrame({'subject_key': sorted(records.model_name.unique())})
        subjects['raw_label'] = 'RClicks ' + subjects.subject_key
        subjects['features'] = subjects.subject_key.map(lambda name: dict(native_identifier=name, algorithm=parameters['models'][name]))
        responses = long[['response_key', 'subject_key', 'item_key', 'response']].assign(test_condition=parameters['labels']['condition'])

        # 5. Keep source metric rows as audit evidence, without inventing model traces.
        evidence = records[['source_file', 'source_row', 'dataset', 'full_stem', 'model_name', 'click_type', *metrics]].to_dict('records')
        records['native_row'] = [json.dumps(row, sort_keys=True, allow_nan=False) for row in evidence]
        traces = long[['response_key', 'source_file', 'source_row']].merge(
            records[['source_file', 'source_row', 'native_row']], on=['source_file', 'source_row'], validate='many_to_one')
        traces = traces[['response_key', 'native_row']].rename(columns={'native_row': 'trace'})
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
                    responses=responses, traces=traces)


if __name__ == '__main__':
    RClicks(__file__).main_from_args()
