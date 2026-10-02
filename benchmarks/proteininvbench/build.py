"""Tabulate native ProteinInvBench predictions and their original CATH backbones."""

import io
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.read_native_pickle import read_native_pickle, native_json_value


class ProteinInvBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, labels = parameters['paths'], parameters['labels']

        # 1. Read saved tensors as data, retaining their vocabulary and run configuration.
        frames, configurations = [], []
        with tarfile.open(self.raw_dir / paths['predictions'], 'r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith('/model_param.json'):
                    configurations.append(dict(run=member.name.rsplit('/', 1)[0],
                        configuration=json.load(archive.extractfile(member))))
                match = re.fullmatch(paths['prediction_pattern'], member.name)
                if not match or match['dataset'] not in parameters['datasets']:
                    continue
                source = read_native_pickle(io.BytesIO(archive.extractfile(member).read()))
                frame = pd.DataFrame({key: source[key] for key in ['title', 'true_seq', 'pred_probs']})
                vocabulary = source['tokenizer'].state['all_tokens']
                if source['tokenizer'].state['_token_to_id'] != {token: index for index, token in enumerate(vocabulary)}:
                    raise ValueError('Recorded protein vocabulary mappings disagree')
                frames.append(frame.assign(dataset=match['dataset'], model=match['model'], member=member.name,
                    run=member.name.rsplit('/', 1)[0], source_row=frame.index + 1,
                    vocabulary=[vocabulary] * len(frame)))
        if not frames:
            raise ValueError('No saved unperturbed CATH predictions are available')
        responses = pd.concat(frames, ignore_index=True).merge(pd.DataFrame(configurations),
            on='run', validate='many_to_one')
        if responses.duplicated(['run', 'title']).any() or any(config['augment_eps'] != 0 for config in responses.configuration):
            raise ValueError('Expected one unperturbed saved prediction per run and protein')
        responses['preprocessing'] = responses.model.map(parameters['preprocessing'])
        if responses.preprocessing.isna().any():
            raise ValueError('The model has no documented native coordinate layout')
        responses['subject_key'] = responses.run
        responses['response_key'] = responses.member + ':' + responses.source_row.astype(str)

        # 2. Join the original structures by dataset and protein identifier.
        inputs = []
        with tarfile.open(self.raw_dir / paths['inputs'], 'r|gz') as archive:
            datasets = {member: dataset for dataset, member in parameters['datasets'].items()}
            for member in archive:
                if member.name in datasets:
                    # pandas.read_json changes native NaN coordinates to null.
                    frame = pd.json_normalize(map(json.loads, archive.extractfile(member)), max_level=0)
                    inputs.append(frame.assign(dataset=datasets[member.name], input_member=member.name))
        backbone = pd.concat(inputs, ignore_index=True).rename(columns={'name': 'title', 'seq': 'native_sequence'})
        responses = responses.merge(backbone[['dataset', 'title', 'native_sequence', 'coords', 'input_member']],
            on=['dataset', 'title'], how='left', validate='many_to_one', indicator=True)
        if not responses._merge.eq('both').all():
            raise ValueError('A saved protein prediction has no original backbone')
        responses['item_key'] = responses.dataset + '/' + responses.title + '/' + responses.preprocessing

        # 3. Compute the published notebook's recovery without changing its padding rule.
        for row in responses.itertuples():
            probability, target = np.asarray(row.pred_probs), np.asarray(row.true_seq)
            if (target.ndim != 1 or not len(target) or probability.shape != (len(target), len(row.vocabulary))
                    or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()):
                raise ValueError('Invalid native prediction tensor dimensions or probabilities')
        responses['response'] = [float(np.mean(np.asarray(target) == np.asarray(probability).argmax(axis=1), dtype=np.float32))
                                for target, probability in zip(responses.true_seq, responses.pred_probs)]
        responses['test_condition'] = 'dataset=' + responses.dataset + ';metric=sequence_recovery;scope=' + labels['scope']

        # 4. Keep backbone inputs separate from reference sequences and target layouts.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.dataset + '/' + items.title
        items['content'] = [json.dumps(dict(task=labels['task'], backbone=native_json_value(coords)), allow_nan=False)
                            for coords in items.coords]
        positions = []
        for row in items.itertuples():
            coordinates = np.stack([np.asarray(row.coords[atom], dtype=float) for atom in ['N', 'CA', 'C', 'O']], axis=1)
            valid = np.flatnonzero(np.isfinite(coordinates.sum(axis=(1, 2)))).tolist()
            positions.append(list(range(len(row.native_sequence))) if row.preprocessing == 'full'
                else valid + ([-1] * (len(row.native_sequence) - len(valid)) if row.preprocessing == 'finite_padded' else []))
        items['reference'] = [json.dumps(dict(amino_acid_sequence=row.native_sequence, token_ids=row.true_seq.tolist(),
            vocabulary=row.vocabulary, coordinate_positions=position), allow_nan=False)
            for row, position in zip(items.itertuples(), positions)]
        items['features'] = [dict(dataset=row.dataset, source_protein=row.title) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=reference, rule=self.grading['rule']) for reference in items.reference]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['native'], sort_keys=True))] * len(items)

        # 5. Preserve complete configurations and full native prediction matrices.
        subjects = responses.drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=labels['harness'], source_release=labels['source_release'], source_run=row.run,
            preprocessing=row.preprocessing, configuration=json.dumps(row.configuration, sort_keys=True)
                .replace(';', r'\u003b').replace('=', r'\u003d')) for row in subjects.itertuples()]
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=paths['predictions'], source_member=row.member, source_row=row.source_row,
            input_archive=paths['inputs'], input_member=row.input_member, configuration_member=row.run + '/model_param.json',
            record=dict(title=row.title, true_seq=row.true_seq.tolist(), pred_probs=row.pred_probs.tolist()),
            vocabulary=row.vocabulary), allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    ProteinInvBench(__file__).main_from_args()
