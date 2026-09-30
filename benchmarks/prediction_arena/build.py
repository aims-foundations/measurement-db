#!/usr/bin/env python3
"""Tabulate the captured Prediction Arena settlements and original Kalshi definitions."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PredictionArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the original roster and settlements, preserving each native record.
        agents = pd.read_json(self.raw_dir / parameters['layout']['agents'], convert_dates=False).rename(
            columns={'id': 'agent_id'})
        inputs = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['settlements'])):
            native = json.loads(path.read_text())
            frame = pd.json_normalize(native, max_level=0).rename_axis('source_row').reset_index()
            inputs.append(frame.assign(agent_id=path.stem, native_record=native,
                source_file=str(path.relative_to(self.raw_dir))))
        settlements = pd.concat(inputs, ignore_index=True).rename(columns={'id': 'response_key'})
        settlements = settlements.merge(agents[['agent_id', 'model_id']], on='agent_id', validate='many_to_one')

        # 2. Join each settlement to its original market definition, not a decoded ticker.
        paths = sorted(self.raw_dir.glob(parameters['layout']['markets']))
        documents = [json.loads(path.read_text()) for path in paths]
        if any(document['cursor'] or len(document['markets']) != 1 for document in documents):
            raise ValueError('Each captured market query must contain exactly one complete result')
        definitions = pd.DataFrame(dict(response_key=[path.stem for path in paths],
            market=[document['markets'][0] for document in documents],
            market_source_file=[str(path.relative_to(self.raw_dir)) for path in paths]))
        responses = settlements.merge(definitions, on='response_key', how='left', validate='one_to_one')
        if responses.market.isna().any() or not responses.ticker.eq(responses.market.map(lambda market: market['ticker'])).all():
            raise ValueError('A settlement is missing its matching market definition')
        if not responses.result.eq(responses.market.map(lambda market: market['result'])).all():
            raise ValueError('The market resolution differs from the original settlement record')

        # 3. Separate stable market inputs from later resolution and account statistics.
        responses['content'] = responses.market.map(lambda market: json.dumps(
            {field: market.get(field) for field in ['ticker', 'event_ticker', 'title', 'subtitle', 'yes_sub_title',
                'no_sub_title', 'close_time', 'rules_primary', 'rules_secondary']}, ensure_ascii=False, sort_keys=True))
        if responses.groupby('ticker').content.nunique().gt(1).any():
            raise ValueError('Repeated market references contain conflicting descriptions or rules')
        items = responses.drop_duplicates('ticker').rename(columns={'ticker': 'raw_item_id'}).copy()
        items['item_key'] = items.raw_item_id
        items['features'] = [dict(platform=parameters['labels']['platform'], input_scope=parameters['labels']['input_scope'])
            for _ in items.index]
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['settlement_profit'], sort_keys=True))
        subjects = agents[['agent_id', 'model_id']].rename(columns={'agent_id': 'subject_key', 'model_id': 'raw_label'})
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_agent_id=row.subject_key,
            source_model_id=row.raw_label) for row in subjects.itertuples()]

        # 4. Preserve the paper's profit criterion without equating it with forecast accuracy.
        profit = pd.to_numeric(responses.realized_pnl, errors='raise')
        if profit.isna().any() or profit.isin([float('inf'), float('-inf')]).any():
            raise ValueError('Each recorded settlement must have a finite realized profit')
        responses['response'] = profit.gt(0).astype(float)
        responses['subject_key'] = responses.agent_id
        responses['item_key'] = responses.ticker
        responses['test_condition'] = parameters['labels']['test_condition_prefix'] + responses.response_key

        # 5. Keep complete settlement records and their original market provenance.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_agent_id=row.agent_id, source_model_id=row.model_id, settlement=row.native_record,
            market_source_file=row.market_source_file, market=row.market), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    PredictionArena(__file__).main_from_args()
