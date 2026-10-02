#!/usr/bin/env python3
"""Tabulate released Risk games, player configurations and complete available logs."""

import json
from pathlib import Path
import sys
import tarfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class LiveAgentRisk(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Load the released JSON records; select completed games with AI players.
        with tarfile.open(self.raw_dir / layout['results'], 'r:gz') as archive:
            files = {entry.name.removeprefix('./'): archive.extractfile(entry).read()
                for entry in archive.getmembers() if entry.isfile()}
        documents = pd.DataFrame([dict(source_file=name, record=json.loads(body))
            for name, body in sorted(files.items()) if name.endswith('.json')])
        lookup = documents.set_index('source_file').record.to_dict()
        games = documents.loc[documents.source_file.str.endswith('/end_game_results.json')].copy()
        games['game'] = games.source_file.str.removesuffix('/end_game_results.json')
        games['players'] = games.record.map(lambda record: record['players'])
        if not all(record['players'] and record['winner'] in [player['name'] for player in record['players']]
                and len({player['name'] for player in record['players']}) == len(record['players']) for record in games.record):
            raise ValueError('A completed game must identify its winner within a unique player roster')
        games = games.loc[~games.players.map(lambda players: all(
            player['name'].lower() in parameters['placeholder_seats'] for player in players))].copy()
        games['manifest'] = games.game.map(lambda game: lookup.get(game + '/game_manifest.json'))
        supplements = pd.DataFrame([dict(game=lookup[name]['game_folder'].removeprefix(layout['game_root']),
            supplement_file=name, supplement=lookup[name], setup_players=lookup[name][field])
            for name, field in parameters['supplements'].items()])
        games = games.merge(supplements, on='game', how='left', validate='one_to_one')
        for field in ['manifest', 'supplement', 'supplement_file']:
            games[field] = games[field].astype(object).where(games[field].notna(), None)

        # 2. Join all native model-call and turn logs by game path and player name.
        turns = documents.loc[documents.source_file.str.contains(parameters['patterns']['turn'])].copy()
        turns['game'] = turns.source_file.str.rsplit('/', n=1).str[0]
        turns['seat'] = turns.record.map(lambda record: record['player']['name'])
        calls = documents.loc[documents.source_file.str.contains('/?llm_interactions/', regex=True)].copy()
        parts = calls.source_file.str.extract(parameters['patterns']['interaction_game'])
        calls['game'] = parts.parent.fillna('') + parts.game_name
        calls['seat'] = calls.record.map(lambda record: record['player'])
        turns = turns.merge(games[['game']], on='game', validate='many_to_one')
        calls = calls.merge(games[['game']], on='game', validate='many_to_one')
        for frame in [turns, calls]:
            frame['entry'] = [dict(source_file=row.source_file, record=row.record) for row in frame.itertuples()]
        turn_groups = turns.groupby(['game', 'seat'], sort=False).entry.agg(list).to_dict()
        call_groups = calls.groupby(['game', 'seat'], sort=False).entry.agg(list).to_dict()

        # 3. Combine compatible declarations, preserving unknown settings as unknown.
        # Null means unspecified; two different non-null declarations are an error.
        players = games[['game', 'players']].explode('players', ignore_index=True)
        players['seat'] = players.players.map(lambda player: player['name'])
        players['seat_position'] = players.groupby('game', sort=False).cumcount()
        players = players.merge(games.drop(columns=['players']), on='game', validate='many_to_one')
        configurations, features, labels = [], [], []
        for row in players.itertuples():
            declared = row.manifest['players'] if row.manifest else (row.setup_players if isinstance(row.setup_players, list) else [])
            evidence = [player for player in declared if player['name'] == row.seat]
            evidence += [entry['record']['player'] for entry in turn_groups.get((row.game, row.seat), [])]
            evidence += [{key: entry['record'][key] for key in ['model', 'provider']}
                for entry in call_groups.get((row.game, row.seat), [])]
            configuration = {}
            for description in evidence:
                for key, value in description.items():
                    if key == 'name':
                        continue
                    if configuration.get(key) is not None and value is not None and configuration[key] != value:
                        raise ValueError('Conflicting released player configuration: ' + row.game + '/' + row.seat + '/' + key)
                    if key not in configuration or configuration[key] is None:
                        configuration[key] = value
            configurations.append(configuration)
            labels.append(configuration.get('model') or row.seat)
            features.append(dict(harness=parameters['labels']['harness'],
                reasoning_effort=configuration.get('reasoning_effort'),
                source_configuration=json.dumps(configuration, sort_keys=True),
                identity_evidence='declared_model' if configuration.get('model') else 'source_seat_label',
                source_seat_label=None if configuration.get('model') else row.seat,
                reported_harness_revision=row.manifest.get('git_revision') if row.manifest else None))
        players['configuration'], players['features'], players['raw_label'] = configurations, features, labels
        players['subject_key'] = [json.dumps(dict(label=label, features=feature), sort_keys=True)
            for label, feature in zip(labels, features)]
        subjects = players[['subject_key', 'raw_label', 'features']].drop_duplicates('subject_key')
        players['roster_entry'] = [dict(name=row.seat, configuration=row.configuration) for row in players.itertuples()]
        rosters = players.groupby('game', sort=False).roster_entry.agg(list).to_dict()

        # 4. Describe the released setup; post-placement boards stay in the trace.
        with tarfile.open(self.raw_dir / layout['source'], 'r:gz') as archive:
            resources = {name: archive.extractfile(layout['source_prefix'] + name).read()
                for name in sorted(parameters['resources'].values())}
        content = []
        for row in games.itertuples():
            rules = row.manifest['rules'] if row.manifest else (row.supplement['config'] if row.supplement else None)
            content.append(json.dumps(dict(task=parameters['labels']['task'], game_instance=row.game,
                released_roster=rosters[row.game], known_rules=rules,
                include_initial_troop_placement=row.manifest.get('include_initial_troop_placement') if row.manifest else None,
                reported_harness_revision=row.manifest.get('git_revision') if row.manifest else None,
                reference_code_revision=parameters['labels']['reference_revision'],
                context_scope=parameters['labels']['context_scope']), ensure_ascii=False, sort_keys=True))
        items = games.assign(item_key=games.game, raw_item_id=games.game, content=content)
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in range(len(items))]
        items['verifier'] = [ExactMatcher(spec=json.dumps(self.grading['verifiers']['win'], sort_keys=True)) for _ in range(len(items))]
        items['attachments'] = [[dict(path=name, data=body, media_type='text/x-python', role='task_program_reference')
            for name, body in resources.items()] for _ in range(len(items))]

        # 5. Retain every seat's win/loss and all available original interaction records.
        # A repeated model may occupy several seats; these are separate observations.
        history = {game: {name: body.decode('utf-8') for name, body in files.items()
            if name.startswith(game + '/') and name.endswith(('.csv', '.json'))
            and not name.endswith(('/end_game_results.json', '/game_manifest.json'))
            and '/turn_summary_' not in name} for game in games.game}
        players['response_key'] = players.source_file + '#seat=' + players.seat
        players['item_key'] = players.game
        players['response'] = [float(row.record['winner'] == row.seat) for row in players.itertuples()]
        players['trial'] = players.groupby(['subject_key', 'item_key'], sort=False).cumcount() + 1
        players['test_condition'] = players.response_key
        players['trace'] = [json.dumps(dict(source_file=row.source_file, seat=row.seat, seat_position=row.seat_position,
            record=row.record, manifest=row.manifest, supplement_file=row.supplement_file, supplement=row.supplement,
            game_history=history[row.game], turns=turn_groups.get((row.game, row.seat), []),
            interactions=call_groups.get((row.game, row.seat), [])), ensure_ascii=False, allow_nan=False)
            for row in players.itertuples()]
        return dict(subjects=subjects,
            items=items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier', 'attachments']],
            responses=players[['response_key', 'subject_key', 'item_key', 'response', 'trial', 'test_condition']],
            traces=players[['response_key', 'trace']])


if __name__ == '__main__':
    LiveAgentRisk(__file__).main_from_args()
