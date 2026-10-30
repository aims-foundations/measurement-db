"""Check recovered Reasoning Gym comparisons and every original completion."""

from collections import Counter
import hashlib
import json
import math
import re
import unicodedata

import pandas as pd

from .reasoning_native_checks import NativeComparisons


_FILES = dict(polynomial_equations='algebra/polynomial_equations.py', cryptarithm='algorithmic/cryptarithm.py', game_of_life='algorithmic/game_of_life.py',
    letter_jumble='algorithmic/letter_jumble.py', sentence_reordering='algorithmic/sentence_reordering.py',
    figlet_font='cognition/figlet_fonts.py', emoji_mystery='games/emoji_mystery.py',
    futoshiki='games/futoshiki.py', mini_sudoku='games/mini_sudoku.py', sudoku='games/sudoku.py')


def _references(name, text):
    if name == 'polynomial_equations':
        roots = []
        for value in text.split(','):
            try:
                roots.append(float(value.strip()))
            except ValueError:
                continue
        return {**{f'root:{i}': dict(position=i, expected=v) for i, v in enumerate(sorted(roots))},
            'extra_roots': dict(expected=0, measurement='unmatched predicted roots')}
    if name == 'cryptarithm':
        return {'mapping:' + pair.split('=')[0]: dict(letter=pair.split('=')[0], expected=int(pair.split('=')[1]))
            for pair in text.split(',')}
    if name in {'futoshiki', 'mini_sudoku', 'sudoku'}:
        grid = [[int(value) for value in re.findall('[1-9]', line)] for line in text.splitlines()]
        grid = [row for row in grid if row]
        return {f'cell:{i}:{j}': dict(row=i, column=j, expected=value) for i, row in enumerate(grid) for j, value in enumerate(row)}
    if name == 'game_of_life':
        grid = json.loads(text)
        if grid and isinstance(grid[0], list):
            return {f'cell:{i}:{j}': dict(row=i, column=j, expected=value) for i, row in enumerate(grid) for j, value in enumerate(row)}
        return {f'cell:{i}': dict(position=i, expected=value) for i, value in enumerate(grid)}
    if name in {'letter_jumble', 'sentence_reordering'}:
        words = text.lower().split() if name == 'letter_jumble' else text.split()
        return {f'word:{i}': dict(position=i, expected=value) for i, value in enumerate(words)}
    if name in {'figlet_font', 'emoji_mystery'}:
        text = text.strip().lower() if name == 'figlet_font' else text
        return {f'character:{i}': dict(position=i, expected=value) for i, value in enumerate(text)}
    return {}


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _reasoning_gym
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.hash_measurement_ids import canonical_grading_criterion, item_id_from_content
    from ..build_measurement_tables.normalize_evaluation_settings import canonicalize_features, features_string
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='Reasoning Gym recovered native comparisons')
    parameters = metadata['build']['parameters']
    protocol = metadata['grading']['verifiers']['components']
    raw = directory / 'raw'
    root = raw / parameters['layout']['results']
    definitions, original_ids, native, scorers = {}, {}, {}, {}
    roster = set()
    for path in sorted(root.glob('*/*/summary.json')):
        run = path.parent.relative_to(root).as_posix()
        if run == parameters['exclusions']['ambiguous_identity_run']:
            continue
        summary = json.loads(path.read_bytes())
        for source_file in sorted(path.parent.glob('*/*.json')):
            data = json.loads(source_file.read_bytes())
            dataset, revision = data['name'], summary['git_hash']
            source_path = 'protocol/' + revision + '/reasoning_gym/' + _FILES[dataset] if dataset in _FILES else None
            if source_path and source_path not in scorers:
                payload = (raw / source_path).read_bytes()
                scorers[source_path] = NativeComparisons(dataset, payload.decode()), hashlib.sha256(payload).hexdigest()
            for position, record in enumerate(data['results']):
                alias = source_file.relative_to(raw).as_posix() + ':' + str(position)
                content = json.dumps(dict(system_prompt=data['system_prompt'], question=record['question']), ensure_ascii=False, sort_keys=True)
                criterion = dict(reference_answer=str(record['expected_answer']), rule=metadata['grading']['rule'])
                spec = dict(metadata['grading']['verifiers']['algorithmic'], dataset=dataset,
                    configuration=data['config'], revision=revision)
                verifier = {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)}
                features = features_string(canonicalize_features(dict(dataset=dataset, category=data['category'], difficulty=run.split('/')[0])))
                parent = item_id_from_content('reasoning_gym', content, features, verifier=json.dumps(verifier),
                    grading_criterion=criterion, response_scale=protocol['aggregate_scale'])
                original_ids[alias] = parent
                checks = _references(dataset, record['expected_answer'])
                definitions.setdefault(parent, dict(alias=alias, content=content, criterion=criterion, verifier=verifier,
                    features=features, checks=checks, source_path=source_path, dataset=dataset,
                    configuration=data['config'], revision=revision))
                roster.update((parent, component) for component in checks)
                for index, completion in enumerate(record['completions']):
                    key = alias + ':' + str(index)
                    grade = None if completion.get('error') is not None else min(1., float(completion['score']))
                    flags = None
                    if checks:
                        if grade is None:
                            flags = dict.fromkeys(checks, None)
                        else:
                            computed, flags = scorers[source_path][0].score(completion['model_answer'], record['expected_answer'], checks, data['config'])
                            _check(math.isclose(computed, completion['score'], abs_tol=1e-12, rel_tol=1e-12), True,
                                'Reasoning Gym pinned scorer reproduces the original aggregate before extraction')
                    if flags is None:
                        component = ''
                        flags = {component: grade}
                        roster.add((parent, component))
                    native[key] = parent, flags

    parents, mapping, actual_roster = {}, {}, Counter()
    for item in tables['items'].to_dict('records'):
        alias, _, component = item['raw_item_id'].partition('::')
        parent = original_ids[alias]
        original = definitions[parent]
        # A fallback task item may first appear in a later completion export.
        # Its recorded alias must resolve to this same complete original task.
        _check((item['content'], item['item_features']), (original['content'], original['features']),
            'Reasoning Gym complete prompt and attributes on every comparison')
        _check(item['testlet_id'], hashlib.sha256(unicodedata.normalize('NFC', item['content']).encode()).hexdigest(),
            'Reasoning Gym shared original stimulus')
        criterion = dict(original['criterion'], response_scale=protocol['aggregate_scale'])
        verifier = original['verifier']
        if component:
            _check(component in original['checks'], True, 'Reasoning Gym source comparison coordinate')
            criterion = dict(reference_answer=json.dumps(original['checks'][component], ensure_ascii=False, sort_keys=True),
                rule=protocol['root_rule'] if original['dataset'] == 'polynomial_equations' else protocol['rule'],
                response_scale=(protocol['extra_count_scale'] if component == 'extra_roots' else protocol['aggregate_scale'])
                    if original['dataset'] == 'polynomial_equations' else protocol['binary_scale'])
            verifier = {'class': 'exact_matcher', 'spec': json.dumps(dict(kind='reconstructed_native_comparison',
                dataset=original['dataset'], configuration=original['configuration'], revision=original['revision'],
                source_file=original['source_path'], source_sha256=scorers[original['source_path']][1], component_id=component), sort_keys=True)}
        _check(json.loads(item['grading_criterion']), json.loads(canonical_grading_criterion(criterion)),
            'Reasoning Gym exact component reference, rule and scale')
        _check(json.loads(item['verifier']), verifier, 'Reasoning Gym pinned parser and source hash')
        _check(pd.isna(item['asset_manifest']), True, 'Reasoning Gym no invented assets')
        parents.setdefault(parent, dict(item, item_id=parent, raw_item_id=original['alias'],
            grading_criterion=canonical_grading_criterion(original['criterion']), verifier=json.dumps(original['verifier'])))
        mapping[item['item_id']] = parent
        actual_roster[parent, component] += 1
    _check(actual_roster, Counter({key: 1 for key in roster}), 'Reasoning Gym complete source comparison roster and retained aggregates')
    attempts = tables['attempts'].set_index('attempt_id').raw_attempt_id.to_dict()
    _check(Counter(attempts.values()), Counter({key: 1 for key in native}), 'Reasoning Gym each original completion exactly once')
    expected = Counter((attempt, native[key][0], component, grade)
        for attempt, key in attempts.items() for component, grade in native[key][1].items())
    items = tables['items'].set_index('item_id').raw_item_id.to_dict()
    actual = Counter((row.attempt_id, mapping[row.item_id], items[row.item_id].partition('::')[2],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'Reasoning Gym every native comparison, format fallback and unknown result')
    projected = parent_view(tables, list(parents.values()), mapping)
    projected['benchmarks'] = projected['benchmarks'].assign(response_type='continuous_bounded',
        response_scale=json.dumps(protocol['aggregate_scale']))
    stats = _reasoning_gym(directory, projected, metadata)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=tables['items'].testlet_id.nunique())
