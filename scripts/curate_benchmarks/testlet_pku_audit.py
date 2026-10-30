"""Audit all category labels and preserve the full original PKU comparison audit."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import unicodedata

import pandas as pd


def verify(directory, tables, metadata, source=None):
    from .native_result_audits import _check, _pku_saferlhf, _pku_saferlhf_sources
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='PKU-SafeRLHF released category labels')
    source = source or _pku_saferlhf_sources(directory, metadata)
    labels = metadata['build']['parameters']['labels']
    safety, protocol = (metadata['grading']['verifiers'][key] for key in ['safety', 'harm_category'])
    categories = safety['harm_categories']
    items = tables['items'].set_index('item_id').to_dict('index')
    definitions = Counter()
    for item in items.values():
        prompt = unicodedata.normalize('NFC', item['content']).strip()
        raw, category = item['raw_item_id'].rsplit('::', 1)
        _check(category in categories, True, 'PKU original category vocabulary')
        _check((raw, item['content']), source['prompts'][prompt], 'PKU complete original prompt for every category')
        _check(item['testlet_id'], hashlib.sha256(prompt.encode()).hexdigest(), 'PKU shared prompt membership')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=None,
            rule=json.dumps(dict(rule=protocol['rule'], category=category), sort_keys=True)), 'PKU named category rule')
        _check(json.loads(item['verifier']), {'class': 'judge', 'judge': labels['judge'], 'judged_by': 'human',
            'spec': json.dumps(dict(protocol, category=category), sort_keys=True)}, 'PKU annotation provenance')
        definitions[prompt, category] += 1
    _check(definitions, Counter({(prompt, category): 1 for prompt in source['prompts'] for category in categories}),
        'PKU each original category exactly once per prompt')

    originals, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    original_records = {}
    for row in originals.itertuples():
        trace = json.loads(traces[row.response_id])
        name, index, side = trace['source_file'], trace['source_row'], trace['side']
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], name + ':' + str(index) + ':' + str(side),
            'PKU original annotation occasion, without invented generation identities')
        original_records[row.attempt_id] = source['native'][name, index], side
    seen = Counter()
    for row in tables['responses'].itertuples():
        record, side = original_records[row.attempt_id]
        item = items[row.item_id]
        category = item['raw_item_id'].rsplit('::', 1)[1]
        _check(unicodedata.normalize('NFC', item['content']).strip(),
            unicodedata.normalize('NFC', record['prompt']).strip(), 'PKU response-to-prompt association')
        _check(row.response, float(not record[f'response_{side}_harm_category'][category]),
            'PKU literal native category flag, without overriding the overall label')
        seen[row.attempt_id, category] += 1
    _check(seen, Counter({(key, category): 1 for key in original_records for category in categories}),
        'PKU every source-side category exactly once')

    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    parent_features = parents.set_index('testlet_id').item_features.to_dict()
    for item in items.values():
        _check(item['item_features'], parent_features[item['testlet_id']], 'PKU attributes retained on every category')
        _check(pd.isna(item['asset_manifest']), True, 'PKU no invented input resources')
    parents['raw_item_id'] = parents.content.map(lambda text: source['prompts'][unicodedata.normalize('NFC', text).strip()][0])
    parents['grading_criterion'] = json.dumps(dict(reference_answer=None, rule=safety['aggregate_rule']))
    parents['verifier'] = json.dumps({'class': 'judge', 'judge': labels['judge'], 'judged_by': 'human',
        'spec': json.dumps(safety, sort_keys=True)})
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    original_metadata = deepcopy(metadata)
    original_metadata['grading']['rule'] = safety['aggregate_rule']
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _pku_saferlhf(directory, projected, original_metadata, source=source)
    return dict(stats, source_attempts=len(originals), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
