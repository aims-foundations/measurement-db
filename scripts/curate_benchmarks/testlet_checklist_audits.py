"""Independent source reconciliation for published subanswers and rubric dimensions."""

import hashlib
import json
import math
import re


def source_attempts(directory, metadata):
    if directory.name == 'atmossci_bench':
        from .native_result_audits import _atmossci_source_records
        records, items, _, _ = _atmossci_source_records(directory, metadata)
        for key, row in records.items():
            group = hashlib.sha256(items[row['item']]['content'].encode()).hexdigest()
            judgment = json.loads(row['trace']['evaluation_json']) if row['trace']['evaluation_json'] else None
            checks = [(group + '::' + check['id'], group, None if check['is_correct'] is None else float(check['is_correct']))
                for check in judgment['evaluation']] if judgment else [(group + '::ungraded_task_outcome', group, None)]
            yield key, row['response'], checks
    elif directory.name == 'llmail_inject':
        yield from _llmail(directory, metadata)
    elif directory.name == 'mmdocrag':
        yield from _mmdocrag(directory, metadata)
    else:
        raise ValueError('Unsupported checklist release')


def _llmail(directory, metadata):
    parameters = metadata['build']['parameters']
    root = directory / 'raw' / parameters['layout']['root']
    scenarios = json.loads((root / parameters['paths']['scenarios']).read_text())
    names = set(json.loads((root / parameters['paths']['objectives']).read_text()))
    aliases = {}
    for filename in parameters['results'].values():
        with (root / filename).open() as stream:
            for line in stream:
                record = json.loads(line)
                if record['started_time'] is None or record['output'] == parameters['request_status']['missing_scenario']:
                    continue
                scenario = 'scenario_' + record['scenario'][5]
                context = scenarios[scenario]
                key = scenario, record['subject'], record['body']
                aliases.setdefault(key, record['job_id'])
                content = dict(scenario_key=scenario, user_query=context['user_query'],
                    released_insertion_position=context['position'], email_subject=record['subject'], email_body=record['body'])
                group = hashlib.sha256(json.dumps(content, ensure_ascii=True, sort_keys=True).encode()).hexdigest()
                flags = json.loads(record['objectives'])
                if flags:
                    assert set(flags) == names and all(type(v) is bool for v in flags.values())
                yield record['job_id'], float(all(flags.values())) if flags else None, [
                    (aliases[key] + '::' + name, group, None if value is None else float(value))
                    for name, value in (flags or dict.fromkeys(names)).items()]


def _mmdocrag(directory, metadata):
    parameters = metadata['build']['parameters']
    rubric = metadata['grading']['verifiers']['release']
    noise = parameters['patterns']['dimension_noise']
    normalize = lambda name: re.sub(noise, '', name.lower())
    expected = {normalize(name) for name in rubric['dimensions']}
    index = 0
    for path in sorted((directory / 'raw' / parameters['layout']['evaluation_directory']).iterdir()):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            current, index = str(index), index + 1
            if row.get('q_id') is None or not isinstance(row.get('response'), dict):
                continue
            dimensions = {}
            for key, value in row['response'].items():
                name = normalize(key)
                if name in expected and name not in dimensions and type(value) in (int, float) and 0 <= value <= rubric['maximum']:
                    dimensions[name] = value
            aggregate = sum(dimensions.values()) / len(expected) / rubric['maximum'] if set(dimensions) == expected else None
            group = 'q_id::' + str(row['q_id'])
            yield current, aggregate, [(group + '::' + name, group, dimensions.get(name)) for name in expected]
