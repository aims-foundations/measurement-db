"""Decode the two released SWE-bench report layouts without re-running tests."""


def named_checks(payload, instance_id):
    record = payload.get(instance_id, payload)
    if record.get('instance_id', instance_id) != instance_id:
        raise ValueError('Test report and source path identify different tasks')
    status = record.get('tests_status', record)
    checks = {}
    for group in ('FAIL_TO_PASS', 'PASS_TO_PASS', 'FAIL_TO_FAIL', 'PASS_TO_FAIL'):
        if group not in status:
            continue
        table = status[group]
        if not isinstance(table, dict) or set(table) - {'success', 'failure'}:
            raise ValueError('Unknown named-test status vocabulary')
        passed, failed = set(table.get('success', [])), set(table.get('failure', []))
        for name in passed | failed:
            checks[group, name] = None if name in passed and name in failed else float(name in passed)
    return checks
