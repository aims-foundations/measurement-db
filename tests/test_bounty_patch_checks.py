import copy

import pytest

from benchmarks.bountybench.patch_checks import final_patch_checks
from scripts.curate_benchmarks.testlet_bounty_audit import native_checks


def workflow(repetitions, success=True):
    def action(resource, **metadata):
        return dict(resource_id=resource, additional_metadata=metadata)
    actions = [action('exploit_docker'), action('verify_subprocess', exit_code=0)]
    for scores in repetitions:
        actions.extend([action('repo_invariants_subprocess', repo_invariant_scores=scores),
            action('bounty_invariants_subprocess', bounty_invariant_scores={})])
    actions += [action('exploit_docker'), action('verify_subprocess', exit_code=1)]
    return dict(workflow_metadata=dict(workflow_name='PatchWorkflow', workflow_summary=dict(success=success)),
        phase_messages=[dict(agent_messages=[dict(agent_id='patch_agent', success=success, action_messages=actions)])],
        additional_metadata=dict(repo_metadata=dict(invariant_thresholds=dict(a=1, b=1)), bounty_metadata={}))


def test_repetitions_keep_the_same_check_and_execution():
    record = workflow([dict(a=0, b=1), dict(a=1, b=1), dict(a=1, b=0)])
    checks = list(final_patch_checks(record))
    assert checks == native_checks(record)
    assert len(checks) == 7
    assert [value for key, _, value, _ in checks if key == 'invariant:a'] == [0, 1, 1]
    assert {condition for key, _, _, condition in checks if key == 'invariant:a'} == {
        'grader_repeat=1', 'grader_repeat=2', 'grader_repeat=3'}


def test_each_check_passing_in_different_repetitions_is_not_a_success():
    record = workflow([dict(a=0, b=1), dict(a=1, b=0), dict(a=0, b=1)], success=False)
    assert list(final_patch_checks(record)) == native_checks(record)
    record['workflow_metadata']['workflow_summary']['success'] = True
    record['phase_messages'][0]['agent_messages'][-1]['success'] = True
    with pytest.raises(ValueError, match='do not reconstruct'):
        list(final_patch_checks(record))


def test_intermediate_patch_does_not_change_final_results():
    record = workflow([dict(a=1, b=1)] * 3)
    expected = list(final_patch_checks(record))
    previous = copy.deepcopy(record['phase_messages'][0]['agent_messages'][0])
    previous.update(success=False, action_messages=[])
    record['phase_messages'][0]['agent_messages'].insert(0, previous)
    assert list(final_patch_checks(record)) == expected


def test_missing_verdict_stays_null_and_unknown_sequence_is_rejected():
    record = workflow([dict(a=1), dict(a=1, b=1), dict(a=1, b=1)])
    assert list(final_patch_checks(record))[1][2] is None
    record['phase_messages'][0]['agent_messages'][0]['action_messages'].pop(0)
    with pytest.raises(ValueError, match='sequence'):
        list(final_patch_checks(record))
