import pytest

from benchmarks.aider.test_cases import case_names, final_outcomes
from scripts.curate_benchmarks.testlet_coding_audit import catch_roster, recorded_tests


def log(text):
    return '```\n-- Configuring done\n' + text + '\n```\n'


def failed(name):
    return '-' * 40 + '\n' + name + '\n' + '-' * 40 + '\nfile.cpp:7: FAILED:\n'


def check(history, cycles, names, expected):
    assert final_outcomes(history, cycles, names) == expected
    assert recorded_tests(dict(history=history, native_record=dict(tests_outcomes=cycles)), names) == expected


def test_comments_and_literal_slashes_in_test_names():
    files = {'tests.cpp': 'TEST_CASE( /* comment */ "https://host/x") {}\n'
        '// TEST_CASE("ignored")\nTEST_CASE(\n// formatting note\n"second") {}'}
    expected = {'https://host/x': 'tests.cpp', 'second': 'tests.cpp'}
    assert case_names(files) == catch_roster(files) == expected


def test_final_cycle_replaces_earlier_failed_patch():
    history = log(failed('a') + 'test cases: 2 | 1 passed | 1 failed')
    history += '```cpp\n// model edit\n```\n'
    history += log('All tests passed (8 assertions in 2 test cases)')
    check(history, [False, True], {'a': 'tests.cpp', 'b': 'tests.cpp'}, {'a': 1., 'b': 1.})


def test_complete_roster_infers_unprinted_passes():
    history = log(failed('a\n  nested section') + 'test cases:   2 |   1 passed | 1 failed')
    check(history, [False], {'a': 'tests.cpp', 'b': 'tests.cpp'}, {'a': 0., 'b': 1.})


def test_status_message_can_follow_the_closing_fence_without_a_newline():
    history = log('All tests passed (2 assertions in 2 test cases)').rstrip('\n') + '> Finished summarizing chat history.\n'
    check(history, [True], {'a': 'tests.cpp', 'b': 'tests.cpp'}, {'a': 1., 'b': 1.})


def test_early_abort_only_records_observed_failure():
    history = log(failed('a') + 'test cases: 1 | 1 failed')
    check(history, [False], {'a': 'tests.cpp', 'b': 'tests.cpp'}, {'a': 0.})


def test_unknown_header_does_not_turn_an_unobserved_case_into_a_pass():
    history = log(failed('a') + failed('corrupted name') + 'test cases: 3 | 1 passed | 2 failed')
    check(history, [False], dict.fromkeys(['a', 'b', 'c'], 'tests.cpp'), {'a': 0.})


@pytest.mark.parametrize('history,cycles', [
    (log('All tests passed (1 assertion in 1 test case)'), [True]),
    (log(failed('a') + 'test cases: 2 | 1 passed | 1 failed'), [False, False]),
    (log('error: compilation failed'), [False]),
])
def test_incomplete_rosters_missing_final_logs_and_compile_errors_have_no_invented_grades(history, cycles):
    check(history, cycles, {'a': 'tests.cpp', 'b': 'tests.cpp'}, {})
