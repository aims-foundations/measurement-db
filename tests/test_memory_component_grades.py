"""Only explicit source verdicts can replace native memory-task aggregates."""
from benchmarks.goodai_ltm_benchmark.component_grades import component_grades


def test_trigger_positions_survive_repeated_reference_text_and_normalized_score():
    record = dict(expected_responses=['reply'] * 3,
        reasoning=["'reply' is in the response.", "'reply' is not in the response.", "'reply' is in the response."],
        score=2/3, max_score=1)
    assert component_grades('Trigger Response', record) == [('0', 'reply', 1.), ('1', 'reply', 0.), ('2', 'reply', 1.)]
    record['score'] = 1
    assert component_grades('Trigger Response', record) == []  # A manual aggregate revision is not a check verdict.


def test_spy_distractor_penalty_is_preserved_with_named_checks():
    record = dict(expected_responses=[['car'], ['night'], ['beach']], reasoning=[
        'Answer contains expected keyword(s)', "['night'] not found in answer.", 'Answer contains expected keyword(s)',
        'Answer also contains `morning`, which indicates that the agent has recalled something incorrect.'], score=1/3, max_score=1)
    checks = component_grades('Spy Meeting', record)
    assert [r[2] for r in checks] == [1, 0, 1, 0]
    assert checks[-1][0] == 'no_conflicting_keywords'


def test_missing_detail_is_not_inferred_from_an_aggregate():
    record = dict(expected_responses=['a', 'b'], reasoning=['One answer was right'], score=1, max_score=2)
    assert component_grades('Instruction Recall', record) == []
