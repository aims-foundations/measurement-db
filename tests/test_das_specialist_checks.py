"""Unknown specialist judgments must never become negative detection labels."""

import pytest

from benchmarks.das_med_hallucination.specialist_checks import subtype_grades


VOCABULARY = {'1A': 'false claim', '1B': 'interaction', '2A': 'fabricated citation'}


def test_definite_detection_is_multilabel_and_preserves_unasked_specialist():
    judgment = {'agent_decisions': [dict(code=1, called=True, classification=['1A', '1A']),
        dict(code=2, called=False)]}
    assert subtype_grades(judgment, VOCABULARY) == {'1A': 1., '1B': 0., '2A': None}


@pytest.mark.parametrize('classification', [None, '0.5'])
def test_uncertainty_and_missing_judgment_are_not_clean(classification):
    judgment = {'agent_decisions': [dict(code=1, called=True, classification=classification),
        dict(code=2, called=True, classification='0')]}
    assert subtype_grades(judgment, VOCABULARY) == {'1A': None, '1B': None, '2A': 0.}
    assert subtype_grades(None, VOCABULARY) == dict.fromkeys(VOCABULARY)


@pytest.mark.parametrize('decisions', [
    [dict(code=1, called=True, classification=['2A'])],
    [dict(code=1, called=False, classification='0')],
    [dict(code=1, called='false', classification=None)],
    [dict(code=1, called=True, classification='1A')],
    [dict(code=1, called=True, classification='0')] * 2,
])
def test_invalid_or_ambiguous_native_classifications_require_review(decisions):
    with pytest.raises(ValueError):
        subtype_grades({'agent_decisions': decisions}, VOCABULARY)
