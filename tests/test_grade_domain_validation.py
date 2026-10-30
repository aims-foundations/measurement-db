"""Shared domains never admit a grade belonging only to another item's scale."""
import json

import pandas as pd
import pytest

from scripts.build_measurement_tables.validate_measurement_tables import validate_response_grades


def test_shared_domains_keep_each_items_scale_and_missing_grades():
    binary = dict(kind='discrete', values=[0, 1])
    scaled = dict(kind='interval', min=0, max=5)
    items = pd.DataFrame([dict(item_id=str(i), grading_criterion=json.dumps(dict(
        reference_answer=str(i), response_scale=binary if i < 100 else scaled))) for i in range(101)])
    responses = pd.DataFrame(dict(item_id=['0', '99', '100', '100'], response=[1., None, 3., 5.]))
    validate_response_grades(responses, {'kind':'mixed'}, items=items)
    responses.loc[0, 'response'] = 3.
    with pytest.raises(ValueError):
        validate_response_grades(responses, {'kind':'mixed'}, items=items)


def test_unobserved_items_still_require_a_valid_explicit_domain():
    items = pd.DataFrame(dict(item_id=['unobserved'], grading_criterion=['{}']))
    responses = pd.DataFrame(dict(item_id=pd.Series(dtype=str), response=pd.Series(dtype=float)))
    with pytest.raises(ValueError, match='response_scale'):
        validate_response_grades(responses, {'kind':'mixed'}, items=items)
