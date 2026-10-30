"""Native predicate weights, unavailable tables and unsafe output text."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_db.benchmarks.qatch.components import predicates, prediction_table


class QATCHTests(unittest.TestCase):
    def test_equal_value_occurrences_share_a_weighted_predicate(self):
        actual = predicates("[['a', 'a', 'b']]", "[['a']]", 'cell_recall', .667)
        self.assertEqual(actual, [(dict(value='a', target_occurrences=2), 1., 2),
                                  (dict(value='b', target_occurrences=1), 0., 1)])

    def test_row_multiplicity_is_a_check_not_just_membership(self):
        actual = predicates("[['a'], ['a'], ['b']]", "[['a'], ['b']]", 'tuple_constraint', .5)
        self.assertEqual([grade for _, grade, _ in actual], [0., 1.])

    def test_replay_discrepancy_retains_the_original_metric(self):
        self.assertEqual(predicates("[['a']]", "[['a']]", 'cell_recall', 0.), [])

    def test_unreleased_table_marker_does_not_invent_cells(self):
        self.assertEqual(predicates("[['a']]", 'EQUAL', 'cell_recall', 1.), [])

    def test_model_output_is_never_executed(self):
        self.assertIsNone(prediction_table("[[__import__('os').system('false')]]"))


if __name__ == '__main__':
    unittest.main()
