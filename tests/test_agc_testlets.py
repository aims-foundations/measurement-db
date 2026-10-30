"""Connections parsing, partial outcomes and repeated-group matching."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_db.benchmarks.agc_bench.components import groups, group_outcomes


class ConnectionsTests(unittest.TestCase):
    reference = 'a, b, c, d\ne, f, g, h\ni, j, k, l\nm, n, o, p'

    def test_partial_recovery_keeps_reference_positions(self):
        self.assertEqual(group_outcomes('h, g, f, e\nm, n, o, p\nwrong\nwrong', self.reference), [0., 1., 0., 1.])

    def test_group_order_prefixes_and_punctuation(self):
        text = 'Group 1: D, C, B, A. Connection: letters\nGroup 2: e, f, g, h\ni, j, k, l\nm, n, o, p'
        self.assertEqual(group_outcomes(text, self.reference), [1.] * 4)

    def test_repeated_prediction_does_not_credit_another_reference(self):
        self.assertEqual(group_outcomes(('a, b, c, d\n' * 4), self.reference), [1., 0., 0., 0.])

    def test_only_first_four_nonempty_lines_and_tokens_count(self):
        self.assertEqual(groups('\n' + self.reference + '\nignored')[0], ('a', 'b', 'c', 'd'))
        self.assertEqual(len(groups('')), 4)


if __name__ == '__main__':
    unittest.main()
