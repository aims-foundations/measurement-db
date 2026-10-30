"""Source assertion predictions retain unknown labels and duplicate positions."""

import io
import csv
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_db.benchmarks.llm4ir.components import assertion_rows


class AssertionPredictionTests(unittest.TestCase):
    def test_reported_labels_do_not_invent_verdicts_for_unknown_answers(self):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['assert_statement', 'final_result', 'comment'])
        for label in ['pass', 'fail', 'unable to determine']:
            writer.writerow(['assert(f(1) == 2);', label, 'Full answer\nwith another line.'])
        rows = assertion_rows(stream.getvalue())
        self.assertEqual([row[3] for row in rows], [1., 0., None])
        self.assertEqual([row[2] for row in rows], [0, 1, 2])
        self.assertEqual(len({row[0] for row in rows}), 3)


if __name__ == '__main__':
    unittest.main()
