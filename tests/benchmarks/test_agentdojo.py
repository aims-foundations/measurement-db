"""Read-only release validation; set MEASUREMENT_DB_FULL_TEST=1 after building."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "agentdojo"


class ReleaseTests(shared.DatasetTests):
    directory = DIRECTORY
    def test_traces_are_complete_json_message_lists(self):
        for row in self.tables['traces'].itertuples():
            with self.subTest(response_id=row.response_id):
                self.assertIsInstance(json.loads(row.trace), list)


if __name__ == "__main__":
    unittest.main()
