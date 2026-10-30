"""Validate complete native inputs, configurations, executions and component grades."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared
from measurement_db.scripts.build_measurement_tables.validate_characterization import check_source_claims
from measurement_db.scripts.curate_benchmarks.testlet_audits import verify_component_results


class ReleaseTests(shared.DatasetTests):
    directory = Path(__file__).resolve().parents[2] / "benchmarks" / "multi_swebench"

    def test_complete_native_evidence(self):
        observed = verify_component_results(self.directory, self.table_dir)
        check_source_claims(self.expected, observed)


if __name__ == "__main__":
    unittest.main()
