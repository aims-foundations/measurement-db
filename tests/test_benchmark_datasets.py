"""Run shared dataset validation and source-specific checks for this repository."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from measurement_db.scripts.build_measurement_tables import validate_benchmark_datasets as shared


def load_tests(loader, tests, pattern):
    return shared.dataset_suite(ROOT)


if __name__ == "__main__":
    raise SystemExit(shared.main(ROOT))
