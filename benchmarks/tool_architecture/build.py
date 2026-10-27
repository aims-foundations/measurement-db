"""Rebuild the shared released-data pilot and export the Tool-interface study study package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analyze_measurements.intervention_pilot import main


if __name__ == "__main__":
    main(package_study="tool_architecture")
