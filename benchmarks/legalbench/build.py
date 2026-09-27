"""Import the released LegalBench runs through the shared HELM table pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base_helm import HelmTabularBuild


class LegalBench(HelmTabularBuild):
    def download(self):
        return self.fetch_sources("runs", "grader", "task_documentation")


if __name__ == "__main__":
    LegalBench(__file__).main_from_args()
