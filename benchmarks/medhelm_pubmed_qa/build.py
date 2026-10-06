#!/usr/bin/env python3
"""Curate this MedHELM scenario from the pinned native run tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_base_helm import HelmTabularBuild


class MedhelmPubmedQa(HelmTabularBuild):
    """Use the shared six-stage HELM table pipeline and metadata declarations."""

    def download(self):
        return self.fetch_sources("runs", "grader")


if __name__ == "__main__":
    MedhelmPubmedQa(__file__).main_from_args()
