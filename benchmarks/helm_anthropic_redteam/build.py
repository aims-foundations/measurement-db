#!/usr/bin/env python3
"""Curate the pinned HELM release using the shared table pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_base_helm import HelmTabularBuild


class HelmAnthropicRedTeam(HelmTabularBuild):
    """Source selection and grading declarations are in metadata.yaml."""

    def download(self):
        return self.fetch_sources("release", "runs", "grader")


if __name__ == "__main__":
    HelmAnthropicRedTeam(__file__).main_from_args()
