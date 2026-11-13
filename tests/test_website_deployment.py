"""Exercise the deployment workflow's change detector against Git history."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/prepare-gallery.yml"


class WebsiteDeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Website CI test")
        self.git("config", "user.email", "test@example.invalid")
        self.write("README.md")
        self.write("benchmarks/example/metadata.yaml")
        self.base = self.commit()
        workflow = yaml.safe_load(WORKFLOW.read_text())
        self.script = next(step["run"] for step in workflow["jobs"]["changes"]["steps"]
                           if step.get("id") == "check")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True).strip()

    def write(self, path):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def check(self, *, event="push", before=None, after=None):
        output = self.root / "result.txt"
        output.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", self.script], cwd=self.root,
            env={**os.environ, "GITHUB_EVENT_NAME": event, "BEFORE": before or self.base,
                 "AFTER": after or self.git("rev-parse", "HEAD"),
                 "GITHUB_SHA": self.git("rev-parse", "HEAD"), "GITHUB_OUTPUT": str(output)},
            text=True, capture_output=True,
        )
        return result, output.read_text().strip() if output.exists() else ""

    def test_benchmark_beyond_300_changed_files_and_across_multiple_commits(self):
        for index in range(305):
            self.write(f"aaa-docs/{index:03}.md")
        self.commit()
        self.write("benchmarks/new_benchmark/nested/metadata.yaml")
        self.commit()
        result, output = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "deploy=true")

    def test_website_changes_and_deleted_benchmark_files_trigger_refresh(self):
        self.write("website/components/example.tsx")
        self.commit()
        self.assertEqual(self.check()[1], "deploy=true")
        self.base = self.git("rev-parse", "HEAD")
        (self.root / "benchmarks/example/metadata.yaml").unlink()
        self.commit()
        self.assertEqual(self.check()[1], "deploy=true")

    def test_unrelated_changes_skip_deployment(self):
        self.write("docs/notes.md")
        self.commit()
        self.assertEqual(self.check()[1], "deploy=false")

    def test_manual_refresh_and_initial_branch_push(self):
        self.assertEqual(self.check(event="workflow_dispatch")[1], "deploy=true")
        self.assertEqual(self.check(before="0" * 40)[1], "deploy=true")

    def test_invalid_git_base_fails_instead_of_skipping_deployment(self):
        result, output = self.check(before="missing-commit")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output, "")

    def test_pull_request_checks_all_commits_not_just_the_latest_push(self):
        self.write("benchmarks/new_benchmark/metadata.yaml")
        self.commit()
        self.write("docs/notes.md")
        head = self.commit()
        result, output = self.check(event="pull_request", after=head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "deploy=true")

    def test_pull_request_ignores_changes_only_on_the_base_branch(self):
        self.write("docs/notes.md")
        head = self.commit()
        self.git("checkout", "-q", "--detach", self.base)
        self.write("website/components/added_on_main.tsx")
        base = self.commit()
        result, output = self.check(event="pull_request", before=base, after=head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "deploy=false")

    def test_pull_request_detects_deleted_benchmark_with_diverged_base(self):
        (self.root / "benchmarks/example/metadata.yaml").unlink()
        head = self.commit()
        self.git("checkout", "-q", "--detach", self.base)
        self.write("docs/notes.md")
        base = self.commit()
        result, output = self.check(event="pull_request", before=base, after=head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "deploy=true")


class GalleryMergeGateTests(unittest.TestCase):
    def test_reproduction_is_manual_and_gallery_deploys_after_preparation(self):
        reproduction = yaml.load((WORKFLOW.parent / "benchmark-reproduction.yml").read_text(), Loader=yaml.BaseLoader)
        gallery = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        deployment = yaml.load((WORKFLOW.parent / "deploy-website.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(set(reproduction["on"]), {"workflow_dispatch"})
        self.assertNotIn("prepare-gallery", reproduction["jobs"])
        self.assertEqual(set(gallery["on"]), {"pull_request", "push", "workflow_dispatch"})
        self.assertEqual(gallery["on"]["push"]["branches"], ["main"])
        self.assertEqual(gallery["jobs"]["prepare-gallery"]["needs"], "changes")
        self.assertEqual(gallery["jobs"]["deploy"]["needs"], "prepare-gallery")
        self.assertEqual(gallery["jobs"]["deploy"]["uses"], "./.github/workflows/deploy-website.yml")
        self.assertEqual(set(deployment["on"]), {"workflow_call", "workflow_dispatch"})
        preparation = gallery["jobs"]["prepare-gallery"]
        self.assertIn("head.repo.full_name == github.repository", preparation["if"])
        self.assertNotIn("benchmark_reproduction", str(preparation["steps"]))

    def test_gallery_cleanup_removes_only_this_runs_scratch(self):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        cleanup = workflow["jobs"]["prepare-gallery"]["steps"][-1]
        self.assertEqual(cleanup["if"], "always()")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "gallery-123-1"
            job.mkdir()
            (job / "partial-bundle.tar").write_text("partial")
            cache = root / "gallery-cache"
            cache.mkdir()
            (cache / "verified.tar").write_text("reusable")
            subprocess.run(["bash", "-e", "-c", cleanup["run"]], check=True,
                           env={**os.environ, "GALLERY_JOB_DIR": str(job)})
            self.assertFalse(job.exists())
            self.assertEqual((cache / "verified.tar").read_text(), "reusable")
            subprocess.run(["bash", "-e", "-c", cleanup["run"]], check=True,
                           env={**os.environ, "GALLERY_JOB_DIR": ""})

    def test_gate_requires_requested_preparation_and_deployment(self):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        steps = workflow["jobs"]["ready"]["steps"]
        self.assertFalse(any("uses" in step for step in steps))
        script = steps[0]["run"]
        valid = {"CHANGES": "success", "REQUIRED": "true", "PREPARED": "success", "DEPLOYED": "success"}
        cases = [({}, True), ({"REQUIRED": "false", "PREPARED": "skipped", "DEPLOYED": "skipped"}, True)]
        cases += [({key: state}, False) for key in ("CHANGES", "PREPARED", "DEPLOYED")
                  for state in ("failure", "cancelled", "skipped")]
        for overrides, success in cases:
            with self.subTest(overrides=overrides):
                result = subprocess.run(["bash", "-e", "-c", script], env={**os.environ, **valid, **overrides})
                self.assertEqual(result.returncode == 0, success)


if __name__ == "__main__":
    unittest.main()
