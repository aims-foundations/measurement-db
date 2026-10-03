"""Coverage and checkpointing must not confuse a grouped job with its benchmarks."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import benchmark_reproduction as reproduction
from scripts.ci import benchmark_reproduction_report as reports


REVISION = "a" * 40


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "artifacts"
        self.environment = patch.dict(os.environ, {
            "GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "b" * 40,
            "RUNNER_NAME": "runner-one", "HF_REVISION": REVISION,
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def batch(self, slugs, statuses, attempt=1, message="details"):
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": str(attempt)}):
            data = reports.new_batch(slugs, REVISION)
        for row, status in zip(data["results"], statuses):
            row.update(status=status, duration_seconds=1.25, message=message)
        path = self.artifacts / f"batch-{attempt}-{'-'.join(slugs)}" / "results.json"
        reports.save(path, data)
        return data, path

    def collect(self, groups, jobs=None, changes_result="success"):
        return reports.collect({"include": [{"benchmarks": group} for group in groups]},
                               self.artifacts, REVISION, jobs or [], changes_result)

    def test_mixed_batch_is_checkpointed_before_the_next_benchmark(self):
        path = self.root / "batch.json"
        observed = []

        def verify(root, slug, revision):
            observed.append([row["status"] for row in json.loads(path.read_text())["results"]])
            if slug == "alpha":
                raise ValueError("Byte mismatch: responses.parquet")

        with patch.object(sys, "argv", ["ci", "verify", "alpha", "beta", "--report", str(path)]), \
                patch.object(reproduction, "resolve_revision", return_value=REVISION), \
                patch.object(reproduction, "verify_benchmark", side_effect=verify), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reproduction.main(), 1)
        self.assertEqual(observed, [["running", "not_run"], ["failed", "running"]])
        rows = json.loads(path.read_text())["results"]
        self.assertEqual([row["status"] for row in rows], ["failed", "passed"])
        self.assertIn("Byte mismatch", rows[0]["message"])
        self.assertIsNotNone(rows[1]["duration_seconds"])

    def test_interruption_preserves_passes_and_not_started_benchmarks(self):
        path = self.root / "batch.json"
        with patch.object(sys, "argv", ["ci", "verify", "alpha", "beta", "gamma", "--report", str(path)]), \
                patch.object(reproduction, "resolve_revision", return_value=REVISION), \
                patch.object(reproduction, "verify_benchmark", side_effect=[None, KeyboardInterrupt()]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                reproduction.main()
        self.assertEqual([row["status"] for row in json.loads(path.read_text())["results"]],
                         ["passed", "interrupted", "not_run"])

    def test_reference_resolution_failure_is_not_a_benchmark_failure(self):
        path = self.root / "batch.json"
        with patch.object(sys, "argv", ["ci", "verify", "alpha", "--report", str(path)]), \
                patch.object(reproduction, "resolve_revision", side_effect=PermissionError("access denied")):
            with self.assertRaises(PermissionError):
                reproduction.main()
        row = json.loads(path.read_text())["results"][0]
        self.assertEqual(row["status"], "not_run")
        self.assertIn("reference resolution failed", row["message"])

    def test_setup_failure_keeps_every_benchmark_not_run(self):
        batch = reports.finalize_batch(reports.new_batch(["alpha", "beta"], REVISION), "failure")
        self.assertEqual([row["status"] for row in batch["results"]], ["not_run", "not_run"])
        self.assertIn("job: failure", batch["results"][0]["message"])

    def test_unfinished_checkpoint_is_interrupted_not_failed(self):
        self.batch(["alpha", "beta"], ["passed", "running"])
        result = self.collect([["alpha", "beta"]])
        self.assertEqual([row["status"] for row in result["results"]], ["passed", "interrupted"])

    def test_missing_group_artifact_still_lists_every_selected_benchmark(self):
        self.batch(["alpha"], ["passed"])
        result = self.collect([["alpha"], ["beta", "gamma"]])
        self.assertEqual([row["status"] for row in result["results"]],
                         ["passed", "not_reported", "not_reported"])
        self.assertEqual(result["counts"]["passed"], 1)
        self.assertEqual(result["counts"]["not_reported"], 2)

    def test_failed_prerequisite_reports_not_run(self):
        result = self.collect([["alpha", "beta"]], changes_result="failure")
        self.assertEqual(result["counts"]["not_run"], 2)
        self.assertTrue(result["warnings"])

    def test_corrupt_artifact_does_not_hide_missing_results(self):
        path = self.artifacts / "broken" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text("{incomplete")
        result = self.collect([["alpha"]])
        self.assertEqual(result["results"][0]["status"], "not_reported")
        self.assertTrue(result["warnings"])

    def test_report_for_a_different_reference_is_not_reused(self):
        data, path = self.batch(["alpha"], ["passed"])
        data["hf_revision"] = "c" * 40
        reports.save(path, data)
        result = self.collect([["alpha"]])
        self.assertEqual(result["results"][0]["status"], "not_reported")
        self.assertIn("HF revision", result["warnings"][0])

    def test_rerun_uses_latest_group_and_retains_prior_successful_groups(self):
        self.batch(["alpha"], ["passed"], attempt=1)
        self.batch(["beta"], ["failed"], attempt=1)
        self.batch(["beta"], ["passed"], attempt=2)
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            result = self.collect([["alpha"], ["beta"]])
        self.assertEqual(result["counts"]["passed"], 2)
        self.assertEqual([row["run_attempt"] for row in result["results"]], [1, 2])

    def test_missing_rerun_artifact_never_falls_back_to_an_old_pass(self):
        self.batch(["alpha"], ["passed"], attempt=1)
        job = {"id": 200, "name": "Reproduce alpha", "run_attempt": 2,
               "conclusion": "cancelled", "html_url": "https://github.com/jobs/200"}
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            result = self.collect([["alpha"]], [job])
        self.assertEqual(result["results"][0]["status"], "not_reported")
        self.assertEqual(result["results"][0]["job_url"], job["html_url"])

    def test_duplicate_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            self.collect([["alpha"], ["alpha"]])

    def test_secrets_are_redacted_and_markdown_cells_are_escaped(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_example_secret"}):
            message = reports.safe_message("Error: hf_example_secret | <script>\n[link](url)")
        self.assertNotIn("hf_example_secret", message)
        self.batch(["alpha"], ["failed"], message=message)
        text = reports.markdown(self.collect([["alpha"]]))
        self.assertIn("&#124;", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("<br>", text)
        self.assertNotIn("[link](url)", text)

    def test_cli_publishes_downloadable_report_even_when_a_benchmark_fails(self):
        self.batch(["alpha", "beta"], ["failed", "passed"])
        output, summary = self.root / "output", self.root / "summary.md"
        matrix = {"include": [{"benchmarks": ["alpha", "beta"]}]}
        with patch.dict(os.environ, {"MATRIX_JSON": json.dumps(matrix), "GITHUB_STEP_SUMMARY": str(summary)}), \
                patch.object(sys, "argv", ["report", "aggregate", "--reports-dir", str(self.artifacts),
                                           "--output-dir", str(output)]), \
                patch.object(reports, "github_jobs", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reports.main(), 1)
        with (output / "results.csv").open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["status"] for row in rows], ["failed", "passed"])
        self.assertEqual(summary.read_text(), (output / "summary.md").read_text())
        self.assertEqual(json.loads((output / "results.json").read_text())["counts"]["passed"], 1)

    def test_no_selected_benchmarks_is_successful_when_download_is_skipped(self):
        with patch.dict(os.environ, {"MATRIX_JSON": '{"include":[]}', "DOWNLOAD_OUTCOME": "skipped"}), \
                patch.object(sys, "argv", ["report", "aggregate", "--reports-dir", str(self.artifacts),
                                           "--output-dir", str(self.root / "output")]), \
                patch.object(reports, "github_jobs", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reports.main(), 0)

    def test_job_lookup_paginates_beyond_one_hundred_jobs(self):
        responses = [io.StringIO(json.dumps({"jobs": [{"id": i} for i in range(100)]})),
                     io.StringIO(json.dumps({"jobs": [{"id": 100}]}))]
        with patch.dict(os.environ, {"GITHUB_TOKEN": "example", "GITHUB_REPOSITORY": "owner/repo"}), \
                patch.object(reports.urllib.request, "urlopen", side_effect=responses) as fetch:
            jobs = reports.github_jobs()
        self.assertEqual(len(jobs), 101)
        self.assertIn("page=2", fetch.call_args_list[1].args[0].full_url)


if __name__ == "__main__":
    unittest.main()
