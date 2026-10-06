"""CI selection, isolated execution, and failures that must never pass silently."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.ci import benchmark_reproduction as reproduction


REVISION = "a" * 40


class RemoteEntryNotFoundError(Exception):
    """Offline stand-in for huggingface_hub.errors.RemoteEntryNotFoundError."""


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "measurement_db"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "CI fixture")
        self.git("config", "user.email", "ci@example.invalid")
        for slug in ("alpha", "beta", "_template"):
            self.write(f"benchmarks/{slug}/build.py", "# builder\n")
            self.write(f"benchmarks/{slug}/metadata.yaml", "benchmark:\n  release: public\n")
        self.write("README.md", "fixture\n")
        self.base = self.commit()

    def git(self, *args):
        return reproduction.git(self.root, *args).strip()

    def write(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def push(self, before, after):
        return reproduction.changed_benchmarks(self.root, "push", {"before": before, "after": after})


class ChangedBenchmarksTests(RepositoryTest):
    def test_push_includes_every_commit_and_nested_non_code_changes(self):
        self.write("benchmarks/alpha/docs/notes.md", "first commit")
        self.commit()
        self.write("benchmarks/beta/metadata.yaml", "second commit")
        self.assertEqual(self.push(self.base, self.commit()), ["alpha", "beta"])

    def test_pr_uses_merge_base_and_excludes_unrelated_base_changes(self):
        self.write("benchmarks/alpha/metadata.yaml", "PR edit")
        head = self.commit()
        self.git("checkout", "-q", "--detach", self.base)
        self.write("benchmarks/beta/metadata.yaml", "base branch edit")
        base = self.commit()
        event = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
        self.assertEqual(reproduction.changed_benchmarks(self.root, "pull_request", event), ["alpha"])

    def test_rename_selects_old_and_new_folders(self):
        (self.root / "benchmarks/alpha").rename(self.root / "benchmarks/gamma")
        self.assertEqual(self.push(self.base, self.commit()), ["alpha", "gamma"])

    def test_deleted_builder_is_not_ignored(self):
        (self.root / "benchmarks/alpha/build.py").unlink()
        self.assertEqual(self.push(self.base, self.commit()), ["alpha"])

    def test_unrelated_edits_select_nothing(self):
        self.write("README.md", "unrelated")
        self.assertEqual(self.push(self.base, self.commit()), [])

    def test_new_branch_and_template_edits_select_all_builders(self):
        self.assertEqual(self.push(reproduction.ZERO_SHA, self.base), ["alpha", "beta"])
        self.write("benchmarks/_template/metadata.yaml", "shared template")
        self.assertEqual(self.push(self.base, self.commit()), ["alpha", "beta"])

    def test_deleted_branch_selects_nothing(self):
        event = {"deleted": True, "before": self.base, "after": reproduction.ZERO_SHA}
        self.assertEqual(reproduction.changed_benchmarks(self.root, "push", event), [])

    def test_dispatch_selects_named_benchmark_or_all(self):
        for name, expected in [("", ["alpha", "beta"]), ("alpha", ["alpha"])]:
            event = {"inputs": {"benchmark": name}}
            self.assertEqual(reproduction.changed_benchmarks(self.root, "workflow_dispatch", event), expected)
        with self.assertRaisesRegex(ValueError, "Invalid benchmark"):
            reproduction.changed_benchmarks(self.root, "workflow_dispatch", {"inputs": {"benchmark": "../alpha"}})

    def test_dispatch_selects_a_deduplicated_comma_separated_subset(self):
        self.write("benchmarks/gamma/build.py", "# unselected builder\n")
        event = {"inputs": {"benchmark": " beta, alpha ,beta "}}
        selected = reproduction.changed_benchmarks(self.root, "workflow_dispatch", event)
        self.assertEqual(selected, ["alpha", "beta"])
        self.assertEqual([job["benchmarks"] for job in reproduction.build_matrix(selected)["include"]],
                         [["alpha"], ["beta"]])

    def test_dispatch_rejects_invalid_entries_in_a_subset(self):
        for value in ("alpha,../beta", "alpha,", ",", "alpha,,beta", "alpha beta"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid benchmark"):
                reproduction.changed_benchmarks(self.root, "workflow_dispatch", {"inputs": {"benchmark": value}})

    def test_large_migration_checks_every_benchmark_within_matrix_limit(self):
        slugs = [f"benchmark_{n}" for n in range(600)]
        matrix = reproduction.build_matrix(slugs)["include"]
        self.assertLessEqual(len(matrix), 256)
        self.assertEqual([slug for job in matrix for slug in job["benchmarks"]], slugs)
        self.assertEqual(reproduction.build_matrix([]), {"include": []})

    def test_runtime_pins_are_read_without_importing_builders(self):
        self.write("benchmarks/alpha/metadata.yaml", "benchmark:\n  release: public\n"
                   "build:\n  parameters:\n    runtime:\n      python_version: '3.12.12'\n")
        versions = reproduction.python_versions(self.root, ["alpha", "beta"])
        self.assertEqual(versions, {"alpha": "3.12.12", "beta": "3.11"})
        self.assertEqual(reproduction.build_matrix(["alpha", "beta"], versions), {"include": [
            {"benchmarks": ["alpha"], "python_version": "3.12.12"},
            {"benchmarks": ["beta"], "python_version": "3.11"},
        ]})

    def test_mixed_runtime_batches_stay_within_matrix_limit(self):
        slugs = [f"benchmark_{n}" for n in range(511)] + ["lawbench"]
        versions = {"lawbench": "3.12.12"}
        matrix = reproduction.build_matrix(slugs, versions)["include"]
        self.assertLessEqual(len(matrix), 256)
        self.assertCountEqual([slug for job in matrix for slug in job["benchmarks"]], slugs)
        for job in matrix:
            self.assertEqual({versions.get(slug, "3.11") for slug in job["benchmarks"]}, {job["python_version"]})
        self.assertEqual([job for job in matrix if "lawbench" in job["benchmarks"]],
                         [{"benchmarks": ["lawbench"], "python_version": "3.12.12"}])

    def test_invalid_runtime_pin_fails_selection(self):
        for value in ("3.12", "'latest'", "'[3.11, 3.12]'", "null"):
            with self.subTest(value=value):
                self.write("benchmarks/alpha/metadata.yaml", "benchmark:\n  release: public\n"
                           f"build:\n  parameters:\n    runtime:\n      python_version: {value}\n")
                with self.assertRaisesRegex(ValueError, "quoted Python version"):
                    reproduction.python_versions(self.root, ["alpha"])


class ReleaseSelectionTests(RepositoryTest):
    def withhold(self, slug="beta"):
        return self.write(f"benchmarks/{slug}/metadata.yaml",
                          "benchmark:\n  release: withheld\n  release_reason: Permission pending.\n")

    def test_withheld_benchmarks_are_excluded_from_the_job_matrix(self):
        self.withhold()
        public, withheld = reproduction.select_releases(self.root, ["alpha", "beta"])
        self.assertEqual(public, ["alpha"])
        self.assertEqual(withheld, {"beta": "Permission pending."})
        self.assertEqual(reproduction.build_matrix(public),
                         {"include": [{"benchmarks": ["alpha"], "python_version": "3.11"}]})

    def test_dispatch_subset_still_obeys_release_policy(self):
        self.withhold("beta")
        selected = reproduction.changed_benchmarks(self.root, "workflow_dispatch",
                                                  {"inputs": {"benchmark": "alpha,beta"}})
        self.assertEqual(reproduction.select_releases(self.root, selected),
                         (["alpha"], {"beta": "Permission pending."}))

    def test_selection_cli_reports_all_withheld_without_scheduling_jobs(self):
        self.withhold("alpha")
        self.withhold("beta")
        event = self.write("event.json", '{"inputs": {"benchmark": ""}}')
        output = self.root / "output"
        with patch.dict(os.environ, {"GITHUB_EVENT_PATH": str(event), "GITHUB_EVENT_NAME": "workflow_dispatch",
                                     "GITHUB_OUTPUT": str(output)}), \
                patch.object(reproduction, "ROOT", self.root), patch.object(sys, "argv", ["ci", "select"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reproduction.main(), 0)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        self.assertEqual(values["has_benchmarks"], "false")
        self.assertEqual(json.loads(values["matrix"]), {"include": []})
        self.assertEqual(set(json.loads(values["withheld"])), {"alpha", "beta"})

    def test_invalid_missing_private_and_duplicate_decisions_fail_closed(self):
        for text in ("benchmark: {}", "benchmark:\n  release: private", "benchmark:\n  release: invalid",
                     "benchmark:\n  release: withheld", "benchmark:\n  release: withheld\n  release_reason: ' '",
                     "benchmark:\n  release: withheld\n  release: public", "benchmark: [", "benchmark: null"):
            with self.subTest(text=text):
                self.write("benchmarks/alpha/metadata.yaml", text)
                with self.assertRaises(ValueError):
                    reproduction.select_releases(self.root, ["alpha"])
        (self.root / "benchmarks/alpha/metadata.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "metadata.yaml"):
            reproduction.select_releases(self.root, ["alpha"])

    def test_direct_verification_skips_before_workspace_build_or_reference_access(self):
        self.withhold("alpha")
        with patch.object(reproduction, "copy_build_source") as copy, \
                patch.object(reproduction.subprocess, "run") as build, \
                patch.object(reproduction, "resolve_revision") as reference:
            with self.assertRaises(reproduction.WithheldBenchmark):
                reproduction.verify_benchmark(self.root, "alpha", REVISION)
        copy.assert_not_called()
        build.assert_not_called()
        reference.assert_not_called()

    def test_direct_cli_records_skip_and_continues_the_public_benchmark(self):
        self.withhold("alpha")
        report = self.root / "report.json"
        original = reproduction.verify_benchmark

        def verify(root, slug, revision, **kwargs):
            if slug == "alpha":
                return original(root, slug, revision, **kwargs)
            kwargs["progress"]("build")
            kwargs["progress"]("comparison")

        with patch.object(reproduction, "ROOT", self.root), \
                patch.object(reproduction, "verify_benchmark", side_effect=verify) as runner, \
                patch.object(sys, "argv", ["ci", "verify", "alpha", "beta", "--report", str(report)]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reproduction.main(), 0)
        rows = json.loads(report.read_text())["results"]
        self.assertEqual([row["status"] for row in rows], ["skipped_withheld", "passed"])
        self.assertEqual(rows[0]["build_status"], "not_run")
        self.assertEqual(runner.call_count, 2)


def file(path):
    return SimpleNamespace(path=path, blob_id="blob")


class ReferenceLayoutTests(unittest.TestCase):
    def test_modern_layout_takes_precedence_over_older_flat_files(self):
        api = Mock()
        api.list_repo_tree.side_effect = [
            [SimpleNamespace(path="alpha/formatted_tables"), file("alpha/response.parquet")],
            [file("alpha/formatted_tables/responses.parquet"), file("alpha/formatted_tables/assets.parquet")],
        ]
        self.assertEqual(reproduction.reference_files(api, "alpha", REVISION), {
            "responses.parquet": "alpha/formatted_tables/responses.parquet",
            "assets.parquet": "alpha/formatted_tables/assets.parquet",
        })
        self.assertTrue(all(call.kwargs["revision"] == REVISION for call in api.list_repo_tree.call_args_list))

    def test_flat_layout_renames_only_the_legacy_response_filename(self):
        api = Mock()
        api.list_repo_tree.return_value = [file("alpha/response.parquet"), file("alpha/metadata.yaml")]
        self.assertEqual(reproduction.reference_files(api, "alpha", REVISION),
                         {"responses.parquet": "alpha/response.parquet"})

    def test_no_tables_and_ambiguous_response_names_fail(self):
        api = Mock()
        api.list_repo_tree.return_value = [file("alpha/metadata.yaml")]
        with self.assertRaisesRegex(ValueError, "No published"):
            reproduction.reference_files(api, "alpha", REVISION)
        api.list_repo_tree.return_value = [file("alpha/response.parquet"), file("alpha/responses.parquet")]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            reproduction.reference_files(api, "alpha", REVISION)

    def test_empty_modern_folder_does_not_fall_back_to_stale_flat_tables(self):
        api = Mock()
        api.list_repo_tree.side_effect = [[SimpleNamespace(path="alpha/formatted_tables"),
                                           file("alpha/response.parquet")], []]
        with self.assertRaisesRegex(ValueError, "No published"):
            reproduction.reference_files(api, "alpha", REVISION)

    def test_network_errors_are_not_treated_as_missing_optional_tables(self):
        api = Mock()
        api.list_repo_tree.side_effect = PermissionError("gated dataset")
        with self.assertRaises(PermissionError):
            reproduction.reference_files(api, "alpha", REVISION)


class ByteComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.generated = self.root / "generated"
        self.generated.mkdir()
        self.reference = self.root / "expected.parquet"
        self.reference.write_bytes(b"PAR1same bytesPAR1")
        self.expected = {"responses.parquet": self.reference}

    def compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            reproduction.compare_tables(self.generated, self.expected)

    def test_identical_bytes_pass(self):
        (self.generated / "responses.parquet").write_bytes(self.reference.read_bytes())
        self.compare()

    def test_same_size_and_timestamp_with_different_bytes_fails(self):
        actual = self.generated / "responses.parquet"
        actual.write_bytes(b"PAR1diff bytesPAR1")
        timestamp = self.reference.stat().st_mtime_ns
        os.utime(actual, ns=(timestamp, timestamp))
        with self.assertRaisesRegex(ValueError, "Byte mismatch"):
            self.compare()

    def test_missing_optional_table_fails(self):
        self.expected = {"traces.parquet": self.reference}
        with self.assertRaisesRegex(ValueError, "Missing generated tables: traces.parquet"):
            self.compare()

    def test_extra_table_fails(self):
        (self.generated / "responses.parquet").write_bytes(self.reference.read_bytes())
        (self.generated / "assets.parquet").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected generated tables: assets.parquet"):
            self.compare()

    def test_empty_comparison_fails(self):
        self.expected = {}
        with self.assertRaisesRegex(ValueError, "empty comparison"):
            self.compare()


class BuildExecutionTests(RepositoryTest):
    def mock_hub(self, hub=None):
        return patch.dict(sys.modules, {
            "huggingface_hub": hub or self.hub(),
            "huggingface_hub.errors": SimpleNamespace(RemoteEntryNotFoundError=RemoteEntryNotFoundError),
        })

    def hub(self):
        api = Mock()
        api.list_repo_tree.return_value = [file("alpha/responses.parquet")]

        def download(repo, filename, *, repo_type, revision, local_dir):
            self.assertEqual((repo, repo_type, revision), (reproduction.HF_REPOSITORY, "dataset", REVISION))
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fresh build")
            return str(path)

        return SimpleNamespace(HfApi=lambda: api, hf_hub_download=download)

    def test_actual_builder_runs_in_clean_workspace_without_touching_local_data(self):
        self.write("benchmarks/alpha/build.py", """
import os
from pathlib import Path
assert Path.cwd().name == 'measurement_db'
for key in ('MEASUREMENT_DB_SOURCE_REPO', 'MEASUREMENT_DB_SOURCE_REVISION',
            'MEASUREMENT_DB_SOURCE_MANIFEST'):
    assert key not in os.environ, key
assert os.environ['GITHUB_TOKEN'] == 'test-builder-token'
directory = Path(__file__).parent
assert not (directory / 'raw').exists()
assert not (directory / 'formatted_tables').exists()
output = directory / 'formatted_tables'
output.mkdir()
(output / 'responses.parquet').write_bytes(b'fresh build')
""")
        old_raw = self.write("benchmarks/alpha/raw/input.json", "local input")
        old_output = self.write("benchmarks/alpha/formatted_tables/responses.parquet", "local output")
        with self.mock_hub(), \
                patch.dict(os.environ, {
                    "MEASUREMENT_DB_SOURCE_REPO": "other/archive",
                    "MEASUREMENT_DB_SOURCE_REVISION": "stale-archive-revision",
                    "MEASUREMENT_DB_SOURCE_MANIFEST": "stale-local-manifest",
                    "GITHUB_TOKEN": "test-builder-token",
                }), \
                contextlib.redirect_stdout(io.StringIO()):
            reproduction.verify_benchmark(self.root, "alpha", REVISION)
        self.assertEqual(old_raw.read_text(), "local input")
        self.assertEqual(old_output.read_text(), "local output")

    def test_nonzero_builder_exit_fails_even_when_outputs_exist(self):
        self.write("benchmarks/alpha/build.py", "raise SystemExit(3)\n")
        hub = self.hub()
        with self.mock_hub(hub), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                reproduction.verify_benchmark(self.root, "alpha", REVISION)
        hub.HfApi().list_repo_tree.assert_not_called()

    def test_deleted_builder_fails(self):
        with self.mock_hub():
            with self.assertRaisesRegex(ValueError, "Missing builder"):
                reproduction.verify_benchmark(self.root, "missing", REVISION)

    def successful_builder(self):
        marker = self.root / "builder-ran"
        self.write("benchmarks/alpha/build.py", f"""
from pathlib import Path
output = Path(__file__).parent / 'formatted_tables'
output.mkdir()
(output / 'responses.parquet').write_bytes(b'fresh build')
Path({str(marker)!r}).write_text(str(output))
""")
        return marker

    def test_missing_reference_is_checked_only_after_the_real_builder_finishes(self):
        for error in (None, RemoteEntryNotFoundError("folder absent")):
            with self.subTest(error=error):
                marker = self.successful_builder()
                marker.unlink(missing_ok=True)
                hub = self.hub()

                def listing(*args, **kwargs):
                    self.assertTrue(marker.is_file())
                    self.assertTrue(Path(marker.read_text()).is_dir())
                    if error:
                        raise error
                    return []

                hub.HfApi().list_repo_tree.side_effect = listing
                phases = []
                with self.mock_hub(hub), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(reproduction.MissingReferenceError):
                        reproduction.verify_benchmark(self.root, "alpha", REVISION,
                                                      progress=lambda phase, revision=None: phases.append(phase))
                self.assertEqual(phases, ["build", "reference", "reference"])
                self.assertFalse(Path(marker.read_text()).exists())  # Scratch cleanup still runs.

    def test_unavailable_reference_revision_does_not_prevent_the_build(self):
        marker = self.successful_builder()
        hub = self.hub()
        with self.mock_hub(hub), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "could not be pinned"):
                reproduction.verify_benchmark(self.root, "alpha", "")
        self.assertTrue(marker.is_file())
        hub.HfApi().list_repo_tree.assert_not_called()

    def test_cli_reference_resolution_failure_still_builds_every_benchmark(self):
        alpha_marker = self.successful_builder()
        beta_marker = self.root / "beta-ran"
        self.write("benchmarks/beta/build.py", (self.root / "benchmarks/alpha/build.py").read_text()
                   .replace(str(alpha_marker), str(beta_marker)))
        path = self.root / "report.json"

        def resolve(revision):
            self.assertTrue(alpha_marker.exists())
            raise PermissionError("reference access denied")

        with self.mock_hub(), patch.object(reproduction, "ROOT", self.root), \
                patch.object(sys, "argv", ["ci", "verify", "alpha", "beta", "--report", str(path)]), \
                patch.object(reproduction, "resolve_revision", side_effect=resolve), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reproduction.main(), 1)
        rows = json.loads(path.read_text())["results"]
        self.assertTrue(beta_marker.exists())
        self.assertEqual([row["status"] for row in rows], ["reference_failed", "reference_failed"])
        self.assertTrue(all(row["build_status"] == "passed" for row in rows))

    def test_missing_reference_does_not_skip_pair_and_both_use_one_pinned_revision(self):
        alpha_marker = self.successful_builder()
        beta_marker = self.root / "beta-ran"
        self.write("benchmarks/beta/build.py", (self.root / "benchmarks/alpha/build.py").read_text()
                   .replace(str(alpha_marker), str(beta_marker)))
        path = self.root / "report.json"
        hub = self.hub()
        hub.HfApi().repo_info.return_value = SimpleNamespace(sha=REVISION)

        def listing(repo, *, repo_type, revision, path_in_repo):
            self.assertEqual(revision, REVISION)
            if path_in_repo == "alpha":
                self.assertTrue(alpha_marker.exists())
                raise RemoteEntryNotFoundError("alpha has no published reference")
            self.assertTrue(beta_marker.exists())
            return [file("beta/responses.parquet")]

        hub.HfApi().list_repo_tree.side_effect = listing
        with self.mock_hub(hub), patch.object(reproduction, "ROOT", self.root), \
                patch.object(sys, "argv", ["ci", "verify", "alpha", "beta", "--report", str(path)]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reproduction.main(), 1)
        report = json.loads(path.read_text())
        self.assertEqual([row["status"] for row in report["results"]], ["reference_missing", "passed"])
        self.assertTrue(all(row["build_status"] == "passed" for row in report["results"]))
        self.assertEqual(report["hf_revision"], REVISION)
        hub.HfApi().repo_info.assert_called_once()

    def test_reference_access_and_network_errors_are_not_missing_references(self):
        self.successful_builder()
        for error in (PermissionError("gated reference"), ConnectionError("connection reset")):
            with self.subTest(error=error):
                path = self.root / "report.json"
                hub = self.hub()
                hub.HfApi().list_repo_tree.side_effect = error
                with self.mock_hub(hub), patch.object(reproduction, "ROOT", self.root), \
                        patch.object(sys, "argv", ["ci", "verify", "alpha", "--revision", REVISION,
                                                  "--report", str(path)]), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(reproduction.main(), 1)
                row = json.loads(path.read_text())["results"][0]
                self.assertEqual((row["status"], row["build_status"]), ("reference_failed", "passed"))

    def test_zero_exit_without_generated_tables_is_a_build_failure(self):
        hub = self.hub()
        with self.mock_hub(hub), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "produced no Parquet"):
                reproduction.verify_benchmark(self.root, "alpha", REVISION)
        hub.HfApi().list_repo_tree.assert_not_called()

    def test_batch_continues_after_failure_but_exits_unsuccessfully(self):
        with patch.object(sys, "argv", ["benchmark_reproduction.py", "verify", "alpha", "beta"]), \
                patch.object(reproduction, "resolve_revision", return_value=REVISION), \
                patch.object(reproduction, "verify_benchmark", side_effect=[ValueError("mismatch"), None]) as verify, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reproduction.main(), 1)
        self.assertEqual([call.args[1] for call in verify.call_args_list], ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
