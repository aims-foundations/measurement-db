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

    def test_large_migration_checks_every_benchmark_within_matrix_limit(self):
        slugs = [f"benchmark_{n}" for n in range(600)]
        matrix = reproduction.build_matrix(slugs)["include"]
        self.assertLessEqual(len(matrix), 256)
        self.assertEqual([slug for job in matrix for slug in job["benchmarks"]], slugs)
        self.assertEqual(reproduction.build_matrix([]), {"include": []})


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
