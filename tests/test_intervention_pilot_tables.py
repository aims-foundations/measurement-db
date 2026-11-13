"""Guard observation identity, cohort ownership, and missing-result semantics."""
from datetime import date
import json
import unittest

from scripts.analyze_measurements import intervention_pilot_tables as pilot


def fixture_rows():
    rows = {name: [] for name in pilot.SCHEMAS}
    rows["benchmarks"] = [dict(benchmark_id="benchmark", name="Benchmark", version=None,
        protocol_ref="protocol", source_revision="revision", scope="one task",
        n_available_item_records=1, identity_status="test")]
    iid = pilot.item_identity("benchmark", "task", "protocol")
    rows["items"] = [dict(item_id=iid, benchmark_id="benchmark", raw_item_id="task", content=None,
        source_reference_answer=None, protocol_ref="protocol", source_locator="source", metadata_status="test")]
    rows["subjects"] = [dict(subject_id="subject", model="model", harness="harness", harness_version=None,
        configuration_json="{}", metadata_status="test")]
    for eid, grade in [("original", 1.0), ("replication", None)]:
        rows["experiments"].append(dict(experiment_id=eid, study_id="study", name=eid, status="observed",
            paper_url="paper", repository_url="repo", source_revision="revision", evaluation_date=None,
            n_responses=1, n_known_grades=int(grade is not None), n_observed_subjects=1))
        rows["experiment_benchmarks"].append(dict(experiment_id=eid, benchmark_id="benchmark", analysis_scope_id=eid,
            n_cohort_items=1, n_responses=1, n_known_grades=int(grade is not None), n_observed_subjects=1))
        rows["cohort_items"].append(dict(experiment_id=eid, benchmark_id="benchmark", item_id=iid))
        rows["arms"].append(dict(arm_id=eid+":arm", experiment_id=eid, source_arm_id="arm", definition_json="{}",
            definition_source="source", status="observed"))
        response = dict(experiment_id=eid, benchmark_id="benchmark", subject_id="subject", item_id=iid,
                        trial=1, test_condition=None, interactors=None, response=grade)
        response["response_id"] = pilot.response_identity(response)
        rows["responses"].append(response)
        rows["response_provenance"].append(dict(response_id=response["response_id"], arm_id=eid+":arm",
            source_repeat="0", source_run=None, source_locator="source:"+eid, source_revision="revision", trace_member=None))
        rows["response_metrics"].append(dict(response_id=response["response_id"], metric="success_all_attempts",
            value=grade if grade is not None else 0.0, unit="binary", policy="unknown counted as failure for this metric"))
    return rows


def tables(rows):
    return {name: pilot.typed_table(name, values) for name, values in rows.items()}


class PilotIdentityTests(unittest.TestCase):
    def test_two_experiments_reuse_subject_item_and_trial(self):
        rows = fixture_rows()
        first, second = rows["responses"]
        self.assertEqual(first["subject_id"], second["subject_id"])
        self.assertEqual(first["item_id"], second["item_id"])
        self.assertEqual(first["trial"], second["trial"])
        self.assertNotEqual(first["response_id"], second["response_id"])
        self.assertEqual(pilot.validate(tables(rows))["status"], "passed")

    def test_grade_correction_does_not_create_new_attempt(self):
        row = fixture_rows()["responses"][0]
        corrected = dict(row, response=0.0)
        self.assertEqual(pilot.response_identity(row), pilot.response_identity(corrected))

    def test_duplicate_within_experiment_rejected_even_with_new_id(self):
        rows = fixture_rows()
        duplicate = dict(rows["responses"][0], response_id="different-id", response=0.0)
        rows["responses"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate observation key"):
            pilot.validate(tables(rows))

    def test_unknown_grade_stays_null_beside_derived_zero(self):
        data = tables(fixture_rows())
        self.assertEqual(data["responses"]["response"].null_count, 1)
        self.assertEqual(data["response_metrics"]["value"].to_pylist(), [1.0, 0.0])
        pilot.validate(data)

    def test_response_must_belong_to_experiment_cohort(self):
        rows = fixture_rows()
        rows["cohort_items"] = rows["cohort_items"][:1]
        with self.assertRaisesRegex(ValueError, "unknown cohort item"):
            pilot.validate(tables(rows))

    def test_arm_provenance_cannot_point_to_other_experiment(self):
        rows = fixture_rows()
        rows["response_provenance"][0]["arm_id"] = "replication:arm"
        with self.assertRaisesRegex(ValueError, "arm experiment mismatch"):
            pilot.validate(tables(rows))

    def test_design_only_does_not_accept_response_placeholders(self):
        rows = fixture_rows()
        rows["experiments"][1]["status"] = "design_only_results_not_located"
        with self.assertRaisesRegex(ValueError, "fabricated responses"):
            pilot.validate(tables(rows))

    def test_null_identity_and_unexpected_columns_rejected(self):
        row = fixture_rows()["responses"][0]
        with self.assertRaisesRegex(ValueError, "unexpected null"):
            pilot.typed_table("responses", [dict(row, experiment_id=None)])
        with self.assertRaisesRegex(ValueError, "columns differ"):
            pilot.typed_table("responses", [dict(row, unsupported="extra")])

    def test_source_yaml_dates_preserve_iso_value(self):
        self.assertEqual(json.loads(pilot.encode({"date": date(2025, 7, 14)})), {"date": "2025-07-14"})


if __name__ == "__main__":
    unittest.main()
