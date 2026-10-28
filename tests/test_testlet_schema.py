"""Structural links preserve identity, native execution boundaries and v3 data."""
import copy
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_tabular_builder import fixture  # noqa: F401
from scripts.build_measurement_tables import reload, validate_dataset
from scripts.build_measurement_tables.migrate_testlet_schema import migrate_dataset


def build(builder, source, output):
    reload()
    builder.main_from_args(["--source", str(source), "--output", str(output)])
    return {p.stem: pd.read_parquet(p) for p in output.glob("*.parquet")}


def test_structural_links_do_not_change_existing_item_or_response_ids(fixture):
    builder, source, output = fixture
    before = build(builder, source, output)
    builder.inputs["items"]["testlet_id"] = "original-task"
    builder.inputs["attempts"] = pd.DataFrame([
        dict(attempt_key="run", subject_key="s1", raw_attempt_id="original-run", trace="shared answer")])
    builder.inputs["responses"]["attempt_key"] = "run"
    after = build(builder, source, output.parent / "linked")
    for table, key in (("items", "item_id"), ("responses", "response_id")):
        assert before[table][key].tolist() == after[table][key].tolist()
    assert after["items"].item_features.isna().all()
    assert after["responses"].attempt_id.nunique() == 1
    assert len(after["attempts"]) == 1
    validate_dataset(after)
    for mutation in ("missing", "different_subject"):
        invalid = copy.deepcopy(after)
        if mutation == "missing":
            invalid["responses"].loc[0, "attempt_id"] = "absent"
        else:
            invalid["attempts"].loc[0, "subject_id"] = "different-subject"
        with pytest.raises(RuntimeError, match="attempt"):
            validate_dataset(invalid)


def test_ungraded_attempt_can_exist_without_fabricated_item_responses(fixture):
    builder, source, output = fixture
    builder.inputs["attempts"] = pd.DataFrame([
        dict(attempt_key="ungraded", subject_key="s1", raw_attempt_id="no-evaluation", trace="complete execution")])
    tables = build(builder, source, output)
    assert len(tables["responses"]) == 5
    assert tables["responses"].attempt_id.isna().all()
    assert tables["attempts"].raw_attempt_id.tolist() == ["no-evaluation"]
    validate_dataset(tables)


def test_conflicting_membership_cannot_silently_merge_identical_items(fixture):
    builder, source, output = fixture
    builder.inputs["items"]["testlet_id"] = ["one-task", "different-task"]
    with pytest.raises(ValueError, match="conflicting registration.*testlet_id"):
        build(builder, source, output)
    assert not (output / "items.parquet").exists()


@pytest.mark.parametrize("field,value", [("attempt_key", None), ("attempt_key", True), ("subject_key", "absent")])
def test_invalid_attempt_keys_fail_before_writing(fixture, field, value):
    builder, source, output = fixture
    attempt = dict(attempt_key="run", subject_key="s1", raw_attempt_id="run", trace=None)
    attempt[field] = value
    builder.inputs["attempts"] = pd.DataFrame([attempt])
    with pytest.raises(RuntimeError, match="attempts"):
        build(builder, source, output)
    assert not (output / "attempts.parquet").exists()


def test_v3_migration_preserves_all_values_and_does_not_infer_membership(fixture):
    builder, source, output = fixture
    build(builder, source, output)
    originals = {}
    for name, column in (("items", "testlet_id"), ("responses", "attempt_id")):
        path = output / f"{name}.parquet"
        old = pq.read_table(path).drop([column])
        metadata = {**old.schema.metadata, b"measurement_db.schema_version": b"3"}
        pq.write_table(old.replace_schema_metadata(metadata), path)
        originals[name] = pq.read_table(path)
    original_bytes = {p.name: p.read_bytes() for p in output.glob("*.parquet")}
    migrated = output.parent / "migrated"
    receipt = migrate_dataset(output, migrated)
    for name, column in (("items", "testlet_id"), ("responses", "attempt_id")):
        current = pq.read_table(migrated / f"{name}.parquet")
        assert current.column(column).null_count == current.num_rows
        assert current.drop([column]).equals(originals[name], check_metadata=False)
        assert current.schema.metadata[b"measurement_db.schema_version"] == b"4"
        assert receipt[name]["added"] == column
    assert {p.name: p.read_bytes() for p in output.glob("*.parquet")} == original_bytes
    validate_dataset({p.stem: pd.read_parquet(p) for p in migrated.glob("*.parquet")})
    with pytest.raises(FileExistsError):
        migrate_dataset(output, migrated)


def test_migration_failure_leaves_source_and_destination_untouched(fixture):
    builder, source, output = fixture
    build(builder, source, output)
    path = output / "items.parquet"
    invalid = pq.read_table(path).append_column("unknown", pa.array(["extra"]))
    pq.write_table(invalid, path)
    destination = output.parent / "failed"
    with pytest.raises(ValueError, match="canonical"):
        migrate_dataset(output, destination)
    assert not destination.exists()
    assert "unknown" in pq.read_schema(path).names
