"""Large trace corpora retain the canonical schema across Arrow batches."""
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_measurement_tables import write_measurement_tables as writer


@pytest.mark.parametrize("rows", [0, 5])
def test_batched_trace_write_preserves_values_types_and_nulls(tmp_path, monkeypatch, rows):
    # Exercise several batches without allocating the >2 GB corpus that exposed
    # Arrow's string-offset limit. AIR-Bench supplies the full-size integration check.
    monkeypatch.setattr(writer, "_ARROW_BATCH_ROWS", 2)
    data = pd.DataFrame({
        "response_id": [f"response-{i}" for i in range(rows)],
        "subject_id": ["subject"] * rows,
        "item_id": [f"item-{i}" for i in range(rows)],
        "benchmark_id": ["benchmark"] * rows,
        "trial": pd.Series([1] * rows, dtype="int64"),
        "test_condition": [None] * rows,
        "interactors": [None] * rows,
        "trace": [f'{i}: {{"text": "' + "α" * 20_001 + '"}\n' for i in range(rows)],
    })
    output = tmp_path / "traces.parquet"
    writer.write_parquet(data, output)
    recovered = pq.read_table(output)
    writer.validate_parquet_schema("traces", recovered.schema)
    assert recovered.to_pylist() == data.to_dict("records")
    assert recovered.schema.metadata[b"measurement_db.schema_version"] == b"4"
    if rows:
        assert pq.ParquetFile(output).num_row_groups == 3


def test_invalid_later_batch_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_ARROW_BATCH_ROWS", 1)
    schema = writer.canonical_arrow_schema("subjects")
    data = pd.DataFrame([{name: None for name in schema.names} for _ in range(2)])
    data["subject_id"] = ["first", "second"]
    data["display_name"] = ["first", "second"]
    data["subject_features_extra"] = pd.Series(["valid", "invalid Unicode \ud800"], dtype=object)
    output = tmp_path / "subjects.parquet"
    output.write_bytes(b"previous release")
    with pytest.raises(UnicodeEncodeError):
        writer.write_parquet(data, output)
    assert output.read_bytes() == b"previous release"


def test_short_attempts_share_bounded_groups_and_large_answers_stay_isolated(tmp_path):
    traces = ['α' * 20] * 4 + ['β' * 2000] + ['γ' * 20] * 4
    data = pd.DataFrame(dict(attempt_id=[f'a{i}' for i in range(len(traces))],
        benchmark_id='benchmark', subject_id='subject', raw_attempt_id=[str(i) for i in range(len(traces))], trace=traces))
    output = tmp_path / 'attempts.parquet'
    writer.write_parquet(data, output, row_group_bytes=500)
    file = pq.ParquetFile(output)
    assert file.read().to_pylist() == data.to_dict('records')
    assert file.num_row_groups < len(data)
    for index in range(file.num_row_groups):
        group = file.read_row_group(index)
        assert group.nbytes <= 500 or group.num_rows == 1
        assert file.metadata.row_group(index).column(4).statistics is None
