"""Write validated registration tables into a benchmark dataset folder."""

from __future__ import annotations

from pathlib import Path

from .register_measurements import _locked_registration_rows
from .validate_measurement_tables import _validated_registration_tables


def save(out_dir: Path | str) -> None:
    """Validate and write this process's registered measurement tables.

    Non-empty ``subjects``, ``items``, and ``benchmarks`` tables are written
    beside the benchmark's ``responses.parquet``. Validation completes before
    the output directory is created, preventing partial output on failure.
    """
    output_directory = Path(out_dir)
    with _locked_registration_rows() as rows:
        tables = _validated_registration_tables(rows, output_directory.name)
        output_directory.mkdir(parents=True, exist_ok=True)
        for table_name, table in tables.items():
            table.to_parquet(
                output_directory / f"{table_name}.parquet", index=False
            )
