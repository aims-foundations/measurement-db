# Benchmark characterization

Keep `characterization.yaml` beside `metadata.yaml`, `build.py`, and `test.py`.
The metadata describes the benchmark and upstream sources. The characterization
records reviewed output expectations and the source evidence used to check them.
Its contract is defined in [`characterization_schema.yaml`](characterization_schema.yaml).
The builder never reads this file, and tests never update it automatically.

## Required fields

- `format_version`: `1`, identifying both the file format and hashing algorithm.
- `tables`: expected `rows` and `logical_sha256` for each generated canonical table.
  Include subjects, items, benchmarks, responses when released, and traces/assets
  when present. Omit absent optional tables; record present empty tables explicitly.
- `source_claims`: at least one named, independently checkable claim. Every claim
  requires `kind`, `origin`, `source`, `locator`, `scope`, and `expected`.
  Unknown fields and duplicate YAML keys are rejected.

A claim's `kind` is `count` or `aggregate`. Counts are nonnegative integers and
must match exactly. Aggregates are finite numbers or mappings from group names
to finite numbers, and require a justified nonnegative `absolute_tolerance`.
Group names must match exactly; comparison uses absolute tolerance only.

Set `origin` to `reported` for a quantity explicitly stated by the benchmark's
creators, or `derived_from_data` for a quantity independently measured from
released raw data. A reported claim can cite a paper PDF, an official blog post,
a dataset card, or a published results file. `source` is its HTTP(S) URL;
`locator` identifies the page, table, section, file, or field. `scope` specifies
the release, split, inclusion criteria, units, and aggregation as applicable.
For example, a percentage and a fraction require different units and tolerances.
Prefer a versioned URL where available; archived-data locators refer to the
snapshot selected by the shared source loader. Tests do not fetch papers or
blog posts: curators record the cited claim and tests reconstruct its quantity.

The following is an illustrative claim, not a benchmark's measured result:

```yaml
source_claims:
  held_out_items:
    kind: count
    origin: reported
    source: https://example.org/benchmark-report.pdf
    locator: 'Table 2, held-out split'
    scope: 'Version 1, distinct released held-out items'
    expected: 100
```

Do not invent a published claim when no comparable statistic is reported.
Derive an expectation from the archived upstream records and document how it
was counted. Do not obtain source expectations by copying the builder's outputs.
A full-benchmark paper count should not be compared directly with a released
subset unless the difference is explicitly accounted for. Neither a citation
nor matching an overall count alone proves that each response was parsed correctly;
retain source-specific record, grading, and trace checks in `test.py`.

## Review and checks

1. Run the builder and independently reconcile its output with the released
   records and applicable reported results. Document decisions in the benchmark's
   curation record.
2. Record source claims and the reviewed table counts and hashes. The shared
   `characterize_tables(tables)` function computes candidate table expectations;
   review them before saving them. Existing tests must pass before migrating
   an already accepted snapshot to a new format.
3. In `test.py`, use `load_characterization(path)` to enforce the schema,
   `check_tables(characterization, tables)` to compare all produced tables, and
   `check_source_claims(characterization, observations)` to verify source evidence.
   Compute observations independently of the builder and reconcile them with the
   curated records. Missing or extra claim IDs fail; no declared claim is ignored.
4. Run the full benchmark test. After an intentional data change, review its
   source and output differences before updating expectations. A failing hash
   is not a reason by itself to replace the stored hash.

The shared helpers live in
[`validate_characterization.py`](scripts/build_measurement_tables/validate_characterization.py).
Benchmark-specific parsing and historical migration assertions remain in
`test.py`; the YAML has the same closed structure for every benchmark.

```bash
python -m scripts.build_measurement_tables.validate_characterization --benchmarks-dir benchmarks
python benchmarks/<slug>/build.py
MEASUREMENT_DB_FULL_TEST=1 python benchmarks/<slug>/test.py -v
```

The first command validates definitions without downloading data and requires a
file for every non-template benchmark. The full test executes the comparisons.
The template intentionally contains no fabricated characterization; its source
audit must be implemented and its expected values reviewed for the new benchmark.

## Logical hashes

Format 1 hashes all column names and the sorted multiset of canonical row hashes
with SHA-256. It normalizes equivalent missing values and NumPy scalars, preserves
exact text and list order, and hashes asset bytes. Repeated rows contribute
repeated hashes. Row order and Parquet compression do not affect the result.
Column order remains part of the hash and is fixed by the canonical table schema.
The existing dataset validator separately enforces storage types, keys, grading
scales, relationships, and derived statistics.
