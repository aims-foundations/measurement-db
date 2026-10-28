# Testlet schema migration

Branch: `feat/testlet-schema`, based on `feat/coverage-explorer` at
`f7d85b32951ea273b9a467636eb3026e4fc95451`.
Catalog source: `aims-foundations/measurement-db` at immutable HF revision
`15dc9f5365087e720a6af6f3e45af834d283b0f8` (270 published benchmarks).

## Schema and builder changes

- `items.testlet_id` groups items sharing an upstream task. It is scoped by
  benchmark, separate from item features, and excluded from item identity.
- `responses.attempt_id` optionally links grades to a shared execution. It is
  excluded from response identity; the existing observation key still requires
  distinct trials for repeated measurements.
- Optional `attempts.parquet` stores the subject, original execution ID and
  complete available trajectory once. Executions without evaluations can remain
  here without invented item outcomes. Response-specific `traces` remain valid.

Existing builders emit null links without source changes. Null means membership
has not been documented. The shared validator checks keys, ownership, grade
domains and attempt references. Conflicting membership for an identical item is
rejected rather than silently overwriting a relationship.

## Catalog migration

All 269 other benchmarks were migrated into a separate local directory using
`scripts/build_measurement_tables/migrate_testlet_schema.py`. It adds null links
and rereads every original value, including row order, IDs, grades and nulls.
Source files are untouched; subjects and benchmark metadata are copied unchanged.

The migrated tables contain 8,402,387 items and 92,055,532 observations. The
[per-benchmark receipt](testlet-schema-v4-migration.csv) records counts and original
and migrated file hashes. This is a structural migration, not a new audit of each
benchmark's upstream measurements. Existing characterization digests are accepted
only if all original values match and the new relationship columns are all null.

The local catalog directory is an **overlay**: unchanged traces and assets stay
at the pinned source revision. Their absence from the overlay must not be used
as a reason to delete them from a release.

## ProgramBench conversion

The builder uses the same 26 pinned source files as its parent: the registry at
`c19e570f64719c8326ccdcc7af803447cb89d1a6` and all 25 original submission archives.
Every downloaded file matched its declared size and SHA256.

| Record | Count |
| --- | ---: |
| Original program testlets | 200 |
| Distinct items, including prompt and grader variants | 1,483,092 |
| Binary test outcomes | 6,119,913 |
| Shared executions | 4,965 |
| Subject configurations | 28 |

One published active Boolean test is one outcome. The full branch/test name and
grader version identify its verifier. Grader and prompt variants remain distinct
items within the original task's testlet. The 28 subject rows and IDs match both
the pinned HF release and the parent's reviewed fingerprint exactly.

All 4,964 evaluated executions contribute their released individual verdicts.
The single execution without an evaluation is preserved without fabricated
outcomes. Native statuses, errors, repeated test records and original model
settings remain in the complete shared trajectory. Each execution has its own
Parquet row group for independent answer retrieval.

An attempt's mean recovers its nonempty active-test fraction. The original
aggregate convention gives an explicitly published empty map zero; an absent
evaluation remains unavailable. Neither invents an individual test outcome.
An official macro-average
over tasks must average those task fractions; pooling all tests would give
programs with more tests greater weight. This branch records the relationships;
testlet-aware statistical fitting is a separate consumer of them.

## Validation and publication

The full native audit passed: every test outcome, item association, execution and
full trajectory matched independently parsed upstream records. It also checks the
existing explicit model mappings and preserved original labels. The regression
fixture exercises ignored tests, repeated verdicts, empty and missing evaluations,
alias provenance and deliberately corrupted records.

Focused schema, migration, builder, source-audit and transport tests pass. The
website's 52 tests and production build pass. The broader regression run found
an unchanged 3EED subject-label audit failure, reproduced on the parent branch;
DecompileBench's missing test dependency was supplied in an isolated directory
and its checks then passed.

Generated tables are staged locally under
`artifacts/testlet-migration/catalog/<benchmark>/formatted_tables` and
`benchmarks/programbench/formatted_tables`. This branch does not publish HF
tables or update the deployed website. A release must retain unchanged catalog
tables, replace ProgramBench's old task-level tables (including retiring its old
`traces.parquet`), and prepare gallery records, answer previews and analysis from
the new coherent snapshot. The additive migration receipts do not claim that
all 269 upstream builders were re-executed or that the byte-for-byte reproduction
gate has run against a new HF reference.

Reproduction entry points:

```sh
python -m scripts.build_measurement_tables.migrate_testlet_schema SOURCE OUTPUT
python benchmarks/programbench/build.py
python tests/test_benchmark_datasets.py programbench
```
