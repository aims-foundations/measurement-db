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
`benchmarks/programbench/formatted_tables`. The scoped preview described below
publishes ProgramBench's converted tables; the other 269 schema overlays remain
local. A full schema release must retain unchanged catalog values and prepare
analysis from the new coherent snapshot. The additive migration receipts do not claim that
all 269 upstream builders were re-executed or that the byte-for-byte reproduction
gate has run against a new HF reference.

Reproduction entry points:

```sh
python -m scripts.build_measurement_tables.migrate_testlet_schema SOURCE OUTPUT
python benchmarks/programbench/build.py
python tests/test_benchmark_datasets.py programbench
```

## Testlet matrix preview

The matrix keeps one item per column and each released binary observation intact.
Shared headers, neutral borders, and small gaps group adjacent columns by the
explicit `testlet_id`. Sorting works within each group. Both heatmap and table
views show continuation labels across bounded pages; a testlet selector jumps to
its first page. No group-level scores or collapsed aggregate columns are created.

ProgramBench's five canonical tables are pinned to HF preview revision
`a7a2d0b153057662838c26b88f3597413ddbd9c5` on `preview/testlet-schema`, based on
`15dc9f5365087e720a6af6f3e45af834d283b0f8`. The old ProgramBench `traces.parquet`
is removed on that branch; shared traces live in `attempts.parquet`. Upload
SHA256 values match all five local files. HF main was unchanged by this upload.
Two oversized attempt records use bounded previews pinned to
`944a7ce66d74d327eefa486e7b367e51381201e1`; other records remain readable through
the existing Parquet path. Other benchmarks reuse their existing prepared assets.

The complete browser export contains 5,794 pages. Every page obeys the existing
256-item, 4,096-observation, 2 MiB limits; the largest expanded page is 698,053
bytes. Full-record checksums verify all 6,119,913 observations against Parquet
(2,579,560 passing and 3,540,353 failing), all 1,483,092 item indexes, and all 200
testlet memberships and continuation boundaries. Page 3,200 contains 128 items
from each of two adjacent testlets and is the preview's initial comparison view.

Focused tests cover source preservation, null/legacy membership compatibility,
partial items, corrupted membership, sorting/filtering, table subpages, and
pointer/keyboard geometry. Prepared evidence is in `artifacts/testlet-demo/`.
The isolated `testlet-preview.yml` workflow verifies the prepared hashes and
deploys only a Vercel preview. UI-only updates reuse these prepared data packs.
The existing Rasch fit is intentionally unavailable for the converted ProgramBench
matrix because its item/observation counts no longer match the source.
