# Subject identity refresh, November 12, 2026

The refreshed canonical tables are published at
[`88f4a665524562532bb4041e6eea600b275f024d`](https://huggingface.co/datasets/aims-foundations/measurement-db/commit/88f4a665524562532bb4041e6eea600b275f024d),
based on `52603c340627ad32f1591862d3d4970d5e16b23f` and source commit `eaf5fcf6425d0d7d381fd96342a451e02c69e043`.
The [validation evidence](audits/subject-identity-refresh-20261112.json) records
all 77 public builds, independent source audits, preservation checks, source
snapshot reviews, and remote file verification.

The refresh preserves 27,807,479 observations,
including null grades, all task and asset content, complete trace payloads, and
distinct evaluation configurations. Reviewed model labels and agent harnesses
are separated while original source labels and configuration fields remain
available. This uses the current schema and introduces no testlet migration.

The refreshed tables yield 270 catalog benchmarks, 981 canonical subject names,
4,130 observed benchmark-subject links, and 35 unmapped benchmarks. Those 35
remain visible in the network's separate grid and inspection list. FrontierOR
is deferred because its live author results no longer match the pinned snapshot;
its benchmark files and published tables remain unchanged. MedArabiq, OSWorld,
and SWE-Together remain withheld.

The three one-off October preview workflows are retired. New previews use the
normal PR reproduction and gallery pipeline, which compares every generated
Parquet byte against the pinned canonical release before preparing the website.
