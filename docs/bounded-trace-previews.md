# Bounded structured trace previews

Canonical `traces.parquet` records remain complete `trace/1` values. A separate,
generic exporter prepares their display data before a browser reads the source.
No benchmark converters, grades, identities, assets, chart fits or canonical
tables change when this display cache is prepared.

The cache contains independently gzipped, SHA-256 checked ranges in binary packs:

- Hashed response-ID indexes select a small trace header and first step page.
- Headers preserve whole-trace counts and recorded run/evaluation details.
- Step pages contain at most 32 steps and at most 1 MiB expanded/compressed.
  Root and step `native` provenance remains in the original source table.
- Long display strings have an explicit prefix and exact Unicode-safe text
  parts. The viewer can read every part without assembling the whole value.
- Search pages contain the complete message/output text of every step. Search
  scans them progressively, supports cancellation and matches across text-part
  boundaries. Native provenance is outside this declared search scope. A very
  large result set reports its full count and limits navigation to the first
  10,000 matching steps.

The normal browser path never reads the source Parquet row when a matching
display cache exists. Original-source inspection remains separate and lazy;
the existing 16 MiB source-reader guard and original-table download remain in
place. This feature does not change grading labels or Markdown rendering.

## Explicit preparation

Prepare only selected benchmarks, using complete, pinned structured tables:

```sh
python -m scripts.trace_previews \
  --tables /path/to/selected-tables \
  --benchmarks workarena visualwebarena \
  --output /path/to/trace-preview-cache \
  --source-revision <immutable-source-commit>
```

An optional publication adds only preview artifacts to an explicit HF dataset
`preview/` branch. Use `--publish-branch`, `--publish-base` if a different branch
parent is needed, and `--website-manifest
website/content/generated/trace-previews.json`. The generated manifest pins both
the exact source-table URL and the immutable preview-pack revision. A cache for
another source URL cannot supply the selected record. Publishing verifies local
source hashes against HF and uploaded pack hashes against the server.

Cache directories are keyed by source-table SHA-256 and exporter identity.
Verification reconstructs all displayed fields and complete search text and
compares them with the canonical records. Incomplete exports are discarded;
pages that exceed the budget fail preparation instead of producing partial
records. Ordinary frontend builds consume pinned caches without running this
exporter or fitting gallery charts.

## Validation

`tests/test_trace_previews.py` exercises large native payloads, Unicode text,
paging, exact reconstruction, stale caches and corrupt packs. The real Python
wire fixture is consumed by `website/scripts/test-trace-preview.mjs`, covering
bounded first loads, full-text search beyond display prefixes, cancellation and
source isolation. The trace-preview route has independent allowlist/range tests.
Existing gallery/client tests cover benchmarks without a prepared display cache.

## Preview rollout coverage

The PR preview publishes bounded caches for WorkArena, VisualWebArena and
TheAgentCompany. TheAgentCompany's cache covers all released OpenHands, OWL
role-play and MUSE transcript records. The wire test also checks that all three
benchmarks retain published display-cache entries; cached source downloads alone
cannot satisfy this rollout check.
