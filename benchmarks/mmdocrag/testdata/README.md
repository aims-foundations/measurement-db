# MMDocRAG characterization

`characterization.json` contains reviewed output expectations, not provider data.
The builder never reads it. Tests never download sources or rewrite this file.

The snapshot records:

- `legacy`: table shapes, null counts, columns, and logical fingerprints from
  the saved pre-migration Parquet tables.
- `preserved`: response and trace fingerprints calculated from those legacy
  tables and verified against the migrated output. Subject IDs are replaced
  by NFC-normalized, lowercase released labels for this comparison; item keys,
  trials, conditions, scores, references, nulls, and trace text remain intact.
- `release`: the accepted current-schema tables, response categories, coverage,
  ungraded-payload causes, judge labels, and case-only subject alias groups.
- `legacy_subject_ids` and `release.subject_ids`: the reviewed identity crosswalk.
- `registry_enrichment`: the 41 subsequently registered labels, the preceding
  migrated subject IDs/table fingerprints, and the count of dependent response
  IDs that changed. The legacy and preserved-measurement expectations were not
  refreshed during this registry enrichment.

Logical hashing uses ordered column names and JSON-encoded cells, preserving
list order and exact floating-point values, with all missing cells represented
as JSON null. Each row is SHA-256 hashed; sorted, fixed-width row digests are
then hashed together after the JSON column-name header. Row ordering, Parquet
compression, and Arrow encoding do not affect these fingerprints.

The initial record was produced only after inspecting the legacy tables,
replaying the legacy transformation against the cached provider release, and
comparing the migrated tables exactly. The approved subject-identity and
registry differences are explained in `../curation_record.md`. Future output
differences require review before updating expectations; a successful new build
alone is not permission to refresh the snapshot.

Run the build and then the read-only checks:

```bash
python benchmarks/mmdocrag/build.py
MEASUREMENT_DB_FULL_TEST=1 python benchmarks/mmdocrag/test.py -v
```

On a clean checkout, table-dependent tests skip unless full-test mode is set.
Pure parsing tests run without cached sources or generated tables. Source
locations and integrity pins belong exclusively in `../metadata.yaml`.
