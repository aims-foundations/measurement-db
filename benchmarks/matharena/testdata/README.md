# MathArena characterization

`characterization.json` contains reviewed output expectations, not upstream
benchmark data. `legacy_baseline` records the pre-migration tables inspected in
the local historical publication cache; `release` records the corrected build.
The correction decisions and verification history are in `../curation_record.md`.
Source revisions, filenames, byte sizes, hashes, and card references belong only
in `../metadata.yaml`.

Fingerprints use the read-only `characterize` / `logical_digest` functions in
`../test.py`: preserve column order; normalize nulls, NumPy values and lists;
represent binary cells by SHA-256; serialize each row as compact Unicode JSON;
sort rows; prefix each UTF-8 payload by its eight-byte big-endian length; SHA-256
the resulting stream. Row order and Parquet compression do not affect the hash.
Every column is covered, including identities, graders, response extensions,
prompt text, trace attribution, and benchmark statistics.

The legacy observation fingerprint compares released model labels, the old
first-record problem text, response values, reference answers, and judge/criterion
conditions. It deliberately excludes the old collision-renumbered trials. It
includes all old final verdicts and native-list proof grades, not the JSON proof
criteria that the old implementation skipped. This projection is checked against
the pinned sources independently of the migrated builder.

Tests never download, write outputs, or refresh this file:

```sh
python benchmarks/matharena/build.py
MEASUREMENT_DB_FULL_TEST=1 python benchmarks/matharena/test.py -v
```

For an intentional future change, review the source/output difference first,
run the builder and the independent reconciliation tests, then explicitly
regenerate the affected expectations in a separate one-off operation. Do not
make failing tests automatically accept new fingerprints. Keep raw data,
decoded images, and all generated Parquet tables out of git.
