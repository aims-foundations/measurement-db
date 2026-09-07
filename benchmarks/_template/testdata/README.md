# Characterization test data

After the first successful build, inspect the generated tables and create
`characterization.json` beside this file. It records reviewed expectations for
the sibling `test.py`; it is not upstream benchmark data and must never be
ingested by `build.py`.

Start with this shape, replacing every value with one observed and reviewed
from the completed build:

```json
{
  "release": {
    "items": 0,
    "assets": 0,
    "subjects": 0,
    "responses": 0,
    "traces": 0,
    "item_rows_sha256": "replace",
    "asset_rows_sha256": "replace",
    "subject_rows_sha256": "replace",
    "benchmark_rows_sha256": "replace",
    "response_cells_sha256": "replace",
    "trace_rows_sha256": "replace"
  }
}
```

To obtain the values without making the test itself write files:

1. Inspect the generated tables and replace the five row-count values above.
   Use zero when the benchmark has no optional `assets.parquet` or
   `traces.parquet` sidecar.
2. Save the file with temporary values for the six fingerprints.
3. Run `python benchmarks/<slug>/test.py`. The
   `test_reviewed_fingerprints` failure displays the complete observed mapping.
4. Review the corresponding item, optional asset, subject, benchmark, response,
   and optional trace tables before copying those digests into
   `characterization.json` and rerunning the test. The asset fingerprint covers
   the exact file bytes as well as their IDs, sizes, and benchmark ownership.

If generated tables exist but `characterization.json` does not, `test.py`
fails even outside full-test mode. This prevents a copied test scaffold from
silently remaining inactive. On a clean checkout with no generated tables, it
skips normally; `MEASUREMENT_DB_FULL_TEST=1` makes missing tables an error.

Add benchmark-specific expectations when they capture important semantics,
such as split coverage, trial counts, response categories, grader variants, or
which subjects have traces. Keep the corresponding assertions in `../test.py`
short and explicitly named.

Run the complete check with:

```bash
python benchmarks/<slug>/build.py
MEASUREMENT_DB_FULL_TEST=1 python benchmarks/<slug>/test.py
```

Update `characterization.json` only after reviewing and accepting the semantic
change that caused the snapshot to differ.
