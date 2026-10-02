# WorkBench Revisited curation record

- **Coverage.** 24 released model configurations × 690 workplace tasks: 16,560 binary correctness verdicts and 16,507 available full-response transcripts.
- **Reconstruction.** The builder joins the authors’ pinned item-level verdict export to the original run CSVs and model manifest. All previous subject, item, response and benchmark tables are unchanged; the trace table is newly retained. Independent checks reconcile every verdict, task, reference action, recorded action and transcript with the upstream files.
- **Limits.** Grading uses the release’s v2 task definitions and deterministic sandbox scorer. Published full-response strings are preserved verbatim; the optional, richer JSON trace files were not released. These observations do not represent new model runs.

[Sources](metadata.yaml) · [Characterization](characterization.yaml) · [Builder](build.py) · [Checks](../../tests/test_benchmark_datasets.py)
