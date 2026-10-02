# ProX math evaluation curation record

- **Coverage.** Eighteen native files supply 24,298 observations from two TinyLlama/OpenWebMath checkpoints across nine math tasks. All prior model outputs and grades are retained; the released files do not cover ProX-refined counterparts.
- **Identity.** Keep task membership and the actual grading reference. Four repeated question texts have differing targets across source records; the older content-only IDs hid these differences. Some released targets appear incorrect and remain visible for review, without regrading.
- **Limitations.** One Minerva-Math question has an empty parsed target in both checkpoints. Its reference is null and its two published failed outcomes are preserved. Native mathematical-equivalence verdicts include grader timeouts recorded as failures.

[Metadata](metadata.yaml) · [Characterization](characterization.yaml) · [Shared checks](../../tests/test_benchmark_datasets.py)
