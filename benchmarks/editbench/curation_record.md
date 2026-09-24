# EDIT-Bench curation record

- **Coverage.** The captured whole-file editing release contains 23,328 observations, 44 configured subjects, 540 items, and 540 generated-code traces. Item/grader and result sources are pinned and chronologically aligned.

- **Curation decisions.** Success requires the released test-pass fraction to equal 1.0; original fractional scores remain in raw results. Missing expected evaluator outputs receive the provider’s zero score. Both released leaderboard tables reconcile within their four-decimal rounding tolerance. API attribution follows the native adapter maps, correcting four Claude and two GPT-OSS subjects to OpenRouter.

- **Limitations.** Only gpt-o3-mini has released generated-code traces; these are submitted code, not full API conversations. Historical runs do not independently identify the item-bank commit. Missing endpoint, dependency, or inference settings are left unknown. Malformed generated code and refusals are preserved as observed attempts.

[Metadata and sources](metadata.yaml) · [Characterization](characterization.yaml) · [Checks](../../tests/test_benchmark_datasets.py) · [Builder](build.py)
