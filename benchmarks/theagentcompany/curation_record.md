# TheAgentCompany curation record

- **Coverage.** 19 agent configurations, 175 tasks, 3,056 released checkpoint fractions and one additional MUSE attempt without a grading file. The latter has a null grade. Every attempt retains its complete native trajectory.
- **Reconstruction.** Join pinned upstream result JSON, task prompts/evaluators and JSON/text trajectories directly. All previous graded observations and subject/item identities are unchanged. Full trajectories replace the previous extracted text and one-million-character clipping; one identical compressed/uncompressed copy is counted once.
- **Limits.** The captured evaluator commit is versioned but is not asserted to be each historical run’s exact grading-code revision. Separate screenshot files are outside this import. Composite subject labels retain the original run configuration, including its environment-model variant.

[Sources](metadata.yaml) · [Characterization](characterization.yaml) · [Builder](build.py) · [Checks](../../tests/test_benchmark_datasets.py)
