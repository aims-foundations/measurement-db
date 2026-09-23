# Example Benchmark curation record

Replace the prompts below with three concise bullets, aiming for about 150 words.
Keep decisions and limitations specific to this benchmark; omit implementation
walkthroughs, migration chronology, and repeated test logs. Preserve material
caveats even if they require a little more space.

- **Coverage.** Which released observations are included, and what does each grade measure? Mention material exclusions or differences from the paper's scope.
- **Curation decisions.** Which transformations affect meaning or identity? Record consequential choices about grading, variants, repeated attempts, missing values, traces, or attachments, and any deliberate corrections.
- **Limitations.** What remains incomplete, uncertain, or inconsistent with the evaluated tasks? Cite the evidence for consequential caveats; do not infer unavailable configuration details.

Keep sources and grading in [metadata.yaml](metadata.yaml). Record reviewed counts and source claims in `characterization.yaml`, with corresponding checks in [shared dataset checks](../../tests/test_benchmark_datasets.py), and link to those files from the completed record.
