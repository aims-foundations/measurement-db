# Few-Shot TTT curation record

- **Coverage.** The 55 released files contain 7,425 observations across eleven configurations and five seeds. Each task exports only five example predictions; the accompanying aggregate accuracies describe larger evaluation sets.
- **Transformation.** Flatten the native task/example lists, retain method identity and assign separate trials to repeated subject-item observations. All previous questions, reference answers, predictions and grades are preserved.
- **Limitations.** Traces contain the released final predictions. Complete reasoning histories and the actual few-shot demonstration sequences are not included in these example records.

[Metadata](metadata.yaml) · [Characterization](characterization.yaml) · [Shared checks](../../tests/test_benchmark_datasets.py)
