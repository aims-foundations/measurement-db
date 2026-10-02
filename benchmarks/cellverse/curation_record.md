# CellVerse curation record

- **Coverage.** The official repository releases 748 DeepSeek-R1 outputs for the multiple-sclerosis scRNA-seq cell-type subset. All match the task bank; 317 are correct, reproducing the reported 42.38% accuracy after rounding. Other models and task families have no observations in this import.
- **Curation decisions.** Match full source messages and references; exclude the reference assistant message from predictor input. Compare the released extracted answer exactly with the reference, and preserve the complete model output as a trace. Sources are pinned and downloadable directly from upstream.
- **Limitations.** The answer-extraction implementation and exact per-run model revision, endpoint and inference settings are not released. The extracted predictions are used as published. The HF dataset specifies CC-BY-4.0; the code repository includes an MIT license.

[Metadata and sources](metadata.yaml) · [Characterization](characterization.yaml) · [Checks](../../tests/test_benchmark_datasets.py) · [Builder](build.py)
