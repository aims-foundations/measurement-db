# MMDocRAG curation record

- **Coverage.** Includes the provider's released question-level answer judgments and matching answer text. Evaluation filenames identify subjects; the record's `model` field identifies the judge. Input mode and quote count remain response conditions, while fine-tuned and `no-think` subject variants remain distinct.
- **Curation decisions.** Grades are the mean of all five valid 0–5 dimensions, divided by five. Dimension keys are normalized; incomplete judgments remain null, including 129 retained attempts. This differs from the provider aggregate script's literal-key, zero-fill policy. Gold questions prefer the 20-quote source; trace matching preserves exact filename case and the last duplicate question's answer.
- **Limitations.** Items contain questions and reference answers, but omit retrieved passages, page images, and the full inference prompt. Trace coverage is incomplete, partly because filenames differ in case. Exact per-attempt inference settings are unavailable; `no-think` records a released label, not a verified runtime setting.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
