# AfriMed-QA curation record

- **Coverage.** Preserve 302,354 attempts from 117 attributable MCQ result files and 34 source model labels, covering 7,039 underlying questions and 42,494 complete prompt/reference variants. All 281,354 recorded grades and 21,000 ungraded attempts retain their full source records.
- **Corrections.** Remove truncation, restore one failure split by a bare carriage return, and match 12,000 output-only attempts to released question and option text. All 281,353 previous grades remain unchanged; remove three byte-identical file copies and preserve three cross-model copies in raw data without assigning their ambiguous results.
- **Limitations.** Some labels do not establish exact model checkpoints or inference settings. The captured release also contains short-answer and consumer-query results requiring separate metric review; MedQA-USMLE runs belong to another benchmark. Source files remain unchanged.

[Upstream release](https://github.com/intron-innovation/AfriMed-QA/tree/9bb0ffc03ef41a0568c3f47223a23997922afb52) · [Builder](build.py)
