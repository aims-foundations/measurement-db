# C-SEO Bench curation

- **Source:** The [author release](https://huggingface.co/datasets/parameterlab/c-seo-results) supplies 413 response-file paths. Two equivalent-condition copies and five additional filename aliases are counted once. The tabular builder retains paired prompts and full outputs and computes document-level citation-rank changes.
- **Interpretation:** Multiple targets share outputs and Original controls, so comparisons are dependent. Historical prompt variants remain visible; this is a signed rank-change measure, not correctness or an isolated causal effect.
- **Source quirks:** One missing output stays ungraded; four citation-parser discrepancies stay as released. Video-game query suffixes are normalized only for joining. Detailed checks and release-specific limitations are recorded in the migration evidence.
