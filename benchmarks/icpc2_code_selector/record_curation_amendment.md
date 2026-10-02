# ICPC-2 Code Selector curation record

- **Coverage.** The provider’s prediction CSV supplies model/query/top-K outcomes, selected codes, and raw model outputs; the evaluation CSV supplies relevant codes and the ordered search candidates.

- **Curation decisions.** The builder reconstructs the model-facing instruction, Portuguese query, and first K candidates using the upstream prompt template. K = 10, 20, 50, 100, or 200 changes the stimulus and is an item feature, so each query/K combination is a distinct item. This replaces the earlier query-only representation.

- **Limitations.** The reconstructed text does not preserve every model-specific message-role wrapper. Candidate parsing has a fallback path, so an independently reviewed build is still needed to establish that every stored prompt includes the correct candidate list; this amendment alone is not evidence of that verification.

[Metadata and sources](metadata.yaml) · [Builder](build.py)
