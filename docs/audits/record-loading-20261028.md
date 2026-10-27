# Answer and item loading audit

Canonical snapshot: `aims-foundations/measurement-db` at
`15dc9f5365087e720a6af6f3e45af834d283b0f8`. Audit covers 270 published benchmarks,
599 source tables and the current flat Parquet page layouts. The accompanying
CSV lists every table. Counts predict the existing reader's compressed plus
uncompressed page budget; they are not website-traffic or visitor-failure rates.

| Source | Tables affected / audited | Entirely blocked tables | Rows over 16 MiB / total |
| --- | ---: | ---: | ---: |
| Model answers | 104 / 267 | 45 | 4,313,987 / 92,030,180 (4.69%) |
| Items | 38 / 270 | 20 | 525,054 / 8,403,704 (6.25%) |
| Binary assets | 17 / 62 | 1 | 4,368 / 189,264 (2.31%) |

Every SciGym answer (5,754) and ProgramBench answer (4,965) exceeded the old
budget. MMLU had 52,224 blocked answers out of 2,106,300. A separate UI bug
prevented the answer panel from mounting whenever the item failed to load, so
answer-table counts alone understate the affected selections. The audit does
not count genuinely unreleased transcripts as loading failures.

The fix keeps the original browser guard and adds exact, independently readable
records for the 142 affected item/answer tables. All 4,839,041 cached rows were
reconstructed, including all text parts, and checked against original full-row
digests. Source SHA256 hashes matched the pinned HF snapshot for every table.
9,222 records need text paging; other cached records load whole. The packs total
63.82 GB in remote storage; a browser fetches only selected byte ranges. Binary
asset loading is unchanged and remains a separate limitation.

Presentation checks covered 783 sampled items across all 270 benchmarks, with
focused browser review recorded separately. Source-specific fixes separate
SciGym's indented instructions and SBML/XML, select its final model, expose
ProgramBench's conversation messages, label DataClawBench's unreleased transcript,
format Mind2Web's HTML as source code, and separate HealthAdminBench configuration
JSON and evaluator records. Reference answers use the same structured formatting
as grading rules. Original records remain available; sampling is not proof that every item
in the database has ideal formatting.

Published cache: HF branch `preview/record-loading-20261028`, pinned commit
`6a461f46b96bc624fe1f0c5438d1886b8cfd9571`. All 1,311 remote files matched their local
checksums. Canonical Parquet tables were not modified.

Validation: 51 frontend/transport tests, 6 coverage-network tests, 3 storage
roundtrip tests, TypeScript, scoped lint, and the production build passed.
Chromium loaded and checked an exact source text hash or paged-record identity
from every one of the 142 affected tables through the published HF byte-range
path. UI checks covered 20 benchmark pages, including desktop/phone layouts for
SciGym, ProgramBench, Mind2Web, HealthAdminBench and WorkArena. An injected item
failure left its model answer visible; retry recovered the item. No page errors
were recorded. Full raw records, bounded long-text navigation and the original
catalog/coverage data remain available.
