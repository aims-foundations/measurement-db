# Coverage explorer

The standalone `/network` page provides graph and matrix views of benchmark–subject
coverage. The catalog links to this page; browsing the catalog does not request
the graph data or start WebGL. The `/api/coverage` snapshot is generated at website
build time from the existing catalog, canonical subjects, and observed-item
marginals. This feature does not change those data sources, benchmark galleries,
or trace rendering, and does not require rebuilding Parquet tables or publishing HF.

## Responsibilities

- `loader.tsx`: deferred fetch, session cache, and retry state.
- `explorer.tsx`: shared filters, selected identity, and network / matrix views.
- `sidebar.tsx`: search, filters, identity details, and connection lists.
- `canvas.tsx`: graph rendering and selection.
- `labels.ts`: measured label placement and contrast over dense neighborhoods.
- `matrix.tsx`: bounded detail grid, ordering, paging, and keyboard navigation.
- `matrix-overview.tsx`: sparse canvas overview and detail-window navigation.
- `view-tabs.tsx`: shared accessible keyboard navigation between views.
- `../../lib/coverage-network/model.ts`: identity, observed edges, and filtering.
- `../../lib/coverage-network/layout.ts`: deterministic graph positioning.
- `../../lib/coverage-network/appearance.ts`: shared colors and node-size scale.
- `../../lib/coverage-network/matrix.ts`: sparse matrix indexing and window bounds.

## Interpretation and bounds

Each edge / filled cell means recorded items exist for a benchmark–subject pair,
including observations without a grade. Empty cells mean no mapped record in
this snapshot, never failure or a score of zero. Canonical subjects can combine
several harness configurations. Subjects and benchmarks without records outside
the published catalog are not introduced by presentation code.

Unlinked benchmarks remain on the graph in a separate grid, in the searchable
list, and in the matrix as empty rows. Their grid positions imply no connections.
Provider filters cannot attribute these unlinked benchmarks to a provider.
Matrix ordering changes only row and column positions. All rows and columns
remain reachable; the overview draws observed edges and the detail grid mounts
at most 12 × 18 cells. Neither view loads individual response records.

Run `pnpm network:test`, `pnpm typecheck`, `pnpm lint`, and `pnpm build` from
`website/`. Browser checks should cover the catalog link, loading and retry,
selection across views, filters, blank-cell semantics, keyboard navigation, and
responsive paging. Counts reflect the current generated snapshot; subject-identity
corrections belong to the data pipeline and are not inferred by this page.
