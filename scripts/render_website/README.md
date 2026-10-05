# Benchmark gallery

All Python rendering functions live in `generate_benchmark_gallery.py`.
The website gets one chart bundle per detail page and reads individual prompts
and answers from HF tables on click. There are no matrix PNGs, chart JSON files,
item exports, or answer shards to generate.

## Setup

Run from this repository's root. Install Python dependencies, Node.js 22.12+
(or 24), and pnpm 10.33.0:

```bash
python -m pip install -r requirements.txt
pnpm --dir website install --frozen-lockfile
```

Set `HF_TOKEN` in both server environments when using private data or media.
The table service also accepts credentials saved by `hf auth login`.
`website/.env.example` lists the Next server settings; search settings are
optional for a local preview. Additional Rasch fitting dependencies are
`torch` and `scikit-learn` (`python -m pip install torch scikit-learn`).

## Local preview

The current catalog spans two HF banks. Start the table service with both:

```bash
python scripts/render_website/generate_benchmark_gallery.py serve \
  --hf-repo aims-foundations/measurement-db-private@d851ee00de4d32478e24719f6369e3b441621933 \
  --hf-repo aims-foundations/measurement-db@2338e6afea4dd6336e42b7b7bf3f7858398f4e15 \
  --port 3050
```

Then start Next:

```bash
GALLERY_DATA_ORIGIN=http://127.0.0.1:3050 pnpm --dir website dev
```

Open `http://localhost:3000/measurement-db`. On a remote server, forward port
3000 over SSH; the browser reaches the table service through Next.
For a production preview, run `pnpm --dir website build --webpack`, then use
`start` in place of `dev`. Webpack also works on Lustre filesystems.

The service binds to loopback. HF credentials stay on the server. Repeated
`--hf-repo` arguments support a split catalog; the first bank wins duplicate
slugs. Omit `@COMMIT` to resolve its current revision once at startup.
HF's SDK handles downloads and caching. Readers request `responses.parquet`,
falling back to legacy `response.parquet` only when HF reports the file missing.
Local analysis scripts prefer the same plural filename. The uploader still
publishes the singular name, so published snapshots require this compatibility.
A cold answer request may download the complete traces Parquet before filtering.

Every detail page requires the table service at `GALLERY_DATA_ORIGIN`.

## Deployment

`.github/workflows/deploy-website.yml` deploys website-related pushes to `main`
as production and pushes to `migration/tabular-builders-20260924-build-test`
as previews. It can also be run manually on either branch. Both reuse the
existing Vercel project, `measurement-db-private`, with root directory
`website` and Node.js 22. The preview alias is
`measurement-db-build-test.vercel.app/measurement-db`.

GitHub Actions uses the repository secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID`,
and `VERCEL_PROJECT_ID`. Production search sync additionally uses the
`MEILISEARCH_HOST` repository variable and `MEILISEARCH_ADMIN_KEY` secret.
Credentials belong in GitHub/Vercel settings, never in this repository.

Vercel needs `HF_TOKEN` and `GALLERY_DATA_ORIGIN` in both Preview and Production.
The workflow accepts sensitive HF tokens and checks a chart bundle from the
gallery service before deploying. The service must have a reachable HTTPS URL;
the loopback server used for local previews cannot serve Vercel requests.
The repository contains the Python service implementation, but no hosted-service
configuration or production endpoint. Hosting that service is a separate step.

The workflow builds and uploads Next.js, checks the deployed pages, then updates
the preview alias or production search indexes. The old repository's production
workflow remains active until production deployment ownership is switched.

## Rendering and analysis

`load_raw` retains exact observation keys and projects schema features into
chart dimensions. `build_detail` selects the layout and axes; `chart_bundle`
serializes them. Joined, pairwise, faceted, session, and basic matrices share
this path. A cell's original key identifies its answer directly.

`cards`, `details`, and `all` refresh the small catalog metadata used by the
homepage and search. They do not export chart or answer assets. `--web-dir`
selects the website and `--cache-dir` selects the HF download cache.

Card images, logos, and existing audio remain on the media host. Embedded item
images come with the selected prompt. Rasch fits remain separate analyses;
abilities and difficulties are keyed by subject and item ID, so emitting them
no longer requires rendered chart files. Use the same `--hf-repo REPO@COMMIT`
arguments for `fit_rasch_models.py` and the table service. Fit summaries record
those source commits and the fitting settings. Refreshing the data requires
refitting the analyses; it can change abilities, difficulties, and AUC.
For the full catalog, also pass `--out artifacts/analyze_measurements/rasch`;
some published benchmarks have no local builder directory.

The included metadata and 40 visible Rasch fits were refreshed on 2026-09-14
against the revisions above; the layouts and interactions are retained.
Graded responses are never thresholded for this analysis. ARC-AGI-3 and HELM
BOLD reached the iteration limit; the UI retains that indication. MathArena
remains slow with the current layout.

Reviewed website citations are kept in
`website/content/curated/benchmark-references.json`, with links to the merged
source snapshot. They preserve credits for the full website catalog, including
benchmarks without a builder in this repository. The attribution generator
uses these references first and local benchmark metadata for new entries;
it does not rewrite builder metadata.

To refresh the homepage summaries after downloading the catalog tables, run
`generate_chart_marginals.py` and `generate_model_timeline.py` with the same
`--cache-dir` and `--web-dir` as the gallery. They read canonical model fields
from `subjects.parquet`. Saturation uses `benchmark_saturation.py
--benchmarks-dir CACHE --emit-web`; then regenerate the cards to include it.

## Checks

```bash
python -m pytest -q tests/test_ai_subjects.py tests/test_benchmark_saturation.py \
  tests/test_fit_rasch_models.py tests/test_generate_benchmark_attributions.py \
  tests/test_generate_benchmark_gallery.py tests/test_generate_chart_marginals.py \
  tests/test_gallery_output_parity.py tests/test_gallery_response_filenames.py
ruff check scripts/render_website/generate_benchmark_gallery.py scripts/analyze_measurements/fit_rasch_models.py
python scripts/render_website/generate_benchmark_attributions.py --check
pnpm --dir website typecheck
pnpm --dir website build --webpack
pnpm --dir website search:check
```

The bundle snapshots were recorded before removing the exporters, after
checking all ten fixture cases against the original renderer. Tests also cover
raw answer keys, nulls, interactors, trial selection, and absent asset writes.

Migrated from `aims-foundations/measurement-db-pp` commit
`d055c41cec9a838fa72ca9dcdd0afc5711c19bfb` (merged PR #2). Generated chart data
and fits are preserved from that snapshot; the old checkout is not required.
