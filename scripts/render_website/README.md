# Benchmark gallery

The website reads only `aims-foundations/measurement-db` on Hugging Face.
All Python chart rendering stays in `generate_benchmark_gallery.py`.
The build packages charts, identities, and most item content with Next.js.
The 20 largest item bundles, all traces, and all item images load from the
same HF revision on demand. No separate Python service or gallery repository is needed.

## Build and preview

Use Python 3.11+, Node.js 22.12+ (or 24), and pnpm 10.33.0. From the repository root:

```bash
python -m pip install numpy pandas pyarrow fastparquet 'huggingface_hub>=1,<2' scikit-learn PyYAML
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pnpm --dir website install --frozen-lockfile

# Resolve once so all rendering and analysis use the same tables.
export HF_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("aims-foundations/measurement-db", revision="migration/tabular-builders-20260924").sha)')
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/render_website/generate_benchmark_gallery.py build
python scripts/render_website/generate_chart_marginals.py
python scripts/render_website/generate_model_timeline.py
python scripts/render_website/generate_benchmark_attributions.py
python scripts/analyze_measurements/benchmark_saturation.py --benchmarks-dir scripts/render_website/.hf-cache --emit-web --replace
python scripts/render_website/generate_benchmark_gallery.py cards
python scripts/analyze_measurements/fit_rasch_models.py --all --emit-web --no-curves --eval-every 4000 --device cpu --out artifacts/website-rasch
pnpm --dir website build
pnpm --dir website start
```

Open `http://localhost:3000/measurement-db`. For a remote server, forward port
3000 over SSH. After generating the data, `pnpm --dir website dev` also works.
For gated HF downloads, set `HF_TOKEN` in the build environment and in the
Next.js server environment (`website/.env.local` for local previews).
The token stays on the server. A route restricted to the generated source
catalog authorizes byte-range downloads and redirects to HF's signed file URL.

`build` discovers benchmarks with `benchmarks.parquet` under either
`<slug>/formatted_tables/` or `<slug>/` on the public HF repository. It excludes
withheld/private releases and `hidden-benchmarks.json`. New releases need no
catalog allowlist update. It replaces the catalog and removes obsolete
viewer outputs. `build slug1,slug2` makes a smaller preview.

Existing reviewed author credits are retained. New benchmarks initially show
the paper and original source links from their HF metadata. Curated
author and institution records can be added later without blocking the page.
An affiliation entry alone does not imply a reviewed citation; reviewed references
and explicit overrides retain their author-count and source-verification checks.

`--cache-dir` selects the local table links used by the analysis scripts;
downloads use the HF SDK's normal shared cache. Responses prefer
`responses.parquet`, with `response.parquet` accepted only when HF reports the
plural file missing. `--revision` or `HF_REVISION` accepts a branch, tag, or commit;
the renderer resolves it to a commit once per run.
Generated viewer files live in ignored `website/public/benchmark-data/`.
Rasch summaries stay in the catalog; item difficulties are compressed alongside
each benchmark's viewer data so a page loads only its own fitted values.
The browser loads the selected benchmark's chart as a static file. Compressed
cell-key and item dictionaries use exact source keys, divided into up to
256 nonempty buckets. Trace and dynamic-item indexes map those keys to physical
Parquet rows. Page indexes let the browser read the required source pages,
with decoding in a worker so the chart remains responsive. Some source pages
and dictionaries are large, so an individual lookup can still transfer tens of MB.

Each build ranks compressed item bundles after removing inline image bytes and
loads the largest 20 dynamically (or all of them for a build with fewer than 20).
The remaining benchmarks keep their full prompt and reference-answer text static.
`gallery-sources.json` records the pinned source revision and allowed tables;
it contains no data credentials and is generated with the viewer files.

Pairwise, session, faceted, and basic layouts share the same chart functions.
Answer keys preserve subject, item, condition, trial, and interactors. Missing
answers remain unavailable; duplicate lookup keys fail the build. Rasch fits
use only native binary responses; graded benchmarks retain their original scale.
TuMLU and PKU-SafeRLHF use observation strips because their source-row identifiers
are annotation occasions, not useful condition bands. Repeated items keep their
separate recorded answers.
Charts exceeding 10,000 rows or 250 million dense cells also use observation
strips, preserving native grades and exact keys. Large strips page through
5,000 observations at a time; other joined charts page through 200 rows.

Card images and institution logos must exist under `website/public/`; otherwise
the existing generated thumbnail and institution-initial fallbacks are used.
The old HF gallery host is no longer queried. Embedded images remain available;
input images referenced by `asset_manifest` are read on demand from the same
revision's `assets.parquet`. Embedded image bytes are also loaded on demand
from their original item row.

## Deployment

Preview builds read the HF branch `migration/tabular-builders-20260924`;
production builds read HF `main`. Each build resolves that branch to one commit.
Both `<benchmark>/formatted_tables/*.parquet` and the older flat table paths are
supported, with the formatted directory taking precedence. Benchmarks marked
`withheld` in HF metadata and entries in `hidden-benchmarks.json` are excluded.
To build the migrated collection locally, pass
`--revision migration/tabular-builders-20260924` to the gallery command.

`.github/workflows/deploy-website.yml` builds data from a single public HF commit,
then builds and deploys Next.js to the existing Vercel project. Pushes to `main`
that change `benchmarks/`, website code, or rendering/analysis scripts deploy
production; equivalent pushes to `migration/tabular-builders-20260924-build-test`
deploy the preview at `measurement-db-build-test.vercel.app/measurement-db`.

PRs targeting `main` from branches in this repository also deploy previews on
opening, reopening, and new commits. Each PR has its own GitHub deployment
environment and website URL, linked from the **Deploy website to Vercel** check
and its run summary. PR previews build GitHub's merge commit so they include the
target branch's changes. Fork and Dependabot PR deployments are skipped because
GitHub withholds the deployment secrets.

The workflow checks the full push or PR diff, including more than 300 changed
files. Unrelated changes skip deployment. Manual runs refresh the data even
without code changes: choose **Actions → Deploy website to Vercel → Run workflow**
and select a branch. Feature branches deploy previews; `main` deploys production.
Before the workflow is on `main`, use the CLI:

```bash
gh workflow run deploy-website.yml --repo aims-foundations/measurement-db --ref YOUR_BRANCH
```

Push and manual deployments check out the latest branch contents. Every build
pins one public HF revision. Independent branches and PRs do not cancel each
other's deployments or replace the build-test preview alias.

Tables must already be published on HF before the merge. This workflow neither
executes benchmark builders nor uploads tables; reproduction CI remains separate.
If the HF release happens later, run **Deploy website to Vercel** manually on
`main` to refresh production.

GitHub Actions needs `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, and
`HF_TOKEN` for gated downloads. Deployment passes `HF_TOKEN` to the Vercel
server environment; it is never included in client code.
The deployment job uses the existing `measurement-db-reproduction` self-hosted
runners: the migrated tables exceed a standard hosted runner's disk capacity.
Full rendering and analysis can take hours, so the job allows up to 24 hours.
Uploads serialize Vercel API requests and retry connection timeouts on the shared runners.
Its Python dependencies are isolated in a job-local environment; HF maintains
the shared download cache.
The reused Vercel project has the historical name `measurement-db-private`;
that project name does not select the HF data source. Its root is `website`.
Production search sync uses `MEILISEARCH_HOST` and `MEILISEARCH_ADMIN_KEY`.
The old repository's production workflow remains active until production
ownership is switched. `GALLERY_DATA_ORIGIN` is no longer used.

## Checks

```bash
python -m pytest -q tests/test_ai_subjects.py tests/test_benchmark_saturation.py \
  tests/test_fit_rasch_models.py tests/test_generate_benchmark_attributions.py \
  tests/test_generate_benchmark_gallery.py tests/test_generate_chart_marginals.py \
  tests/test_gallery_output_parity.py tests/test_gallery_response_filenames.py \
  tests/test_gallery_hybrid_data.py \
  tests/test_website_deployment.py
python scripts/render_website/generate_benchmark_attributions.py --check
pnpm --dir website lint
pnpm --dir website typecheck
pnpm --dir website search:check
python tests/test_gallery_hybrid_data.py --export /tmp/gallery-fixture
node website/scripts/test-gallery-source.mjs /tmp/gallery-fixture
node website/scripts/test-gallery-source-route.mjs
```

The fixture tests compare all ten chart layouts with the previously verified
renderer and compare exported prompts/answers with direct Parquet lookups.
