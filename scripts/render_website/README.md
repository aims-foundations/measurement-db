# Benchmark gallery

PR validation reads canonical tables from `aims-foundations/measurement-db`.
Parallel workers prepare reusable static bundles; accepted releases are stored
under `v2/` in `aims-foundations/measurement-db-gallery`. Deployment restores
those files and builds Next.js without rendering charts or fitting models.
Charts, identities, and most item content remain packaged with the application.
The existing 20 dynamic item bundles, traces, and images still load from pinned
canonical HF tables on demand. This pipeline adds no browser calls to the
gallery repository. Python chart rendering stays in `generate_benchmark_gallery.py`.

## Build and preview

Use Python 3.11+, Node.js 22.12+ (or 24), and pnpm 10.33.0. From the repository root:

```bash
python -m pip install -r scripts/ci/gallery-requirements.txt
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
pnpm --dir website install --frozen-lockfile

# Resolve once so all rendering and analysis use the same tables.
export HF_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("aims-foundations/measurement-db", revision="main").sha)')
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

CI uses `website/content/curated/dynamic-item-benchmarks.json` to preserve the
existing 20 dynamic benchmarks. Workers skip their static prompt generation
entirely. Other benchmarks keep full prompt and reference-answer text static;
new benchmarks default to static. The standalone gallery CLI retains its
historical size-ranking behavior for manual experiments.
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
The browser does not query the HF gallery repository. Embedded images remain available;
input images referenced by `asset_manifest` are read on demand from the same
revision's `assets.parquet`. Embedded image bytes are also loaded on demand
from their original item row.

## Deployment

1. An in-repository PR runs **Benchmark table reproduction**. Changed builders
   read their declared upstream sources and byte-compare every generated table
   against one pinned commit of canonical HF `main`. Missing references, missing
   or extra tables, and mismatches fail. Withheld benchmarks retain their
   existing skip policy. There is no automatic baseline creation or canonical
   HF upload. Fork PRs never execute repository code on the runners.
2. After the benchmark report succeeds, **Prepare gallery** checks its per-table
   SHA-256/size receipts. Four isolated worker processes reuse matching accepted
   bundles or generate gallery, saturation, marginals, timeline inputs, and
   Rasch output for each benchmark. One parent combines all eight catalogs.
   Preparation emits stage timings and a cache-hit summary. It does not write HF.
3. The same run uploads a `gallery-candidate-<attempt>` artifact and a small
   `gallery-manifest-<attempt>` artifact, retained for 30 days. A reusable Vercel
   workflow restores this candidate and deploys a PR preview. Wait for the
   benchmark report, preparation, and preview checks before merging.
4. A relevant push to `main` runs **Deploy website to Vercel**. It finds a
   successful preparation run for the merged PR and requires an exact Git-tree
   match, which supports merge and squash commits. It verifies the validation
   receipt, whole-archive hashes, indexed file hashes, and source catalogs;
   restores static files; builds Next.js; publishes accepted bundles only to
   the gallery HF repository; and deploys the prebuilt application.

Preparation and deployment inspect full Git diffs, including changes beyond
GitHub's 300-file filter limit. This change does not alter the benchmark
validator's existing push triggers or folder selection. Tightening those
triggers and eliminating duplicate validation reports is a separate follow-up.

The per-benchmark cache key includes table paths and SHA-256 hashes, metadata,
renderer/analysis code, numerical-library versions, fit options, relevant
curated settings and assets, and the dynamic-item policy. It excludes the global
HF revision and unrelated benchmarks. An unrelated HF commit therefore reuses
unchanged output. A cached bundle retains its original pinned source revision;
its table bytes must equal the current release's input hashes. Shared renderer
or numerical-library upgrades deliberately invalidate all affected bundles.

HF `measurement-db-gallery/v2/benchmarks/` holds immutable benchmark archives;
`v2/catalog/` holds catalogs; `v2/receipts/` holds validation reports; and
`v2/releases/` holds release manifests. `v2/latest.json` advances only after
all objects are uploaded. Existing legacy gallery files are preserved. Uploads
use a parent-commit guard and refuse to publish if GitHub `main` has moved.
An accepted HF release matching the current Git tree also supports redeployment
after GitHub's candidate artifacts expire. Neither path regenerates missing data
silently: stale/missing candidates fail with instructions to prepare again.

GitHub artifacts are the temporary PR handoff; HF gallery is the durable cache.
The HF SDK's runner-local download cache and pip/pnpm caches reduce transfers
and installation work but are not required for correctness. Archive upload uses
compression level zero because bulky viewer files are already gzipped. Four
workers and two numerical-library threads per worker bound initial concurrency.
The initial v2 cache fill still prepares the full visible collection and can
take hours; subsequent preparation reuses unchanged bundles. The roughly 3 GB
of retained static website data still has to be transferred to Vercel.

GitHub Actions needs the existing `VERCEL_TOKEN`, `VERCEL_ORG_ID`,
`VERCEL_PROJECT_ID`, and `HF_TOKEN`. The read token must access both HF datasets.
Add `HF_GALLERY_WRITE_TOKEN` with write permission **only** to
`aims-foundations/measurement-db-gallery`; it is used only by the production
publication step. No workflow has an operation that writes canonical tables.
The read token remains server-only for the existing dynamic lookups.
Preparation and deployment use the existing `measurement-db-reproduction`
self-hosted runners. Preparation has a 12-hour timeout; deployment has 90 minutes.
Vercel uploads retain serialized requests and bounded connection-timeout retries.
The existing Vercel project's root is `website` and framework is Next.js.
Production search sync retains `MEILISEARCH_HOST` and `MEILISEARCH_ADMIN_KEY`.

For initial bootstrap, expired PR artifacts, or an intentional HF data refresh:

1. Run **Benchmark table reproduction** on the intended repository branch with
   `prepare_gallery=true`. The benchmark input is the existing validation scope;
   leave it empty for full validation. Preparation always produces a complete
   visible gallery, reusing matching accepted bundles when available.
2. Run **Deploy website to Vercel** on that same tree with `candidate_run_id`
   set to the successful preparation run ID. Feature branches create previews;
   `main` publishes the gallery release and deploys production.

An ordinary redeploy can leave `candidate_run_id` empty. If the base branch
changed since PR preparation, update the PR and rerun its checks before merging.
Make the preparation/preview checks required under repository branch protection
when enabling this flow; this patch does not change repository settings.

## Checks

```bash
python -m pytest -q tests/test_ai_subjects.py tests/test_benchmark_saturation.py \
  tests/test_fit_rasch_models.py tests/test_generate_benchmark_attributions.py \
  tests/test_generate_benchmark_gallery.py tests/test_generate_chart_marginals.py \
  tests/test_gallery_output_parity.py tests/test_gallery_response_filenames.py \
  tests/test_gallery_hybrid_data.py \
  tests/test_gallery_artifacts.py tests/test_gallery_prepare.py \
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
