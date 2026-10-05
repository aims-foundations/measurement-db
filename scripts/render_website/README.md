# Benchmark gallery

The website reads only `aims-foundations/measurement-db` on Hugging Face.
All Python chart rendering stays in `generate_benchmark_gallery.py`.
The build packages chart bundles, prompts, and answers with Next.js; visitors
need neither a Python service nor a separate gallery dataset repository.

## Build and preview

Use Python 3.11+, Node.js 22.12+ (or 24), and pnpm 10.33.0. From the repository root:

```bash
python -m pip install numpy pandas pyarrow 'huggingface_hub>=1,<2' scikit-learn PyYAML
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pnpm --dir website install --frozen-lockfile

# Resolve once so all rendering and analysis use the same tables.
export HF_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("aims-foundations/measurement-db").sha)')
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/render_website/generate_benchmark_gallery.py build
python scripts/render_website/generate_chart_marginals.py
python scripts/render_website/generate_model_timeline.py
python scripts/render_website/generate_benchmark_attributions.py
python scripts/analyze_measurements/benchmark_saturation.py --benchmarks-dir scripts/render_website/.hf-cache --emit-web --replace
python scripts/render_website/generate_benchmark_gallery.py cards
python scripts/analyze_measurements/fit_rasch_models.py --all --emit-web --no-curves --device cpu --out artifacts/website-rasch
pnpm --dir website build --webpack
pnpm --dir website start
```

Open `http://localhost:3000/measurement-db`. For a remote server, forward port
3000 over SSH. After generating the data, `pnpm --dir website dev` also works.
`HF_TOKEN` is optional build authentication; the website does not need it at runtime.

`build` selects available public benchmarks from `published-benchmarks.json`,
excluding `hidden-benchmarks.json`. It replaces the catalog and removes obsolete
viewer outputs. `build slug1,slug2` makes a smaller preview. Existing visibility
rules currently select four of the six public benchmarks; no private-bank fallback
is used to fill the catalog.

`--cache-dir` selects the HF SDK download cache. Responses prefer
`responses.parquet`, with `response.parquet` accepted only when HF reports the
plural file missing. `--revision COMMIT` or `HF_REVISION` pins the source revision.
Generated viewer files live in ignored `website/public/benchmark-data/`.
Their compressed item/answer dictionaries use exact source keys, divided into
256 buckets so a click downloads a small part of the benchmark. The browser
selects a bucket with SHA-256; this is an index, not cache-version tracking.

Pairwise, session, faceted, and basic layouts share the same chart functions.
Answer keys preserve subject, item, condition, trial, and interactors. Missing
answers remain unavailable; duplicate lookup keys fail the build. Rasch fits
use only native binary responses; graded benchmarks retain their original scale.

Card images and institution logos must exist under `website/public/`; otherwise
the existing generated thumbnail and institution-initial fallbacks are used.
The old HF gallery host is no longer queried. Embedded images remain available;
input images referenced by `asset_manifest` are read from the same source
revision's `assets.parquet` and included with the item.

## Deployment

`.github/workflows/deploy-website.yml` builds data from a single public HF commit,
then builds and deploys Next.js to the existing Vercel project. Pushes to `main`
deploy production; pushes to `migration/tabular-builders-20260924-build-test`
deploy the preview at `measurement-db-build-test.vercel.app/measurement-db`.
The workflow can also be run manually. HF updates appear after a new deployment.

GitHub Actions needs `VERCEL_TOKEN`, `VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID`.
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
  tests/test_gallery_output_parity.py tests/test_gallery_response_filenames.py
python scripts/render_website/generate_benchmark_attributions.py --check
pnpm --dir website lint
pnpm --dir website typecheck
pnpm --dir website search:check
```

The fixture tests compare all ten chart layouts with the previously verified
renderer and compare exported prompts/answers with direct Parquet lookups.
