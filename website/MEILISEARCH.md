# Meilisearch Cloud setup

The website is ready to search two public indexes:

- `measurement_db_benchmarks`: one document per visible benchmark
- `measurement_db_results`: one aggregate result per visible benchmark and AI
  model

The current build prepares 61 benchmarks and 7,432 aggregate model results.
Raw model responses, prompts, traces, and hidden competition benchmarks are not
sent to Meilisearch.

The existing on-page gallery filter remains available if Cloud is not yet
configured or temporarily unavailable.

## 1. Create the Cloud project

1. Sign in at [Meilisearch Cloud](https://cloud.meilisearch.com/) and choose
   **New project**.
2. Choose **Resource-Based** billing.
3. Use a recognizable project name such as `aims-measurement-db`.
4. Choose the **SFO** region, which Meilisearch recommends for US West Coast
   users. A region cannot be changed after project creation.
5. Keep the latest stable Meilisearch version.
6. Start with the smallest **XS** resource tier. This search corpus is only a
   few megabytes, so XS has ample memory for an introductory deployment. Check
   the exact hourly and estimated monthly price shown in the Cloud form before
   purchasing; regional rates can change.

Resource-based projects charge a fixed hourly compute rate for the selected
tier, plus storage and bandwidth. They do not impose per-document or per-search
caps, but the selected CPU and memory determine capacity. Expect the dashboard
to request billing details when creating this plan; the advertised trial may
begin on usage-based billing instead.

Current references:

- [Create a Cloud project](https://www.meilisearch.com/docs/capabilities/platform/infrastructure/create_a_project)
- [Cloud regions and resource tiers](https://www.meilisearch.com/docs/capabilities/platform/infrastructure/overview)
- [Current pricing and estimator](https://www.meilisearch.com/pricing)

## 2. Copy the URL and keys

From the Cloud project, copy its project URL and API keys.

The browser key is intentionally public. For the simplest start, use the
project's default **Search API key**. For tighter least-privilege access, create
a custom key with:

- action: `search`
- indexes: `measurement_db_benchmarks` and `measurement_db_results`
- no write, settings, index-management, task, or key-management actions

The indexing key is private. The default **Admin API key** works for initial
setup and must only be stored as a GitHub Actions secret. Never put it in a
variable whose name starts with `NEXT_PUBLIC_`. A custom sync key can replace it
later if desired.

See [Meilisearch API-key security](https://www.meilisearch.com/docs/capabilities/security/how_to/manage_api_keys).

## 3. Configure Vercel

In the Vercel project for the measurement-db website, open **Settings →
Environment Variables** and add these variables to both **Production** and
**Preview**:

| Name                                 | Value                                            |
| ------------------------------------ | ------------------------------------------------ |
| `NEXT_PUBLIC_MEILISEARCH_HOST`       | The Cloud project URL, beginning with `https://` |
| `NEXT_PUBLIC_MEILISEARCH_SEARCH_KEY` | The search-only API key                          |

These two values are compiled into the browser bundle. That is safe only because
the key is search-only and the indexed records are public. Redeploy after adding
or changing either value; Vercel does not retrofit environment changes into an
existing deployment.

For local UI testing, copy the two `NEXT_PUBLIC_` entries from `.env.example`
to `.env.local`, replace their placeholders, and restart the development
server. The private sync variables are not needed by the website process.

## 4. Configure automatic indexing in GitHub

In the GitHub repository, open **Settings → Secrets and variables → Actions**.

Add one repository or `production` environment variable:

| Type     | Name               | Value                      |
| -------- | ------------------ | -------------------------- |
| Variable | `MEILISEARCH_HOST` | The same Cloud project URL |

Add one repository or `production` environment secret:

| Type   | Name                    | Value                                |
| ------ | ----------------------- | ------------------------------------ |
| Secret | `MEILISEARCH_ADMIN_KEY` | The private admin or custom sync key |

The production deployment workflow validates the public corpus, deploys the
site, builds fresh staging indexes, checks their document counts, atomically
swaps both indexes into production, and deletes the old copies. Preview
deployments validate the records but never write to Cloud.

## 5. Run the first import

After the Vercel and GitHub values are configured:

1. Open **GitHub → Actions → Deploy measurement-db to Vercel**.
2. Choose **Run workflow** on the `main` branch.
3. Confirm the `Sync Meilisearch Cloud indexes` step reports 61 benchmarks and
   7,432 results for the current data revision.
4. Redeploy Preview separately if it should receive the new public browser
   variables.
5. Search the site for a benchmark such as `MMLU`, a model such as `Claude`, and
   a use case such as `medical reasoning`.

The counts will change naturally as visible website data changes. Every later
production website deployment runs the same sync automatically.

## Local validation and manual sync

No Cloud credentials are needed to validate the documents:

```bash
cd website
pnpm search:check
```

To perform a manual sync, provide the private values only to that shell:

```bash
cd website
MEILISEARCH_HOST="https://your-project-url.meilisearch.io" \
MEILISEARCH_ADMIN_KEY="your-private-key" \
pnpm search:sync
```

The script never prints either credential.

## Routine maintenance

- Search data is derived from the same generated JSON used by the website.
- Edit `content/curated/hidden-benchmarks.json` to hide or restore a benchmark;
  the website and search sync consume the same list.
- Monitor CPU, memory, and search latency in the Cloud **Infrastructure** tab.
  Increase the tier only if memory remains high or peak search latency rises.
- At the time this guide was written, scaling up is self-service and scaling
  down requires Meilisearch support.
- Rotate the search key by adding the new key in Vercel, redeploying, verifying
  search, and then revoking the old key.

If the new search box says full model-result search is unavailable, first check
that the two Vercel variables exist in the correct environment and that the site
was redeployed. A `403` generally means the browser key lacks `search` access to
one of the two exact index names. Stale results generally mean the production
workflow's sync step was skipped or failed.
