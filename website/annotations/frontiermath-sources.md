# FrontierMath annotation source inventory

This is the full set of external sources named in the three annotators'
FrontierMath submissions. It covers all five annotation parts and includes
sources listed in negative-search rationales (for example, a source checked
before marking a code `Missing`), not only sources quoted as affirmative
evidence.

## Scope and normalization

- Inputs: the five `Part 1` through `Part 5` CSV exports in this directory.
- Records: 15 total (five parts each from Ameen Patel, Max Zhang, and Nhi
  Truong).
- Corpus cutoffs: 2026-09-26 (MZ), 2026-09-27 (AP), and 2026-10-18 (NT).
- Extraction result: 55 literal URL forms, normalized to 38 source records.
- Normalization removes `utm_source`, merges section anchors and access formats
  for the same arXiv paper, and treats the graph view of the FrontierMath v2
  benchmark page as the same source. Distinct benchmark versions and a
  specifically cited comment/revision page remain separate.
- No DOI, ISBN, or bare-domain citation without an `http(s)` URL was found in
  the FrontierMath records.

The `Used in` abbreviations are D = Durability, V = Validity, I = Integrity,
N = Neutrality, and X = Verification. Annotator abbreviations are AP = Ameen
Patel, MZ = Max Zhang, and NT = Nhi Truong.

## Epoch AI properties and benchmark artifacts (24)

| # | Source | Used in | Annotators |
|---:|---|---|---|
| 1 | [Epoch AI — Transparency](https://epoch.ai/about/transparency) | N, X | MZ, NT |
| 2 | [Epoch AI Benchmarks — About](https://epoch.ai/benchmarks/about) | D, V, I, N, X | AP, MZ, NT |
| 3 | [FrontierMath Tiers 1–3 (unversioned benchmark page)](https://epoch.ai/benchmarks/frontiermath-tiers-1-3) | D, V, I, N | AP |
| 4 | [FrontierMath Tiers 1–3 v1](https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1) | D, V, I | AP, MZ |
| 5 | [FrontierMath Tiers 1–3 v2](https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2) | D, V, I, N, X | AP, MZ, NT |
| 6 | [GPQA Diamond benchmark graph](https://epoch.ai/benchmarks/gpqa-diamond?view=graph&tab=release-date) | N | AP |
| 7 | [Epoch AI Benchmarks — Use this data](https://epoch.ai/benchmarks/use-this-data) | I | AP, MZ |
| 8 | [Benchmark reviews documentation — Included benchmarks](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks) | X | MZ |
| 9 | [Epoch benchmark data archive](https://epoch.ai/data/benchmark_data.zip) | I | NT |
| 10 | [FrontierMath — The benchmark](https://epoch.ai/frontiermath/the-benchmark) | D | MZ |
| 11 | [FrontierMath Tiers 1–4](https://epoch.ai/frontiermath/tiers-1-4) | D | AP |
| 12 | [FrontierMath Tiers 1–4 — About](https://epoch.ai/frontiermath/tiers-1-4/about) | D, V, I, N, X | AP, MZ, NT |
| 13 | [FrontierMath Tiers 1–4 — Benchmark problems](https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems) | D, V, X | AP, MZ, NT |
| 14 | [Less than 70% of FrontierMath is within reach for today's models](https://epoch.ai/gradient-updates/less-than-70-percent-of-frontiermath-is-within-reach-for-todays-models) | N | AP |
| 15 | [Benchmarking hub update](https://epoch.ai/latest/benchmarking-hub-update) | D, I | MZ |
| 16 | [OpenAI and FrontierMath](https://epoch.ai/latest/openai-and-frontiermath) | D, V, I, N | AP, MZ, NT |
| 17 | [What is Epoch?](https://epoch.ai/latest/what-is-epoch) | N | MZ |
| 18 | [Deep Think Math](https://epoch.ai/publications/deep-think-math) | D | NT |
| 19 | [Grok 4 Math](https://epoch.ai/publications/grok-4-math) | D | NT |
| 20 | [GPT-5.4 set a new record on FrontierMath](https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath) | D, V, I, N | AP, MZ |
| 21 | [The Epoch Brief — June 12, 2026](https://epochai.substack.com/p/the-epoch-brief-june-12-2026) | D, X | NT |
| 22 | [Public Inspect evaluation log](https://logs.epoch.ai/inspect-viewer/c79c08da/viewer.html?log_file=https%3A%2F%2Flogs.epoch.ai%2Finspect_ai_logs%2FE9AfnKpQrDsFWKmXMhBbJS.eval) | D | MZ |
| 23 | [Epoch AI LinkedIn post announcing FrontierMath Tiers 1–4 v2](https://www.linkedin.com/posts/epochai_frontiermath-tiers-14-v2-is-live-we-activity-7471253846296711168-bRlG) | X | MZ |
| 24 | [Epoch AI Research post on X](https://x.com/EpochAIResearch/status/2065488154086568445) | D, V, N | AP, MZ |

## Papers and code (4)

| # | Source | Used in | Annotators |
|---:|---|---|---|
| 25 | [FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI (arXiv:2411.04872)](https://arxiv.org/abs/2411.04872) | D, V, I, X | AP, MZ, NT |
| 26 | [arXiv:2602.05192 (PDF, cited as a search-result snippet)](https://arxiv.org/pdf/2602.05192) | D | MZ |
| 27 | [FrontierMath scoring/evaluation implementation gist](https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb) | D, V, I, X | AP, MZ, NT |
| 28 | [Revision history for the scoring/evaluation gist](https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb/revisions) | I | NT |

## Third-party reporting and commentary (10)

| # | Source | Used in | Annotators |
|---:|---|---|---|
| 29 | [AI Wiki — FrontierMath](https://aiwiki.ai/wiki/frontiermath) | D | MZ |
| 30 | [What I wish I knew about FrontierMath (Lemmata)](https://lemmata.substack.com/p/what-i-wish-i-knew-about-frontiermath) | D | MZ |
| 31 | [Our World in Data — AI performance on FrontierMath over time](https://ourworldindata.org/grapher/ai-frontiermath-over-time) | D, V, I | MZ |
| 32 | [TechCrunch — AI benchmarking organization criticized for waiting to disclose funding from OpenAI](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai) | D | MZ |
| 33 | [InfoQ — Epoch AI introduces FrontierMath benchmark](https://www.infoq.com/news/2024/11/epochai-frontiermath-benchmark) | D | MZ |
| 34 | [Some lessons from the OpenAI–FrontierMath debacle (LessWrong)](https://www.lesswrong.com/posts/8ZgLYwBmB3vLavjKE/some-lessons-from-the-openai-frontiermath-debacle) | D | MZ |
| 35 | [Meemi's shortform (LessWrong)](https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform) | D, I | MZ |
| 36 | [Specific comment on Meemi's shortform](https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz) | D | AP |
| 37 | [How far behind are open models? (LessWrong)](https://www.lesswrong.com/posts/rJcCrXyEsJKmmDpWG/how-far-behind-are-open-models) | D | AP |
| 38 | [Search Engine Journal — OpenAI secretly funded FrontierMath benchmarking dataset](https://www.searchenginejournal.com/openai-secretly-funded-frontiermath-benchmarking-dataset/537760/) | D | MZ |

## Duplicate URL forms merged above

- The FrontierMath paper appeared in 13 forms: the abstract, PDF, HTML v3,
  HTML v7, eight v7 section anchors, and one tracking-parameter URL.
- The FrontierMath Tiers 1–3 v2 page appeared as the canonical page, a graph
  view, and a tracking-parameter URL.
- The `Use this data` page, GPT-5.4 Substack post, and implementation gist each
  appeared both with and without a tracking parameter.

## Search trails without a discrete source URL

These searches were part of the rationales but are not counted among the 38
source records:

- MZ, Durability/privacy rationale: `FrontierMath kept private misuse risk
  safety reason withheld problems`.
- MZ, Durability/restricted audience: `FrontierMath researchers cannot access
  private problems dataset not available to researchers Epoch`.
- NT, Verification/external verification status: targeted web searches for an
  external review or audit (the exact query was not recorded).

Two evidence-quality comments say that a required web search was *not*
performed. They are audit notes, not sources, and are therefore excluded.
