// ---------------------------------------------------------------------------
//  Benchmark annotations — the private-benchmark verifiability survey.
//
//  WHAT THESE SCORES MEAN. The instrument codes *public verifiability*: how
//  strongly available evidence permits outsiders to assess a benchmark's claims
//  and procedures. Per the codebook's own decision rule, the scores "describe
//  strength of verifiability evidence, not the benchmark's actual quality,
//  fairness, or validity." A low code means no supporting evidence was located
//  after an adequate search — not that the practice is absent in reality. Copy
//  on these pages must not drift into reading them as a quality ranking.
//
//  The framework is organized around three trust claims (validity, integrity,
//  neutrality). Durability is deliberately NOT a fourth claim: it is the
//  constraint that motivates withholding information in the first place, and so
//  determines the disclosure limits under which the three claims must be
//  evidenced. Verifiability is also distinct from verification — making a claim
//  checkable does not mean anyone checked it.
//
//  This is decoupled from content/benchmark-details.ts and the generated card
//  catalog, which are derived from canonical parquet tables on Hugging Face.
//  None of the benchmarks below is in that catalog, and an annotation is about
//  a benchmark's documentation rather than its response data.
//
//  Source: "Trustworthy Private AI Benchmarks Require External Verification"
//  (codebook v4) and the Part 1 annotation form responses. The context fields
//  below are transcribed from coded annotations; the per-claim assessments are
//  placeholders until the scored items (V1–V5, I1–I5, N1–N4) are exported.
// ---------------------------------------------------------------------------

/** How far a benchmark's annotation has progressed. */
export type AnnotationStatus = "pending" | "context-coded" | "reviewed";

export const annotationStatusLabel: Record<AnnotationStatus, string> = {
  pending: "Not yet annotated",
  "context-coded": "Context coded",
  reviewed: "Reviewed",
};

export type ClaimId =
  | "validity"
  | "integrity"
  | "neutrality"
  | "verification";

/** The three trust claims. `question` is the paper's own one-line framing of
 *  each claim and doubles as the hover definition. */
export const claims: readonly {
  id: ClaimId;
  label: string;
  question: string;
  definition: string;
}[] = [
  {
    id: "validity",
    label: "Validity",
    question: "What does the score support?",
    definition:
      "Whether public evidence shows what capability or task distribution the score characterizes, who it is meant to inform, and what conclusions it does and does not support.",
  },
  {
    id: "integrity",
    label: "Integrity",
    question: "Was the score produced as claimed?",
    definition:
      "Whether a reported score can be shown to follow the stated procedure — item pool, sampling, configuration, scoring, exclusions, reruns and reporting decisions.",
  },
  {
    id: "neutrality",
    label: "Neutrality",
    question: "Was the evaluation administered evenhandedly?",
    definition:
      "Whether participants received comparable access and treatment, and whether funding, ownership or other conflicts that could affect results are disclosed.",
  },
  {
    // Not one of the paper's three trust claims: the other three ask whether a
    // claim *could* be checked, this one asks whether anyone did.
    id: "verification",
    label: "Verification",
    question: "Did anyone outside actually check?",
    definition:
      "Whether a party independent of both the operator and the evaluated providers has examined the benchmark's claims or procedures. Making a claim checkable is not the same as someone having checked it.",
  },
];

/** One claim's cells in the summary table. Empty arrays render as "not yet
 *  coded" rather than as an absence of findings.
 *
 *  `channels` is deliberately not called "fixes": a verifiability gap is not
 *  automatically a defect to be repaired by disclosing more. Durability is a
 *  legitimate reason to withhold, and a claim can be made checkable through
 *  controlled inspection or non-disclosing verification instead of publication.
 *  The cell records what would make the claim checkable, not what the operator
 *  ought to release. */
export type ClaimAssessment = {
  strengths: readonly string[];
  concerns: readonly string[];
  channels: readonly string[];
};

const uncoded: ClaimAssessment = { strengths: [], concerns: [], channels: [] };

// ---------------------------------------------------------------------------
//  Item-level codes (the tree)
// ---------------------------------------------------------------------------

/** One leaf's coded state. `desc` carries a descriptive answer with no status:
 *  the Durability band is not scored, so its leaves report what was coded
 *  rather than how strong the evidence is. */
export type LeafStatus = "ok" | "part" | "gap" | "risk" | "desc";

export const leafStatusLabel: Record<LeafStatus, string> = {
  ok: "Documented",
  part: "Partial",
  gap: "Not located",
  risk: "Disclosed risk",
  desc: "Descriptive",
};

/** One response option and how many annotators selected it. `code` is verbatim
 *  from the form's selection string; `k` of `n` annotators chose it. Counting
 *  is case-insensitive (coders typed "items/tasks" and "Items/tasks"), and the
 *  most common spelling is kept.
 *
 *  Note this is a SELECTION TALLY, not the majority share used in the paper's
 *  reliability section. They agree only for codes a majority selected: a code
 *  nobody picked has majority share 1.0 (all three agreed to leave it) but a
 *  tally of 0/n. The tally is what a reader can interpret directly. */
export type Vote = { code: string; k: number; n: number };

export type Leaf = {
  /** Short label for the compact diagram. */
  name: string;
  /** The form's question, verbatim including its item prefix. */
  question: string;
  /** The coded answer, shown under the name. */
  sub: string;
  /** One sentence of detail, revealed on hover/focus. */
  detail: string;
  status: LeafStatus;
  /** Per-option selection tallies, verbatim from the form. */
  votes?: readonly Vote[];
  /** Question annotator agreement: mean majority share over this question's
   *  options (multi-select) or the majority share of the chosen option
   *  (single-select). */
  qaa?: number;
  /** The chance baseline for this question. For multi-select it is always
   *  0.75 — each checkbox is an independent y/n with expected majority share
   *  (2 unanimous + 6 split x 2/3) / 8. For single-select with N options it is
   *  (N^2 + 3N - 1) / 3N^2, so it falls as the option list grows. */
  qraa?: number;
  kind?: "multi" | "single";
  /** Number of options the form offered for this question. */
  options?: number;
};

/** Agreement across the whole instrument, kept separate for multi- and
 *  single-select because their chance baselines differ. */
export type Agreement = {
  multiBoxes: number;
  multiObserved: number;
  multiRandom: number;
  singleCount: number;
  singleObserved: number;
  singleRandom: number;
};

/** One part of the codebook. `scored` is false for Durability, which the
 *  codebook explicitly excludes from any scale score. */
export type Band = {
  id: string;
  label: string;
  scored: boolean;
  leaves: readonly Leaf[];
};

export type AnnotationLink = { label: string; href: string };

/** E1 — whether a party outside the operator and the evaluated provider
 *  actually performed a relevant check. `sources` is empty when none was
 *  located; otherwise each entry links the verifier's own page. */
export type ExternalVerification = {
  status: string;
  sources: readonly AnnotationLink[];
};

export type Annotation = {
  /** URL segment under /annotations. */
  slug: string;
  name: string;
  status: AnnotationStatus;
  /** Eyebrow line: operator · domain · version. */
  operator: string | null;
  domain: string | null;
  version: string | null;
  /** C1 — consequential components intentionally withheld. */
  hidden: readonly string[];
  /** C3 — how evaluation proceeds while material stays withheld. */
  runBy: string | null;
  /** C4 — secrecy lifecycle: when protected material becomes accessible. */
  release: string | null;
  /** E1 — null while the verification items are uncoded. */
  externalVerification: ExternalVerification | null;
  summary: string | null;
  /** Guidance on reading the benchmark's published scores. */
  readingScores: string | null;
  /** The five codebook parts and their coded items. Empty until coded. */
  bands: readonly Band[];
  claims: Record<ClaimId, ClaimAssessment>;
  links: readonly AnnotationLink[];
  /** Annotator initials. Names are not published; the coding team is
   *  recorded in the annotation form, not on the page. */
  annotators: readonly string[];
  /** Frozen corpus cutoff — every code is tied to evidence available by then. */
  cutoff: string | null;
  /** Instrument-wide annotator agreement, shown under the root node. */
  agreement?: Agreement;
  /** WORKING NOTE — remove before publication. Open issues where the tree, the
   *  summary table and the coded tallies do not yet agree. Rendered under the
   *  tree so they stay in view while being resolved; an empty or absent list
   *  renders nothing. */
  reconcile?: readonly string[];
};

const uncodedClaims: Record<ClaimId, ClaimAssessment> = {
  validity: uncoded,
  integrity: uncoded,
  neutrality: uncoded,
  verification: uncoded,
};

// Context fields hold the majority code across the three annotators; `cutoff`
// is the latest frozen corpus date among them. Where coders split, the value
// taken by two of three is recorded and the disagreement stays in the form.
export const annotations: readonly Annotation[] = [
  {
    slug: "frontiermath",
    name: "FrontierMath",
    status: "context-coded",
    operator: "Epoch AI",
    domain: "Research-level mathematics",
    version: "Tiers 1–3, v2",
    hidden: ["Problems", "Reference answers"],
    runBy: "Epoch runs the model",
    release: "None planned",
    externalVerification: { status: "None located", sources: [] },
    summary:
      "Its scoring code, dated versions and run records are public, so on paper FrontierMath is mostly checkable. Its weak point is independence: OpenAI funded the problems and has every one of them, plus most solutions. This is a possible conflict of interest.",
    readingScores: null,
    bands: [
      {
        id: "durability",
        label: "Durability",
        scored: false,
        leaves: [
          {
            name: "Why it's hidden",
            question:
              "Privacy rationale — Why does the benchmark documentation say information is withheld?",
            sub: "Leakage, gaming, legal, fairness",
            detail:
              "Contamination, adaptive gaming, confidentiality and fair evaluation.",
            status: "desc",
            qaa: 0.875,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "adaptive gaming/overfitting", k: 3, n: 3 },
              { code: "contamination/leakage", k: 3, n: 3 },
              { code: "confidentiality/legal", k: 2, n: 3 },
              { code: "fair or blind evaluation", k: 2, n: 3 },
              { code: "proprietary/IP", k: 1, n: 3 },
              { code: "other", k: 0, n: 3 },
              { code: "security/safety", k: 0, n: 3 },
            ],
          },
          {
            name: "What's hidden",
            question:
              "Protected component — Which consequential parts of the evaluation are intentionally withheld?",
            sub: "Problems, answers",
            detail: "The problems and their reference answers.",
            status: "desc",
            qaa: 0.9583,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "items/tasks", k: 3, n: 3 },
              { code: "labels/reference answers", k: 3, n: 3 },
              { code: "run logs/configuration", k: 1, n: 3 },
              { code: "aggregation/reporting rules", k: 0, n: 3 },
              { code: "environment/tools/data", k: 0, n: 3 },
              { code: "other", k: 0, n: 3 },
              { code: "rubric/judge/scorer", k: 0, n: 3 },
              { code: "test membership/distribution", k: 0, n: 3 },
            ],
          },
          {
            name: "Hidden from",
            question:
              "Restricted audience — Who lacks ordinary access to the protected component?",
            sub: "Developers, public, clients",
            detail: "Evaluated developers, the public and customers.",
            status: "desc",
            qaa: 0.8571,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "Evaluated provider/team", k: 3, n: 3 },
              { code: "general public", k: 3, n: 3 },
              { code: "customer/client", k: 2, n: 3 },
              { code: "benchmark participant/user", k: 1, n: 3 },
              { code: "independent researcher", k: 1, n: 3 },
              { code: "other", k: 0, n: 3 },
              { code: "unclear", k: 0, n: 3 },
            ],
          },
          {
            name: "Release plan",
            question:
              "Secrecy lifecycle — When, if ever, is the protected material intended to become accessible?",
            sub: "None planned",
            detail: "No release of the hidden set is planned.",
            status: "desc",
            qaa: 0.6667,
            qraa: 0.4907,
            kind: "single",
            options: 6,
            votes: [
              { code: "Indefinite/no planned release", k: 2, n: 3 },
              { code: "not reported", k: 1, n: 3 },
            ],
          },
          {
            name: "How models run",
            question:
              "Evaluation access — How is evaluation performed while the protected material remains withheld?",
            sub: "Epoch runs them",
            detail: "Epoch runs every model itself.",
            status: "desc",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "evaluator runs the model/system", k: 3, n: 3 },
              { code: "hosted task/evaluation API", k: 0, n: 3 },
              { code: "other", k: 0, n: 3 },
              { code: "Prediction submission to server", k: 0, n: 3 },
              { code: "secure/controlled environment", k: 0, n: 3 },
              { code: "submitted code/container", k: 0, n: 3 },
              { code: "third-party platform/custodian", k: 0, n: 3 },
            ],
          },
          {
            name: "Durability controls",
            question:
              "Durability controls — What mechanisms are used to preserve benchmark usefulness across repeated evaluation?",
            sub: "Access limits, leak checks",
            detail: "Access controls and contamination monitoring.",
            status: "desc",
            qaa: 0.9524,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "access controls", k: 3, n: 3 },
              { code: "contamination/overlap monitoring", k: 3, n: 3 },
              { code: "retirement/replacement", k: 1, n: 3 },
              { code: "fresh/on-demand generation", k: 0, n: 3 },
              { code: "refresh/rotation", k: 0, n: 3 },
              { code: "Submission/rate limits", k: 0, n: 3 },
            ],
          },
        ],
      },
      {
        id: "validity",
        label: "Validity",
        scored: true,
        leaves: [
          {
            name: "Claims & limits",
            question:
              "Claim framing — Which elements needed to interpret the benchmark claim are explicitly stated?",
            sub: "Documented",
            detail: "Excludes proofs and open research.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 5,
            votes: [
              { code: "intended use or decision purpose", k: 3, n: 3 },
              { code: "limits or out-of-scope interpretation", k: 3, n: 3 },
              { code: "Measured capability/construct", k: 3, n: 3 },
              { code: "target domain/population", k: 3, n: 3 },
            ],
          },
          {
            name: "Sample tasks",
            question:
              "Task transparency — What lets an outsider understand the task without seeing the live private set?",
            sub: "Documented",
            detail: "12 public sample problems.",
            status: "ok",
            qaa: 0.8889,
            qraa: 0.75,
            kind: "multi",
            options: 3,
            votes: [
              { code: "Representative public items/dev set", k: 3, n: 3 },
              { code: "detailed task blueprint/schema", k: 2, n: 3 },
            ],
          },
          {
            name: "Hidden-set coverage",
            question:
              "Hidden-set coverage evidence — What evidence supports assessing whether the hidden set covers the stated task space?",
            sub: "Partial",
            detail:
              "Expert-reviewed, but never compared with the public problems.",
            status: "part",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "documented expert or independent review of hidden set", k: 3, n: 3 },
              { code: "Sampling/source method documented", k: 3, n: 3 },
              { code: "public-private comparability evidence", k: 0, n: 3 },
            ],
          },
          {
            name: "Scoring method",
            question:
              "Scoring transparency — What lets an outsider understand how model outputs become reported scores?",
            sub: "Documented",
            detail: "Answer-checking code is public.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 6,
            votes: [
              { code: "aggregation/threshold rules public", k: 3, n: 3 },
              { code: "judge/scorer details public", k: 3, n: 3 },
              { code: "Metric/formula public", k: 3, n: 3 },
              { code: "controlled inspection only", k: 0, n: 3 },
              { code: "rubric/judge criteria public", k: 0, n: 3 },
            ],
          },
        ],
      },
      {
        id: "integrity",
        label: "Integrity",
        scored: true,
        leaves: [
          {
            name: "Run rules",
            question:
              "Protocol rules — Which consequential evaluation rules are documented?",
            sub: "Documented",
            detail: "Token limits, forced submission, zero for no answer.",
            status: "ok",
            qaa: 0.8889,
            qraa: 0.75,
            kind: "multi",
            options: 6,
            votes: [
              { code: "Model/system eligibility or configuration", k: 3, n: 3 },
              { code: "result reporting/publication rules", k: 3, n: 3 },
              { code: "task selection/randomization", k: 3, n: 3 },
              { code: "failure/exclusion handling", k: 2, n: 3 },
              { code: "submission/retry/rerun rules", k: 2, n: 3 },
            ],
          },
          {
            name: "Fixed in advance",
            question:
              "Ex-ante fixation — What evidence indicates consequential rules or content were fixed before the relevant results?",
            sub: "Partial",
            detail: "Dated versions, but no hash or preregistration.",
            status: "part",
            qaa: 1.0,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "Dated/versioned protocol", k: 3, n: 3 },
              { code: "preregistration or immutable commitment", k: 0, n: 3 },
              { code: "third-party/escrow confirmation", k: 0, n: 3 },
            ],
          },
          {
            name: "Traceable to a run",
            question:
              "Run provenance — Which details identify the evaluation run that produced a reported score?",
            sub: "Documented",
            detail: "Version, model, settings and date. No scorer version.",
            status: "ok",
            qaa: 0.9048,
            qraa: 0.75,
            kind: "multi",
            options: 7,
            votes: [
              { code: "Benchmark version/snapshot", k: 3, n: 3 },
              { code: "inference/system configuration", k: 3, n: 3 },
              { code: "model version/checkpoint", k: 3, n: 3 },
              { code: "task subset/seed/session", k: 3, n: 3 },
              { code: "run date", k: 2, n: 3 },
              { code: "evaluator/scorer version", k: 1, n: 3 },
            ],
          },
          {
            name: "Run records",
            question:
              "Run audit evidence — What retained evidence could be used to check whether the reported run followed the stated procedure?",
            sub: "Partial",
            detail: "Logs only for the 10 public problems.",
            status: "part",
            qaa: 0.75,
            qraa: 0.75,
            kind: "multi",
            options: 4,
            votes: [
              { code: "controlled-access run records", k: 2, n: 3 },
              { code: "evaluator or third-party replay possible", k: 2, n: 3 },
              { code: "Public outputs/logs/run record", k: 2, n: 3 },
            ],
          },
        ],
      },
      {
        id: "neutrality",
        label: "Neutrality",
        scored: true,
        leaves: [
          {
            name: "Entry rules",
            question:
              "Participation-rule visibility — Who can inspect the rules governing eligibility and access to the benchmark?",
            sub: "Documented",
            detail: "Epoch reports only models it can run itself.",
            status: "ok",
            qaa: 1.0,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "Public", k: 3, n: 3 },
            ],
          },
          {
            name: "Unequal treatment",
            question:
              "Differential-treatment disclosure — Does the documentation state whether participants/providers may receive different access or evaluation treatment?",
            sub: "Disclosed risk",
            detail: "OpenAI has exclusive access to some problems.",
            status: "risk",
            qaa: 1.0,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "differential/privileged treatment disclosed or permitted", k: 3, n: 3 },
            ],
          },
          {
            name: "Ties to developers",
            question:
              "Evaluator–provider ties — What material relationships between the evaluator/operator and evaluated providers are documented?",
            sub: "Disclosed risk",
            detail: "OpenAI funded and commissioned the problems.",
            status: "risk",
            qaa: 0.9583,
            qraa: 0.75,
            kind: "multi",
            options: 8,
            votes: [
              { code: "customer/vendor relationship", k: 3, n: 3 },
              { code: "funding/sponsorship", k: 3, n: 3 },
              { code: "privileged model/data/API access", k: 3, n: 3 },
              { code: "research/technical partnership", k: 3, n: 3 },
              { code: "other material tie", k: 1, n: 3 },
              { code: "Same organization/affiliate", k: 0, n: 3 },
            ],
          },
        ],
      },
      {
        id: "verification",
        label: "Verification",
        scored: true,
        leaves: [
          {
            name: "Independent check",
            question:
              "External verification status — What evidence shows that a party outside the benchmark operator and evaluated provider actually performed a relevant check?",
            sub: "Not located",
            detail:
              "Reviews so far were commissioned by Epoch or done by OpenAI.",
            status: "gap",
            qaa: 0.6667,
            qraa: 0.6296,
            kind: "single",
            options: 3,
            votes: [
              { code: "no external verification located", k: 2, n: 3 },
              { code: "operator reports external verification only", k: 1, n: 3 },
            ],
          },
        ],
      },
    ],
    claims: {
      validity: {
        strengths: [
          "Scope and limits are documented, including the exclusion of proof-based problems.",
          "Public sample problems and the answer format are published.",
        ],
        concerns: [
          "The hidden set has never been compared with the public problems.",
        ],
        channels: [
          "A published comparison of the v2 public and hidden distributions.",
        ],
      },
      integrity: {
        strengths: [
          "The scoring harness is public; per-item verifiers are not.",
          "Every version is dated, and corrections are logged.",
          "Submission, rerun, exclusion and reporting rules are all documented.",
        ],
        concerns: ["Scores on hidden problems can't be checked item by item."],
        channels: [
          "A hash of the held-out set and the scorer version with every score.",
        ],
      },
      neutrality: {
        strengths: [
          "OpenAI's funding and access are stated on the benchmark page.",
        ],
        concerns: [
          "OpenAI has all 290 problems and solutions to 237 of them.",
        ],
        channels: [],
      },
      verification: {
        strengths: [],
        concerns: ["No independent party has checked the hidden set."],
        channels: [
          "An audit of the held-out problems by a party with no OpenAI tie.",
        ],
      },
    },
    links: [
      {
        label: "Epoch AI — Transparency",
        href: "https://epoch.ai/about/transparency",
      },
      {
        label: "Epoch AI Benchmarks — About",
        href: "https://epoch.ai/benchmarks/about",
      },
      {
        label: "FrontierMath Tiers 1–3 (unversioned benchmark page)",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3",
      },
      {
        label: "FrontierMath Tiers 1–3 v1",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1",
      },
      {
        label: "FrontierMath Tiers 1–3 v2",
        href: "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2",
      },
      {
        label: "GPQA Diamond benchmark graph",
        href: "https://epoch.ai/benchmarks/gpqa-diamond?view=graph&tab=release-date",
      },
      {
        label: "Epoch AI Benchmarks — Use this data",
        href: "https://epoch.ai/benchmarks/use-this-data",
      },
      {
        label: "Benchmark reviews documentation — Included benchmarks",
        href: "https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks",
      },
      {
        label: "Epoch benchmark data archive",
        href: "https://epoch.ai/data/benchmark_data.zip",
      },
      {
        label: "FrontierMath — The benchmark",
        href: "https://epoch.ai/frontiermath/the-benchmark",
      },
      {
        label: "FrontierMath Tiers 1–4",
        href: "https://epoch.ai/frontiermath/tiers-1-4",
      },
      {
        label: "FrontierMath Tiers 1–4 — About",
        href: "https://epoch.ai/frontiermath/tiers-1-4/about",
      },
      {
        label: "FrontierMath Tiers 1–4 — Benchmark problems",
        href: "https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems",
      },
      {
        label: "Less than 70% of FrontierMath is within reach for today's models",
        href: "https://epoch.ai/gradient-updates/less-than-70-percent-of-frontiermath-is-within-reach-for-todays-models",
      },
      {
        label: "Benchmarking hub update",
        href: "https://epoch.ai/latest/benchmarking-hub-update",
      },
      {
        label: "OpenAI and FrontierMath",
        href: "https://epoch.ai/latest/openai-and-frontiermath",
      },
      {
        label: "What is Epoch?",
        href: "https://epoch.ai/latest/what-is-epoch",
      },
      {
        label: "Deep Think Math",
        href: "https://epoch.ai/publications/deep-think-math",
      },
      {
        label: "Grok 4 Math",
        href: "https://epoch.ai/publications/grok-4-math",
      },
      {
        label: "GPT-5.4 set a new record on FrontierMath",
        href: "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath",
      },
      {
        label: "The Epoch Brief — June 12, 2026",
        href: "https://epochai.substack.com/p/the-epoch-brief-june-12-2026",
      },
      {
        label: "Public Inspect evaluation log",
        href: "https://logs.epoch.ai/inspect-viewer/c79c08da/viewer.html?log_file=https%3A%2F%2Flogs.epoch.ai%2Finspect_ai_logs%2FE9AfnKpQrDsFWKmXMhBbJS.eval",
      },
      {
        label: "Epoch AI LinkedIn post announcing FrontierMath Tiers 1–4 v2",
        href: "https://www.linkedin.com/posts/epochai_frontiermath-tiers-14-v2-is-live-we-activity-7471253846296711168-bRlG",
      },
      {
        label: "Epoch AI Research post on X",
        href: "https://x.com/EpochAIResearch/status/2065488154086568445",
      },
      {
        label: "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI (arXiv:2411.04872)",
        href: "https://arxiv.org/abs/2411.04872",
      },
      {
        label: "arXiv:2602.05192 (PDF, cited as a search-result snippet)",
        href: "https://arxiv.org/pdf/2602.05192",
      },
      {
        label: "FrontierMath scoring/evaluation implementation gist",
        href: "https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb",
      },
      {
        label: "Revision history for the scoring/evaluation gist",
        href: "https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb/revisions",
      },
      {
        label: "AI Wiki — FrontierMath",
        href: "https://aiwiki.ai/wiki/frontiermath",
      },
      {
        label: "What I wish I knew about FrontierMath (Lemmata)",
        href: "https://lemmata.substack.com/p/what-i-wish-i-knew-about-frontiermath",
      },
      {
        label: "Our World in Data — AI performance on FrontierMath over time",
        href: "https://ourworldindata.org/grapher/ai-frontiermath-over-time",
      },
      {
        label: "TechCrunch — AI benchmarking organization criticized for waiting to disclose funding from OpenAI",
        href: "https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai",
      },
      {
        label: "InfoQ — Epoch AI introduces FrontierMath benchmark",
        href: "https://www.infoq.com/news/2024/11/epochai-frontiermath-benchmark",
      },
      {
        label: "Some lessons from the OpenAI–FrontierMath debacle (LessWrong)",
        href: "https://www.lesswrong.com/posts/8ZgLYwBmB3vLavjKE/some-lessons-from-the-openai-frontiermath-debacle",
      },
      {
        label: "Meemi's shortform (LessWrong)",
        href: "https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform",
      },
      {
        label: "Specific comment on Meemi's shortform",
        href: "https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz",
      },
      {
        label: "How far behind are open models? (LessWrong)",
        href: "https://www.lesswrong.com/posts/rJcCrXyEsJKmmDpWG/how-far-behind-are-open-models",
      },
      {
        label: "Search Engine Journal — OpenAI secretly funded FrontierMath benchmarking dataset",
        href: "https://www.searchenginejournal.com/openai-secretly-funded-frontiermath-benchmarking-dataset/537760/",
      },
    ],
    annotators: ["MZ", "AP", "NT"],
    cutoff: "2026-10-18",
    agreement: {
      multiBoxes: 92,
      multiObserved: 0.9348,
      multiRandom: 0.75,
      singleCount: 4,
      singleObserved: 0.8333,
      singleRandom: 0.5949,
    },
    reconcile: [
      "Run records: each annotator picked a different pair of options (2/3 on all three). The leaf reads “Partial — Logs only for the 10 public problems”, which hides a three-way split about what run evidence exists at all.",
      "Ties to developers: unanimous on four relationship types — funding, customer/vendor, research partnership and privileged model/API access — but the Neutrality summary bullet names only funding.",
      "Scoring method: the leaf detail still says “Answer-checking code is public”. The summary bullet was tightened to note that per-item verifiers are withheld; the leaf should match.",
      "Options nobody selected are invisible here, because the tally is built from what was chosen. A 0/3 line would show what was considered and rejected — for example, no public–private comparability evidence under Hidden-set coverage.",
      "Verification is 2–1, not settled. One annotator coded “operator reports external verification only”, citing Epoch-commissioned blind reviews and mathematician interviews; the other two excluded both as construction-stage and operator-commissioned.",
      "Band order differs between views: the tree runs Durability first, the summary table starts at Validity. Pick one order.",
      "Box universe: the 92 count includes the “other” and “unclear” options, which are free text — agreement on a free-text box is not well defined, and the Part 1 working sheet leaves Restricted audience’s “Other” blank and averages over the remaining five. Decide whether they are scored or excluded, and make both views match.",
      "Verification scope is counted in the instrument-wide BAA but has no leaf in the tree, because it is conditional on an external check having occurred. For FrontierMath two annotators left it blank, so its eight boxes score as near-unanimous agreement about a question that did not apply. Decide whether conditional questions enter the denominator at all.",
      "BAA does not reconcile with the working sheet: 92 boxes agreed, but the share totals differ (86.0 here against 85.3 there, giving 0.9348 against 0.92717). 85.3 cannot be produced by majority shares of 1 and 2/3 over 92 binary boxes — it would need 20.1 split boxes — so one side is treating about two boxes differently, or rounding mid-calculation. Worth tracing before either number is published.",
    ],
  },
  {
    slug: "swebench_pro",
    name: "SWE-Bench Pro",
    status: "context-coded",
    operator: "Scale AI",
    domain: "Repository-level software engineering",
    version: "Commercial set, 276 tasks",
    hidden: [
      "Tasks",
      "Reference answers",
      "Rubric and scorer",
      "Environment and tooling",
    ],
    runBy: "Scale AI runs the model",
    release: "None planned",
    externalVerification: {
      status: "None located for the commercial set",
      sources: [],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      {
        label: "Benchmark page",
        href: "https://labs.scale.com/leaderboard/swe_bench_pro?tab=private",
      },
      { label: "Paper", href: "https://arxiv.org/abs/2509.16941" },
    ],
    annotators: ["MZ", "AP", "NT"],
    cutoff: "2026-10-17",
  },
  {
    slug: "hcast",
    name: "METR Time Horizon / HCAST",
    status: "context-coded",
    operator: "METR",
    domain: "Autonomous software and research tasks",
    version: "Time Horizon 1.1",
    hidden: [
      "Tasks",
      "Reference answers",
      "Rubric and scorer",
      "Environment and tooling",
    ],
    runBy: "METR runs the model",
    release: "None planned",
    externalVerification: {
      status: "Located",
      sources: [
        {
          label: "Nguyen & Fithian, arXiv:2610.12466",
          href: "https://arxiv.org/abs/2610.12466",
        },
        {
          label: "METR time-horizon reanalysis",
          href: "https://github.com/ofir-reich/metr-time-horizon-reanalysis",
        },
      ],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      {
        label: "Benchmark page",
        href: "https://metr.org/blog/2026-01-29-time-horizon-1-1/",
      },
      { label: "HCAST paper", href: "https://metr.org/hcast.pdf" },
    ],
    annotators: ["MZ", "AP", "NT"],
    cutoff: "2026-10-26",
  },
  {
    slug: "gdpval",
    name: "GDPval",
    status: "context-coded",
    operator: "OpenAI",
    domain: "Economically valuable occupational tasks",
    version: "v2, hidden split of 1,100 of 1,320 tasks",
    hidden: ["Tasks", "Reference answers", "Run logs and configuration"],
    runBy: "OpenAI runs the model",
    release: "None planned",
    externalVerification: {
      status: "Located",
      sources: [
        {
          label: "Inspect Evals: GDPval (UK AISI)",
          href: "https://ukgovernmentbeis.github.io/inspect_evals/evals/gdpval/",
        },
        {
          label: "Epoch AI on economic-value benchmarks",
          href: "https://epoch.ai/publications/what-do-economic-value-benchmarks-tell-us",
        },
      ],
    },
    summary: null,
    readingScores: null,
    bands: [],
    claims: uncodedClaims,
    links: [
      { label: "Benchmark page", href: "https://openai.com/index/gdpval/" },
      { label: "Paper", href: "https://arxiv.org/abs/2510.04374" },
      {
        label: "Dataset",
        href: "https://huggingface.co/datasets/openai/gdpval",
      },
    ],
    annotators: ["MZ", "AP", "NT"],
    cutoff: "2026-10-26",
  },
];

export function getAnnotation(slug: string): Annotation | undefined {
  return annotations.find((annotation) => annotation.slug === slug);
}

// ---------------------------------------------------------------------------
//  Evidence quality (Rubric v1)
//
//  Every SELECTED code carries four scores, one per dimension. The rubric is
//  explicit that they are never combined: "Record each score separately. Do not
//  add them up or assign a band." Bands are to be set only after the first
//  annotation round, once the score distribution is known. So a mean is taken
//  per dimension — mean V, mean S, mean P, mean C — and never across them.
//
//  Codes that were NOT selected carry a single Search completeness score
//  instead: how hard the annotator looked before concluding there was nothing.
//  That is what makes a negative finding credible, so it is shown too.
//
//  `N/A`, `none documented` and `not reported` codes are exempt from rating by
//  the rubric, which is why they carry no scores here.
// ---------------------------------------------------------------------------

/** One dimension of the rubric, for the reference pulldown. */
export type RubricDimension = {
  key: "v" | "s" | "p" | "c" | "search";
  letter: string;
  name: string;
  question: string;
  /** Level -> short name, indexed from `min`. */
  min: number;
  levels: readonly string[];
  rule: string;
};

export const evidenceRubric: readonly RubricDimension[] = [
  {
    key: "v",
    letter: "V",
    name: "Verifiability",
    question: "Does the evidence show the fact holds, or only state it?",
    min: 1,
    levels: ["Stated", "Inspectable", "Confirmed"],
    rule: "Rate what an outsider could check today. Plans and promises count as Stated. Runs 1–3 because every cited source at least states the fact.",
  },
  {
    key: "s",
    letter: "S",
    name: "Source independence",
    question: "Who made the claim, and do they have a stake in it?",
    min: 0,
    levels: [
      "Unattributed",
      "Interested or informal",
      "Operator official",
      "Independent",
    ],
    rule: "Rate who wrote this piece of evidence, not who hosts it. This rates the source, not the benchmark's ties.",
  },
  {
    key: "p",
    letter: "P",
    name: "Precision and fit",
    question: "Does the evidence cover this code, for this version?",
    min: 0,
    levels: ["Off-target", "Adjacent", "On-target", "Exact"],
    rule: "Evidence that needs an inference scores at most 1. Exact means this version, with checkable specifics.",
  },
  {
    key: "c",
    letter: "C",
    name: "Corroboration",
    question: "Do other sources agree?",
    min: 0,
    levels: ["Contradicted", "Single", "Same party", "Cross-party"],
    rule: "Count sources, not copies. Silence is not agreement.",
  },
  {
    key: "search",
    letter: "Search",
    name: "Search completeness",
    question:
      "How thoroughly did you search before concluding a code has no support?",
    min: 0,
    levels: ["Minimal", "Partial", "Complete", "Targeted"],
    rule: "Core sources: benchmark page, paper, code repository, docs or FAQ, changelog, operator posts, dataset card. Targeted adds a keyword search and a web search.",
  },
];

/** The evidence an annotator cited for one code, as recorded in the form's
 *  notes field. Roughly four in five rated codes have one; the rest were rated
 *  without a per-code note, so the page says so rather than inventing one. */
export type Rationale = {
  sources: readonly string[];
  quotes: readonly string[];
  note: string | null;
};

export type QualityEntry = {
  code: string;
  by: string;
  v: number;
  s: number;
  p: number;
  c: number;
  rationale?: readonly Rationale[];
};

export type QuestionQuality = {
  /** Joins to the tree leaf of the same name. */
  leaf: string;
  question: string;
  entries: readonly QualityEntry[];
  search: readonly { by: string; score: number }[];
  comments: readonly { by: string; text: string }[];
};

export type BandQuality = {
  id: string;
  label: string;
  rated: number;
  /** Per-dimension means. Never averaged together — see the note above. */
  means: { v: number; s: number; p: number; c: number } | null;
  questions: readonly QuestionQuality[];
};

/** Look up a dimension's level name for a score, e.g. V2 -> "Inspectable". */
export function levelName(
  key: RubricDimension["key"],
  score: number,
): string | null {
  const dim = evidenceRubric.find((d) => d.key === key);
  if (!dim) return null;
  return dim.levels[score - dim.min] ?? null;
}

export const frontierMathEvidenceQuality: readonly BandQuality[] = [
  {
    id: "durability",
    label: "Durability",
    rated: 40,
    means: { v: 1.18, s: 1.95, p: 1.48, c: 1.65 },
    questions: [
      {
        leaf: "Why it's hidden",
        question:
          "Privacy rationale — Why does the benchmark documentation say information is withheld?",
        entries: [
          {
            code: "contamination/leakage",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/abs/2411.04872", "https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["FrontierMath uses new, unpublished problems and automated verification to reliably evaluate models while minimizing risk of data contamination.", "All problems are novel and unpublished to ensure AI models have not previously encountered them during training."],
                note: null,
              },
            ],
          },
          {
            code: "fair or blind evaluation",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["This allows us to independently test OpenAI and other AI models on solutions no AI developer has access to."],
                note: "Despite this, it appears it is semi-biased in favor of OpenAI.",
              },
            ],
          },
          {
            code: "proprietary/IP",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["As is typical of commissioned work, OpenAI retains ownership of these questions and has access to the problems and solutions, with the exception of a holdout set.", "However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
            ],
          },
          {
            code: "adaptive gaming/overfitting",
            by: "MZ",
            v: 1, s: 1, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform"],
                quotes: ["…a separate, unseen holdout set—an extra safeguard to prevent overfitting and ensure accurate progress measurement."],
                note: null,
              },
            ],
          },
          {
            code: "contamination/leakage",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about", "https://arxiv.org/pdf/2411.04872", "https://epoch.ai/benchmarks/about"],
                quotes: ["All problems are novel and unpublished to ensure AI models have not previously encountered them during training.", "by using exclusively novel, unreleased problems, it helps prevent data contamination", "To mitigate the risk of accidental leakage into LLM training corpora"],
                note: null,
              },
            ],
          },
          {
            code: "adaptive gaming/overfitting",
            by: "AP",
            v: 1, s: 1, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz"],
                quotes: ["an extra safeguard to prevent overfitting and ensure accurate progress measurement"],
                note: null,
              },
            ],
          },
          {
            code: "fair or blind evaluation",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["We are finalizing a 50-problem set for which OpenAI will only receive the problem statements and not the solutions. This allows us to independently test OpenAI and other AI models on solutions no AI developer has access to."],
                note: null,
              },
            ],
          },
          {
            code: "confidentiality/legal",
            by: "AP",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["Epoch AI is free to conduct and publish evaluations of any models using the FrontierMath problem set commissioned by OpenAI. However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
            ],
          },
          {
            code: "contamination/leakage",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/publications/grok-4-math"],
                quotes: ["Epoch’s FrontierMath benchmark gives us an independent data point on Grok 4’s overall math capabilities: while concerns about overfitting and data contamination may loom over public benchmarks, the private FrontierMath problems should be safe from that."],
                note: "Epoch explicitly links private FrontierMath problems to protection against both contamination and overfitting.",
              },
              {
                sources: ["https://arxiv.org/html/2411.04872v3"],
                quotes: ["FrontierMath uses new, unpublished problems and automated verification to reliably evaluate models while minimizing risk of data contamination."],
                note: "Unpublished problems are intended to reduce the risk that models encounter test material during training. This sentence supports contamination/leakage",
              },
            ],
          },
          {
            code: "adaptive gaming/overfitting",
            by: "NT",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/publications/grok-4-math"],
                quotes: ["Epoch’s FrontierMath benchmark gives us an independent data point on Grok 4’s overall math capabilities: while concerns about overfitting and data contamination may loom over public benchmarks, the private FrontierMath problems should be safe from that."],
                note: "Epoch explicitly links private FrontierMath problems to protection against both contamination and overfitting.",
              },
            ],
          },
          {
            code: "confidentiality/legal",
            by: "NT",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: "This documents a permission requirement that restricts sharing.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc. The only source is a forum comment posted during the January 2025 controversy over OpenAI's access; no official Epoch page states this rationale." },
          { by: "AP", text: "Promise: written while the holdout was still being finalized; the About page later reports it done. Red flag: numbers differ (a \"50-problem set\" here; 53 withheld solutions on the About page). Red flag: post-hoc, published in response to the January 2025 controversy." },
          { by: "AP", text: "Red flag: post-hoc. The sharing restriction was disclosed only in the January 2025 clarification; Epoch says its contract barred earlier disclosure." },
          { by: "AP", text: "Code repository (Epoch's FrontierMath gist) not checked for this question; no dataset card exists." },
          { by: "NT", text: "The supporting sources predate v2. They document earlier reasons for restricting information, but do not confirm unchanged arrangements for v2. Fair/blind evaluation is described for an earlier holdout; its continuation in v2 was not established." },
        ],
      },
      {
        leaf: "What's hidden",
        question:
          "Protected component — Which consequential parts of the evaluation are intentionally withheld?",
        entries: [
          {
            code: "items/tasks",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["We have made twelve problems public: ten from Tiers 1-3 and two from Tier 4."],
                note: null,
              },
            ],
          },
          {
            code: "labels/reference answers",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["…except for 53 solutions randomly withheld in version FrontierMath_02-28-25 for holdout evaluation."],
                note: null,
              },
              {
                sources: [],
                quotes: [" the problem has a unique integer solution, it checks for an exact match. If the problem has a unique symbolic real answer, it uses SymPy evaluation to check that the difference between the submitted answer and the actual answer simplifies to 0. In all other cases, a custom verification script is necessary to check validity of a submitted answer (see Figure 3)."],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "The selection method is disclosed (\"randomly withheld\"). The subject distribution (MSC table) and tier shares are public, though for v1-era sets. Which specific problems are in the holdout, and which were corrected or removed in v2, is never stated, but undisclosed is not the same as documented as withheld.",
              },
            ],
          },
          {
            code: "items/tasks",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["all the numbers on this hub correspond to evaluations on the private sets", "However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
            ],
          },
          {
            code: "labels/reference answers",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about", "https://epoch.ai/latest/openai-and-frontiermath", "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath"],
                quotes: ["except for 53 solutions randomly withheld in version FrontierMath_02-28-25 for holdout evaluation", "OpenAI will only receive the problem statements and not the solutions", "FrontierMath was funded by OpenAI, who has exclusive access to: all 290 problems in Tiers 1–3; solutions to 237 of these problems; 28 of the 48 problems in Tier 4; solutions to these 28 problems. Epoch holds out the rest."],
                note: null,
              },
            ],
          },
          {
            code: "run logs/configuration",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For FrontierMath, the log viewer is only available to see model traces on the 10 public questions in frontiermath-2025-02-28-public."],
                note: "The configuration is published: the v2 methodology section points readers to the public-question log viewer as the best way to understand evaluation settings, and publishes the full prompt, token limits, and pinned tool libraries.",
              },
              {
                sources: [],
                quotes: [],
                note: "The scorer itself is public. It builds each scoring run from the item's own verification code stored in dataset metadata, and marks the answer correct only if that problem's verify() returns True. Problems without a unique integer or symbolic answer need a custom verification script. So item-level judging criteria for private problems are not public. However, no document states this as a restriction; it follows from the items being private.",
              },
              {
                sources: [],
                quotes: [],
                note: "The sampling rule is disclosed: the 53 held-out solutions were chosen at random. Two things are not published: which problems are in the holdout; which problems were changed in v2. The changelog reports 123 corrected and 5 removed Tiers 1-3 problems but does not identify them.",
              },
            ],
          },
          {
            code: "items/tasks",
            by: "NT",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Unless stated otherwise, all the numbers on this hub correspond to evaluations on the private sets."],
                note: "The v2 page identifies a private evaluation set, alongside a small public sample of questions.",
              },
            ],
          },
          {
            code: "labels/reference answers",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: "The sharing restriction explicitly covers both the questions and their answers.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 2 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: numbers differ (a 50-problem holdout in the January 2025 post; 53 withheld solutions on the About page)." },
          { by: "AP", text: "Excerpt: the X announcement (test membership) was read only as a search snippet. Dataset card does not exist; add it to the checked list." },
        ],
      },
      {
        leaf: "Hidden from",
        question:
          "Restricted audience — Who lacks ordinary access to the protected component?",
        entries: [
          {
            code: "Evaluated provider/team",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["This allows us to independently test OpenAI and other AI models on solutions no AI developer has access to."],
                note: null,
              },
            ],
          },
          {
            code: "general public",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://ourworldindata.org/grapher/ai-frontiermath-over-time"],
                quotes: ["Only 12 of the problems are publicly available: 10 from Tiers 1–3 and 2 from Tier 4."],
                note: null,
              },
            ],
          },
          {
            code: "customer/client",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: [],
                quotes: ["They have access to all problem statements and solutions up to FrontierMath_12-04-24,” / “except for 53 solutions randomly withheld in version FrontierMath_02-28-25 for holdout evaluation."],
                note: null,
              },
            ],
          },
          {
            code: "evaluated provider/team",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath", "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath"],
                quotes: ["solutions no AI developer has access to\"; \"However, we cannot share the questions and answers with other parties without written permission from OpenAI.", "FrontierMath was funded by OpenAI, who has exclusive access to"],
                note: null,
              },
            ],
          },
          {
            code: "customer/client",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["OpenAI commissioned 300 questions for the core FrontierMath dataset"],
                note: null,
              },
            ],
          },
          {
            code: "general public",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://arxiv.org/abs/2411.04872"],
                quotes: ["We have made twelve problems public: ten from Tiers 1-3", "FrontierMath uses new, unpublished problems"],
                note: null,
              },
            ],
          },
          {
            code: "Evaluated provider/team",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/publications/deep-think-math"],
                quotes: ["While most model developers also only have access to the twelve public samples, OpenAI, which funded FrontierMath, has wider access."],
                note: "Most model developers had access only to public samples; OpenAI was a disclosed exception.",
              },
            ],
          },
          {
            code: "benchmark participant/user",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
          },
          {
            code: "independent researcher",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
          },
          {
            code: "general public",
            by: "NT",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["Epoch AI is free to conduct and publish evaluations of any models using the FrontierMath problem set commissioned by OpenAI. However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "Ordinary access is restricted; OpenAI has special subset access. The developer-access report and ownership clarification predate v2. Their historical sample and problem counts should not be applied to v2. No separate customer/client policy located.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (January 2025 clarification)." },
          { by: "AP", text: "Red flags: post-hoc; numbers differ (50-problem holdout vs. 53 withheld solutions)." },
          { by: "AP", text: "Code repository not checked; dataset card not named as absent." },
        ],
      },
      {
        leaf: "Release plan",
        question:
          "Secrecy lifecycle — When, if ever, is the protected material intended to become accessible?",
        entries: [
          {
            code: "Indefinite/no planned release",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
            ],
          },
          {
            code: "indefinite/no planned release",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath", "https://epoch.ai/frontiermath/tiers-1-4", "https://x.com/EpochAIResearch/status/2065488154086568445"],
                quotes: ["but will not have access to the solutions", "A benchmark of several hundred unpublished, highly challenging mathematics problems", "is now approaching saturation"],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "Public samples have grown over time:",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Excerpt: the X announcement was read only as a search snippet. Red flag: post-hoc (January 2025 clarification)." },
          { by: "AP", text: "Code repository not checked; dataset card not named as absent. Excerpt: the X announcement and the v1 page were read only as search snippets." },
        ],
      },
      {
        leaf: "How models run",
        question:
          "Evaluation access — How is evaluation performed while the protected material remains withheld?",
        entries: [
          {
            code: "evaluator runs the model/system",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For Epoch AI-administered benchmarks, we only report results for models where we can run evaluations ourselves."],
                note: null,
              },
            ],
          },
          {
            code: "evaluator runs the model/system",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about", "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath", "https://epoch.ai/benchmarks/frontiermath-tiers-1-3"],
                quotes: ["we only report results for models where we can run evaluations ourselves", "Our implementation allows the model to reason and run Python code.", "We had pre-release access to evaluate the model.", "We used the streaming API"],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "The model's Python tool and the scorer run in a Docker sandbox with no network access and a 512 MB memory cap. That sandbox limits the model's code execution; it does not protect the benchmark material. The problem text itself goes to the provider's API during each run. Nothing resembles a restricted room.",
              },
            ],
          },
          {
            code: "evaluator runs the model/system",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/publications/grok-4-math"],
                quotes: ["We have thus far run Grok 4 on FrontierMath’s Tier 1-3 problems twice, trying to find API settings that would let us get a complete run."],
                note: "Epoch describes running the model on FrontierMath itself, using the model provider’s API.",
              },
              {
                sources: [],
                quotes: ["We performed a full manual evaluation of Deep Think on the FrontierMath benchmark."],
                note: "This is another example of Epoch conducting the evaluation itself.",
              },
              {
                sources: [],
                quotes: [],
                note: "Both reports predate v2. The final Python answer function is a response format, not a submitted participant container. No public prediction server, task API, custodian, or controlled-room mechanism are found.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 2 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Excerpt: the pre-v2 hub page (\"streaming API\") was read only as a search snippet." },
          { by: "AP", text: "Excerpt: the pre-v2 hub page and the LessWrong post were read only as search snippets. Dataset card not named as absent." },
        ],
      },
      {
        leaf: "Durability controls",
        question:
          "Durability controls — What mechanisms are used to preserve benchmark usefulness across repeated evaluation?",
        entries: [
          {
            code: "access controls",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["They have access to all problem statements and solutions up to FrontierMath_12-04-24, except for 53 solutions randomly withheld in version FrontierMath_02-28-25 for holdout evaluation."],
                note: null,
              },
            ],
          },
          {
            code: "contamination/overlap monitoring",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7"],
                quotes: ["To provide further confidence in our problems' originality, we ran them through the plagiarism detection tools Quetext and Copyscape."],
                note: null,
              },
            ],
          },
          {
            code: "retirement/replacement",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: [],
                quotes: ["We released v2, which corrected 123 problems in Tiers 1-3 and 12 problems in Tier 4. We also removed 5 problems from Tiers 1-3 and 7 problems from Tier 4."],
                note: null,
              },
            ],
          },
          {
            code: "access controls",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epoch.ai/frontiermath/tiers-1-4/about", "https://epoch.ai/benchmarks/about"],
                quotes: ["all the numbers on this hub correspond to evaluations on the private sets", "53 solutions from this version are held out from OpenAI for evaluation purposes", "bots are prevented from accessing the log viewer"],
                note: null,
              },
            ],
          },
          {
            code: "contamination/overlap monitoring",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7#S2.SS4", "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath"],
                quotes: ["we ran them through the plagiarism detection tools Quetext and Copyscape", "Neither of these differences is statistically significant."],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "The holdout came from problems added in the 02-28-25 version, and the paper promises new problems. But these were one-time expansions, not generation per run or per release, and v2 added no new Tiers 1-3 problems.",
              },
            ],
          },
          {
            code: "contamination/overlap monitoring",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v3"],
                quotes: ["To provide further confidence in our problems’ originality, we ran them through the plagiarism detection tools Quetext and Copyscape."],
                note: "The authors describe checking whether newly constructed problems overlap with existing material.",
              },
            ],
          },
          {
            code: "access controls",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: [],
                quotes: ["To minimize the risk of problems and solutions being disseminated online, we encouraged all submissions to be conducted through secure, encrypted channels.", "We employed encrypted communication platforms to coordinate with contributors and requested that any written materials stored online be encrypted—for instance, by placing documents in password-protected archives shared only via secure methods."],
                note: "The authors describe secure transfer and storage requests, while acknowledging exceptions.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: self-vouching (the plagiarism check's \"no significant matches\" result is reported with no output shown)." },
          { by: "AP", text: "Code repository not checked; dataset card not named as absent. Excerpt: the X announcement and the v1 page were read only as search snippets." },
          { by: "NT", text: "Earlier construction procedures; no independently inspectable enforcement records." },
        ],
      },
    ],
  },
  {
    id: "validity",
    label: "Validity",
    rated: 32,
    means: { v: 1.62, s: 1.97, p: 1.91, c: 1.62 },
    questions: [
      {
        leaf: "Claims & limits",
        question:
          "Claim framing — Which elements needed to interpret the benchmark claim are explicitly stated?",
        entries: [
          {
            code: "Measured capability/construct",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["FrontierMath is a benchmark we created to evaluate the mathematical capabilities of frontier AI models."],
                note: null,
              },
            ],
          },
          {
            code: "intended use or decision purpose",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["This remains our core mission—to help the AI community and the public at large accurately understand and measure AI capabilities."],
                note: null,
              },
            ],
          },
          {
            code: "target domain/population",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["The questions cover most major branches of modern mathematics – from computationally intensive problems in number theory and real analysis to abstract questions in algebraic geometry and category theory."],
                note: null,
              },
            ],
          },
          {
            code: "limits or out-of-scope interpretation",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7"],
                quotes: ["First, the practical focus on automatically verifiable and numerical answers excludes proof-writing and open-ended exploration, which are significant parts of modern math research."],
                note: null,
              },
            ],
          },
          {
            code: "Measured capability/construct",
            by: "AP",
            v: 1, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["each problem is carefully crafted to evaluate genuine mathematical understanding"],
                note: null,
              },
            ],
          },
          {
            code: "intended use or decision purpose",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["the AI community and the public at large accurately understand and measure AI capabilities"],
                note: null,
              },
            ],
          },
          {
            code: "target domain/population",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Expert-written math problems covering advanced undergrad through early career research."],
                note: null,
              },
            ],
          },
          {
            code: "limits or out-of-scope interpretation",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7#S4.SS2.SSS1", "https://epoch.ai/benchmarks/about"],
                quotes: ["this does not mean that its reasoning was correct", "it may not always generalize to other contexts or use cases"],
                note: null,
              },
            ],
          },
          {
            code: "Measured capability/construct",
            by: "NT",
            v: 1, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Expert-written math problems covering advanced undergrad through early career research.", "FrontierMath is a benchmark of hundreds of original, exceptionally challenging mathematics problems crafted and vetted by expert mathematicians."],
                note: "The page identifies both the advanced mathematical task and the intended difficulty range.",
              },
            ],
          },
          {
            code: "intended use or decision purpose",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v3"],
                quotes: ["As AI systems advance toward expert-level mathematical abilities, FrontierMath offers a rigorous testbed that quantifies their progress."],
                note: "The stated purpose is to measure progress toward expert-level mathematics.",
              },
              {
                sources: ["https://arxiv.org/html/2411.04872v3"],
                quotes: ["While this approach enables efficient and scalable evaluation, it does impose certain limitations. Most notably, we cannot include problems that require mathematical proofs or formal reasoning steps, as these would demand human evaluation to assess correctness and clarity."],
                note: "The paper explains that automatically checking final answers leaves proof-writing outside this evaluation format.",
              },
              {
                sources: [],
                quotes: [],
                note: "This describes advanced mathematical reasoning and progress measurement. The earlier paper states the proof-writing limitation.",
              },
            ],
          },
          {
            code: "target domain/population",
            by: "NT",
            v: 1, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Expert-written math problems covering advanced undergrad through early career research.", "FrontierMath is a benchmark of hundreds of original, exceptionally challenging mathematics problems crafted and vetted by expert mathematicians."],
                note: "The page identifies both the advanced mathematical task and the intended difficulty range.",
              },
            ],
          },
          {
            code: "limits or out-of-scope interpretation",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
          },
        ],
        search: [
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (January 2025 clarification)." },
        ],
      },
      {
        leaf: "Sample tasks",
        question:
          "Task transparency — What lets an outsider understand the task without seeing the live private set?",
        entries: [
          {
            code: "Representative public items/dev set",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems"],
                quotes: ["These sample problems span the difficulty levels and subject areas covered in the benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "Representative public items/dev set",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems", "https://arxiv.org/html/2411.04872v7#A1", "https://epoch.ai/benchmarks/about"],
                quotes: ["span the difficulty levels and subject areas covered in the benchmark", "with one chosen at random from each quintile of difficulty", "only available to see model traces on the 10 public questions"],
                note: null,
              },
            ],
          },
          {
            code: "detailed task blueprint/schema",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about", "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["natural-language math problems with solutions expressed as closed-form mathematical expressions\" Also includes the three problem requirements (definite verifiable answer, guessproof, computationally tractable) and tier definitions based on background, creativity, and execution.", "the model needs to submit a Python function answer() that returns the answer"],
                note: null,
              },
            ],
          },
          {
            code: "Representative public items/dev set",
            by: "NT",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems"],
                quotes: ["These sample problems span the difficulty levels and subject areas covered in the benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "detailed task blueprint/schema",
            by: "NT",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["For each FrontierMath question, the model needs to submit a Python function `answer()` that returns the answer. The answer is a Python object, often (although not always) an integer or a `sympy` object. Our implementation allows the model to reason and run Python code."],
                note: "The page publishes the answer format and the prompt specifying Python tools and resource limits.",
              },
            ],
          },
        ],
        search: [
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: numbers differ (the About page says 300 core and 350 total problems; the v2 page says 295 and 338)." },
        ],
      },
      {
        leaf: "Hidden-set coverage",
        question:
          "Hidden-set coverage evidence — What evidence supports assessing whether the hidden set covers the stated task space?",
        entries: [
          {
            code: "Sampling/source method documented",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7"],
                quotes: ["Each mathematician created new problems following guidelines designed to ensure clarity, verifiability, and definitive answers across various difficulty levels and mathematical domains."],
                note: null,
              },
            ],
          },
          {
            code: "documented expert or independent review of hidden set",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["Each problem undergoes peer review by expert mathematicians who verify correctness, check for ambiguities, and assess difficulty ratings."],
                note: null,
              },
            ],
          },
          {
            code: "Sampling/source method documented",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/pdf/2411.04872"],
                quotes: ["FrontierMath contains problems across 70% of the top-level subjects in the MSC2020 classification\" ; \"Each mathematician created new problems following guidelines designed to ensure clarity, verifiability, and definitive answers across various difficulty levels and mathematical domains."],
                note: null,
              },
            ],
          },
          {
            code: "documented expert or independent review of hidden set",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about", "https://x.com/EpochAIResearch/status/2065488154086568445"],
                quotes: ["peer review by expert mathematicians who verify correctness", "engaged mathematicians to review these flags"],
                note: null,
              },
              {
                sources: [],
                quotes: [],
                note: "The paper's public samples were drawn at random, one from each difficulty quintile, but the public and private splits are never compared directly. The GPT-5.4 post compares two private splits (held-out and non-held-out), not public against private.",
              },
            ],
          },
          {
            code: "Sampling/source method documented",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v3"],
                quotes: ["Each mathematician created new problems following guidelines designed to ensure clarity, verifiability, and definitive answers across various difficulty levels and mathematical domains."],
                note: null,
              },
            ],
          },
          {
            code: "documented expert or independent review of hidden set",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about", "https://arxiv.org/html/2411.04872v3"],
                quotes: ["Each problem undergoes peer review by expert mathematicians who verify correctness, check for ambiguities, and assess difficulty ratings."],
                note: null,
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Excerpt: the X announcement was read only as a search snippet." },
          { by: "AP", text: "v2 page, changelog and code repository not listed as checked; dataset card not named as absent." },
        ],
      },
      {
        leaf: "Scoring method",
        question:
          "Scoring transparency — What lets an outsider understand how model outputs become reported scores?",
        entries: [
          {
            code: "Metric/formula public",
            by: "MZ",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["If your answer is correct you will get 1 point. If it is incorrect, or if you don't submit an answer, you will get 0 points."],
                note: null,
              },
            ],
          },
          {
            code: "judge/scorer details public",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7"],
                quotes: ["If the problem has a unique integer solution, it checks for an exact match. If the problem has a unique symbolic real answer, it uses SymPy evaluation to check that the difference between the submitted answer and the actual answer simplifies to 0."],
                note: null,
              },
            ],
          },
          {
            code: "aggregation/threshold rules public",
            by: "MZ",
            v: 1, s: 1, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://ourworldindata.org/grapher/ai-frontiermath-over-time"],
                quotes: ["This indicator shows the share of FrontierMath problems that AI models solve correctly, based on Epoch AI's evaluation."],
                note: null,
              },
            ],
          },
          {
            code: "Metric/formula public",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb"],
                quotes: ["If your answer is correct you will get 1 point.", "@scorer(metrics=[accuracy(), stderr()], name=\"verification_code\")"],
                note: null,
              },
            ],
          },
          {
            code: "judge/scorer details public",
            by: "AP",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7#S2.SS2"],
                quotes: ["If the problem has a unique integer solution, it checks for an exact match."],
                note: null,
              },
            ],
          },
          {
            code: "aggregation/threshold rules public",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about", "https://epoch.ai/benchmarks/frontiermath-tiers-1-3"],
                quotes: ["a confidence interval of plus and minus one standard error", "We allow up to 1% of samples to fail without being scored"],
                note: null,
              },
            ],
          },
          {
            code: "Metric/formula public",
            by: "NT",
            v: 2, s: 2, p: 3, c: 2,
          },
          {
            code: "judge/scorer details public",
            by: "NT",
            v: 2, s: 2, p: 2, c: 2,
          },
          {
            code: "aggregation/threshold rules public",
            by: "NT",
            v: 2, s: 2, p: 2, c: 1,
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Excerpt: the pre-v2 hub page (1% failure rule) was read only as a search snippet." },
          { by: "AP", text: "Operator posts not checked; dataset card not named as absent. Excerpt: pre-v2 hub page." },
        ],
      },
    ],
  },
  {
    id: "integrity",
    label: "Integrity",
    rated: 37,
    means: { v: 1.57, s: 2.0, p: 1.92, c: 1.16 },
    questions: [
      {
        leaf: "Run rules",
        question:
          "Protocol rules — Which consequential evaluation rules are documented?",
        entries: [
          {
            code: "Model/system eligibility or configuration",
            by: "AP",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["We only consider the chat/instruct version of each model, rather than the base model"],
                note: null,
              },
            ],
          },
          {
            code: "task selection/randomization",
            by: "AP",
            v: 1, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["all the numbers on this hub correspond to evaluations on the private sets"],
                note: null,
              },
            ],
          },
          {
            code: "submission/retry/rerun rules",
            by: "AP",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://gist.github.com/tadamcz/7b3a45d0d5d35f758bb6deb8000731bb"],
                quotes: ["Your answer has been recorded. No feedback is provided."],
                note: null,
              },
            ],
          },
          {
            code: "result reporting/publication rules",
            by: "AP",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["retains the right to conduct and publish evaluations using the full FrontierMath dataset"],
                note: null,
              },
            ],
          },
          {
            code: "Model/system eligibility or configuration",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["We only consider the chat/instruct version of each model, rather than the base model (see the model identifier)."],
                note: null,
              },
            ],
          },
          {
            code: "task selection/randomization",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Unless stated otherwise, all the numbers on this hub correspond to evaluations on the private sets."],
                note: null,
              },
            ],
          },
          {
            code: "failure/exclusion handling",
            by: "MZ",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["If it is incorrect, or if you don't submit an answer, you will get 0 points."],
                note: null,
              },
            ],
          },
          {
            code: "result reporting/publication rules",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For Epoch AI-administered benchmarks, we only report results for models where we can run evaluations ourselves."],
                note: null,
              },
            ],
          },
          {
            code: "Model/system eligibility or configuration",
            by: "NT",
            v: 1, s: 2, p: 3, c: 1,
          },
          {
            code: "task selection/randomization",
            by: "NT",
            v: 1, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: [],
                note: null,
              },
            ],
          },
          {
            code: "submission/retry/rerun rules",
            by: "NT",
            v: 1, s: 2, p: 3, c: 1,
          },
          {
            code: "failure/exclusion handling",
            by: "NT",
            v: 1, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: [],
                note: null,
              },
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For Epoch AI-administered benchmarks, we only report results for models where we can run evaluations ourselves."],
                note: null,
              },
            ],
          },
          {
            code: "result reporting/publication rules",
            by: "NT",
            v: 1, s: 2, p: 1, c: 1,
          },
        ],
        search: [
          { by: "AP", score: 0 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (the conflict-of-interest statement was added after the January 2025 controversy)." },
          { by: "AP", text: "Only the v2 page checked; dataset card not named as absent." },
        ],
      },
      {
        leaf: "Fixed in advance",
        question:
          "Ex-ante fixation — What evidence indicates consequential rules or content were fixed before the relevant results?",
        entries: [
          {
            code: "Dated/versioned protocol",
            by: "AP",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epoch.ai/frontiermath/tiers-1-4/about"],
                quotes: ["On 2026-06-12, we released a major update, addressing errors in 42% of problems.", "FrontierMath_02-28-25: Full 300 core dataset."],
                note: null,
              },
            ],
          },
          {
            code: "Dated/versioned protocol",
            by: "MZ",
            v: 1, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["2026-06-12. We released v2, which corrected 123 problems in Tiers 1-3 and 12 problems in Tier 4. We also removed 5 problems from Tiers 1-3 and 7 problems from Tier 4."],
                note: null,
              },
              {
                sources: [],
                quotes: ["In the future, we plan to pin each benchmark run to the exact git revision for full auditability” ; \"This implementation is different from the code we used to run preliminary evaluations in the paper."],
                note: null,
              },
            ],
          },
          {
            code: "Dated/versioned protocol",
            by: "NT",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["On 2026-06-12, we released a major update, addressing errors in 42% of problems."],
                note: "Epoch says it released the updated benchmark on June 12, 2026. The example evaluation run in its data is dated July 9, 2026. The stated release date therefore comes before that run. Source for the run date: https://epoch.ai/data/benchmark_data.zip",
              },
              {
                sources: [],
                quotes: [],
                note: "There is a limitation: this establishes a timeline, but does not prove that all evaluation rules were set before the results were known. I did not find a preserved copy showing exactly which rules were in place before the run, or another record showing those rules had been locked in.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 2 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Excerpt: the pre-v2 hub page was read only as a search snippet. Dataset card not named as absent." },
        ],
      },
      {
        leaf: "Traceable to a run",
        question:
          "Run provenance — Which details identify the evaluation run that produced a reported score?",
        entries: [
          {
            code: "Benchmark version/snapshot",
            by: "AP",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["Unless stated otherwise, all the numbers on this hub correspond to evaluations on the private sets."],
                note: null,
              },
            ],
          },
          {
            code: "model version/checkpoint",
            by: "AP",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["We only consider the chat/instruct version of each model, rather than the base model (see the model identifier)."],
                note: null,
              },
            ],
          },
          {
            code: "inference/system configuration",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epoch.ai/benchmarks/about"],
                quotes: ["There is a hard limit of 1,000,000 tokens.", "We use each model’s API-default temperature"],
                note: null,
              },
            ],
          },
          {
            code: "task subset/seed/session",
            by: "AP",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath"],
                quotes: ["On Tiers 1–3 GPT-5.4 Pro solved 52% of the non-held-out set and 42% of the held-out set."],
                note: null,
              },
            ],
          },
          {
            code: "Benchmark version/snapshot",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://ourworldindata.org/grapher/ai-frontiermath-over-time"],
                quotes: ["In June 2026, Epoch AI reissued FrontierMath after addressing errors in 42% of its problems. This indicator uses the corrected problem set.", "FrontierMath Tiers 1-3 (v1, old) Update (2026-06-12): We've released v2. This page is no longer updated."],
                note: null,
              },
            ],
          },
          {
            code: "inference/system configuration",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["There is a hard limit of 1,000,000 tokens. If you exceed this, the conversation will end immediately (even if you haven't submitted an answer)."],
                note: null,
              },
            ],
          },
          {
            code: "model version/checkpoint",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: [],
                quotes: ["gpt-5.2-2025-12-11_medium"],
                note: null,
              },
            ],
          },
          {
            code: "run date",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
          },
          {
            code: "task subset/seed/session",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
          },
          {
            code: "evaluator/scorer version",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
          },
          {
            code: "Benchmark version/snapshot",
            by: "NT",
            v: 2, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/data/benchmark_data.zip"],
                quotes: [],
                note: null,
              },
            ],
          },
          {
            code: "model version/checkpoint",
            by: "NT",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/data/benchmark_data.zip"],
                quotes: [],
                note: null,
              },
            ],
          },
          {
            code: "inference/system configuration",
            by: "NT",
            v: 2, s: 2, p: 3, c: 1,
          },
          {
            code: "run date",
            by: "NT",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/data/benchmark_data.zip"],
                quotes: [],
                note: null,
              },
            ],
          },
          {
            code: "task subset/seed/session",
            by: "NT",
            v: 2, s: 2, p: 3, c: 1,
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 1 },
        ],
        comments: [
          { by: "AP", text: "Paper not checked; data download not opened (blocked). Dataset card not named as absent." },
        ],
      },
      {
        leaf: "Run records",
        question:
          "Run audit evidence — What retained evidence could be used to check whether the reported run followed the stated procedure?",
        entries: [
          {
            code: "Public outputs/logs/run record",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about", "https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["lets you see the full details of every LLM interaction, as well as how each answer was scored", "For FrontierMath, we recommend using the log viewer on the public questions as the best way to understand the evaluation settings"],
                note: null,
              },
            ],
          },
          {
            code: "controlled-access run records",
            by: "AP",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1"],
                quotes: ["The record ID is gda5UeWrA8HcbDCRuLJ56H."],
                note: null,
              },
            ],
          },
          {
            code: "evaluator or third-party replay possible",
            by: "AP",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1"],
                quotes: ["In one case the issue was with our grader, so we could re-grade solutions given by models on prior runs."],
                note: null,
              },
            ],
          },
          {
            code: "controlled-access run records",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/benchmarking-hub-update"],
                quotes: ["Internally, our system provides full auditability by linking each evaluation with the exact git revision it was run with."],
                note: null,
              },
            ],
          },
          {
            code: "evaluator or third-party replay possible",
            by: "MZ",
            v: 1, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v1"],
                quotes: ["In one case the issue was with our grader, so we could re-grade solutions given by models on prior runs."],
                note: null,
              },
            ],
          },
          {
            code: "Public outputs/logs/run record",
            by: "NT",
            v: 2, s: 2, p: 3, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/data/benchmark_data.zip"],
                quotes: [],
                note: null,
              },
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For FrontierMath, the log viewer is only available to see model traces on the 10 public questions in frontiermath-2025-02-28-public."],
                note: "I could inspect the basic run record, but the full model responses and grading logs for this private run are not available.",
              },
            ],
          },
        ],
        search: [
          { by: "NT", score: 1 },
        ],
        comments: [
        ],
      },
    ],
  },
  {
    id: "neutrality",
    label: "Neutrality",
    rated: 19,
    means: { v: 1.53, s: 2.0, p: 1.84, c: 1.58 },
    questions: [
      {
        leaf: "Entry rules",
        question:
          "Participation-rule visibility — Who can inspect the rules governing eligibility and access to the benchmark?",
        entries: [
          {
            code: "Public",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about", "https://epoch.ai/frontiermath/tiers-1-4/about", "https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["Our top priority is to track state of the art AI performance. For this reason, we prioritize leading model releases from the main AI labs.", "They have access to all problem statements and solutions up to FrontierMath_12-04-24, except for 53 solutions randomly withheld in version FrontierMath_02-28-25 for holdout evaluation.", "proactively publicly disclose benchmark sponsorship and data access agreements"],
                note: null,
              },
            ],
          },
          {
            code: "Public",
            by: "MZ",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["For Epoch AI-administered benchmarks, we only report results for models where we can run evaluations ourselves. This means that we do not have data for models that are only deployed internally, such as PaLM or Gopher."],
                note: null,
              },
            ],
          },
          {
            code: "Public",
            by: "NT",
            v: 2, s: 2, p: 1, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/about"],
                quotes: ["We only consider the chat/instruct version of each model, rather than the base model (see the model identifier).", "For Epoch AI-administered benchmarks, we only report results for models where we can run evaluations ourselves."],
                note: "Anyone can read this requirement on Epoch’s website. It states one condition for evaluating a model. However, I didn't find a complete policy for FrontierMath v2 explaining who can request an evaluation or how private benchmark access is granted.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 0 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (the OpenAI access terms and the disclosure commitment were published only after the January 2025 controversy)." },
          { by: "AP", text: "No sources-checked list for the codes not selected; only Epoch's About/FAQ pages and one operator post are evident." },
        ],
      },
      {
        leaf: "Unequal treatment",
        question:
          "Differential-treatment disclosure — Does the documentation state whether participants/providers may receive different access or evaluation treatment?",
        entries: [
          {
            code: "differential/privileged treatment disclosed or permitted",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2", "https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath", "https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["FrontierMath was developed with funding from OpenAI, who has exclusive access to a subset of the benchmark.", "who has exclusive access to: all 290 problems in Tiers 1–3; solutions to 237 of these problems", "However, we cannot share the questions and answers with other parties without written permission from OpenAI."],
                note: null,
              },
            ],
          },
          {
            code: "differential/privileged treatment disclosed or permitted",
            by: "MZ",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/what-is-epoch"],
                quotes: ["OpenAI is the only AI company with access to FrontierMath (and the upcoming Tier 4 of this benchmark, an even more difficult set of problems), which has diminished confidence in FrontierMath results for OpenAI models."],
                note: null,
              },
            ],
          },
          {
            code: "differential/privileged treatment disclosed or permitted",
            by: "NT",
            v: 1, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["FrontierMath was developed with funding from OpenAI, who has exclusive access to a subset of the benchmark."],
                note: "This explicitly discloses special benchmark-data access for OpenAI.",
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 0 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (the January 2025 clarification post)." },
          { by: "AP", text: "No sources-checked list for the codes not selected; only the v2 page and two operator posts are evident." },
        ],
      },
      {
        leaf: "Ties to developers",
        question:
          "Evaluator–provider ties — What material relationships between the evaluator/operator and evaluated providers are documented?",
        entries: [
          {
            code: "funding/sponsorship",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["FrontierMath was developed with funding from OpenAI"],
                note: null,
              },
            ],
          },
          {
            code: "customer/vendor relationship",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath", "https://epoch.ai/benchmarks/gpqa-diamond?view=graph&tab=release-date"],
                quotes: ["OpenAI commissioned Epoch AI to produce 300 advanced math problems for AI evaluation that form the core of the FrontierMath benchmark.", "xAI compensated us for this evaluation and provided compute credits."],
                note: null,
              },
            ],
          },
          {
            code: "research/technical partnership",
            by: "AP",
            v: 2, s: 2, p: 2, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath", "https://x.com/EpochAIResearch/status/2065488154086568445"],
                quotes: ["we partnered with OpenAI, who provided both the necessary funding and technical expertise to develop the benchmark.", "This project began in April when OpenAI shared with us that they had found more errors than expected when conducting an internal review."],
                note: null,
              },
            ],
          },
          {
            code: "privileged model/data/API access",
            by: "AP",
            v: 2, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epochai.substack.com/p/gpt-54-set-a-new-record-on-frontiermath", "https://epoch.ai/gradient-updates/less-than-70-percent-of-frontiermath-is-within-reach-for-todays-models"],
                quotes: ["We had pre-release access to evaluate the model.", "Thanks to OpenAI for API credits used for the GPT-5 experiments."],
                note: null,
              },
            ],
          },
          {
            code: "Other material tie",
            by: "AP",
            v: 2, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["As is typical of commissioned work, OpenAI retains ownership of these questions"],
                note: null,
              },
            ],
          },
          {
            code: "funding/sponsorship",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["FrontierMath was developed with funding from OpenAI, who has exclusive access to a subset of the benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "customer/vendor relationship",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["OpenAI commissioned Epoch AI to produce 300 advanced math problems for AI evaluation that form the core of the FrontierMath benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "research/technical partnership",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["After approaching several potential funders, we partnered with OpenAI, who provided both the necessary funding and technical expertise to develop the benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "privileged model/data/API access",
            by: "MZ",
            v: 1, s: 2, p: 2, c: 1,
            rationale: [
              {
                sources: [],
                quotes: ["We had pre-release access to evaluate the model."],
                note: null,
              },
            ],
          },
          {
            code: "funding/sponsorship",
            by: "NT",
            v: 1, s: 2, p: 3, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/benchmarks/frontiermath-tiers-1-3-v2"],
                quotes: ["FrontierMath was developed with funding from OpenAI, who has exclusive access to a subset of the benchmark."],
                note: "The current benchmark page discloses both funding and special data access.",
              },
            ],
          },
          {
            code: "customer/vendor relationship",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["OpenAI commissioned Epoch AI to produce 300 advanced math problems for AI evaluation that form the core of the FrontierMath benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "research/technical partnership",
            by: "NT",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://epoch.ai/latest/openai-and-frontiermath"],
                quotes: ["After approaching several potential funders, we partnered with OpenAI, who provided both the necessary funding and technical expertise to develop the benchmark."],
                note: null,
              },
            ],
          },
          {
            code: "privileged model/data/API access",
            by: "NT",
            v: 1, s: 2, p: 3, c: 2,
          },
        ],
        search: [
          { by: "AP", score: 1 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: post-hoc (the January 2025 clarification)." },
          { by: "AP", text: "Excerpt: the X announcement was read only as a search snippet. Red flag: post-hoc (the January 2025 clarification)." },
          { by: "AP", text: "Excerpt: the Gradient Updates post was read only as a search snippet." },
          { by: "AP", text: "Red flag: post-hoc (the January 2025 clarification)." },
          { by: "AP", text: "Paper and code repository not checked; no web search (required for a 2 on this variable)." },
        ],
      },
    ],
  },
  {
    id: "verification",
    label: "Verification",
    rated: 1,
    means: { v: 1.0, s: 2.0, p: 1.0, c: 2.0 },
    questions: [
      {
        leaf: "Independent check",
        question:
          "External verification status — What evidence shows that a party outside the benchmark operator and evaluated provider actually performed a relevant check?",
        entries: [
          {
            code: "operator reports external verification only",
            by: "AP",
            v: 1, s: 2, p: 1, c: 2,
            rationale: [
              {
                sources: ["https://arxiv.org/html/2411.04872v7#S2.SS3", "https://arxiv.org/html/2411.04872v7#S6", "https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems"],
                quotes: ["We commissioned blind reviews from second reviewers on 30 randomly chosen questions", "We conducted interviews with four prominent mathematicians to gather expert perspectives on FrontierMath’s difficulty, significance, and prospects", "We thank Will McLafferty, Timothy Smits, and Frederick Vu for identifying errors in the code and solution previously posted on the website, which has since been updated."],
                note: null,
              },
            ],
          },
        ],
        search: [
          { by: "AP", score: 0 },
          { by: "NT", score: 2 },
        ],
        comments: [
          { by: "AP", text: "Red flag: self-vouching (Epoch reports the second reviews but publishes no reviewer reports or names)." },
          { by: "AP", text: "No sources-checked list for the codes not selected; Section 5 also needs a web search for a 2." },
          { by: "NT", text: "No separate v2 dataset card located. This records a search outcome, not proof that no outside check ever occurred." },
        ],
      },
    ],
  },
];

/** The Evidence Quality Rubric, verbatim from `annotations/Evidence quality
 *  rubric.pdf`. Kept as the source text rather than a restructured summary so
 *  the page shows exactly what the annotators coded against. The structured
 *  `evidenceRubric` above is only for naming a score inline (V2 ->
 *  "Inspectable"); this is the document. */
export const evidenceRubricText = `
**Evidence Quality Rubric (v1)**

Each selected code gets four scores, one per dimension; codes not selected get one search score.

### How to rate

1. For each selected code, score V, S and P on the source you would quote first, and C across all its sources.
2. Record each score separately. Do not add them up.
3. Give one Search score for the codes not selected.
4. N/A, none documented, and not reported, codes get no rating.

### The four dimensions

| Dimension | Coding question | Allowed scores | Decision rule | Score definitions |
|---|---|---|---|---|
| **V. Verifiability** | Does the evidence show the fact holds, or only state it? | 1 Stated; 2 Inspectable; 3 Confirmed | Rate what an outsider could check today. Plans and promises count as Stated; write "promise" in the comment. | 1 Stated: a claim or promise no outsider can check; 2 Inspectable: an outsider can check it (code, public items, logs, or the page itself for disclosure codes); 3 Confirmed: someone other than the source checked it and documented the result |
| **S. Source independence** | Who made the claim, and do they have a stake in it? | 0 Unattributed; 1 Interested or informal; 2 Operator official; 3 Independent | Rate who wrote this piece of evidence, not who hosts it. This rates the source, not the benchmark's ties (those are coded in Evaluator–provider ties). | 0 Unattributed: author unknown; 1 Interested or informal: the evaluated provider or funder, or operator staff on a forum or social media; 2 Operator official: the operator's site, paper, repository or changelog; 3 Independent: an outside party you checked and found no ties |
| **P. Precision and fit** | Does the evidence cover this code, for this version? | 0 Off-target; 1 Adjacent; 2 On-target; 3 Exact | Evidence that needs an inference scores at most 1. | 0 Off-target: a different benchmark or property; 1 Adjacent: an earlier version, a sibling tier, operator-wide practice, or unclear whether it counts as evidence for the code; 2 On-target: this benchmark and code, without specifics; 3 Exact: this version, with checkable specifics (counts, dates, IDs) |
| **C. Corroboration** | Do other sources agree? | 0 Contradicted; 1 Single; 2 Same party; 3 Cross-party | Count sources, not copies. Silence is not agreement. | 0 Contradicted: an unresolved conflict (tag the code Unclear); 1 Single: one source; 2 Same party: separate sources from the same party agree; 3 Cross-party: a source from a different party agrees |

### Search completeness

| Dimension | Coding question | Allowed scores | Decision rule | Score definitions |
|---|---|---|---|---|
| **Search completeness** | How thoroughly did you search before concluding a code has no support? | 0 Minimal; 1 Partial; 2 Complete; 3 Targeted | Core sources: benchmark page, paper, code repository, docs or FAQ, changelog, operator posts, dataset card. Name any that don't exist. Evaluator–provider ties and Section 5 also need a web search for a 2. | 0 Minimal: one or two core sources; 1 Partial: at least one core source skipped; 2 Complete: every core source checked; 3 Targeted: as 2, plus a keyword search for the missing concept and a web search |

### Worked examples (FrontierMath Tiers 1–3 v2)

| Code | Scores | Deciding reason |
|---|---|---|
| Scoring transparency: Metric/formula public | V2 S2 P3 C2 | The scoring rule and scorer code are public and checkable |
| Protected component: Labels/reference answers | V1 S2 P1 C2 | The withheld solutions can't be inspected, and the count is for a pre-v2 version |
| Privacy rationale: Adaptive gaming/overfitting | V1 S1 P2 C1 | The only source is a co-founder's forum comment, written during the January 2025 controversy |
| Durability controls: codes not selected | Search 2 | Every core source checked; no keyword or web search |
`;
