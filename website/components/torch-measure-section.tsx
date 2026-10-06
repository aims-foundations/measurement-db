import { ActionLink } from "@/components/action-link";
import { ScrollReveal } from "@/components/scroll-reveal";
import { torchMeasure } from "@/content/measurement-db";

const quickstart = `import torch
from torch_measure.models import Rasch

# Binary AI systems × items; NaN marks a missing response.
responses = torch.tensor(
    [
        [1, 0, 1, float("nan")],
        [0, 1, 1, 0],
        [1, 1, float("nan"), 1],
    ],
    dtype=torch.float32,
)

model = Rasch(
    n_subjects=responses.shape[0],
    n_items=responses.shape[1],
)
model.fit(responses, max_epochs=500, verbose=False)

print(model.ability.detach())
print(model.difficulty.detach())`;

const capabilities = [
  {
    label: "Fit",
    description:
      "Fit latent-variable models, including IRT and factor models, to compatible response tensors.",
  },
  {
    label: "Diagnose",
    description:
      "Study reliability, item fit, calibration, scalability, differential item functioning, and uncertainty.",
  },
  {
    label: "Design",
    description:
      "Build adaptive tests with Fisher-information, spanning, or random item-selection strategies.",
  },
] as const;

export function TorchMeasureSection() {
  return (
    <section
      id="analysis"
      className="rd-beige-band rd-band scroll-mt-16"
      aria-labelledby="torch-measure-title"
    >
      <div className="rd-container">
        <ScrollReveal>
          <div className="grid gap-12 lg:grid-cols-[minmax(0,0.88fr)_minmax(30rem,1.12fr)] lg:items-start lg:gap-20">
            <div>
              <p className="rd-mono opacity-60">Analysis package</p>
              <h2 id="torch-measure-title" className="rd-h2 mt-4">
                Analyze the bank with torch_measure
              </h2>
              <p className="rd-prose mt-6 text-lg font-light leading-relaxed opacity-75">
                {torchMeasure.tagline} The Data Bank supplies curated item-level
                evidence; torch_measure works downstream on compatible
                subject-by-item data for modeling and diagnostics.
              </p>

              <dl className="mt-9 border-b border-black/20">
                {capabilities.map((capability) => (
                  <div
                    key={capability.label}
                    className="grid gap-2 border-t border-black/20 py-4 sm:grid-cols-[5rem_1fr] sm:gap-5"
                  >
                    <dt className="rd-mono text-[var(--rd-cardinal)]">
                      {capability.label}
                    </dt>
                    <dd className="text-sm font-light leading-relaxed opacity-70">
                      {capability.description}
                    </dd>
                  </div>
                ))}
              </dl>

              <div className="mt-8 flex flex-wrap items-center gap-3">
                <ActionLink href={torchMeasure.docsHref} external>
                  Read the docs
                </ActionLink>
                <ActionLink
                  href={torchMeasure.repoHref}
                  variant="secondary"
                  external
                >
                  GitHub
                </ActionLink>
                <ActionLink
                  href={torchMeasure.packageHref}
                  variant="ghost"
                  external
                >
                  PyPI
                </ActionLink>
              </div>
            </div>

            <div className="min-w-0 border border-black/20 bg-[var(--rd-dark)] text-white">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/20 px-5 py-4">
                <span className="rd-mono opacity-60">Analysis quickstart</span>
                <span className="font-mono text-xs opacity-70">
                  Python 3.10+
                </span>
              </div>
              <div className="border-b border-white/15 bg-white/[0.055] px-5 py-4 font-mono text-sm">
                <span className="select-none opacity-45" aria-hidden="true">
                  ${" "}
                </span>
                <a
                  href={torchMeasure.packageHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline decoration-white/30 underline-offset-4 hover:decoration-white"
                >
                  {torchMeasure.installCommand}
                </a>
              </div>
              <pre className="overflow-x-auto p-5 text-[0.78rem] leading-6 sm:p-7 sm:text-sm">
                <code>{quickstart}</code>
              </pre>
              <p className="border-t border-white/15 px-5 py-4 text-xs font-light leading-relaxed opacity-65 sm:px-7">
                This Rasch example is for binary responses. When preparing a
                compatible tensor from the Data Bank, retain the selected
                benchmark, trial, and test-condition context alongside it.
              </p>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
