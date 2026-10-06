import { ActionLink } from "@/components/action-link";
import { databaseTotals, measurementDb } from "@/content/measurement-db";

function DatabaseSnapshot() {
  return (
    <div className="rd-measurement-hero-stats">
      <dl className="rd-measurement-stat-puzzle" aria-label="Data bank totals">
        {databaseTotals.map((stat) => (
          <div key={stat.key} className="rd-measurement-stat-tile">
            <dt>{stat.label}</dt>
            <dd>{stat.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * Landing-only hero for measurement-db.
 *
 * This version keeps the landing page's first screen focused on the project
 * promise and actions; benchmark discovery starts in the searchable catalog
 * immediately below.
 */
export function MeasurementDbHero() {
  return (
    <section
      className="rd-page-hero rd-measurement-hero"
      aria-labelledby="measurement-db-title"
    >
      <div className="rd-container relative">
        <div className="rd-measurement-hero-grid">
          <div className="rd-measurement-hero-content">
            <h1 id="measurement-db-title" className="rd-measurement-hero-title">
              Discover, share, and analyze{" "}
              <span className="whitespace-nowrap">fine-grained</span> AI
              measurement data
            </h1>
            <p className="rd-measurement-hero-copy">{measurementDb.tagline}</p>

            <div className="rd-measurement-hero-actions">
              <div className="rd-measurement-hero-action-buttons">
                <ActionLink href="#schema" variant="primary">
                  View schema
                </ActionLink>
                <ActionLink href="#analysis" variant="secondary">
                  Analyze with torch_measure
                </ActionLink>
              </div>
              <div className="rd-measurement-hero-action-utilities">
                <ActionLink
                  href={measurementDb.contributeHref}
                  variant="ghost"
                  external
                >
                  Contribute data
                </ActionLink>
                <ActionLink
                  href={measurementDb.datasetHref}
                  variant="ghost"
                  external
                >
                  Download data
                </ActionLink>
                <ActionLink href="#citation" variant="ghost">
                  Cite this work
                </ActionLink>
              </div>
            </div>
          </div>

          <DatabaseSnapshot />
        </div>
      </div>
    </section>
  );
}
