import { ActionLink } from "@/components/action-link";

export default function NotFound() {
  return (
    <main id="main-content" className="rd-page-hero rd-band">
      <div className="rd-container max-w-4xl">
        <div className="rd-card text-center">
          <p className="rd-mono opacity-60">404</p>
          <h1 className="rd-h2 mt-4">That benchmark does not exist.</h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg font-light leading-relaxed opacity-75">
            Return to the Measurement Data Bank and continue browsing the
            current catalog.
          </p>
          <div className="mt-8">
            <ActionLink href="/">Return to the catalog</ActionLink>
          </div>
        </div>
      </div>
    </main>
  );
}
