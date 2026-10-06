"use client";

import {
  type FormEvent,
  type MouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import { ArrowRight, MagnifyingGlass, X } from "@phosphor-icons/react";
import { withBase } from "@/lib/base-path";
import { mainSiteHref } from "@/lib/main-site-url";
import type { SiteSearchResponse } from "@/lib/site-search-types";

type SiteSearchDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: () => void;
};

function resultCountLabel(total: number, shown: number) {
  if (total === 0) return "No results";
  if (total === 1) return "1 result";
  if (total === shown) return `${total} results`;
  return `${shown} of ${total} results`;
}

function isPlainClick(event: MouseEvent<HTMLAnchorElement>) {
  return (
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey
  );
}

export function SiteSearchDialog({
  open,
  onOpenChange,
  onNavigate,
}: SiteSearchDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const requestSequence = useRef(0);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SiteSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const trimmedQuery = query.trim();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open && !dialog.open) {
      returnFocusRef.current =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      dialog.showModal();
      window.requestAnimationFrame(() => inputRef.current?.focus());
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    const sequence = ++requestSequence.current;

    if (!open || trimmedQuery.length < 2) {
      return;
    }

    const controller = new AbortController();

    const timeout = window.setTimeout(async () => {
      try {
        const result = await fetch(
          `${withBase("/api/site-search")}?q=${encodeURIComponent(trimmedQuery)}`,
          {
            headers: { Accept: "application/json" },
            signal: controller.signal,
          },
        );

        if (!result.ok) throw new Error("Search request failed");
        const nextResponse = (await result.json()) as SiteSearchResponse;

        if (sequence === requestSequence.current) {
          setResponse(nextResponse);
        }
      } catch (requestError) {
        if (
          !(requestError instanceof DOMException) ||
          requestError.name !== "AbortError"
        ) {
          if (sequence === requestSequence.current) setError(true);
        }
      } finally {
        if (sequence === requestSequence.current) setLoading(false);
      }
    }, 200);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [open, trimmedQuery]);

  const closeDialog = () => onOpenChange(false);

  const handleQueryChange = (nextQuery: string) => {
    requestSequence.current += 1;
    setQuery(nextQuery);
    setResponse(null);
    setLoading(nextQuery.trim().length >= 2);
    setError(false);
  };

  const handleDialogClose = () => {
    onOpenChange(false);
    setQuery("");
    setResponse(null);
    setLoading(false);
    setError(false);
    window.requestAnimationFrame(() => {
      const returnTarget = returnFocusRef.current;
      if (returnTarget?.isConnected) returnTarget.focus();
    });
  };

  const handleResultClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (isPlainClick(event)) {
      onNavigate();
      closeDialog();
    }
  };

  const handleSearchSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const firstResult = response?.results[0];
    if (!firstResult || loading || error) return;

    onNavigate();
    closeDialog();
    window.location.assign(mainSiteHref(firstResult.href));
  };

  const statusMessage = loading
    ? `Searching for ${trimmedQuery}`
    : error
      ? "Search is temporarily unavailable"
      : response
        ? `${resultCountLabel(response.total, response.results.length)} for ${trimmedQuery}`
        : trimmedQuery.length === 1
          ? "Enter at least two characters to search"
          : "";
  const showResultPanel =
    trimmedQuery.length >= 2 && (error || response !== null);
  const showFooter = trimmedQuery.length >= 2 && (error || response !== null);

  return (
    <dialog
      ref={dialogRef}
      id="rd-site-search-dialog"
      className={`rd-search-dialog${showResultPanel ? " rd-search-dialog--expanded" : ""}`}
      aria-labelledby="rd-site-search-title"
      aria-describedby="rd-site-search-help"
      onClose={handleDialogClose}
      onCancel={(event) => {
        event.preventDefault();
        closeDialog();
      }}
      onKeyDownCapture={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          closeDialog();
        }
      }}
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) closeDialog();
      }}
    >
      <div className="rd-search-dialog-shell">
        <h2 id="rd-site-search-title" className="sr-only">
          Search AIMS
        </h2>
        <p id="rd-site-search-help" className="sr-only">
          Find pages, publications, events, course projects, and software.
          Results update as you type; press Enter to open the first result.
        </p>

        <form
          action={mainSiteHref("/search")}
          method="get"
          role="search"
          className="rd-search-dialog-form"
          onSubmit={handleSearchSubmit}
        >
          <label htmlFor="rd-dialog-search-input" className="sr-only">
            Search AIMS
          </label>
          <div className="rd-search-dialog-input-wrap">
            <MagnifyingGlass
              className="rd-search-dialog-icon"
              size={30}
              weight="regular"
              aria-hidden="true"
            />
            <input
              ref={inputRef}
              id="rd-dialog-search-input"
              name="q"
              type="search"
              minLength={2}
              maxLength={120}
              autoComplete="off"
              enterKeyHint="go"
              value={query}
              placeholder="Search AIMS"
              onChange={(event) => handleQueryChange(event.target.value)}
            />
            {loading ? (
              <span className="rd-search-dialog-loader" aria-hidden="true" />
            ) : null}
            <button
              type="button"
              className="rd-search-dialog-close"
              aria-label="Close search"
              onClick={closeDialog}
            >
              <span className="rd-search-dialog-escape" aria-hidden="true">
                esc
              </span>
              <X
                className="rd-search-dialog-mobile-close"
                size={20}
                weight="regular"
                aria-hidden="true"
              />
            </button>
          </div>
        </form>

        <p className="sr-only" role="status" aria-live="polite" aria-atomic>
          {statusMessage}
        </p>

        {showResultPanel ? (
          <div
            className="rd-search-dialog-body"
            aria-busy={loading ? "true" : undefined}
          >
            {error ? (
              <div className="rd-search-dialog-message" role="alert">
                <h3>Search is temporarily unavailable.</h3>
                <p>Continue on the full search page.</p>
              </div>
            ) : null}

            {!error && response?.results.length === 0 ? (
              <div className="rd-search-dialog-message">
                <h3>No results for “{trimmedQuery}”</h3>
                <p>Try fewer words or a broader topic.</p>
              </div>
            ) : null}

            {!error && response?.results.length ? (
              <ol className="rd-search-dialog-results">
                {response.results.map((result) => (
                  <li key={result.id}>
                    <a
                      href={mainSiteHref(result.href)}
                      onClick={handleResultClick}
                    >
                      <div className="rd-search-dialog-result-copy">
                        <h3>{result.title}</h3>
                        <p>{result.summary}</p>
                        <div className="rd-search-dialog-result-meta">
                          <span>{result.kind}</span>
                          <span aria-hidden="true">·</span>
                          <span>{result.section}</span>
                        </div>
                      </div>
                      <ArrowRight
                        size={18}
                        weight="regular"
                        aria-hidden="true"
                      />
                    </a>
                  </li>
                ))}
              </ol>
            ) : null}
          </div>
        ) : null}

        {showFooter ? (
          <div className="rd-search-dialog-footer">
            <a
              href={`${mainSiteHref("/search")}?q=${encodeURIComponent(trimmedQuery)}`}
              onClick={handleResultClick}
            >
              {!loading && response && response.total > 0
                ? `View all ${response.total} result${response.total === 1 ? "" : "s"}`
                : "Search all results"}
              <ArrowRight size={18} weight="bold" aria-hidden="true" />
            </a>
          </div>
        ) : null}
      </div>
    </dialog>
  );
}
