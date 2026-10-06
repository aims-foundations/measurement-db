"use client";

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

type ScrollRevealProps = {
  children: ReactNode;
  className?: string;
  stagger?: boolean;
  as?: "div" | "section" | "article" | "li";
};

export function ScrollReveal({
  children,
  className = "",
  stagger = false,
  as: Tag = "div",
}: ScrollRevealProps) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (prefersReducedMotion) {
      el.classList.add("is-visible");
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      // Reveal as soon as the element's edge enters the viewport (threshold 0),
      // with the trigger line pulled up 10% of the viewport via rootMargin. A
      // percentage-of-element threshold (e.g. 0.1) silently fails for elements
      // taller than ~10x the viewport — the visible ratio can never reach it,
      // so a long section (e.g. the full benchmark gallery) would stay at
      // opacity:0 (present and clickable, but invisible).
      { threshold: 0, rootMargin: "0px 0px -10% 0px" },
    );

    if (stagger) {
      const children = el.querySelectorAll(".reveal");
      for (const child of children) {
        observer.observe(child);
      }
    } else {
      observer.observe(el);
    }

    return () => observer.disconnect();
  }, [stagger]);

  const classes = stagger
    ? `reveal-stagger ${className}`.trim()
    : `reveal ${className}`.trim();

  return (
    // @ts-expect-error -- Tag is a valid HTML element
    <Tag ref={ref} className={classes}>
      {children}
    </Tag>
  );
}
