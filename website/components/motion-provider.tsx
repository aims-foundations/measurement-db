"use client";

import type { ReactNode } from "react";
import { MotionConfig } from "motion/react";

/**
 * Disables motion's built-in reduced-motion detection so the SSR and client
 * initial renders always produce the same markup. Components use the
 * hydration-safe `useReducedMotionSafe` hook to branch after mount instead.
 */
export function MotionProvider({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="never">{children}</MotionConfig>;
}
