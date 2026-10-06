"use client";

import type { ReactNode } from "react";
import { motion } from "motion/react";
import { useReducedMotionSafe } from "@/lib/use-reduced-motion";

type MotionRevealProps = {
  children: ReactNode;
  className?: string;
  /** Delay before the first child animates (seconds) */
  delay?: number;
  /** Stagger interval between children (seconds) */
  stagger?: number;
  /** Tag to render as container */
  as?: "div" | "section" | "article" | "ul";
};

const containerVariants = (delay: number, stagger: number) => ({
  hidden: {},
  visible: {
    transition: {
      delayChildren: delay,
      staggerChildren: stagger,
    },
  },
});

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring" as const,
      stiffness: 100,
      damping: 20,
    },
  },
};

export function MotionReveal({
  children,
  className = "",
  delay = 0,
  stagger = 0.08,
  as = "div",
}: MotionRevealProps) {
  const prefersReduced = useReducedMotionSafe();

  if (prefersReduced) {
    const Tag = as;
    return <Tag className={className}>{children}</Tag>;
  }

  const Tag = motion[as];

  return (
    <Tag
      className={className}
      variants={containerVariants(delay, stagger)}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.1 }}
    >
      {children}
    </Tag>
  );
}

/** Wrap each child item with this for stagger animation */
export function MotionRevealItem({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  const prefersReduced = useReducedMotionSafe();

  if (prefersReduced) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div className={className} variants={itemVariants}>
      {children}
    </motion.div>
  );
}
