import type { Route } from "next";
import type { ReactNode } from "react";
import { ArrowRight } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";

type ActionLinkProps = {
  href: string;
  children: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  external?: boolean;
  // A `proxied` target is served under our own domain by a Next rewrite (see
  // next.config.ts) but is not a typed app route, so it can't use next/link.
  // Render it as a plain same-tab anchor (full navigation), like the nav does.
  proxied?: boolean;
  disabled?: boolean;
  className?: string;
};

const baseClasses = "touch-manipulation";

const variantClasses = {
  primary: "rd-btn",
  secondary: "rd-btn rd-btn-ghost",
  ghost: "rd-arrow-link text-sm",
} as const;

function ArrowIcon() {
  return (
    <ArrowRight
      size={14}
      weight="regular"
      className="transition-transform duration-200 group-hover:translate-x-0.5"
      aria-hidden="true"
    />
  );
}

export function ActionLink({
  href,
  children,
  variant = "primary",
  external,
  proxied = false,
  disabled = false,
  className = "",
}: ActionLinkProps) {
  const isExternal = external ?? href.startsWith("http");
  const classes =
    `group ${baseClasses} ${variantClasses[variant]} ${className}`.trim();

  if (disabled) {
    return (
      <span
        aria-disabled="true"
        className={`${classes} cursor-not-allowed opacity-55`}
      >
        {children}
      </span>
    );
  }

  if (proxied) {
    return (
      <a className={classes} href={href}>
        {children}
        {variant === "ghost" && <ArrowIcon />}
      </a>
    );
  }

  if (isExternal) {
    return (
      <a className={classes} href={href} rel="noopener noreferrer" target="_blank">
        {children}
        {variant === "ghost" && <ArrowIcon />}
      </a>
    );
  }

  return (
    <Link href={href as Route} className={classes}>
      {children}
      {variant === "ghost" && <ArrowIcon />}
    </Link>
  );
}
