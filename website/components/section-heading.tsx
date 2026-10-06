type SectionHeadingProps = {
  eyebrow?: string;
  title?: string;
  description?: string;
  // Preserved for API compatibility. The redesign uses a single restrained
  // accent, so per-call colors are no longer mapped.
  accent?: "cardinal" | "palo-alto" | "lagunita" | "poppy" | "plum" | "default";
  center?: boolean;
};

export function SectionHeading({
  eyebrow,
  title,
  description,
  center = false,
}: SectionHeadingProps) {
  return (
    <div className={center ? "mx-auto text-center" : ""}>
      {eyebrow ? <p className="rd-mono opacity-60">{eyebrow}</p> : null}
      {title ? (
        <h2 className={`rd-h2 ${eyebrow ? "mt-4" : ""}`}>{title}</h2>
      ) : null}
      {description ? (
        <p
          className={`rd-prose mt-4 text-lg font-light leading-relaxed opacity-75 ${
            center ? "mx-auto" : ""
          }`}
        >
          {description}
        </p>
      ) : null}
    </div>
  );
}
