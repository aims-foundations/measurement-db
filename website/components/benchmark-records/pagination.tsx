export function Pagination({
  label,
  page,
  total,
  size,
  onChange,
}: {
  label: string;
  page: number;
  total: number;
  size: number;
  onChange: (page: number) => void;
}) {
  return (
    <nav
      aria-label={`${label} pages`}
      className="flex flex-wrap items-center gap-3 text-xs"
    >
      <button
        className="rounded border border-[var(--line)] px-3 py-2 text-xs hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40"
        disabled={page === 0}
        onClick={() => onChange(page - 1)}
      >
        Previous {label}
      </button>
      <span className="tabular-nums text-[var(--muted)]">
        {label}: {total ? page * size + 1 : 0}–
        {Math.min((page + 1) * size, total)} of {total.toLocaleString()}
      </span>
      <button
        className="rounded border border-[var(--line)] px-3 py-2 text-xs hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40"
        disabled={(page + 1) * size >= total}
        onClick={() => onChange(page + 1)}
      >
        Next {label}
      </button>
    </nav>
  );
}
