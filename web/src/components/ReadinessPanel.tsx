import type { ReadinessReport, ReadinessStatus } from "../api";

const TONE: Record<ReadinessStatus, string> = {
  pass: "text-teal border-teal",
  review: "text-orange-text border-orange",
  block: "text-orange border-orange",
};

const LABEL: Record<ReadinessStatus, string> = {
  pass: "ready",
  review: "review",
  block: "blocked",
};

export function ReadinessBadge({
  readiness,
  title,
}: {
  readiness: ReadinessReport;
  title?: string;
}) {
  const issues = readiness.checks.filter((check) => check.status !== "pass");
  return (
    <span
      className={`badge ${TONE[readiness.status]}`}
      title={title ?? issues.map((check) => check.message).join("\n")}
    >
      {LABEL[readiness.status]}
    </span>
  );
}

export function ReadinessPanel({
  readiness,
  compact = false,
  className = "",
}: {
  readiness: ReadinessReport;
  compact?: boolean;
  className?: string;
}) {
  const issues = readiness.checks.filter((check) => check.status !== "pass");
  const heading = readiness.status === "pass"
    ? "Ready"
    : readiness.status === "review"
      ? "Review recommended"
      : "Not ready";

  return (
    <section
      className={`border bg-paper ${TONE[readiness.status]} ${compact ? "p-3" : "p-5"} ${className}`}
      style={{ borderRadius: 2 }}
      aria-label="Print readiness"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="grp-label" style={{ color: "currentColor" }}>
          Print readiness
        </div>
        <ReadinessBadge readiness={readiness} />
      </div>
      {!compact && (
        <p className="font-mono text-xs mt-2" style={{ color: "currentColor" }}>
          {heading}
        </p>
      )}
      {issues.length ? (
        <ul className={`font-mono text-[11px] mt-2 space-y-1 ${compact ? "max-w-64" : ""}`}>
          {issues.map((check) => (
            <li key={`${check.code}:${check.message}`}>
              <span className="uppercase">{check.source}</span>
              {" · "}
              {check.message}
              {check.confidence != null && ` (${Math.round(check.confidence * 100)}%)`}
            </li>
          ))}
        </ul>
      ) : (
        !compact && (
          <p className="font-mono text-[11px] text-muted mt-2">
            All available checks passed.
          </p>
        )
      )}
    </section>
  );
}
