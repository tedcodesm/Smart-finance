/** Small presentational helpers shared across pages. */

import type { ReactNode } from "react";

export function money(value: number | null | undefined, currency = "KES"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const activeCurrency = !currency || currency === "USD" ? "KES" : currency;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: activeCurrency,
    maximumFractionDigits: Math.abs(value) >= 1000 ? 0 : 2,
  }).format(value);
}

export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(1)}%`;
}

export function shortDate(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function relativeTime(value: string | null | undefined): string {
  if (!value) return "—";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "—";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return shortDate(value);
}

export function Card({
  title,
  children,
  action,
  style,
}: {
  title?: string;
  children: ReactNode;
  action?: ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div className="card" style={style}>
      {(title || action) && (
        <div className="spread" style={{ marginBottom: "0.75rem" }}>
          {title && <h3 style={{ margin: 0 }}>{title}</h3>}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "pos" | "neg";
}) {
  return (
    <div className="card">
      <h3>{label}</h3>
      <div className={`stat-value ${tone ?? ""}`}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Banner({ kind, children }: { kind: "error" | "info" | "success"; children: ReactNode }) {
  return <div className={`alert-banner ${kind}`}>{children}</div>;
}

export function Bar({ value, max }: { value: number; max: number }) {
  const ratio = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  const tone = ratio >= 100 ? "over" : ratio >= 80 ? "warn" : "";
  return (
    <div className="bar" role="progressbar" aria-valuenow={Math.round(ratio)} aria-valuemin={0} aria-valuemax={100}>
      <span className={tone} style={{ width: `${ratio}%` }} />
    </div>
  );
}

/** Renders an ApiError (or anything else) as readable text. */
export function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
