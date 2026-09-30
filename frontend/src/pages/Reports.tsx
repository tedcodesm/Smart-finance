import { useState } from "react";
import { api } from "../api/client";
import type { Page, Report } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Card, Empty, errorText, money, relativeTime } from "../components/ui";

const REPORT_TYPES = [
  { value: "spending_by_category", label: "Spending by category" },
  { value: "income_vs_expense", label: "Income vs expense" },
  { value: "budget_performance", label: "Budget performance" },
  { value: "net_cash_flow", label: "Net cash flow" },
  { value: "goal_progress", label: "Goal progress" },
];

const PERIODS = ["daily", "weekly", "monthly", "yearly", "custom"];

/** Best-effort pretty-printer for the free-form report payloads. */
function ReportBody({ data }: { data: Record<string, unknown> | undefined }) {
  if (!data) return <Empty>This report has no stored payload.</Empty>;
  const rows = Object.entries(data).filter(([, v]) => v !== null && v !== undefined);
  if (!rows.length) return <Empty>This report has no stored payload.</Empty>;

  return (
    <div className="stack">
      {rows.map(([key, value]) => {
        if (Array.isArray(value)) {
          if (!value.length) return null;
          const first = value[0] as Record<string, unknown>;
          if (typeof first === "object" && first !== null) {
            const columns = Object.keys(first).slice(0, 6);
            return (
              <div key={key}>
                <h3 style={{ marginBottom: "0.4rem" }}>{key.replace(/_/g, " ")}</h3>
                <div style={{ maxHeight: 260, overflow: "auto" }}>
                  <table>
                    <thead>
                      <tr>
                        {columns.map((col) => (
                          <th key={col}>{col.replace(/_/g, " ")}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(value as Record<string, unknown>[]).slice(0, 25).map((row, i) => (
                        <tr key={i}>
                          {columns.map((col) => (
                            <td key={col}>
                              {typeof row[col] === "number" ? money(row[col] as number) : String(row[col] ?? "—")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          }
          return (
            <div key={key}>
              <h3>{key.replace(/_/g, " ")}</h3>
              <div className="inline">
                {(value as unknown[]).slice(0, 20).map((v, i) => (
                  <span className="pill" key={i}>
                    {typeof v === "number" ? money(v) : String(v)}
                  </span>
                ))}
              </div>
            </div>
          );
        }
        if (typeof value === "object") {
          return (
            <div key={key}>
              <h3>{key.replace(/_/g, " ")}</h3>
              <div className="inline">
                {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
                  <span className="pill" key={k}>
                    {k.replace(/_/g, " ")}: {typeof v === "number" ? money(v) : String(v)}
                  </span>
                ))}
              </div>
            </div>
          );
        }
        return (
          <div className="spread" key={key}>
            <span className="muted">{key.replace(/_/g, " ")}</span>
            <strong>{typeof value === "number" ? money(value) : String(value)}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default function Reports() {
  const [draft, setDraft] = useState({ report_type: "spending_by_category", period: "monthly", period_key: "" });
  const [selected, setSelected] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { data, loading, error: loadError, reload } = useAsync(
    () => api.get<Page<Report>>("/reports", { page_size: 50 }),
    [],
  );

  function defaultPeriodKey(): string {
    const now = new Date();
    if (draft.period === "monthly") return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
    if (draft.period === "yearly") return String(now.getFullYear());
    if (draft.period === "daily") return now.toISOString().slice(0, 10);
    return "";
  }

  async function generate(persist: boolean) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const payload = {
        report_type: draft.report_type,
        period: draft.period,
        period_key: draft.period_key || defaultPeriodKey() || undefined,
        persist,
      };
      const result = await api.post<Report>(persist ? "/reports" : "/reports/generate", payload);
      setNotice(persist ? "Report saved." : "Report generated (not saved).");
      setSelected(result);
      if (persist) reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await api.delete(`/reports/${id}`);
      if (selected?.id === id) setSelected(null);
      setNotice("Report deleted.");
      reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Reports</h1>
          <p className="sub">Generate a report, then export it as PDF or JSON.</p>
        </div>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {notice && <Banner kind="success">{notice}</Banner>}
      {loadError && <Banner kind="error">{errorText(loadError)}</Banner>}

      <Card title="Generate a report" style={{ marginBottom: "1rem" }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void generate(true);
          }}
          className="row"
        >
          <div className="field">
            <label htmlFor="r-type">Report</label>
            <select
              id="r-type"
              value={draft.report_type}
              onChange={(e) => setDraft({ ...draft, report_type: e.target.value })}
            >
              {REPORT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="r-period">Period</label>
            <select id="r-period" value={draft.period} onChange={(e) => setDraft({ ...draft, period: e.target.value })}>
              {PERIODS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="r-key">Period key (optional)</label>
            <input
              id="r-key"
              placeholder={defaultPeriodKey() || "e.g. 2026-09"}
              value={draft.period_key}
              onChange={(e) => setDraft({ ...draft, period_key: e.target.value })}
            />
          </div>
          <button className="btn" type="button" disabled={busy} onClick={() => void generate(false)}>
            Preview
          </button>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? "Working…" : "Generate & save"}
          </button>
        </form>
      </Card>

      {selected && (
        <Card
          title={`${REPORT_TYPES.find((t) => t.value === selected.report_type)?.label ?? selected.report_type} · ${selected.period_key ?? selected.period}`}
          action={
            <div className="inline">
              <button
                className="btn btn-sm"
                onClick={() => api.download(`/reports/${selected.id}/export`, { format: "pdf" }, "report.pdf")}
              >
                PDF
              </button>
              <button
                className="btn btn-sm"
                onClick={() => api.download(`/reports/${selected.id}/export`, { format: "json" }, "report.json")}
              >
                JSON
              </button>
            </div>
          }
          style={{ marginBottom: "1rem" }}
        >
          <ReportBody data={selected.data} />
        </Card>
      )}

      <Card title="Saved reports">
        {loading ? (
          <div className="loading">Loading…</div>
        ) : !data?.items.length ? (
          <Empty>No saved reports yet.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th>Period</th>
                <th>Generated</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {data.items.map((report) => (
                <tr key={report.id}>
                  <td>{REPORT_TYPES.find((t) => t.value === report.report_type)?.label ?? report.report_type}</td>
                  <td className="mono muted">{report.period_key ?? report.period}</td>
                  <td className="muted nowrap">{relativeTime(report.generated_at)}</td>
                  <td className="right">
                    <div className="inline" style={{ justifyContent: "flex-end" }}>
                      <button
                        className="btn btn-sm"
                        onClick={async () => {
                          try {
                            setSelected(await api.get<Report>(`/reports/${report.id}`));
                          } catch (err) {
                            setError(errorText(err));
                          }
                        }}
                      >
                        View
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => api.download(`/reports/${report.id}/export`, { format: "pdf" }, "report.pdf")}
                      >
                        PDF
                      </button>
                      <button className="btn btn-sm btn-danger" onClick={() => remove(report.id)}>
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}
