import { useState } from "react";
import { api } from "../api/client";
import type { Alert, Page } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Card, Empty, errorText, relativeTime } from "../components/ui";

const TYPE_LABELS: Record<string, string> = {
  budget_threshold: "Budget threshold",
  budget_exceeded: "Budget exceeded",
  savings_deadline: "Savings deadline",
  goal_achieved: "Goal achieved",
  goal_milestone: "Goal milestone",
  unusual_spending: "Unusual spending",
  overspend_risk: "Overspend risk",
  forecast_warning: "Forecast warning",
};

export default function Alerts() {
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data, loading, error: loadError, reload } = useAsync(
    () => api.get<Page<Alert>>("/alerts", { page_size: 100, unread_only: unreadOnly || undefined }),
    [unreadOnly],
  );
  const { data: counts, reload: reloadCounts } = useAsync(
    () => api.get<Record<string, number>>("/alerts/counts"),
    [],
  );

  async function markAll() {
    setError(null);
    try {
      await api.patch("/alerts/read", { all: true });
      reload();
      reloadCounts();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function markOne(id: string) {
    setError(null);
    try {
      await api.patch("/alerts/read", { alert_ids: [id] });
      reload();
      reloadCounts();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await api.delete(`/alerts/${id}`);
      reload();
      reloadCounts();
    } catch (err) {
      setError(errorText(err));
    }
  }

  const unread = counts?.unread ?? 0;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Alerts</h1>
          <p className="sub">{unread > 0 ? `${unread} unread` : "All caught up"}</p>
        </div>
        <div className="inline">
          <button className={`btn ${unreadOnly ? "btn-primary" : ""}`} onClick={() => setUnreadOnly((v) => !v)}>
            {unreadOnly ? "Showing unread" : "Show unread only"}
          </button>
          <button className="btn" onClick={markAll} disabled={unread === 0}>
            Mark all read
          </button>
        </div>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {loadError && <Banner kind="error">{errorText(loadError)}</Banner>}

      {loading ? (
        <div className="loading">Loading…</div>
      ) : !data?.items.length ? (
        <Empty>{unreadOnly ? "No unread alerts." : "No alerts."}</Empty>
      ) : (
        <div className="stack">
          {data.items.map((alert) => (
            <Card key={alert.id}>
              <div className="spread" style={{ alignItems: "flex-start", gap: "1rem" }}>
                <div style={{ minWidth: 0 }}>
                  <div className="inline" style={{ marginBottom: "0.25rem" }}>
                    <span className={`pill ${alert.severity}`}>{alert.severity}</span>
                    <span className="pill">{TYPE_LABELS[alert.type] ?? alert.type}</span>
                    {!alert.is_read && <span className="pill info">new</span>}
                    <span className="muted" style={{ fontSize: "0.78rem" }}>
                      {relativeTime(alert.triggered_at)}
                    </span>
                  </div>
                  <div style={{ fontWeight: 600, marginBottom: "0.15rem" }}>{alert.title}</div>
                  <div className="muted" style={{ fontSize: "0.88rem" }}>
                    {alert.message}
                  </div>
                </div>
                <div className="inline">
                  {!alert.is_read && (
                    <button className="btn btn-sm" onClick={() => markOne(alert.id)}>
                      Mark read
                    </button>
                  )}
                  <button className="btn btn-sm btn-danger" onClick={() => remove(alert.id)}>
                    Dismiss
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
