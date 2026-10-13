import { useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { Goal, Page } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Bar, Card, Empty, errorText, money, pct, shortDate } from "../components/ui";
import Loader from "../components/Loader";

const BLANK = { name: "", target_amount: "", current_amount: "0", target_date: "", color: "#6366f1", description: "" };

export default function Goals() {
  const [draft, setDraft] = useState({ ...BLANK });
  const [contributions, setContributions] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { data, loading, error: loadError, reload } = useAsync(
    () => api.get<Page<Goal>>("/goals", { page_size: 100 }),
    [],
  );
  const { data: summary } = useAsync(
    () => api.get<Record<string, number>>("/goals/summary").catch(() => null),
    [],
  );

  async function create(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await api.post("/goals", {
        name: draft.name,
        target_amount: Number(draft.target_amount),
        current_amount: Number(draft.current_amount || 0),
        target_date: draft.target_date || undefined,
        color: draft.color,
        description: draft.description || undefined,
      });
      setDraft({ ...BLANK });
      setNotice("Goal created.");
      reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  async function contribute(goalId: string) {
    const amount = Number(contributions[goalId]);
    if (!amount || amount <= 0) return;
    setError(null);
    try {
      await api.post(`/goals/${goalId}/contribute`, { amount });
      setContributions({ ...contributions, [goalId]: "" });
      setNotice("Contribution recorded.");
      reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await api.delete(`/goals/${id}`);
      setNotice("Goal deleted.");
      reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Goals</h1>
          <p className="sub">
            {summary ? `${money(summary.total_saved)} saved of ${money(summary.total_target)}` : "Loading…"}
          </p>
        </div>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {notice && <Banner kind="success">{notice}</Banner>}
      {loadError && <Banner kind="error">{errorText(loadError)}</Banner>}

      <Card title="Create a goal" style={{ marginBottom: "1rem" }}>
        <form onSubmit={create} className="row">
          <div className="field">
            <label htmlFor="g-name">Name</label>
            <input id="g-name" required value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="g-target">Target amount</label>
            <input
              id="g-target"
              type="number"
              step="0.01"
              min="0.01"
              required
              value={draft.target_amount}
              onChange={(e) => setDraft({ ...draft, target_amount: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="g-current">Already saved</label>
            <input
              id="g-current"
              type="number"
              step="0.01"
              min="0"
              value={draft.current_amount}
              onChange={(e) => setDraft({ ...draft, current_amount: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="g-date">Target date</label>
            <input id="g-date" type="date" value={draft.target_date} onChange={(e) => setDraft({ ...draft, target_date: e.target.value })} />
          </div>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? "Creating…" : "Create goal"}
          </button>
        </form>
      </Card>

      {loading ? (
        <Loader label="Loading goals…" />
      ) : !data?.items.length ? (
        <Empty>No goals yet — create one above.</Empty>
      ) : (
        <div className="grid grid-2">
          {data.items.map((goal) => (
            <div className="card" key={goal.id}>
              <div className="spread" style={{ marginBottom: "0.6rem" }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: "1rem", color: "var(--text)" }}>{goal.name}</h3>
                  <div className="muted" style={{ fontSize: "0.8rem" }}>
                    {goal.target_date ? `target ${shortDate(goal.target_date)}` : "no target date"}
                    {goal.days_remaining !== null && goal.days_remaining >= 0 ? ` · ${goal.days_remaining} days left` : ""}
                  </div>
                </div>
                <div className="inline">
                  <span className={`pill ${goal.status === "achieved" ? "success" : goal.status === "active" ? "info" : ""}`}>
                    {goal.status}
                  </span>
                  <button className="btn btn-sm btn-danger" onClick={() => remove(goal.id)}>
                    Delete
                  </button>
                </div>
              </div>

              <div className="spread" style={{ marginBottom: "0.3rem" }}>
                <span className="muted">Progress</span>
                <span>
                  <strong>{money(goal.current_amount)}</strong>{" "}
                  <span className="muted">of {money(goal.target_amount)}</span>
                </span>
              </div>
              <Bar value={goal.current_amount} max={goal.target_amount} />

              <div className="inline" style={{ marginTop: "0.6rem", marginBottom: "0.75rem" }}>
                <span className="pill success">{pct(goal.progress_pct)}</span>
                <span className="muted" style={{ fontSize: "0.82rem" }}>
                  {money(goal.remaining_amount)} to go
                </span>
              </div>

              {goal.status === "active" && (
                <div className="row">
                  <div className="field">
                    <label htmlFor={`c-${goal.id}`}>Contribute</label>
                    <input
                      id={`c-${goal.id}`}
                      type="number"
                      step="0.01"
                      min="0.01"
                      placeholder="0.00"
                      value={contributions[goal.id] ?? ""}
                      onChange={(e) => setContributions({ ...contributions, [goal.id]: e.target.value })}
                    />
                  </div>
                  <button className="btn" onClick={() => contribute(goal.id)}>
                    Add
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}
