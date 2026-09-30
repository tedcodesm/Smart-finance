import { useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { Budget, Category, Page } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Bar, Card, Empty, errorText, money, pct, shortDate } from "../components/ui";

const BLANK = { name: "", limit_amount: "", period: "monthly", category_id: "" };

export default function Budgets() {
  const [draft, setDraft] = useState({ ...BLANK });
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { data, loading, error: loadError, reload } = useAsync(
    () => api.get<Page<Budget>>("/budgets", { page_size: 100 }),
    [],
  );
  const { data: categories } = useAsync(() => api.get<Page<Category>>("/categories", { page_size: 200 }), []);

  async function create(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await api.post("/budgets", {
        name: draft.name || undefined,
        limit_amount: Number(draft.limit_amount),
        period: draft.period,
        category_id: draft.category_id || undefined,
      });
      setDraft({ ...BLANK });
      setNotice("Budget created.");
      reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await api.delete(`/budgets/${id}`);
      setNotice("Budget deleted.");
      reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Budgets</h1>
          <p className="sub">Track spend against a limit per category and period.</p>
        </div>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {notice && <Banner kind="success">{notice}</Banner>}
      {loadError && <Banner kind="error">{errorText(loadError)}</Banner>}

      <Card title="Create a budget" style={{ marginBottom: "1rem" }}>
        <form onSubmit={create} className="row">
          <div className="field">
            <label htmlFor="b-name">Name</label>
            <input id="b-name" placeholder="optional" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="b-limit">Limit amount</label>
            <input
              id="b-limit"
              type="number"
              step="0.01"
              min="0.01"
              required
              value={draft.limit_amount}
              onChange={(e) => setDraft({ ...draft, limit_amount: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="b-period">Period</label>
            <select id="b-period" value={draft.period} onChange={(e) => setDraft({ ...draft, period: e.target.value })}>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="yearly">Yearly</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="b-cat">Category</label>
            <select id="b-cat" value={draft.category_id} onChange={(e) => setDraft({ ...draft, category_id: e.target.value })}>
              <option value="">Overall (no category)</option>
              {(categories?.items ?? [])
                .filter((c) => c.type === "expense")
                .map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
            </select>
          </div>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? "Creating…" : "Create budget"}
          </button>
        </form>
      </Card>

      {loading ? (
        <div className="loading">Loading…</div>
      ) : !data?.items.length ? (
        <Empty>No budgets yet — create one above.</Empty>
      ) : (
        <div className="grid grid-2">
          {data.items.map((budget) => {
            const ratio = budget.limit_amount > 0 ? (budget.spent_amount / budget.limit_amount) * 100 : 0;
            return (
              <div className="card" key={budget.id}>
                <div className="spread" style={{ marginBottom: "0.6rem" }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: "1rem", color: "var(--text)" }}>
                      {budget.name ?? budget.category_name ?? "Overall budget"}
                    </h3>
                    <div className="muted" style={{ fontSize: "0.8rem" }}>
                      {budget.period} · {budget.category_name ?? "all categories"}
                      {budget.start_date ? ` · from ${shortDate(budget.start_date)}` : ""}
                    </div>
                  </div>
                  <button className="btn btn-sm btn-danger" onClick={() => remove(budget.id)}>
                    Delete
                  </button>
                </div>

                <div className="spread" style={{ marginBottom: "0.3rem" }}>
                  <span className="muted">Spent</span>
                  <span>
                    <strong>{money(budget.spent_amount)}</strong>{" "}
                    <span className="muted">of {money(budget.limit_amount)}</span>
                  </span>
                </div>
                <Bar value={budget.spent_amount} max={budget.limit_amount} />

                <div className="inline" style={{ marginTop: "0.6rem" }}>
                  <span className={`pill ${ratio >= 100 ? "critical" : ratio >= 80 ? "warning" : "success"}`}>
                    {pct(ratio)} used
                  </span>
                  <span className="muted" style={{ fontSize: "0.82rem" }}>
                    {budget.remaining_amount > 0
                      ? `${money(budget.remaining_amount)} left`
                      : `${money(Math.abs(budget.remaining_amount))} over`}
                  </span>
                  {!budget.is_active && <span className="pill">inactive</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
