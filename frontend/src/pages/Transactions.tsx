import { useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { Category, Page, Transaction } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Card, Empty, errorText, money, shortDate } from "../components/ui";

const BLANK = { amount: "", type: "expense", category_id: "", date: "", description: "", notes: "" };

export default function Transactions() {
  const [page, setPage] = useState(1);
  const [type, setType] = useState("");
  const [categoryId, setCategoryId] = useState("");
  const [search, setSearch] = useState("");
  const [draft, setDraft] = useState({ ...BLANK });
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const limit = 15;
  const query = { page, page_size: limit, type: type || undefined, category_id: categoryId || undefined, search: search || undefined };

  const { data, loading, error: loadError, reload } = useAsync(
    () => api.get<Page<Transaction>>("/transactions", query),
    [page, type, categoryId, search],
  );
  const { data: categories } = useAsync(() => api.get<Page<Category>>("/categories", { page_size: 200 }), []);

  function reset(nextPage = 1) {
    setPage(nextPage);
    setError(null);
    setNotice(null);
  }

  async function create(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await api.post("/transactions", {
        amount: Number(draft.amount),
        type: draft.type,
        category_id: draft.category_id,
        date: draft.date || undefined,
        description: draft.description || undefined,
        notes: draft.notes || undefined,
      });
      setDraft({ ...BLANK });
      setNotice("Transaction added.");
      reset(1);
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
      await api.delete(`/transactions/${id}`);
      setNotice("Transaction deleted.");
      reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  const meta = data?.meta;
  const expenseCategories = (categories?.items ?? []).filter((c) => c.type === draft.type);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Transactions</h1>
          <p className="sub">{meta ? `${meta.total} total` : "Loading…"}</p>
        </div>
        <div className="inline">
          <button className="btn" onClick={() => api.download("/transactions/export", { format: "csv" }, "transactions.csv")}>
            Export CSV
          </button>
          <button className="btn" onClick={() => api.download("/transactions/export", { format: "pdf" }, "transactions.pdf")}>
            Export PDF
          </button>
        </div>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {notice && <Banner kind="success">{notice}</Banner>}
      {loadError && <Banner kind="error">{errorText(loadError)}</Banner>}

      <Card title="Add a transaction" style={{ marginBottom: "1rem" }}>
        <form onSubmit={create} className="row">
          <div className="field">
            <label htmlFor="tx-type">Type</label>
            <select id="tx-type" value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value, category_id: "" })}>
              <option value="expense">Expense</option>
              <option value="income">Income</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="tx-amount">Amount</label>
            <input
              id="tx-amount"
              type="number"
              step="0.01"
              min="0.01"
              required
              value={draft.amount}
              onChange={(e) => setDraft({ ...draft, amount: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="tx-category">Category</label>
            <select id="tx-category" required value={draft.category_id} onChange={(e) => setDraft({ ...draft, category_id: e.target.value })}>
              <option value="">Select…</option>
              {expenseCategories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="tx-date">Date</label>
            <input id="tx-date" type="date" value={draft.date} onChange={(e) => setDraft({ ...draft, date: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="tx-desc">Description</label>
            <input id="tx-desc" value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} />
          </div>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? "Adding…" : "Add"}
          </button>
        </form>
      </Card>

      <Card title="Filters">
        <div className="row">
          <div className="field">
            <label htmlFor="f-type">Type</label>
            <select
              id="f-type"
              value={type}
              onChange={(e) => {
                setType(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All</option>
              <option value="expense">Expense</option>
              <option value="income">Income</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="f-cat">Category</label>
            <select
              id="f-cat"
              value={categoryId}
              onChange={(e) => {
                setCategoryId(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All</option>
              {(categories?.items ?? []).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.type})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="f-search">Search</label>
            <input
              id="f-search"
              placeholder="description contains…"
              defaultValue={search}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setSearch((e.target as HTMLInputElement).value);
                  setPage(1);
                }
              }}
            />
          </div>
          <button
            className="btn"
            onClick={() => {
              setSearch("");
              setType("");
              setCategoryId("");
              setPage(1);
            }}
          >
            Clear
          </button>
        </div>
      </Card>

      <div style={{ marginTop: "1rem" }}>
        {loading ? (
          <div className="loading">Loading…</div>
        ) : !data?.items.length ? (
          <Empty>No transactions match these filters.</Empty>
        ) : (
          <>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Description</th>
                  <th>Category</th>
                  <th>Type</th>
                  <th className="num">Amount</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.items.map((txn) => (
                  <tr key={txn.id}>
                    <td className="nowrap muted">{shortDate(txn.date)}</td>
                    <td>{txn.description ?? "—"}</td>
                    <td>
                      {txn.category_name && (
                        <span className="inline">
                          <span
                            aria-hidden
                            style={{ width: 8, height: 8, borderRadius: 2, background: txn.category_color ?? "var(--accent)", display: "inline-block" }}
                          />
                          {txn.category_name}
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`pill ${txn.type}`}>{txn.type}</span>
                    </td>
                    <td className={`num ${txn.type === "income" ? "pos" : "neg"}`}>
                      {txn.type === "income" ? "+" : "−"}
                      {money(txn.amount)}
                    </td>
                    <td className="right">
                      <button className="btn btn-sm btn-danger" onClick={() => remove(txn.id)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {meta && (
              <div className="pager">
                <span>
                  Page {meta.page} of {Math.max(1, meta.pages)} · showing {data.items.length} of {meta.total}
                </span>
                <span className="inline">
                  <button className="btn btn-sm" disabled={!meta.has_prev} onClick={() => setPage((p) => Math.max(1, p - 1))}>
                    Previous
                  </button>
                  <button className="btn btn-sm" disabled={!meta.has_next} onClick={() => setPage((p) => p + 1)}>
                    Next
                  </button>
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
