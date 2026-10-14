import { api } from "../api/client";
import type { Dashboard, MetricCard } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Card, Empty, errorText, money, pct, shortDate } from "../components/ui";
import { useAuth } from "../auth/AuthContext";
import Loader from "../components/Loader";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** `period_key` is `YYYY-MM`, `YYYY-Www` or `YYYY` depending on the period. */
function periodLabel(key: string): string {
  if (/^\d{4}-W\d{1,2}$/.test(key)) return `W${key.split("-W")[1]}`;
  const month = /^(\d{4})-(\d{1,2})$/.exec(key);
  if (month) return `${MONTHS[Number(month[2]) - 1]} ${month[1]}`;
  return key;
}

function periodName(period: string): string {
  if (period === "yearly") return "this year";
  if (period === "weekly") return "this week";
  return "this month";
}

/** Signed percentage change, or null when there is no comparable prior value. */
function changeOf(card: MetricCard): number | null {
  if (card.change_pct !== null) return card.change_pct;
  if (card.previous_value === null || card.previous_value === 0) return null;
  return ((card.value - card.previous_value) / Math.abs(card.previous_value)) * 100;
}

function Delta({ value, invert }: { value: number | null; invert?: boolean }) {
  if (value === null || !Number.isFinite(value)) {
    return <span className="stat-hint">no prior period</span>;
  }
  const rising = value > 0;
  const good = invert ? !rising : rising;
  const arrow = rising ? "↑" : value < 0 ? "↓" : "→";
  return (
    <span className={`delta ${good ? "pos" : "neg"}`}>
      {arrow} {pct(Math.abs(value))} vs last {value === 0 ? "period" : "period"}
    </span>
  );
}

type CashFlowRow = NonNullable<Dashboard["cash_flow"]>[number];

function CashFlowChart({ rows }: { rows: CashFlowRow[] }) {
  const width = 760;
  const height = 190;
  const pad = 10;

  if (rows.length < 2) {
    return <div className="empty small">Not enough history to chart yet.</div>;
  }

  const values = rows.flatMap((row) => [row.income, row.expense]);
  const min = Math.min(0, ...values);
  const max = Math.max(1, ...values);
  const span = Math.max(max - min, 1);

  const x = (index: number) => (index / (rows.length - 1)) * width;
  const y = (value: number) => height - pad - ((value - min) / span) * (height - pad * 2);
  const line = (pick: (row: CashFlowRow) => number) =>
    rows.map((row, index) => `${x(index)},${y(pick(row))}`).join(" ");

  const income = line((row) => row.income);
  const expense = line((row) => row.expense);
  const area = `${income} ${width},${height} 0,${height}`;
  const first = rows[0].period_key;
  const last = rows[rows.length - 1].period_key;

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Income and spending by period">
        <defs>
          <linearGradient id="incomeFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <line x1="0" y1={y(0)} x2={width} y2={y(0)} stroke="var(--border)" strokeWidth="1" />
        <polygon points={area} fill="url(#incomeFill)" />
        <polyline
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          points={income}
        />
        <polyline
          fill="none"
          stroke="var(--pos)"
          strokeWidth="2"
          strokeDasharray="4 3"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          points={expense}
        />
      </svg>
      <div className="chart-legend">
        <span>
          <i /> Income
        </span>
        <span>
          <i className="alt" /> Spending
        </span>
        <span style={{ marginLeft: "auto" }}>
          {periodLabel(first)} – {periodLabel(last)}
        </span>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { user } = useAuth();
  const { data, error, loading, reload } = useAsync(() => api.get<Dashboard>("/dashboard"), []);

  if (loading) return <Loader label="Loading dashboard…" />;
  if (error) return <Banner kind="error">{errorText(error)}</Banner>;
  if (!data) return <Empty>No data.</Empty>;

  const { cards, currency } = data;
  const period = periodName(data.period);
  const categories = data.top_categories ?? [];
  const goals = data.goals_summary;
  const alerts = data.alerts ?? [];
  const unread = alerts.filter((alert) => !alert.is_read).length;

  return (
    <>
      <header className="topbar">
        <div>
          <h1>Welcome back, {user?.name ?? "there"}</h1>
          <p>Your position for {period}.</p>
        </div>
        <div className="topbar-actions">
          <div className="profile-pill">
            <div className="profile-avatar">{(user?.name ?? "U").slice(0, 1).toUpperCase()}</div>
            <span>{(!user?.base_currency || user.base_currency === "USD") && (!currency || currency === "USD") ? "KES" : (user?.base_currency ?? currency)}</span>
          </div>
        </div>
      </header>

      <div className="stat-grid">
        <Card title="Income">
          <div className="stat-value">{money(cards.income.value, currency)}</div>
          <Delta value={changeOf(cards.income)} />
        </Card>
        <Card title="Spending">
          <div className="stat-value">{money(cards.expense.value, currency)}</div>
          <Delta value={changeOf(cards.expense)} invert />
        </Card>
        <Card title="Net">
          <div className={`stat-value ${cards.net.value < 0 ? "neg" : ""}`}>{money(cards.net.value, currency)}</div>
          <Delta value={changeOf(cards.net)} />
        </Card>
        <Card title="Savings rate">
          <div className="stat-value">{cards.savings_rate.value === 0 ? "—" : pct(cards.savings_rate.value)}</div>
          <Delta value={changeOf(cards.savings_rate)} />
        </Card>
      </div>

      <div className="dash-grid">
        <Card title="Cash flow">
          <CashFlowChart rows={data.cash_flow ?? []} />
        </Card>

        <Card title="Spending by category">
          {categories.length ? (
            <div className="cat-list">
              {categories.map((category) => (
                <div className="cat-row" key={category.category_id}>
                  <div className="cat-head">
                    <span className="cat-name">
                      <i className="cat-dot" style={{ background: category.category_color ?? "var(--accent)" }} />
                      {category.category_name}
                    </span>
                    <span className="cat-amount">{money(category.total, currency)}</span>
                  </div>
                  <div className="bar">
                    <span style={{ width: `${Math.min(100, Math.max(0, category.share_pct))}%` }} />
                  </div>
                  <div className="cat-meta">
                    <span>{pct(category.share_pct)} of spending</span>
                    <span>
                      {category.change_pct === null
                        ? "no prior period"
                        : `${category.change_pct > 0 ? "↑" : category.change_pct < 0 ? "↓" : "→"} ${pct(
                            Math.abs(category.change_pct),
                          )} vs last period`}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty small">No spending recorded for {period}.</div>
          )}
        </Card>
      </div>

      <div className="dash-grid">
        <Card title="Recent transactions">
          {data.recent_transactions?.length ? (
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Description</th>
                  <th>Category</th>
                  <th className="num">Amount</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_transactions.map((txn) => (
                  <tr key={txn.id}>
                    <td className="nowrap muted">{shortDate(txn.date)}</td>
                    <td>{txn.description ?? txn.category_name ?? "—"}</td>
                    <td className="muted">{txn.category_name ?? "—"}</td>
                    <td className={`num ${txn.type === "income" ? "pos" : ""}`}>
                      {txn.type === "income" ? "+" : "−"}
                      {money(txn.amount, currency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty small">No transactions yet.</div>
          )}
        </Card>

        <div className="stack">
          {goals && (
            <Card title="Savings goals">
              <div className="spread" style={{ marginBottom: "0.5rem" }}>
                <span className="muted">Progress</span>
                <span>
                  <strong>{money(goals.total_saved, currency)}</strong>{" "}
                  <span className="muted">of {money(goals.total_target, currency)}</span>
                </span>
              </div>
              <div className="bar">
                <span style={{ width: `${Math.min(100, Math.max(0, goals.overall_progress_pct))}%` }} />
              </div>
              <div className="inline" style={{ marginTop: "0.6rem" }}>
                <span className="pill success">{pct(goals.overall_progress_pct)}</span>
                <span className="muted">
                  {goals.achieved_goals} of {goals.total_goals} achieved
                </span>
              </div>
            </Card>
          )}

          {alerts.length > 0 && (
            <Card title="Alerts">
              <div className="insight-list">
                {alerts.slice(0, 4).map((alert) => (
                  <div className="insight-item" key={alert.id}>
                    <span style={{ minWidth: 0 }}>
                      <span className={`pill ${alert.severity}`} style={{ marginRight: 6 }}>
                        {alert.severity}
                      </span>
                      {alert.title}
                    </span>
                  </div>
                ))}
              </div>
              {unread > 0 && (
                <div className="inline" style={{ marginTop: "0.75rem" }}>
                  <button className="btn btn-sm" onClick={reload}>
                    Refresh
                  </button>
                  <span className="muted">{unread} unread</span>
                </div>
              )}
            </Card>
          )}

          {data.insights?.length ? (
            <Card title="Insights">
              <div className="insight-list">
                {data.insights.map((insight) => (
                  <div className="insight-item" key={insight}>
                    {insight}
                  </div>
                ))}
              </div>
            </Card>
          ) : null}
        </div>
      </div>
    </>
  );
}
