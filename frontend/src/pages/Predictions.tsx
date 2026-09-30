import { useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type {
  BudgetRecommendationsResponse,
  ForecastResponse,
  ModelsStatus,
  OverspendResponse,
  SavingsResponse,
} from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Banner, Bar, Card, Empty, errorText, money, pct, relativeTime } from "../components/ui";

type Tab = "forecast" | "overspend" | "budgets" | "savings";

const TABS: { id: Tab; label: string }[] = [
  { id: "forecast", label: "Expense forecast" },
  { id: "overspend", label: "Overspend risk" },
  { id: "budgets", label: "Budget suggestions" },
  { id: "savings", label: "Savings" },
];

/**
 * A new account has nothing scored yet, so the on-demand endpoints answer 404
 * (`not_found`) instead of 200 with an empty body. That is an empty state, not a
 * fault, so it gets onboarding copy rather than a red error banner.
 */
function notScoredYet(err: Error | null): boolean {
  return err instanceof ApiError && err.status === 404;
}

/** Shows a handful of headline metrics from a model run without dumping all of them. */
function MetricPills({ metrics }: { metrics: Record<string, number> | null }) {
  if (!metrics) return null;
  const wanted = [
    ["mae", "MAE", 2],
    ["rmse", "RMSE", 2],
    ["r2", "R²", 4],
    ["accuracy", "Accuracy", 4],
    ["roc_auc", "ROC-AUC", 4],
    ["f1", "F1", 4],
    ["silhouette_score", "Silhouette", 4],
    ["mae_skill_vs_naive", "Skill vs naive", 4],
  ] as const;
  const shown = wanted.filter(([key]) => metrics[key] !== undefined);
  if (!shown.length) return null;
  return (
    <div className="inline" style={{ marginTop: "0.4rem" }}>
      {shown.map(([key, label, digits]) => (
        <span className="pill" key={key} title={key}>
          {label} {Number(metrics[key]).toFixed(digits)}
        </span>
      ))}
    </div>
  );
}

export default function Predictions() {
  const [tab, setTab] = useState<Tab>("forecast");
  const [error, setError] = useState<string | null>(null);

  const { data: models, error: modelsError } = useAsync(() => api.get<ModelsStatus>("/predictions/models/status"), []);

  // On-demand endpoints always return fresh output, so the page is never empty
  // even before the nightly batch has written any `predictions` documents.
  const forecast = useAsync(
    () => (tab === "forecast" ? api.post<ForecastResponse>("/predictions/expense-forecast", {}) : Promise.resolve(null)),
    [tab],
  );
  const overspend = useAsync(
    () => (tab === "overspend" ? api.post<OverspendResponse>("/predictions/overspend-risk", {}) : Promise.resolve(null)),
    [tab],
  );
  const budgets = useAsync(
    () =>
      tab === "budgets"
        ? api.post<BudgetRecommendationsResponse>("/predictions/budget-recommendations", {})
        : Promise.resolve(null),
    [tab],
  );
  const savings = useAsync(
    () =>
      tab === "savings"
        ? api.post<SavingsResponse>("/predictions/savings-recommendation", {})
        : Promise.resolve(null),
    [tab],
  );

  const active = tab === "forecast" ? forecast : tab === "overspend" ? overspend : tab === "budgets" ? budgets : savings;
  const awaitingData = notScoredYet(active.error);

  async function runInference() {
    setError(null);
    try {
      await api.post("/predictions/inference/run", {});
      active.reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Predictions</h1>
          <p className="sub">
            {models?.available
              ? `4 pooled models · artifacts in ${models.artifact_dir}`
              : "Models unavailable — run mercy-ml-train"}
          </p>
        </div>
        <button className="btn" onClick={runInference}>
          Re-run batch inference
        </button>
      </div>

      {error && <Banner kind="error">{error}</Banner>}
      {modelsError && <Banner kind="error">{errorText(modelsError)}</Banner>}

      <Card title="Trained models" style={{ marginBottom: "1rem" }}>
        {!models ? (
          <Empty>Loading model status…</Empty>
        ) : (
          <div className="grid grid-2">
            {models.models.map((model) => (
              <div key={model.model_type} style={{ borderLeft: "2px solid var(--border)", paddingLeft: "0.75rem" }}>
                <div className="spread">
                  <strong>{model.model_type}</strong>
                  <span className={`pill ${model.is_active ? "success" : ""}`}>
                    {model.is_active ? "active" : "inactive"}
                  </span>
                </div>
                <div className="muted mono" style={{ fontSize: "0.76rem" }}>
                  {model.model_version} · trained {relativeTime(model.trained_at)} · n={model.n_train}
                </div>
                <MetricPills metrics={model.metrics} />
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="inline" style={{ marginBottom: "1rem" }}>
        {TABS.map((t) => (
          <button key={t.id} className={`btn ${tab === t.id ? "btn-primary" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {active.loading && <div className="loading">Running model…</div>}
      {active.error && !awaitingData && <Banner kind="error">{errorText(active.error)}</Banner>}

      {awaitingData && (
        <div className="empty">
          <strong>Nothing to model yet.</strong> Predictions need a few weeks of transactions to learn
          from — add some on the <Link to="/transactions">Transactions</Link> page and reload, or sign in
          as the demo account to see this page populated.
        </div>
      )}

      {!active.loading && !awaitingData && !active.error && (
        <>
          {tab === "forecast" && <ForecastView data={forecast.data} onReload={forecast.reload} />}
          {tab === "overspend" && <OverspendView data={overspend.data} />}
          {tab === "budgets" && <BudgetSuggestions data={budgets.data} />}
          {tab === "savings" && <SavingsView data={savings.data} />}
        </>
      )}
    </>
  );
}

function ForecastView({ data, onReload }: { data: ForecastResponse | null; onReload: () => void }) {
  if (!data) return <Empty>No forecast available.</Empty>;
  const items = data.items ?? [];
  if (!items.length) return <Empty>The model needs more history to forecast.</Empty>;
  const peak = Math.max(...items.map((i) => i.predicted_amount), 1);
  return (
    <Card
      title={`Predicted spending · ${data.model_version} · as of ${relativeTime(data.as_of)}`}
      action={
        <button className="btn btn-sm" onClick={onReload}>
          Refresh
        </button>
      }
    >
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Period</th>
            <th className="num">Predicted</th>
            <th className="num">Range</th>
            <th>Band</th>
            <th style={{ width: "22%" }} />
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => (
            <tr key={`${item.category_id ?? "all"}-${item.period_key}-${i}`}>
              <td>{item.category_name ?? "All spending"}</td>
              <td className="muted mono">{item.period_key}</td>
              <td className="num"><strong>{money(item.predicted_amount)}</strong></td>
              <td className="num muted">
                {item.lower_bound !== null && item.upper_bound !== null
                  ? `${money(item.lower_bound)} – ${money(item.upper_bound)}`
                  : "—"}
              </td>
              <td>
                {item.band && (
                  <span className={`pill ${item.band === "high" ? "critical" : item.band === "low" ? "" : "warning"}`}>
                    {item.band}
                  </span>
                )}
              </td>
              <td>
                <Bar value={item.predicted_amount} max={peak} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function OverspendView({ data }: { data: OverspendResponse | null }) {
  if (!data) return <Empty>No risk assessment available.</Empty>;
  const items = data.items ?? [];
  if (!items.length) return <Empty>No budgets to assess against.</Empty>;
  return (
    <Card title={`Overspend risk · ${data.model_version}`}>
      <Banner kind={data.global_probability >= 0.5 ? "error" : "info"}>
        Overall probability of overspending this period: <strong>{pct(data.global_probability * 100)}</strong>
      </Banner>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th className="num">Probability</th>
            <th>Risk</th>
            <th style={{ width: "32%" }} />
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.category_id}>
              <td>{item.category_name}</td>
              <td className="num">{pct(item.probability * 100)}</td>
              <td>
                <span className={`pill ${item.label === "high" ? "critical" : item.label === "medium" ? "warning" : "success"}`}>
                  {item.label}
                </span>
              </td>
              <td>
                <Bar value={item.probability * 100} max={100} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function BudgetSuggestions({ data }: { data: BudgetRecommendationsResponse | null }) {
  if (!data) return <Empty>No suggestions available.</Empty>;
  const items = data.items ?? [];
  if (!items.length) return <Empty>Not enough history to suggest budgets.</Empty>;
  return (
    <Card title="Suggested budget limits">
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th className="num">Current</th>
            <th className="num">Suggested</th>
            <th className="num">Change</th>
            <th>Basis</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => {
            const current = item.current_limit ?? null;
            const delta = current !== null ? item.suggested_limit - current : null;
            return (
              <tr key={`${item.category_id ?? "all"}-${i}`}>
                <td>{item.category_name ?? "Overall"}</td>
                <td className="num muted">{current !== null ? money(current) : "—"}</td>
                <td className="num"><strong>{money(item.suggested_limit)}</strong></td>
                <td className={`num ${delta === null ? "muted" : delta > 0 ? "neg" : "pos"}`}>
                  {delta === null ? "—" : `${delta > 0 ? "+" : ""}${money(delta)}`}
                </td>
                <td className="muted" style={{ fontSize: "0.82rem" }}>
                  {item.basis ?? item.rationale ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

function SavingsView({ data }: { data: SavingsResponse | null }) {
  if (!data) return <Empty>No savings recommendation available.</Empty>;
  return (
    <Card title="Savings recommendation">
      <div className="stat-value pos">{money(data.recommended_savings)}</div>
      <p className="muted" style={{ marginTop: "0.25rem" }}>
        {data.rationale ?? "Recommended monthly savings based on your spending pattern."}
      </p>
      {data.items && data.items.length > 0 && (
        <table style={{ marginTop: "0.75rem" }}>
          <thead>
            <tr>
              <th>Category</th>
              <th className="num">Suggested saving</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item, i) => (
              <tr key={`${item.category_id ?? "all"}-${i}`}>
                <td>{item.category_name ?? "Overall"}</td>
                <td className="num pos">{money(item.recommended)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
