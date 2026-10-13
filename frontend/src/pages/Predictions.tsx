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
import Loader from "../components/Loader";

type Tab = "forecast" | "overspend" | "budgets" | "savings";

const TABS: { id: Tab; label: string }[] = [
  { id: "forecast", label: "Expense forecast" },
  { id: "overspend", label: "Overspend risk" },
  { id: "budgets", label: "Budget suggestions" },
  { id: "savings", label: "Savings" },
];

/**
 * A new account has nothing scored yet (404), or models might be awaiting training (503).
 * These get friendly guided state rather than a generic red failure.
 */
function notScoredYet(err: Error | null): boolean {
  return err instanceof ApiError && (err.status === 404 || err.status === 503);
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
    <div className="inline" style={{ marginTop: "0.4rem", flexWrap: "wrap", gap: "0.35rem" }}>
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
  const [runningInference, setRunningInference] = useState(false);
  const [inferenceNotice, setInferenceNotice] = useState<string | null>(null);

  const {
    data: models,
    error: modelsError,
    loading: modelsLoading,
    reload: reloadModels,
  } = useAsync(() => api.get<ModelsStatus>("/predictions/models/status"), []);

  // On-demand endpoints always return fresh output
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
    setInferenceNotice(null);
    setRunningInference(true);
    try {
      const res = await api.post<{ predictions_written?: number; users_scored?: number }>(
        "/predictions/inference/run",
        {},
      );
      setInferenceNotice(
        `Batch inference complete: scored ${res.users_scored ?? 1} user(s), generated ${res.predictions_written ?? 0} predictions.`,
      );
      active.reload();
      reloadModels();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setRunningInference(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Predictions</h1>
          <p className="sub">
            {modelsLoading
              ? "Checking model status…"
              : models?.available
              ? `${models.models.length} active models · artifacts in ${models.artifact_dir}`
              : modelsError
              ? "Unable to reach prediction models service"
              : models?.message || "Models unavailable — run ml-train"}
          </p>
        </div>
        <button className="btn" onClick={runInference} disabled={runningInference}>
          {runningInference ? "Running inference…" : "Re-run batch inference"}
        </button>
      </div>

      {inferenceNotice && <Banner kind="info">{inferenceNotice}</Banner>}
      {error && <Banner kind="error">{error}</Banner>}
      {modelsError && <Banner kind="error">{errorText(modelsError)}</Banner>}

      <Card title="Trained models" style={{ marginBottom: "1rem" }}>
        {modelsLoading ? (
          <Empty>Loading model status…</Empty>
        ) : modelsError ? (
          <Empty>Unable to load model registry: {errorText(modelsError)}</Empty>
        ) : !models || models.models.length === 0 ? (
          <Empty>{models?.message || "No trained models found."}</Empty>
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

      {active.loading && <Loader label="Running model…" />}
      {active.error && !awaitingData && <Banner kind="error">{errorText(active.error)}</Banner>}

      {awaitingData && (
        <div className="empty">
          {active.error instanceof ApiError && active.error.status === 503 ? (
            <>
              <strong>Models are currently unavailable.</strong> ML model artifacts have not been trained yet.
              Click &ldquo;Re-run batch inference&rdquo; or run model training to generate predictions.
            </>
          ) : (
            <>
              <strong>Nothing to model yet.</strong> Predictions need a few weeks of transactions to learn
              from — add some on the <Link to="/transactions">Transactions</Link> page and reload, or sign in
              as the demo account to see this page populated.
            </>
          )}
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
  const periodLabel = data.target_period_key || (data.as_of ? relativeTime(data.as_of) : "Current period");
  return (
    <Card
      title={`Predicted spending · ${data.model_version} · ${periodLabel}`}
      action={
        <button className="btn btn-sm" onClick={onReload}>
          Refresh
        </button>
      }
    >
      {data.total_predicted > 0 && (
        <div style={{ marginBottom: "1rem" }}>
          <span className="muted">Total Predicted: </span>
          <strong style={{ fontSize: "1.2rem" }}>{money(data.total_predicted)}</strong>
          {data.total_lower_bound !== null && data.total_upper_bound !== null && (
            <span className="muted" style={{ marginLeft: "0.5rem" }}>
              ({money(data.total_lower_bound)} – {money(data.total_upper_bound)})
            </span>
          )}
        </div>
      )}
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
  const prob = (data.global_probability ?? 0) * 100;
  return (
    <Card title={`Overspend risk · ${data.model_version}`}>
      <Banner kind={prob >= 50 ? "error" : "info"}>
        Overall probability of overspending this period: <strong>{pct(prob)}</strong>
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
            const suggested = item.recommended_limit ?? item.suggested_limit ?? 0;
            const delta = current !== null ? suggested - current : null;
            const basisText =
              typeof item.basis === "string"
                ? item.basis
                : item.basis && typeof item.basis === "object"
                ? Object.entries(item.basis)
                    .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(2) : String(v)}`)
                    .join(", ")
                : item.rationale ?? "—";

            return (
              <tr key={`${item.category_id ?? "all"}-${i}`}>
                <td>{item.category_name ?? "Overall"}</td>
                <td className="num muted">{current !== null ? money(current) : "—"}</td>
                <td className="num"><strong>{money(suggested)}</strong></td>
                <td className={`num ${delta === null ? "muted" : delta > 0 ? "neg" : "pos"}`}>
                  {delta === null ? "—" : `${delta > 0 ? "+" : ""}${money(delta)}`}
                </td>
                <td className="muted" style={{ fontSize: "0.82rem" }}>
                  {basisText}
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
      <div className="grid grid-2" style={{ marginBottom: "1rem" }}>
        <div>
          <div className="muted" style={{ fontSize: "0.85rem" }}>Recommended Monthly Savings</div>
          <div className="stat-value pos" style={{ fontSize: "1.8rem" }}>{money(data.recommended_savings)}</div>
          <div className="muted mono" style={{ fontSize: "0.78rem", marginTop: "0.2rem" }}>
            Target Savings Rate: {pct(data.savings_rate_pct)}
          </div>
        </div>
        <div>
          <div className="muted" style={{ fontSize: "0.85rem" }}>Financial Overview</div>
          <div style={{ fontSize: "0.9rem", marginTop: "0.25rem", lineHeight: "1.6" }}>
            <div>Avg. Income: <strong>{money(data.average_income)}</strong></div>
            <div>Predicted Expenses: <strong>{money(data.predicted_expenses)}</strong></div>
            <div>Discretionary Spend: <strong>{money(data.discretionary_spend)}</strong></div>
          </div>
        </div>
      </div>
      <p className="muted" style={{ marginTop: "0.5rem" }}>
        {data.rationale ?? "Recommended monthly savings calculated from your income and forecast expenses."}
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
