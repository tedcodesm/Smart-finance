/** Response shapes returned by the Mercy API. Mirrors the OpenAPI schemas. */

export interface PageMeta {
  total: number;
  skip: number;
  limit: number;
  page: number;
  page_size: number;
  pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface Page<T> {
  items: T[];
  meta: PageMeta;
}

export interface User {
  id: string;
  name: string;
  email: string;
  base_currency: string;
  is_active: boolean;
  monthly_income_estimate: number | null;
  created_at?: string;
  last_login_at?: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthResponse {
  user: User;
  tokens: TokenPair;
}

export interface Category {
  id: string;
  name: string;
  type: "income" | "expense";
  icon: string | null;
  color: string | null;
  monthly_typical_amount: number | null;
  is_archived: boolean;
  is_system?: boolean;
}

export interface Transaction {
  id: string;
  amount: number;
  signed_amount: number;
  type: "income" | "expense";
  category_id: string | null;
  category_name: string | null;
  category_color: string | null;
  category_icon: string | null;
  date: string;
  description: string | null;
  notes: string | null;
  is_recurring: boolean;
}

export interface TransactionSummary {
  income: number;
  expense: number;
  net: number;
  count: number;
  by_category?: { category_id: string; category_name: string; total: number; count: number }[];
}

export interface Budget {
  id: string;
  name: string | null;
  limit_amount: number;
  spent_amount: number;
  remaining_amount: number;
  period: "weekly" | "monthly" | "yearly";
  category_id: string | null;
  category_name: string | null;
  category_color: string | null;
  alert_threshold_pct: number;
  rollover: boolean;
  is_active: boolean;
  start_date: string | null;
  end_date: string | null;
  period_key?: string;
}

export interface BudgetProgress extends Budget {
  usage_pct: number;
  days_remaining: number | null;
  period_elapsed_pct: number | null;
  is_exceeded: boolean;
  is_threshold_crossed: boolean;
  projected_total: number | null;
  pace_status: string | null;
  last_alerted_at: string | null;
}

export interface Goal {
  id: string;
  name: string;
  target_amount: number;
  current_amount: number;
  remaining_amount: number;
  progress_pct: number;
  target_date: string | null;
  days_remaining: number | null;
  status: "active" | "achieved" | "paused" | "cancelled";
  color: string | null;
  description: string | null;
  linked_category_id: string | null;
}

export interface Alert {
  id: string;
  type: string;
  severity: "info" | "warning" | "critical";
  title: string;
  message: string;
  is_read: boolean;
  read_at: string | null;
  triggered_at: string;
  expires_at: string | null;
  context?: Record<string, unknown> | null;
}

export interface Prediction {
  id: string;
  type: string;
  scope: Record<string, unknown>;
  period_key: string | null;
  predicted_amount: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  confidence: number | null;
  band: string | null;
  generated_at: string;
  expires_at: string | null;
}

export interface ModelStatus {
  name: string;
  available: boolean;
  kind?: string;
  metrics?: Record<string, unknown> | null;
  trained_at?: string | null;
  message?: string | null;
}

export interface Report {
  id: string;
  report_type: string;
  title: string | null;
  period: string;
  period_key: string | null;
  generated_at: string;
  data?: Record<string, unknown>;
}

/** `GET /dashboard` is a single round-trip read model. */
export interface MetricCard {
  label: string;
  value: number;
  previous_value: number | null;
  change_pct: number | null;
  trend: "up" | "down" | null;
  currency: string;
}

export interface Dashboard {
  generated_at: string;
  currency: string;
  period: string;
  period_key: string;
  cards: {
    income: MetricCard;
    expense: MetricCard;
    net: MetricCard;
    savings_rate: MetricCard;
  };
  cash_flow?: {
    period_key: string;
    income: number;
    expense: number;
    net: number;
    transaction_count: number;
    cumulative_net: number;
  }[];
  net_worth?: { period_key: string; cumulative_income: number; cumulative_expense: number; balance: number }[];
  top_categories?: {
    category_id: string;
    category_name: string;
    category_color: string | null;
    total: number;
    share_pct: number;
    change_pct: number | null;
  }[];
  budgets?: BudgetProgress[];
  recent_transactions?: Transaction[];
  predictions?: Prediction[];
  alerts?: Alert[];
  goals_summary?: {
    total_goals: number;
    active_goals: number;
    achieved_goals: number;
    total_target: number;
    total_saved: number;
    total_remaining: number;
    overall_progress_pct: number;
  };
  insights?: string[];
}

/* ---------- ML ---------- */

export interface ModelsStatus {
  available: boolean;
  artifact_dir: string;
  message: string | null;
  latest_run: Record<string, unknown> | null;
  models: {
    model_type: string;
    model_version: string;
    trained_at: string;
    is_active: boolean;
    n_train: number;
    n_test: number;
    metrics: Record<string, number> | null;
  }[];
}

export interface ForecastResponse {
  run_id: string;
  /** ISO datetime string or period key; backend sends `target_period_key` */
  target_period_key: string;
  as_of?: string; // not present in backend – kept for backwards compat
  model_version: string;
  period: string;
  total_predicted: number;
  total_lower_bound?: number | null;
  total_upper_bound?: number | null;
  items: {
    category_id: string | null;
    category_name: string | null;
    period_key: string;
    predicted_amount: number;
    lower_bound: number | null;
    upper_bound: number | null;
    confidence_score: number;
    trend: string | null;
    band?: string | null;
  }[];
}

export interface OverspendResponse {
  run_id: string;
  as_of: string;
  model_version: string;
  global_probability: number;
  items: {
    category_id: string;
    category_name: string;
    budget_id: string;
    probability: number;
    label: "low" | "medium" | "high";
    confidence_score: number;
  }[];
}

export interface BudgetRecommendationsResponse {
  run_id?: string;
  model_version?: string;
  period?: string;
  items: {
    category_id: string | null;
    category_name: string | null;
    /** Backend field name is recommended_limit */
    recommended_limit: number;
    /** Alias kept so old renders still work */
    suggested_limit?: number;
    basis?: Record<string, unknown> | string;
    rationale?: string;
    current_limit?: number | null;
    confidence_score?: number;
  }[];
}

export interface SavingsResponse {
  run_id?: string;
  model_version?: string;
  period?: string;
  average_income: number;
  predicted_expenses: number;
  discretionary_spend: number;
  recommended_savings: number;
  savings_rate_pct: number;
  confidence_score?: number;
  basis?: Record<string, unknown>;
  rationale?: string;
  items?: { category_id: string | null; category_name: string | null; recommended: number }[];
}
