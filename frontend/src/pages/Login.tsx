import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { Banner, errorText } from "../components/ui";

// Must match DEMO_EMAIL / DEMO_PASSWORD in the backend `.env`; override in
// `frontend/.env` when the seeded demo account differs from the defaults.
const DEMO_EMAIL = import.meta.env.VITE_DEMO_EMAIL || "demo@example.com";
const DEMO_PASSWORD = import.meta.env.VITE_DEMO_PASSWORD || "demopassword1";

export default function Login() {
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email.trim(), password);
      navigate("/", { replace: true });
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <div className="card auth-card">
        <div className="brand">
          <div className="brand-mark">U</div>
          <div>
            <div className="brand-name">user</div>
            <div className="brand-sub">personal finance</div>
          </div>
        </div>

        <h1>Sign in</h1>
        <p>Track spending, budgets and forecasts in one place.</p>

        {error && <Banner kind="error">{error}</Banner>}

        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div className="auth-demo">
          <div className="auth-demo-head">
            <span className="eyebrow">Demo account</span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                setEmail(DEMO_EMAIL);
                setPassword(DEMO_PASSWORD);
              }}
            >
              Fill
            </button>
          </div>
          <div className="auth-demo-creds mono">
            <span>{DEMO_EMAIL}</span>
            <span>{DEMO_PASSWORD}</span>
          </div>
        </div>

        <p className="hint">
          No account? <Link to="/register">Create one</Link>
        </p>
      </div>
    </div>
  );
}
