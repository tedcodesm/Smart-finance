import { NavLink, useNavigate } from "react-router-dom";
import { useEffect, useState, type ReactNode } from "react";
import { useAuth } from "../auth/AuthContext";
import { useAsync } from "../hooks/useAsync";
import { api } from "../api/client";
import { currentTheme, toggleTheme, type Theme } from "../theme";
import type { Alert } from "../api/types";

const LINKS = [
  { to: "/", label: "Dashboard", icon: "◧", end: true },
  { to: "/transactions", label: "Transactions", icon: "⇄", end: false },
  { to: "/budgets", label: "Budgets", icon: "◔", end: false },
  { to: "/goals", label: "Goals", icon: "◎", end: false },
  { to: "/predictions", label: "Predictions", icon: "◈", end: false },
  { to: "/reports", label: "Reports", icon: "▤", end: false },
  { to: "/alerts", label: "Alerts", icon: "◉", end: false },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [theme, setTheme] = useState<Theme>(() => currentTheme());

  // Follow the OS while the user has not made an explicit choice.
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => {
      try {
        if (window.localStorage.getItem("mercy-theme")) return;
      } catch {
        /* storage unavailable: fall through to following the OS */
      }
      const next: Theme = event.matches ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      setTheme(next);
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const { data: counts } = useAsync(
    () => api.get<{ unread: number }>("/alerts/counts").catch(() => ({ unread: 0 })),
    [],
  );
  const unread = counts?.unread ?? 0;

  return (
    <div className="shell">
      <div className="dashboard-shell">
        <aside className="dashboard-sidebar">
          <div className="brand-row">
            <div className="brand-mark">U</div>
            <div className="brand-copy">
              <div className="brand-name">User</div>
              <div className="brand-sub">personal finance</div>
            </div>
          </div>

          <nav className="sidebar-nav">
            {LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
              >
                <span aria-hidden>{link.icon}</span>
                {link.label}
                {link.to === "/alerts" && unread > 0 && <span className="pill critical badge">{unread}</span>}
              </NavLink>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="user-block">
              <div className="user-name">{user?.name ?? "User"}</div>
              <div className="user-email">{user?.email ?? "user@example.com"}</div>
            </div>
            <button
              className="nav-item small"
              onClick={() => setTheme(toggleTheme())}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              <span aria-hidden>{theme === "dark" ? "☀" : "☾"}</span>
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
            <button
              className="nav-item small"
              onClick={async () => {
                await logout();
                navigate("/login", { replace: true });
              }}
            >
              Sign out
            </button>
          </div>
        </aside>

        <main className="dashboard-main">{children}</main>
      </div>
    </div>
  );
}

export type { Alert };
