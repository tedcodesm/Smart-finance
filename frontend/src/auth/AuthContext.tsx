import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, getAccessToken, setTokens, setUnauthorizedHandler } from "../api/client";
import type { AuthResponse, User } from "../api/types";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Restore the session on first paint: a stored access token means we only need
  // to confirm it is still valid by reading the profile.
  useEffect(() => {
    let cancelled = false;
    if (!getAccessToken()) {
      setLoading(false);
      return;
    }
    api
      .get<User>("/users/me")
      .then((me) => {
        if (!cancelled) setUser(me);
      })
      .catch(() => setTokens(null, null))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A refresh failure anywhere in the app drops us back to the login screen.
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
  }, []);

  const adopt = useCallback((data: AuthResponse) => {
    setTokens(data.tokens.access_token, data.tokens.refresh_token);
    setUser(data.user);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      adopt(await api.post<AuthResponse>("/auth/login", { email, password }));
    },
    [adopt],
  );

  const register = useCallback(
    async (name: string, email: string, password: string) => {
      adopt(
        await api.post<AuthResponse>("/auth/register", {
          name,
          email,
          password,
          base_currency: "USD",
        }),
      );
    },
    [adopt],
  );

  const logout = useCallback(async () => {
    const refresh = localStorage.getItem("mercy.refresh_token");
    if (refresh) {
      // Best effort: the local session is cleared regardless of the outcome.
      await api.post("/auth/logout", { refresh_token: refresh }).catch(() => undefined);
    }
    setTokens(null, null);
    setUser(null);
  }, []);

  const value = useMemo<AuthState>(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}
