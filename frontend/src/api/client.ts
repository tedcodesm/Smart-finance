/**
 * Thin typed wrapper over the Mercy REST API.
 *
 * Two behaviours the rest of the app depends on:
 *  - the access token is held in memory and mirrored to `localStorage` so a page
 *    refresh does not sign the user out;
 *  - a 401 triggers a single refresh-token exchange, and the original request is
 *    replayed once. Concurrent 401s share that single refresh.
 */

// In production there is no Vite dev-server proxy, so we need to prefix every
// request with the backend origin. VITE_BACKEND_ORIGIN is set in the Render
// environment; it is empty (undefined) in local dev where the proxy handles it.
const ORIGIN = (import.meta.env.VITE_BACKEND_ORIGIN ?? "").replace(/\/$/, "");
const BASE = `${ORIGIN}/api/v1`;
const ACCESS_KEY = "mercy.access_token";
const REFRESH_KEY = "mercy.refresh_token";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

let accessToken: string | null = localStorage.getItem(ACCESS_KEY);
let refreshToken: string | null = localStorage.getItem(REFRESH_KEY);
let onUnauthorized: (() => void) | null = null;

export function setTokens(access: string | null, refresh: string | null): void {
  accessToken = access;
  refreshToken = refresh;
  if (access) localStorage.setItem(ACCESS_KEY, access);
  else localStorage.removeItem(ACCESS_KEY);
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
  else localStorage.removeItem(REFRESH_KEY);
}

export function getAccessToken(): string | null {
  return accessToken;
}

export function setUnauthorizedHandler(fn: () => void): void {
  onUnauthorized = fn;
}

type Query = Record<string, string | number | boolean | undefined | null>;

function buildUrl(path: string, query?: Query): string {
  const url = `${BASE}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

async function parseError(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `Request failed (${response.status})`;
  let details: unknown = null;
  try {
    const body = await response.json();
    if (body && typeof body === "object") {
      code = body.code ?? code;
      message = body.message ?? message;
      details = body.details ?? null;
    }
  } catch {
    /* non-JSON error body; keep the defaults */
  }
  return new ApiError(response.status, code, message, details);
}

let refreshInFlight: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  if (!refreshToken) return false;
  // Collapse parallel 401s into one refresh round-trip.
  refreshInFlight ??= (async () => {
    try {
      const response = await fetch(buildUrl("/auth/refresh"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!response.ok) return false;
      const data = await response.json();
      setTokens(data.tokens.access_token, data.tokens.refresh_token);
      return true;
    } catch {
      return false;
    } finally {
      // Cleared on the next tick so callers awaiting this promise see the result.
      setTimeout(() => {
        refreshInFlight = null;
      }, 0);
    }
  })();
  return refreshInFlight;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  query?: Query;
  raw?: boolean;
  retry?: boolean;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, query, raw = false, retry = true } = options;
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;

  const response = await fetch(buildUrl(path, query), {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (response.status === 401 && retry && refreshToken) {
    if (await refreshAccessToken()) {
      return request<T>(path, { ...options, retry: false });
    }
    setTokens(null, null);
    onUnauthorized?.();
  }

  if (!response.ok) throw await parseError(response);
  if (raw) return (await response.blob()) as T;
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  get: <T,>(path: string, query?: Query) => request<T>(path, { query }),
  post: <T,>(path: string, body?: unknown, query?: Query) => request<T>(path, { method: "POST", body, query }),
  patch: <T,>(path: string, body?: unknown) => request<T>(path, { method: "PATCH", body }),
  put: <T,>(path: string, body?: unknown) => request<T>(path, { method: "PUT", body }),
  delete: <T,>(path: string) => request<T>(path, { method: "DELETE" }),
  /** Streams a binary export, preserving the filename from Content-Disposition. */
  download: async (path: string, query?: Query, fallbackName = "export") => {
    const headers: Record<string, string> = {};
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
    const response = await fetch(buildUrl(path, query), { headers });
    if (!response.ok) throw await parseError(response);
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") ?? "";
    const match = /filename="?([^";]+)"?/.exec(disposition);
    const name = match ? match[1] : fallbackName;
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(href);
  },
};
