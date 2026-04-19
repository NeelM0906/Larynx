import { getToken, invalidateToken } from "./token";

/**
 * Thin wrapper around fetch that attaches the bearer token from localStorage.
 * Real typed surface lands once openapi-typescript generation is wired up
 * against the gateway's /openapi.json — see `npm run gen:api`.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const BASE_URL = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";

export async function apiFetch<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  if (!res.ok) throw await buildApiError(res);

  const contentType = res.headers.get("Content-Type") ?? "";
  if (contentType.includes("application/json")) return (await res.json()) as T;
  return (await res.blob()) as unknown as T;
}

/**
 * Convert a non-OK `Response` into an `ApiError`, and as a side effect
 * clear the stored token if the gateway returned 401. Used by apiFetch
 * and by the handful of raw `fetch` / `XMLHttpRequest` paths that can't
 * go through apiFetch (streaming audio, upload progress, etc.).
 *
 * Every path that can receive a 401 from the gateway must funnel through
 * here (or call `invalidateToken("rejected")` directly) so the AuthGate
 * reopens instead of leaving the user looking at a bare "token expired"
 * page error.
 */
export async function buildApiError(res: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    try {
      body = await res.text();
    } catch {
      body = null;
    }
  }
  if (res.status === 401) invalidateToken("rejected");
  return new ApiError(`${res.status} ${res.statusText}`, res.status, body);
}

/**
 * Convenience wrapper for code that already has a parsed body + status
 * (e.g. XHR handlers). Mirrors `buildApiError`'s token-invalidation
 * side effect.
 */
export function apiErrorFrom(status: number, statusText: string, body: unknown): ApiError {
  if (status === 401) invalidateToken("rejected");
  return new ApiError(`${status} ${statusText}`, status, body);
}
