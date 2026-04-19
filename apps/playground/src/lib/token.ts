const TOKEN_KEY = "larynx.token";
const EVENT = "larynx:token-changed";

export type TokenChangeReason =
  | "set"
  | "cleared"
  | "rejected"
  | "ws-rejected"
  | "validation-failed";

export interface TokenChangeDetail {
  reason: TokenChangeReason;
}

function emit(reason: TokenChangeReason): void {
  window.dispatchEvent(new CustomEvent<TokenChangeDetail>(EVENT, { detail: { reason } }));
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
  emit("set");
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
  emit("cleared");
}

/**
 * Clear the stored token because it was rejected by the gateway.
 * No-op if nothing is stored. Fires token-changed with a reason the
 * AuthGate surfaces in the dialog.
 */
export function invalidateToken(
  reason: Exclude<TokenChangeReason, "set" | "cleared">,
): void {
  if (typeof window === "undefined") return;
  if (window.localStorage.getItem(TOKEN_KEY) === null) return;
  window.localStorage.removeItem(TOKEN_KEY);
  emit(reason);
}
