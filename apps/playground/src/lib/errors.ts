import { ApiError } from "./api-client";

export interface HumanizedError {
  headline: string;
  detail?: string;
  raw: string;
}

/**
 * Map an ApiError (or any thrown value) into a short, tester-readable
 * headline + optional detail line. Always returns `raw` so callers can
 * surface the original payload behind a "Show details" expander.
 */
export function humanizeApiError(
  err: unknown,
  overrides?: { [status: number]: string },
): HumanizedError {
  if (err instanceof ApiError) {
    const extracted = extractMessage(err.body);
    const raw = stringifyBody(err.body, err.status);
    const override = overrides?.[err.status];
    if (override) return { headline: override, detail: extracted, raw };

    switch (true) {
      case err.status === 401:
        return { headline: "Token expired or invalid — sign in again", detail: extracted, raw };
      case err.status === 403:
        return { headline: "Not authorised for that request", detail: extracted, raw };
      case err.status === 404:
        return { headline: extracted ?? "Not found", raw };
      case err.status === 413:
        return { headline: "File too large", detail: extracted, raw };
      case err.status === 422:
        return { headline: "Request didn’t pass validation", detail: extracted, raw };
      case err.status >= 500:
        return { headline: "The gateway had a problem — try again", detail: extracted, raw };
      default:
        return { headline: extracted ?? `Request failed (HTTP ${err.status})`, raw };
    }
  }

  if (err instanceof Error) {
    return { headline: err.message, raw: err.stack ?? err.message };
  }
  return { headline: "Unknown error", raw: String(err) };
}

function extractMessage(body: unknown): string | undefined {
  if (body == null) return undefined;
  if (typeof body === "string") return body.trim() || undefined;
  if (typeof body !== "object") return String(body);

  const obj = body as Record<string, unknown>;

  // FastAPI convention: {"detail": "<string>"} OR {"detail": {code, message}}
  // OR {"detail": [{loc, msg, ...}]} for 422 validation errors.
  if ("detail" in obj) {
    const d = obj.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      const first = d[0];
      if (first && typeof first === "object" && "msg" in first) {
        return String((first as { msg: unknown }).msg);
      }
      return undefined;
    }
    if (d && typeof d === "object") {
      const dd = d as Record<string, unknown>;
      if (typeof dd.message === "string") return dd.message;
      if (typeof dd.detail === "string") return dd.detail;
    }
  }

  // OpenAI-shaped: {"error": {"message": "..."}}
  if ("error" in obj && obj.error && typeof obj.error === "object") {
    const e = obj.error as Record<string, unknown>;
    if (typeof e.message === "string") return e.message;
  }
  if (typeof obj.message === "string") return obj.message;

  return undefined;
}

function stringifyBody(body: unknown, status: number): string {
  if (body == null) return `HTTP ${status}`;
  if (typeof body === "string") return body;
  try {
    return JSON.stringify(body, null, 2);
  } catch {
    return String(body);
  }
}
