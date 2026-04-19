"use client";

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  getToken,
  setToken,
  clearToken,
  type TokenChangeDetail,
  type TokenChangeReason,
} from "@/lib/token";
import { apiFetch, ApiError } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { ErrorPanel } from "@/components/error-panel";

type RejectReason = Exclude<TokenChangeReason, "set" | "cleared">;

const REJECT_COPY: Record<RejectReason, { headline: string; detail: string }> = {
  rejected: {
    headline: "Your previous token was rejected.",
    detail: "The gateway returned 401 for a protected endpoint. Paste a fresh token to continue.",
  },
  "ws-rejected": {
    headline: "The conversation endpoint rejected your token.",
    detail:
      "The WebSocket closed with a policy-violation code. The token has been cleared; paste a fresh one to reconnect.",
  },
  "validation-failed": {
    headline: "The stored token is no longer valid.",
    detail:
      "The gateway didn't accept the token that was saved in this browser. Paste a current one.",
  },
};

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const [hasToken, setHasToken] = useState(false);
  const [rejectReason, setRejectReason] = useState<RejectReason | null>(null);
  const [draft, setDraft] = useState("");
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState<HumanizedError | null>(null);

  useEffect(() => {
    setMounted(true);
    const initial = Boolean(getToken());
    setHasToken(initial);

    // Revalidate any pre-existing token on mount. If it turns out to be
    // stale (e.g. the gateway's LARYNX_API_TOKEN was rotated since the
    // browser last cached one) apiFetch will invalidate it on 401 and
    // the token-changed handler below reopens the dialog.
    if (initial) {
      void apiFetch("/v1/voices?limit=1").catch(() => {
        // apiFetch side-effects invalidate on 401. Anything else
        // (CORS, offline) we ignore here — page-level error panels
        // will surface it.
      });
    }

    const onChange = (e: Event) => {
      const detail = (e as CustomEvent<TokenChangeDetail>).detail;
      const nowHas = Boolean(getToken());
      setHasToken(nowHas);
      if (!nowHas) {
        if (
          detail?.reason === "rejected" ||
          detail?.reason === "ws-rejected" ||
          detail?.reason === "validation-failed"
        ) {
          setRejectReason(detail.reason);
        } else {
          setRejectReason(null);
        }
      } else {
        setRejectReason(null);
        setValidationError(null);
      }
    };
    window.addEventListener("larynx:token-changed", onChange);
    return () => window.removeEventListener("larynx:token-changed", onChange);
  }, []);

  if (!mounted) {
    // Pre-hydration: render nothing to avoid a flash of un-gated content.
    return null;
  }

  const onSave = async () => {
    const trimmed = draft.trim();
    if (!trimmed || validating) return;
    setValidating(true);
    setValidationError(null);
    // Stash in localStorage first — apiFetch reads from there — then
    // validate by hitting a bearer-auth'd read endpoint. On 401 apiFetch
    // also invalidates via the api-client helper, so we don't have to
    // duplicate clearToken() here.
    setToken(trimmed);
    try {
      await apiFetch("/v1/voices?limit=1");
      setDraft("");
      setRejectReason(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        setValidationError({
          headline: "That token didn’t work — the gateway rejected it.",
          detail: "Double-check you copied the whole token without trailing whitespace.",
          raw: "401 Unauthorized from GET /v1/voices",
        });
      } else {
        // Not a 401 — apiFetch didn't invalidate. Keep the token out of
        // storage so the user isn't trapped on a possibly-bad value.
        clearToken();
        setValidationError(
          humanizeApiError(e, {
            // Gateway unreachable / CORS / DNS — treat as config issue.
            0: "Couldn’t reach the gateway — is it running?",
          }),
        );
      }
    } finally {
      setValidating(false);
    }
  };

  const reject = rejectReason ? REJECT_COPY[rejectReason] : null;

  return (
    <>
      {children}
      <Dialog open={!hasToken} onOpenChange={() => {}}>
        <DialogContent
          showCloseButton={false}
          className="sm:max-w-md border-border/80"
        >
          <DialogHeader>
            <DialogTitle className="font-display text-3xl italic font-normal leading-none pt-2">
              {reject ? "Token rejected" : "Set API token"}
            </DialogTitle>
            <DialogDescription className="pt-1">
              {reject
                ? reject.detail
                : "Paste your gateway bearer token. It's stored in this browser's localStorage and sent with every request."}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2 pt-2">
            {reject && (
              <p className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
                {reject.headline}
              </p>
            )}
            <Label
              htmlFor="token"
              className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground"
            >
              Bearer token
            </Label>
            <Input
              id="token"
              type="password"
              autoFocus
              spellCheck={false}
              autoComplete="off"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void onSave();
              }}
              placeholder="sk-larynx-…"
              className="font-mono text-sm"
              disabled={validating}
            />
            {validationError && <ErrorPanel error={validationError} />}
          </div>
          <DialogFooter className="mt-2">
            <Button onClick={() => void onSave()} disabled={!draft.trim() || validating}>
              {validating ? "Checking…" : "Save token"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/**
 * "Clear token" CTA for the global nav. Confirms before wiping so a
 * stray click doesn't force the tester to re-paste their token.
 */
export function SignOutButton() {
  const [confirming, setConfirming] = useState(false);
  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setConfirming(true)}
        className="text-muted-foreground hover:text-foreground"
      >
        Sign out
      </Button>
      <Dialog open={confirming} onOpenChange={setConfirming}>
        <DialogContent className="sm:max-w-md border-border/80">
          <DialogHeader>
            <DialogTitle className="font-display text-2xl italic font-normal leading-none pt-2">
              Clear token?
            </DialogTitle>
            <DialogDescription className="pt-1">
              Removes the token from this browser&apos;s localStorage. You&apos;ll
              need to paste it again to use the playground.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-2 gap-2">
            <Button variant="outline" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                clearToken();
                setConfirming(false);
              }}
            >
              Sign out
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
