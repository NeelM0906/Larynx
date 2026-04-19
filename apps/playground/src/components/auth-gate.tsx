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
import { getToken, setToken, clearToken } from "@/lib/token";
import { apiFetch, ApiError } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { ErrorPanel } from "@/components/error-panel";

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const [hasToken, setHasToken] = useState(false);
  const [draft, setDraft] = useState("");
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState<HumanizedError | null>(null);

  useEffect(() => {
    setMounted(true);
    setHasToken(Boolean(getToken()));
    const onChange = () => setHasToken(Boolean(getToken()));
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
    // validate by hitting a bearer-auth'd read endpoint. If that 401s
    // we pull the token back out again.
    setToken(trimmed);
    try {
      await apiFetch("/v1/voices?limit=1");
      setDraft("");
    } catch (e) {
      clearToken();
      if (e instanceof ApiError && e.status === 401) {
        setValidationError({
          headline: "That token didn’t work — the gateway rejected it.",
          detail: "Double-check you copied the whole token without trailing whitespace.",
          raw: "401 Unauthorized from GET /v1/voices",
        });
      } else {
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
              Set API token
            </DialogTitle>
            <DialogDescription className="pt-1">
              Paste your gateway bearer token. It&apos;s stored in this
              browser&apos;s localStorage and sent with every request.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2 pt-2">
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
