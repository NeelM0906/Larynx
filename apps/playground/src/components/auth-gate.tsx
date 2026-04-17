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

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const [hasToken, setHasToken] = useState(false);
  const [draft, setDraft] = useState("");

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

  const onSave = () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    setToken(trimmed);
    setDraft("");
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
                if (e.key === "Enter") onSave();
              }}
              placeholder="sk-larynx-…"
              className="font-mono text-sm"
            />
          </div>
          <DialogFooter className="mt-2">
            <Button onClick={onSave} disabled={!draft.trim()}>
              Save token
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function SignOutButton() {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => clearToken()}
      className="text-muted-foreground"
    >
      Clear token
    </Button>
  );
}
