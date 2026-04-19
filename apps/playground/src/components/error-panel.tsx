"use client";

import { useState } from "react";
import type { HumanizedError } from "@/lib/errors";

/**
 * Inline error surface used across feature pages. Shows the humanised
 * headline prominently; the raw body is tucked behind a "Show details"
 * toggle so devs can still inspect the original payload without
 * scaring testers.
 */
export function ErrorPanel({ error }: { error: HumanizedError | null }) {
  const [open, setOpen] = useState(false);
  if (!error) return null;
  return (
    <div
      role="alert"
      className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
    >
      <p className="font-medium">{error.headline}</p>
      {error.detail && (
        <p className="mt-1 text-destructive/80 text-xs font-mono">{error.detail}</p>
      )}
      {error.raw && error.raw !== error.headline && (
        <>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-2 inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-widest text-destructive/80 hover:text-destructive"
          >
            {open ? "Hide details" : "Show details"}
          </button>
          {open && (
            <pre className="mt-2 max-h-40 overflow-auto rounded border border-destructive/30 bg-background/60 p-2 text-[11px] text-destructive/90 whitespace-pre-wrap">
              {error.raw}
            </pre>
          )}
        </>
      )}
    </div>
  );
}
