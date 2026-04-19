"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { RefreshCwIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { apiFetch } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";

type VoiceSource = "uploaded" | "designed" | "seed" | "lora";

interface Voice {
  id: string;
  name: string;
  description: string | null;
  source: VoiceSource;
  sample_rate_hz: number | null;
  duration_ms: number | null;
  prompt_text: string | null;
  design_prompt: string | null;
  created_at: string;
  updated_at: string;
}

interface VoiceListResponse {
  voices: Voice[];
  total: number;
  limit: number;
  offset: number;
}

export default function LibraryPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const highlightId = searchParams.get("voice");

  const [voices, setVoices] = useState<Voice[] | null>(null);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<HumanizedError | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Voice | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<HumanizedError | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch<VoiceListResponse>("/v1/voices?limit=200");
      setVoices(resp.voices);
      setTotal(resp.total);
    } catch (e) {
      setError(humanizeApiError(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const onTest = useCallback(
    (voice: Voice) => {
      router.push(`/tts?voice=${encodeURIComponent(voice.id)}`);
    },
    [router],
  );

  const onConfirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await apiFetch(`/v1/voices/${encodeURIComponent(pendingDelete.id)}`, {
        method: "DELETE",
      });
      setVoices((prev) => prev?.filter((v) => v.id !== pendingDelete.id) ?? null);
      setTotal((n) => Math.max(0, n - 1));
      setPendingDelete(null);
    } catch (e) {
      setDeleteError(humanizeApiError(e));
    } finally {
      setDeleting(false);
    }
  }, [pendingDelete]);

  return (
    <PageShell
      kicker="04 · Voice library"
      title="The collection."
      intro="Every voice in the gateway's Postgres. Test one, delete one, or clone a fresh one."
    >
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            {loading
              ? "Loading…"
              : voices
                ? `${total} voice${total === 1 ? "" : "s"}`
                : ""}
          </p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void load()}
            disabled={loading}
            aria-label="Refresh"
          >
            <RefreshCwIcon className={loading ? "animate-spin" : undefined} />
            Refresh
          </Button>
        </div>

        <ErrorPanel error={error} />

        {loading && !voices && <LoadingSkeleton />}

        {!loading && voices && voices.length === 0 && <EmptyState />}

        {voices && voices.length > 0 && (
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-px bg-border/60 border border-border/60">
            {voices.map((v) => (
              <VoiceCard
                key={v.id}
                voice={v}
                highlighted={v.id === highlightId}
                onTest={() => onTest(v)}
                onDelete={() => setPendingDelete(v)}
              />
            ))}
          </ul>
        )}
      </div>

      <DeleteDialog
        voice={pendingDelete}
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(null);
            setDeleteError(null);
          }
        }}
        deleting={deleting}
        error={deleteError}
        onConfirm={onConfirmDelete}
      />
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function VoiceCard({
  voice,
  highlighted,
  onTest,
  onDelete,
}: {
  voice: Voice;
  highlighted: boolean;
  onTest: () => void;
  onDelete: () => void;
}) {
  const cardRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    if (highlighted && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlighted]);

  const meta = useMemo(() => {
    const parts: string[] = [];
    if (voice.sample_rate_hz) parts.push(`${(voice.sample_rate_hz / 1000).toFixed(0)} kHz`);
    if (voice.duration_ms) parts.push(`${(voice.duration_ms / 1000).toFixed(1)} s`);
    return parts.join(" · ");
  }, [voice.sample_rate_hz, voice.duration_ms]);

  return (
    <li
      ref={cardRef}
      className={`relative bg-background p-6 transition-colors ${
        highlighted ? "ring-1 ring-primary/50" : ""
      }`}
      data-voice-id={voice.id}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-display text-2xl italic text-foreground truncate">
            {voice.name}
          </h3>
          <p className="mt-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground/80">
            {voice.id.slice(0, 12)}
          </p>
        </div>
        <SourceBadge source={voice.source} />
      </div>
      {voice.description && (
        <p className="mt-3 text-sm text-muted-foreground">{voice.description}</p>
      )}
      {voice.design_prompt && (
        <p className="mt-3 text-sm italic text-muted-foreground">
          “{voice.design_prompt}”
        </p>
      )}
      <dl className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
        {meta && <div>{meta}</div>}
        <div>Created {formatDate(voice.created_at)}</div>
      </dl>
      <div className="mt-5 flex items-center gap-2">
        <Button size="sm" onClick={onTest}>
          Test
        </Button>
        <Button size="sm" variant="destructive" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </li>
  );
}

function SourceBadge({ source }: { source: VoiceSource }) {
  const styles: Record<VoiceSource, string> = {
    uploaded: "border-foreground/30 text-foreground",
    designed: "border-primary/40 text-primary",
    seed: "border-border text-muted-foreground",
    lora: "border-chart-2 text-chart-2",
  };
  return (
    <span
      className={`shrink-0 rounded-sm border px-2 py-0.5 text-[10px] font-mono uppercase tracking-widest ${styles[source]}`}
    >
      {source}
    </span>
  );
}

function LoadingSkeleton() {
  return (
    <ul className="grid grid-cols-1 md:grid-cols-2 gap-px bg-border/60 border border-border/60">
      {[0, 1, 2, 3].map((i) => (
        <li key={i} className="bg-background p-6 animate-pulse">
          <div className="h-6 w-2/3 rounded bg-muted" />
          <div className="mt-2 h-3 w-1/4 rounded bg-muted/60" />
          <div className="mt-6 h-3 w-full rounded bg-muted/40" />
          <div className="mt-2 h-3 w-4/5 rounded bg-muted/40" />
          <div className="mt-5 flex gap-2">
            <div className="h-7 w-16 rounded bg-muted" />
            <div className="h-7 w-20 rounded bg-muted/60" />
          </div>
        </li>
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <div className="rounded-md border border-dashed border-border p-12 text-center">
      <p className="font-display text-2xl italic text-foreground">No voices yet.</p>
      <p className="mt-3 text-sm text-muted-foreground">
        Clone your first voice from a reference clip, or describe a new voice
        from scratch.
      </p>
      <div className="mt-6 flex items-center justify-center gap-3">
        <Link
          href="/clone"
          className="inline-flex h-8 items-center rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Clone a voice →
        </Link>
        <Link
          href="/design"
          className="inline-flex h-8 items-center rounded-lg border border-border px-3 text-sm text-muted-foreground hover:bg-muted"
        >
          Design a voice
        </Link>
      </div>
    </div>
  );
}

function DeleteDialog({
  voice,
  open,
  onOpenChange,
  deleting,
  error,
  onConfirm,
}: {
  voice: Voice | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  deleting: boolean;
  error: HumanizedError | null;
  onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md border-border/80">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl italic font-normal leading-none pt-2">
            Delete voice?
          </DialogTitle>
          <DialogDescription className="pt-1">
            {voice ? (
              <>
                <span className="font-medium text-foreground">{voice.name}</span>{" "}
                and its cached latents + reference audio will be removed. This
                can&apos;t be undone.
              </>
            ) : null}
          </DialogDescription>
        </DialogHeader>
        {error && (
          <div className="pt-2">
            <ErrorPanel error={error} />
          </div>
        )}
        <DialogFooter className="mt-2 gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={deleting}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onConfirm} disabled={deleting}>
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}
