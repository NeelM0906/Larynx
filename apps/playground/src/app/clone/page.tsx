"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch, ApiError } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";

const MAX_FILE_BYTES = 50 * 1024 * 1024; // 50 MB — UI-side cap under the gateway's 100 MB.
const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"];
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.concat([
  "audio/wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/flac",
  "audio/x-flac",
  "audio/mp4",
  "audio/m4a",
  "audio/ogg",
]).join(",");

type VoiceSource = "uploaded" | "designed" | "seed" | "lora";

interface VoiceResponse {
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

export default function ClonePage() {
  const router = useRouter();

  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [promptText, setPromptText] = useState("");
  const [description, setDescription] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<HumanizedError | null>(null);
  const [created, setCreated] = useState<VoiceResponse | null>(null);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragHover, setDragHover] = useState(false);

  const fileTooLarge = file !== null && file.size > MAX_FILE_BYTES;
  const fileBadType = file !== null && !hasAcceptedExt(file.name);
  const canSubmit =
    !submitting &&
    file !== null &&
    !fileTooLarge &&
    !fileBadType &&
    name.trim().length > 0;

  const onPickFile = useCallback((f: File | null) => {
    setFile(f);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setDragHover(false);
      const f = e.dataTransfer.files?.[0] ?? null;
      if (f) onPickFile(f);
    },
    [onPickFile],
  );

  const onSubmit = useCallback(async () => {
    if (!canSubmit || !file) return;
    setSubmitting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("name", name.trim());
      if (promptText.trim()) form.append("prompt_text", promptText.trim());
      if (description.trim()) form.append("description", description.trim());
      form.append("audio", file, file.name);
      const resp = await apiFetch<VoiceResponse>("/v1/voices", {
        method: "POST",
        body: form,
      });
      setCreated(resp);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError({
          headline: "A voice with that name already exists",
          detail: "Pick a unique name — voice names are used as keys.",
          raw: JSON.stringify(e.body, null, 2),
        });
      } else {
        setError(
          humanizeApiError(e, {
            413: "Audio file too large — trim to under 50 MB or compress to MP3.",
          }),
        );
      }
    } finally {
      setSubmitting(false);
    }
  }, [canSubmit, file, name, promptText, description]);

  const onReset = useCallback(() => {
    setFile(null);
    setName("");
    setPromptText("");
    setDescription("");
    setCreated(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  if (created) {
    return (
      <PageShell
        kicker="02 · Voice cloning"
        title="Voice is ready."
        intro="Encoded, stored, and wired up. Give it a test or jump to the library."
      >
        <SuccessPanel
          voice={created}
          onClone={onReset}
          onTest={() => router.push(`/tts?voice=${encodeURIComponent(created.id)}`)}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      kicker="02 · Voice cloning"
      title="Borrow a voice."
      intro="Upload a clean 10–30 s reference clip, name it, and we’ll add it to the library."
    >
      <div className="flex flex-col gap-6">
        <label
          htmlFor="clone-file"
          onDragOver={(e) => {
            e.preventDefault();
            setDragHover(true);
          }}
          onDragLeave={() => setDragHover(false)}
          onDrop={onDrop}
          className={`block cursor-pointer rounded-md border border-dashed p-10 text-center transition ${
            dragHover ? "border-primary bg-muted/40" : "border-primary/40 hover:bg-muted/30"
          }`}
        >
          <p className="font-display text-2xl text-primary">
            {file ? "Replace reference clip" : "Drop a reference clip"}
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            .wav · .mp3 · .flac · .m4a · .ogg — max 50 MB. Clean audio,
            single speaker, 10–30 seconds is ideal.
          </p>
          <input
            id="clone-file"
            ref={inputRef}
            type="file"
            accept={ACCEPT_ATTR}
            className="hidden"
            onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
          />
        </label>

        {file && <FileMetaCard file={file} tooLarge={fileTooLarge} badType={fileBadType} />}

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          <Field
            id="clone-name"
            label="Voice name *"
            hint="Must be unique across the library."
            required
          >
            <input
              id="clone-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. amy-natural"
              maxLength={128}
              className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
            />
          </Field>

          <Field
            id="clone-desc"
            label="Description"
            hint="Optional. Free-form note for the library card."
          >
            <input
              id="clone-desc"
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Warm alto, podcast tone"
              maxLength={500}
              className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
            />
          </Field>

          <Field
            id="clone-prompt"
            label="Reference transcript"
            hint="Optional but recommended — exactly what's said in the clip. Unlocks VoxCPM2’s higher-quality cloning mode."
          >
            <textarea
              id="clone-prompt"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="Type out exactly what the speaker says in the reference clip."
              rows={3}
              maxLength={500}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
            />
          </Field>
        </div>

        <ErrorPanel error={error} />

        <div className="flex items-center gap-3">
          <Button size="lg" onClick={() => void onSubmit()} disabled={!canSubmit}>
            {submitting ? "Processing reference…" : "Clone voice"}
          </Button>
          {submitting && (
            <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <span className="inline-block h-2 w-2 animate-ping rounded-full bg-primary" />
              Uploading and encoding.
            </span>
          )}
        </div>
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function FileMetaCard({
  file,
  tooLarge,
  badType,
}: {
  file: File;
  tooLarge: boolean;
  badType: boolean;
}) {
  const sizeKB = (file.size / 1024).toFixed(1);
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  const label = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
  const warn = tooLarge || badType;
  return (
    <div
      className={`rounded-md border p-3 text-xs font-mono ${
        warn
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-border bg-muted/30"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate">{file.name}</span>
        <span className="shrink-0 text-muted-foreground">{label}</span>
      </div>
      {tooLarge && (
        <p className="mt-2">File is over 50 MB — trim or compress before uploading.</p>
      )}
      {badType && (
        <p className="mt-2">
          Unsupported extension — use .wav, .mp3, .flac, .m4a, or .ogg.
        </p>
      )}
    </div>
  );
}

function Field({
  id,
  label,
  hint,
  required,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={id}
        className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
      >
        {label}
        {required && <span className="ml-1 text-primary">·</span>}
      </label>
      {children}
      {hint && <span className="text-[11px] text-muted-foreground/90">{hint}</span>}
    </div>
  );
}

function SuccessPanel({
  voice,
  onClone,
  onTest,
}: {
  voice: VoiceResponse;
  onClone: () => void;
  onTest: () => void;
}) {
  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-md border border-primary/40 bg-primary/5 p-5">
        <p className="font-display text-xl italic text-foreground">
          {voice.name} is in the library.
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-3 text-[11px] font-mono uppercase tracking-widest text-muted-foreground md:grid-cols-4">
          <div>
            <dt>Voice id</dt>
            <dd className="text-foreground font-semibold truncate" title={voice.id}>
              {voice.id.slice(0, 12)}
            </dd>
          </div>
          {voice.sample_rate_hz && (
            <div>
              <dt>Sample rate</dt>
              <dd className="text-foreground font-semibold">{voice.sample_rate_hz / 1000} kHz</dd>
            </div>
          )}
          {voice.duration_ms && (
            <div>
              <dt>Duration</dt>
              <dd className="text-foreground font-semibold">
                {(voice.duration_ms / 1000).toFixed(1)} s
              </dd>
            </div>
          )}
          <div>
            <dt>Source</dt>
            <dd className="text-foreground font-semibold">{voice.source}</dd>
          </div>
        </dl>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button size="lg" onClick={onTest}>
          Test in TTS →
        </Button>
        <Link
          href={`/library?voice=${encodeURIComponent(voice.id)}`}
          className="inline-flex h-9 items-center rounded-lg border border-border px-3 text-sm hover:bg-muted"
        >
          Go to library
        </Link>
        <Button variant="ghost" size="sm" onClick={onClone}>
          Clone another
        </Button>
      </div>
    </div>
  );
}

function hasAcceptedExt(name: string): boolean {
  const lower = name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

