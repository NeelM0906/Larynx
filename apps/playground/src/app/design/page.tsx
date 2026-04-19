"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch, ApiError } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { getToken } from "@/lib/token";

const BASE_URL = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";
const DEFAULT_PREVIEW_TEXT = "Hello. This is a preview of the designed voice.";
const PROMPT_EXAMPLES = [
  "A warm British narrator, unhurried, mid-range male.",
  "Soft intimate female voice, breathy but clear, American English.",
  "Deep resonant storyteller, measured and authoritative.",
];

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

interface DesignPreview {
  preview_id: string;
  expires_in_s: number;
  name: string;
  description: string | null;
  design_prompt: string;
  preview_text: string;
  sample_rate: number;
  duration_ms: number;
  audioUrl: string;
}

export default function DesignPage() {
  const router = useRouter();

  const [prompt, setPrompt] = useState("");
  const [previewText, setPreviewText] = useState("");
  const [workingName, setWorkingName] = useState("");

  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<HumanizedError | null>(null);
  const [preview, setPreview] = useState<DesignPreview | null>(null);

  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<HumanizedError | null>(null);
  const [saved, setSaved] = useState<VoiceResponse | null>(null);

  const prevAudioUrlRef = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (prevAudioUrlRef.current) URL.revokeObjectURL(prevAudioUrlRef.current);
    };
  }, []);

  const canGenerate = !generating && prompt.trim().length > 0;
  const canSave = !saving && preview !== null && saveName.trim().length > 0;

  const onGenerate = useCallback(async () => {
    if (!canGenerate) return;
    setGenerating(true);
    setGenError(null);
    setSaveError(null);
    setSaved(null);
    try {
      const token = getToken();
      const previewName = `preview-${Date.now().toString(36)}`;
      setWorkingName(previewName);
      const body: Record<string, unknown> = {
        name: previewName,
        design_prompt: prompt.trim(),
      };
      const pt = previewText.trim();
      if (pt) body.preview_text = pt;

      const meta = await apiFetch<{
        preview_id: string;
        expires_in_s: number;
        name: string;
        description: string | null;
        design_prompt: string;
        preview_text: string;
        sample_rate: number;
        duration_ms: number;
      }>("/v1/voices/design", {
        method: "POST",
        body: JSON.stringify(body),
      });

      const audioRes = await fetch(
        `${BASE_URL}/v1/voices/design/${encodeURIComponent(meta.preview_id)}/audio`,
        {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        },
      );
      if (!audioRes.ok) {
        let errBody: unknown;
        try {
          errBody = await audioRes.json();
        } catch {
          errBody = await audioRes.text();
        }
        throw new ApiError(`${audioRes.status} ${audioRes.statusText}`, audioRes.status, errBody);
      }
      const blob = await audioRes.blob();
      const url = URL.createObjectURL(blob);
      if (prevAudioUrlRef.current) URL.revokeObjectURL(prevAudioUrlRef.current);
      prevAudioUrlRef.current = url;

      setPreview({ ...meta, audioUrl: url });
      if (!saveName.trim()) setSaveName("");
    } catch (e) {
      setPreview(null);
      setGenError(humanizeApiError(e));
    } finally {
      setGenerating(false);
    }
  }, [canGenerate, prompt, previewText, saveName]);

  const onSave = useCallback(async () => {
    if (!canSave || !preview) return;
    setSaving(true);
    setSaveError(null);
    try {
      const body: Record<string, unknown> = { name: saveName.trim() };
      const voice = await apiFetch<VoiceResponse>(
        `/v1/voices/design/${encodeURIComponent(preview.preview_id)}/save`,
        { method: "POST", body: JSON.stringify(body) },
      );
      setSaved(voice);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setSaveError({
          headline: "A voice with that name already exists",
          detail: "Pick a unique name — voice names are used as keys.",
          raw: JSON.stringify(e.body, null, 2),
        });
      } else {
        setSaveError(humanizeApiError(e));
      }
    } finally {
      setSaving(false);
    }
  }, [canSave, preview, saveName]);

  // When the prompt or preview-text changes, hide the save form so testers
  // don't accidentally commit a stale preview.
  const onPromptChange = (v: string) => {
    setPrompt(v);
    if (preview) setPreview(null);
    if (saved) setSaved(null);
    if (saveError) setSaveError(null);
  };
  const onPreviewTextChange = (v: string) => {
    setPreviewText(v);
    if (preview) setPreview(null);
    if (saved) setSaved(null);
    if (saveError) setSaveError(null);
  };

  if (saved) {
    return (
      <PageShell
        kicker="03 · Voice design"
        title="Voice is in the library."
        intro="Your designed voice is saved and ready to try."
      >
        <SavedPanel
          voice={saved}
          onDesignAnother={() => {
            setPrompt("");
            setPreviewText("");
            setPreview(null);
            setSaved(null);
            setSaveName("");
            setGenError(null);
            setSaveError(null);
          }}
          onTest={() => router.push(`/tts?voice=${encodeURIComponent(saved.id)}`)}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      kicker="03 · Voice design"
      title="Describe a voice. Hear it."
      intro="Prose in. Preview out. Save the keeper to the library."
    >
      <div className="flex flex-col gap-6">
        <Field
          id="design-prompt"
          label="Describe the voice you want *"
          hint="Natural language — timbre, age, accent, pace, mood."
        >
          <textarea
            id="design-prompt"
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            placeholder={PROMPT_EXAMPLES[0]}
            rows={3}
            maxLength={500}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-base leading-relaxed focus:outline-none focus:ring-2 focus:ring-ring/50"
          />
          <ExampleChips onPick={(s) => onPromptChange(s)} examples={PROMPT_EXAMPLES} />
        </Field>

        <Field
          id="design-preview-text"
          label="Sample text"
          hint={`Optional — what the preview should say. Default: "${DEFAULT_PREVIEW_TEXT}"`}
        >
          <input
            id="design-preview-text"
            type="text"
            value={previewText}
            onChange={(e) => onPreviewTextChange(e.target.value)}
            placeholder={DEFAULT_PREVIEW_TEXT}
            maxLength={400}
            className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
          />
        </Field>

        <ErrorPanel error={genError} />

        <div className="flex items-center gap-3">
          <Button size="lg" onClick={() => void onGenerate()} disabled={!canGenerate}>
            {generating ? "Designing voice…" : preview ? "Regenerate preview" : "Generate preview"}
          </Button>
          {generating && (
            <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <span className="inline-block h-2 w-2 animate-ping rounded-full bg-primary" />
              Rendering from prompt.
            </span>
          )}
        </div>

        {preview && (
          <PreviewPanel
            preview={preview}
            workingName={workingName}
            saveName={saveName}
            setSaveName={setSaveName}
            canSave={canSave}
            saving={saving}
            saveError={saveError}
            onSave={onSave}
          />
        )}
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function PreviewPanel({
  preview,
  workingName,
  saveName,
  setSaveName,
  canSave,
  saving,
  saveError,
  onSave,
}: {
  preview: DesignPreview;
  workingName: string;
  saveName: string;
  setSaveName: (v: string) => void;
  canSave: boolean;
  saving: boolean;
  saveError: HumanizedError | null;
  onSave: () => void;
}) {
  return (
    <div className="rounded-md border border-primary/40 bg-primary/5 p-5">
      <div className="flex items-baseline justify-between">
        <p className="font-display text-lg italic text-foreground">
          Preview ready.
        </p>
        <dl className="flex gap-4 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
          <div>{(preview.duration_ms / 1000).toFixed(2)}s</div>
          <div>{preview.sample_rate / 1000}kHz</div>
          {preview.expires_in_s > 0 && (
            <div>expires in {preview.expires_in_s}s</div>
          )}
        </dl>
      </div>
      <audio src={preview.audioUrl} controls className="mt-3 w-full" />

      <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto] md:items-end">
        <div className="flex flex-col gap-2">
          <label
            htmlFor="design-save-name"
            className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
          >
            Voice name *
          </label>
          <input
            id="design-save-name"
            type="text"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder={workingName || "e.g. brit-narrator"}
            maxLength={128}
            className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
          />
        </div>
        <Button size="lg" onClick={onSave} disabled={!canSave}>
          {saving ? "Saving…" : "Save to library"}
        </Button>
      </div>
      {saveError && (
        <div className="mt-3">
          <ErrorPanel error={saveError} />
        </div>
      )}
    </div>
  );
}

function SavedPanel({
  voice,
  onDesignAnother,
  onTest,
}: {
  voice: VoiceResponse;
  onDesignAnother: () => void;
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
        <Button variant="ghost" size="sm" onClick={onDesignAnother}>
          Design another
        </Button>
      </div>
    </div>
  );
}

function Field({
  id,
  label,
  hint,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={id}
        className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
      >
        {label}
      </label>
      {children}
      {hint && <span className="text-[11px] text-muted-foreground/90">{hint}</span>}
    </div>
  );
}

function ExampleChips({
  examples,
  onPick,
}: {
  examples: readonly string[];
  onPick: (s: string) => void;
}) {
  return (
    <div className="mt-1 flex flex-wrap gap-2">
      {examples.map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => onPick(s)}
          className="rounded-md border border-border bg-background px-2 py-1 text-[11px] text-muted-foreground hover:border-primary/60 hover:text-foreground"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
