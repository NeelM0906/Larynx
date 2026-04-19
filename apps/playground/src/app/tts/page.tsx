"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch, ApiError } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { getToken } from "@/lib/token";

const MAX_CHARS = 1000;
const BASE_URL = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";

const SAMPLE_RATES = [16000, 24000, 48000] as const;
const DEFAULT_SAMPLE_RATE = 24000;
const DEFAULT_CFG = 2.0;
const DEFAULT_TEMPERATURE = 1.0;

type VoiceSource = "uploaded" | "designed" | "seed" | "lora";

interface Voice {
  id: string;
  name: string;
  source: VoiceSource;
}

interface VoiceListResponse {
  voices: Voice[];
  total: number;
  limit: number;
  offset: number;
}

interface SynthResult {
  url: string;
  contentType: string;
  durationMs: number | null;
  generationTimeMs: number | null;
  sampleRate: number | null;
  voiceId: string | null;
  sizeBytes: number;
}

export default function TtsPage() {
  const searchParams = useSearchParams();
  const urlVoice = searchParams.get("voice");

  const [text, setText] = useState("");
  const [voices, setVoices] = useState<Voice[] | null>(null);
  const [voiceId, setVoiceId] = useState<string>("");
  const [voicesError, setVoicesError] = useState<HumanizedError | null>(null);

  const [sampleRate, setSampleRate] = useState<number>(DEFAULT_SAMPLE_RATE);
  const [cfgValue, setCfgValue] = useState<number>(DEFAULT_CFG);
  const [temperature, setTemperature] = useState<number>(DEFAULT_TEMPERATURE);

  const [synthesizing, setSynthesizing] = useState(false);
  const [synthError, setSynthError] = useState<HumanizedError | null>(null);
  const [result, setResult] = useState<SynthResult | null>(null);

  // Free any object URL we own when it's replaced or the page unmounts.
  const prevUrlRef = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    };
  }, []);

  // Load voices on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await apiFetch<VoiceListResponse>("/v1/voices?limit=200");
        if (cancelled) return;
        const sorted = [...resp.voices].sort((a, b) => a.name.localeCompare(b.name));
        setVoices(sorted);
      } catch (e) {
        if (!cancelled) setVoicesError(humanizeApiError(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Pre-select from ?voice=<id> if present, else alphabetically-first.
  useEffect(() => {
    if (!voices || voiceId) return;
    if (urlVoice && voices.some((v) => v.id === urlVoice)) {
      setVoiceId(urlVoice);
      return;
    }
    if (voices.length > 0) setVoiceId(voices[0].id);
  }, [voices, urlVoice, voiceId]);

  const charsLeft = MAX_CHARS - text.length;
  const trimmedText = text.trim();
  const canSynth = trimmedText.length > 0 && !synthesizing && text.length <= MAX_CHARS;

  const onSynthesize = useCallback(async () => {
    if (!canSynth) return;
    setSynthesizing(true);
    setSynthError(null);
    try {
      const token = getToken();
      const body: Record<string, unknown> = {
        text: trimmedText,
        sample_rate: sampleRate,
        cfg_value: cfgValue,
        temperature,
      };
      if (voiceId) body.voice_id = voiceId;
      const res = await fetch(`${BASE_URL}/v1/tts`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let errBody: unknown;
        try {
          errBody = await res.json();
        } catch {
          errBody = await res.text();
        }
        throw new ApiError(`${res.status} ${res.statusText}`, res.status, errBody);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
      prevUrlRef.current = url;
      setResult({
        url,
        contentType: res.headers.get("Content-Type") ?? blob.type ?? "audio/wav",
        durationMs: numFromHeader(res.headers.get("X-Audio-Duration-Ms")),
        generationTimeMs: numFromHeader(res.headers.get("X-Generation-Time-Ms")),
        sampleRate: numFromHeader(res.headers.get("X-Sample-Rate")),
        voiceId: res.headers.get("X-Voice-ID"),
        sizeBytes: blob.size,
      });
    } catch (e) {
      setSynthError(
        humanizeApiError(e, {
          404: "Voice no longer exists — pick another",
        }),
      );
    } finally {
      setSynthesizing(false);
    }
  }, [canSynth, trimmedText, voiceId, sampleRate, cfgValue, temperature]);

  const paramsChanged =
    sampleRate !== DEFAULT_SAMPLE_RATE ||
    cfgValue !== DEFAULT_CFG ||
    temperature !== DEFAULT_TEMPERATURE;

  return (
    <PageShell
      kicker="01 · Text to speech"
      title="Say something."
      intro="Type up to a thousand characters, pick a voice from your library, hit synthesize."
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <label
              htmlFor="tts-text"
              className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
            >
              Text
            </label>
            <span
              className={`font-mono text-[11px] ${
                charsLeft < 0 ? "text-destructive" : "text-muted-foreground/70"
              }`}
            >
              {text.length} / {MAX_CHARS}
            </span>
          </div>
          <textarea
            id="tts-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Welcome to Larynx. What would you like to hear?"
            rows={5}
            maxLength={MAX_CHARS * 2} // soft gate above; hard gate via disabled state
            className="min-h-32 w-full rounded-md border border-border bg-background px-3 py-2 text-base leading-relaxed placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/50"
          />
        </div>

        <VoicePicker
          voices={voices}
          voiceId={voiceId}
          onChange={setVoiceId}
          error={voicesError}
        />

        <AdvancedParams
          sampleRate={sampleRate}
          setSampleRate={setSampleRate}
          cfgValue={cfgValue}
          setCfgValue={setCfgValue}
          temperature={temperature}
          setTemperature={setTemperature}
          changed={paramsChanged}
          onReset={() => {
            setSampleRate(DEFAULT_SAMPLE_RATE);
            setCfgValue(DEFAULT_CFG);
            setTemperature(DEFAULT_TEMPERATURE);
          }}
        />

        <ErrorPanel error={synthError} />

        <div className="flex items-center gap-4">
          <Button size="lg" onClick={() => void onSynthesize()} disabled={!canSynth}>
            {synthesizing ? "Synthesizing…" : "Synthesize"}
          </Button>
          {synthesizing && (
            <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <span className="inline-block h-2 w-2 animate-ping rounded-full bg-primary" />
              Audio is rendering — 1–2 s on real hardware.
            </span>
          )}
        </div>

        {result && <ResultPanel result={result} text={trimmedText} />}
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function VoicePicker({
  voices,
  voiceId,
  onChange,
  error,
}: {
  voices: Voice[] | null;
  voiceId: string;
  onChange: (id: string) => void;
  error: HumanizedError | null;
}) {
  if (error) {
    return (
      <div className="flex flex-col gap-2">
        <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Voice
        </label>
        <ErrorPanel error={error} />
      </div>
    );
  }
  if (!voices) {
    return (
      <div className="flex flex-col gap-2">
        <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Voice
        </label>
        <div className="h-8 w-full animate-pulse rounded-md bg-muted" />
      </div>
    );
  }
  if (voices.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border p-6 text-center">
        <p className="font-display text-xl italic text-foreground">No voices yet.</p>
        <p className="mt-2 text-sm text-muted-foreground">
          Clone or design a voice first.
        </p>
        <div className="mt-4 flex items-center justify-center gap-2">
          <Link
            href="/clone"
            className="inline-flex h-8 items-center rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            Clone a voice →
          </Link>
        </div>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor="tts-voice"
        className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
      >
        Voice
      </label>
      <select
        id="tts-voice"
        value={voiceId}
        onChange={(e) => onChange(e.target.value)}
        className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
      >
        {voices.map((v) => (
          <option key={v.id} value={v.id}>
            {v.name} · {v.source}
          </option>
        ))}
      </select>
    </div>
  );
}

function ResultPanel({ result, text }: { result: SynthResult; text: string }) {
  const downloadName = useMemo(() => {
    const slug = text
      .slice(0, 40)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    return `${slug || "synth"}-${stamp}.wav`;
  }, [text]);

  return (
    <div className="rounded-md border border-border bg-card p-5">
      <div className="flex items-baseline justify-between">
        <p className="font-display text-lg italic text-foreground">Audio ready.</p>
        <dl className="flex gap-4 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
          {result.durationMs !== null && (
            <div>{(result.durationMs / 1000).toFixed(2)}s</div>
          )}
          {result.sampleRate !== null && <div>{result.sampleRate / 1000}kHz</div>}
          {result.generationTimeMs !== null && (
            <div>rendered in {result.generationTimeMs}ms</div>
          )}
          <div>{(result.sizeBytes / 1024).toFixed(0)} KB</div>
        </dl>
      </div>
      <audio src={result.url} controls className="mt-3 w-full" />
      <div className="mt-3 flex items-center gap-2">
        <a
          href={result.url}
          download={downloadName}
          className="inline-flex h-8 items-center rounded-lg border border-border bg-background px-3 text-sm hover:bg-muted"
        >
          Download .wav
        </a>
      </div>
    </div>
  );
}

function AdvancedParams(props: {
  sampleRate: number;
  setSampleRate: (n: number) => void;
  cfgValue: number;
  setCfgValue: (n: number) => void;
  temperature: number;
  setTemperature: (n: number) => void;
  changed: boolean;
  onReset: () => void;
}) {
  const {
    sampleRate,
    setSampleRate,
    cfgValue,
    setCfgValue,
    temperature,
    setTemperature,
    changed,
    onReset,
  } = props;
  return (
    <details className="group rounded-md border border-border bg-card/40 [&[open]]:bg-card">
      <summary className="flex cursor-pointer items-center justify-between px-4 py-3 text-sm select-none">
        <span className="flex items-baseline gap-3">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Advanced
          </span>
          {changed && (
            <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest text-primary">
              ● modified
            </span>
          )}
        </span>
        <span className="text-muted-foreground transition-transform group-open:rotate-90">
          ›
        </span>
      </summary>
      <div className="grid grid-cols-1 gap-5 border-t border-border/60 p-5 md:grid-cols-3">
        <div className="flex flex-col gap-2">
          <label
            htmlFor="tts-sr"
            className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
          >
            Sample rate
          </label>
          <select
            id="tts-sr"
            value={sampleRate}
            onChange={(e) => setSampleRate(Number(e.target.value))}
            className="h-9 rounded-md border border-border bg-background px-3 text-sm"
          >
            {SAMPLE_RATES.map((sr) => (
              <option key={sr} value={sr}>
                {sr / 1000} kHz
              </option>
            ))}
          </select>
          <span className="text-[11px] text-muted-foreground">
            Output WAV sample rate. 24 kHz is the model-native default.
          </span>
        </div>
        <NumSlider
          label="CFG value"
          value={cfgValue}
          min={1}
          max={3}
          step={0.1}
          format={(n) => n.toFixed(1)}
          onChange={setCfgValue}
          hint="Prompt adherence. Higher = more on-prompt but less natural."
        />
        <NumSlider
          label="Temperature"
          value={temperature}
          min={0}
          max={2}
          step={0.1}
          format={(n) => n.toFixed(1)}
          onChange={setTemperature}
          hint="Sampling variance. 0 is greedy; defaults to 1.0."
        />
        {changed && (
          <button
            type="button"
            onClick={onReset}
            className="self-start rounded-md border border-border px-3 py-1 text-xs text-muted-foreground hover:bg-muted md:col-span-3"
          >
            Reset to defaults
          </button>
        )}
      </div>
    </details>
  );
}

function NumSlider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (n: number) => string;
  onChange: (n: number) => void;
  hint?: string;
}) {
  const { label, value, min, max, step, format, onChange, hint } = props;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {label}
        </label>
        <span className="font-mono text-sm text-foreground">{format(value)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      {hint && <span className="text-[11px] text-muted-foreground">{hint}</span>}
    </div>
  );
}

function numFromHeader(v: string | null): number | null {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

