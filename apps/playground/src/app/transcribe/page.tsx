"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch, apiErrorFrom } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { getToken } from "@/lib/token";
import { startSTTStream, type STTStreamHandle } from "@/lib/stt-stream";

const MAX_FILE_BYTES = 100 * 1024 * 1024; // gateway body limit
const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".webm", ".ogg"];
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.concat([
  "audio/wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/flac",
  "audio/x-flac",
  "audio/mp4",
  "audio/m4a",
  "audio/webm",
  "audio/ogg",
]).join(",");

const LANGUAGES: Array<{ code: string; label: string; model: "nano" | "mlt" }> = [
  { code: "", label: "Auto-detect (Nano)", model: "nano" },
  { code: "en", label: "English", model: "nano" },
  { code: "zh", label: "Chinese", model: "nano" },
  { code: "ja", label: "Japanese", model: "nano" },
  { code: "yue", label: "Cantonese", model: "mlt" },
  { code: "ko", label: "Korean", model: "mlt" },
  { code: "vi", label: "Vietnamese", model: "mlt" },
  { code: "id", label: "Indonesian", model: "mlt" },
  { code: "th", label: "Thai", model: "mlt" },
  { code: "ms", label: "Malay", model: "mlt" },
  { code: "tl", label: "Filipino", model: "mlt" },
  { code: "ar", label: "Arabic", model: "mlt" },
  { code: "hi", label: "Hindi", model: "mlt" },
  { code: "nl", label: "Dutch", model: "mlt" },
  { code: "pl", label: "Polish", model: "mlt" },
  { code: "pt", label: "Portuguese", model: "mlt" },
  { code: "sv", label: "Swedish", model: "mlt" },
  { code: "fi", label: "Finnish", model: "mlt" },
  { code: "el", label: "Greek", model: "mlt" },
  { code: "hu", label: "Hungarian", model: "mlt" },
  { code: "cs", label: "Czech", model: "mlt" },
  { code: "da", label: "Danish", model: "mlt" },
  { code: "ro", label: "Romanian", model: "mlt" },
];

interface STTResponse {
  text: string;
  language: string;
  model_used: "nano" | "mlt";
  duration_ms: number;
  processing_ms: number;
  punctuated: boolean;
}

type Mode = "upload" | "live";

export default function TranscribePage() {
  const [mode, setMode] = useState<Mode>("live");
  const [language, setLanguage] = useState("en");
  const [hotwords, setHotwords] = useState("");

  return (
    <PageShell
      kicker="06 · Transcribe"
      title="Words from audio."
      intro="Drop a file to transcribe a whole clip, or go live and see Fun-ASR's partials as you speak."
    >
      <div className="flex flex-col gap-6">
        <ModeToggle mode={mode} setMode={setMode} />
        <LanguageHotwords
          language={language}
          setLanguage={setLanguage}
          hotwords={hotwords}
          setHotwords={setHotwords}
        />
        {mode === "upload" ? (
          <UploadFlow language={language} hotwords={hotwords} />
        ) : (
          <LiveFlow language={language} hotwords={hotwords} />
        )}
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function ModeToggle({
  mode,
  setMode,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
}) {
  return (
    <div className="inline-flex self-start rounded-md border border-border bg-card/40 p-0.5 font-mono text-[11px] uppercase tracking-widest">
      {(
        [
          { v: "live", label: "Live" },
          { v: "upload", label: "Upload audio" },
        ] as const
      ).map((m) => (
        <button
          key={m.v}
          type="button"
          onClick={() => setMode(m.v)}
          className={`h-8 min-w-24 rounded-sm px-3 transition-colors ${
            mode === m.v
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

function LanguageHotwords({
  language,
  setLanguage,
  hotwords,
  setHotwords,
}: {
  language: string;
  setLanguage: (v: string) => void;
  hotwords: string;
  setHotwords: (v: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
      <div className="flex flex-col gap-2">
        <label
          htmlFor="stt-language"
          className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
        >
          Language
        </label>
        <select
          id="stt-language"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
        >
          {LANGUAGES.map((l) => (
            <option key={l.code || "auto"} value={l.code}>
              {l.label} {l.model === "mlt" ? "· MLT" : ""}
            </option>
          ))}
        </select>
        <span className="text-[11px] text-muted-foreground">
          Defaults to English. Switch to auto for mixed-language audio; other languages route through MLT.
        </span>
      </div>
      <div className="flex flex-col gap-2">
        <label
          htmlFor="stt-hotwords"
          className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
        >
          Hotwords
        </label>
        <input
          id="stt-hotwords"
          type="text"
          value={hotwords}
          onChange={(e) => setHotwords(e.target.value)}
          placeholder="comma,separated,domain,terms"
          className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50"
        />
        <span className="text-[11px] text-muted-foreground">
          Optional — bias Fun-ASR toward proper nouns or jargon.
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload flow — POST /v1/stt with a whole file
// ---------------------------------------------------------------------------

function UploadFlow({
  language,
  hotwords,
}: {
  language: string;
  hotwords: string;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [audioBlob, setAudioBlob] = useState<{ blob: Blob; name: string } | null>(null);
  const [dragHover, setDragHover] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState<HumanizedError | null>(null);
  const [result, setResult] = useState<STTResponse | null>(null);

  const audioUrl = useMemo(() => (audioBlob ? URL.createObjectURL(audioBlob.blob) : null), [
    audioBlob,
  ]);
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const fileTooLarge = audioBlob !== null && audioBlob.blob.size > MAX_FILE_BYTES;
  const canTranscribe = !transcribing && audioBlob !== null && !fileTooLarge;

  const onPickFile = useCallback((f: File | null) => {
    if (!f) return;
    setAudioBlob({ blob: f, name: f.name });
    setResult(null);
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

  const onTranscribe = useCallback(async () => {
    if (!canTranscribe || !audioBlob) return;
    setTranscribing(true);
    setError(null);
    setResult(null);
    const largeUpload = audioBlob.blob.size > 5 * 1024 * 1024;
    setUploadPct(largeUpload ? 0 : null);
    try {
      const form = new FormData();
      form.append("file", audioBlob.blob, audioBlob.name);
      if (language) form.append("language", language);
      if (hotwords.trim()) form.append("hotwords", hotwords.trim());

      const resp = largeUpload
        ? await xhrUpload<STTResponse>("/v1/stt", form, (pct) => setUploadPct(pct))
        : await apiFetch<STTResponse>("/v1/stt", { method: "POST", body: form });
      setResult(resp);
    } catch (e) {
      setError(
        humanizeApiError(e, {
          413: "Audio file too large — trim or compress before uploading.",
          503: "STT worker is unavailable — try again in a moment.",
        }),
      );
    } finally {
      setTranscribing(false);
      setUploadPct(null);
    }
  }, [canTranscribe, audioBlob, language, hotwords]);

  return (
    <>
      <label
        htmlFor="transcribe-file"
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
          {audioBlob ? "Replace audio" : "Drop audio"}
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          .wav · .mp3 · .flac · .m4a · .webm · .ogg — max 100 MB.
        </p>
        <input
          id="transcribe-file"
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
        />
      </label>
      {audioBlob && (
        <FileMetaCard file={audioBlob} tooLarge={fileTooLarge} />
      )}
      {audioUrl && <audio src={audioUrl} controls className="w-full" />}

      <ErrorPanel error={error} />

      <div className="flex items-center gap-4">
        <Button size="lg" onClick={() => void onTranscribe()} disabled={!canTranscribe}>
          {transcribing ? "Transcribing…" : "Transcribe"}
        </Button>
        {transcribing && (
          <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <span className="inline-block h-2 w-2 animate-ping rounded-full bg-primary" />
            {uploadPct !== null && uploadPct < 100
              ? `Uploading ${uploadPct}%`
              : "Running Fun-ASR."}
          </span>
        )}
      </div>

      {result && <UploadResultPanel result={result} />}
    </>
  );
}

function FileMetaCard({
  file,
  tooLarge,
}: {
  file: { blob: Blob; name: string };
  tooLarge: boolean;
}) {
  const sizeKB = (file.blob.size / 1024).toFixed(1);
  const sizeMB = (file.blob.size / (1024 * 1024)).toFixed(2);
  const label = file.blob.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
  return (
    <div
      className={`rounded-md border p-3 text-xs font-mono ${
        tooLarge
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-border bg-muted/30"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate">{file.name}</span>
        <span className="shrink-0 text-muted-foreground">{label}</span>
      </div>
      {tooLarge && (
        <p className="mt-2">File exceeds 100 MB — split or compress before uploading.</p>
      )}
    </div>
  );
}

function UploadResultPanel({ result }: { result: STTResponse }) {
  const [copied, setCopied] = useState(false);
  const rtf = result.duration_ms > 0 ? result.processing_ms / result.duration_ms : null;
  const wordCount = result.text.trim() ? result.text.trim().split(/\s+/).length : 0;

  const downloadName = useMemo(() => {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    return `transcript-${stamp}.txt`;
  }, []);
  const blobUrl = useMemo(
    () => URL.createObjectURL(new Blob([result.text], { type: "text/plain;charset=utf-8" })),
    [result.text],
  );
  useEffect(() => {
    return () => URL.revokeObjectURL(blobUrl);
  }, [blobUrl]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard permission denied — ignore.
    }
  };

  return (
    <div className="rounded-md border border-border bg-card p-5">
      <div className="flex items-baseline justify-between">
        <p className="font-display text-lg italic text-foreground">Transcript.</p>
        <dl className="flex flex-wrap gap-4 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
          <div>{result.language || "unknown"}</div>
          <div>{result.model_used}</div>
          <div>{wordCount} words</div>
          {result.duration_ms > 0 && <div>{(result.duration_ms / 1000).toFixed(2)}s audio</div>}
          {rtf !== null && <div>{rtf.toFixed(2)}× rtf</div>}
        </dl>
      </div>
      <textarea
        value={result.text}
        readOnly
        rows={8}
        className="mt-3 w-full resize-y rounded-md border border-border bg-background px-3 py-2 font-mono text-sm leading-relaxed text-foreground focus:outline-none"
      />
      {result.text.trim() === "" && (
        <p className="mt-2 text-[11px] text-muted-foreground">
          No speech detected in the clip.
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => void onCopy()}>
          {copied ? "Copied" : "Copy"}
        </Button>
        <a
          href={blobUrl}
          download={downloadName}
          className="inline-flex h-7 items-center rounded-[min(var(--radius-md),12px)] border border-border bg-background px-2.5 text-[0.8rem] hover:bg-muted"
        >
          Download .txt
        </a>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live flow — WS /v1/stt/stream
// ---------------------------------------------------------------------------

interface LiveSegment {
  id: string;
  text: string;
  partial: boolean;
  language?: string;
}

function LiveFlow({ language, hotwords }: { language: string; hotwords: string }) {
  const [running, setRunning] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [segments, setSegments] = useState<LiveSegment[]>([]);
  const [error, setError] = useState<HumanizedError | null>(null);
  const [permDenied, setPermDenied] = useState(false);
  const handleRef = useRef<STTStreamHandle | null>(null);
  const pendingIdRef = useRef<string | null>(null);

  const finalText = useMemo(
    () =>
      segments
        .filter((s) => !s.partial)
        .map((s) => s.text.trim())
        .filter(Boolean)
        .join(" "),
    [segments],
  );
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    return () => {
      const h = handleRef.current;
      handleRef.current = null;
      if (h) void h.stop();
    };
  }, []);

  const upsertPartial = useCallback((text: string) => {
    setSegments((prev) => {
      const id = pendingIdRef.current;
      if (id) {
        const idx = prev.findIndex((s) => s.id === id);
        if (idx !== -1) {
          const next = prev.slice();
          next[idx] = { ...next[idx], text, partial: true };
          return next;
        }
      }
      const newId = `p-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      pendingIdRef.current = newId;
      return [...prev, { id: newId, text, partial: true }];
    });
  }, []);

  const finaliseSegment = useCallback(
    (text: string, language?: string) => {
      setSegments((prev) => {
        const id = pendingIdRef.current;
        if (id) {
          const idx = prev.findIndex((s) => s.id === id);
          if (idx !== -1) {
            const next = prev.slice();
            next[idx] = { ...next[idx], text, partial: false, language };
            pendingIdRef.current = null;
            return next;
          }
        }
        pendingIdRef.current = null;
        return [
          ...prev,
          {
            id: `f-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            text,
            partial: false,
            language,
          },
        ];
      });
    },
    [],
  );

  const onStart = useCallback(async () => {
    setError(null);
    setPermDenied(false);
    setSegments([]);
    pendingIdRef.current = null;
    try {
      const parsedHotwords = hotwords
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const handle = await startSTTStream({
        language: language || undefined,
        hotwords: parsedHotwords,
        onEvent: (ev) => {
          switch (ev.type) {
            case "speech_start":
              setSpeaking(true);
              break;
            case "speech_end":
              setSpeaking(false);
              break;
            case "partial":
              upsertPartial(ev.text);
              break;
            case "final":
              finaliseSegment(ev.punctuated_text || ev.text, ev.language);
              break;
            case "error":
              setError({
                headline: `Gateway: ${ev.code}`,
                detail: ev.message,
                raw: JSON.stringify(ev, null, 2),
              });
              break;
            case "heartbeat":
              break;
          }
        },
        onClose: ({ code, reason, wasAccepted }) => {
          setRunning(false);
          setSpeaking(false);
          if (code !== 1000 && code !== 1005 && wasAccepted === false) {
            setError({
              headline: "Streaming connection closed unexpectedly",
              detail: `code=${code}${reason ? ` · ${reason}` : ""}`,
              raw: JSON.stringify({ code, reason, wasAccepted }, null, 2),
            });
          }
        },
        onError: ({ message, url }) => {
          setError({
            headline: "Streaming connection error",
            detail: message,
            raw: `Failed to open ${url}`,
          });
        },
      });
      handleRef.current = handle;
      setRunning(true);
    } catch (e) {
      const name = (e as { name?: string }).name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        setPermDenied(true);
      } else if (name === "NotFoundError") {
        setError({
          headline: "No microphone found",
          detail: "Plug in a mic and reload the page.",
          raw: String(e),
        });
      } else {
        setError({
          headline: "Couldn't start streaming transcription",
          detail: (e as Error).message,
          raw: String(e),
        });
      }
    }
  }, [language, hotwords, upsertPartial, finaliseSegment]);

  const onStop = useCallback(async () => {
    const h = handleRef.current;
    handleRef.current = null;
    if (h) await h.stop();
    setRunning(false);
    setSpeaking(false);
  }, []);

  const onCopy = async () => {
    if (!finalText) return;
    try {
      await navigator.clipboard.writeText(finalText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  if (permDenied) {
    return (
      <div className="rounded-md border border-border bg-card/40 p-8 text-center">
        <p className="font-display text-xl italic text-foreground">Need mic access.</p>
        <p className="mt-2 text-sm text-muted-foreground">
          The browser declined mic access. Allow microphone for this origin and try again.
        </p>
        <div className="mt-4">
          <Button onClick={() => void onStart()}>Request again</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        {!running ? (
          <Button size="lg" onClick={() => void onStart()}>
            Start live transcription
          </Button>
        ) : (
          <Button size="lg" variant="destructive" onClick={() => void onStop()}>
            Stop
          </Button>
        )}
        <StatePill running={running} speaking={speaking} />
      </div>

      <ErrorPanel error={error} />

      <LiveTranscript segments={segments} />

      {finalText && (
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => void onCopy()}>
            {copied ? "Copied" : "Copy final transcript"}
          </Button>
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            {finalText.split(/\s+/).filter(Boolean).length} words
          </span>
        </div>
      )}
    </div>
  );
}

function StatePill({ running, speaking }: { running: boolean; speaking: boolean }) {
  const tone = !running
    ? "border-border bg-muted/40 text-muted-foreground"
    : speaking
      ? "border-primary/60 bg-primary text-primary-foreground"
      : "border-primary/60 bg-primary/10 text-primary";
  const label = !running ? "Idle" : speaking ? "Speaking" : "Listening";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px] uppercase tracking-widest ${tone}`}
    >
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      {label}
    </span>
  );
}

function LiveTranscript({ segments }: { segments: LiveSegment[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [segments]);
  if (segments.length === 0) {
    return (
      <div className="flex min-h-32 items-center justify-center rounded-md border border-dashed border-border p-8 text-sm text-muted-foreground">
        Start streaming and Fun-ASR partials will land here as you speak.
      </div>
    );
  }
  return (
    <div
      ref={ref}
      className="flex max-h-96 flex-col gap-2 overflow-y-auto rounded-md border border-border bg-card/40 p-5 leading-relaxed"
    >
      {segments.map((s) => (
        <p
          key={s.id}
          className={`text-sm ${s.partial ? "text-muted-foreground italic" : "text-foreground"}`}
        >
          {s.text || <span className="text-muted-foreground/60">…</span>}
          {s.partial && <span className="ml-1 animate-pulse text-primary">▏</span>}
        </p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// XHR upload helper (used only by the upload flow for >5 MB files)
// ---------------------------------------------------------------------------

function xhrUpload<T>(
  path: string,
  form: FormData,
  onProgress: (pct: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const BASE_URL = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";
    const token = getToken();
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}${path}`);
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      const ok = xhr.status >= 200 && xhr.status < 300;
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = xhr.responseText;
      }
      if (ok) {
        resolve(body as T);
      } else {
        reject(apiErrorFrom(xhr.status, xhr.statusText, body));
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(form);
  });
}
