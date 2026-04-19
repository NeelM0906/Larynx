"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch, apiErrorFrom } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { getToken } from "@/lib/token";

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

// Pulled from packages/funasr_worker/.../language_router.py. "auto" is Nano-only.
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

type Mode = "upload" | "record";
type RecState = "idle" | "recording" | "stopped";

export default function TranscribePage() {
  const [mode, setMode] = useState<Mode>("upload");

  // Shared across modes — the actual blob we send to the gateway.
  const [audioBlob, setAudioBlob] = useState<{ blob: Blob; name: string } | null>(null);

  // Upload mode state.
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragHover, setDragHover] = useState(false);

  // Record mode state.
  const [recState, setRecState] = useState<RecState>("idle");
  const [recSeconds, setRecSeconds] = useState(0);
  const mediaRecRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const tickRef = useRef<number | null>(null);
  const [recError, setRecError] = useState<string | null>(null);

  // Shared form state.
  const [language, setLanguage] = useState("");
  const [hotwords, setHotwords] = useState("");
  const [itn, setItn] = useState(true);
  const [punctuate, setPunctuate] = useState(true);

  const [transcribing, setTranscribing] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [transcribeError, setTranscribeError] = useState<HumanizedError | null>(null);
  const [result, setResult] = useState<STTResponse | null>(null);

  const audioUrl = useMemo(() => {
    if (!audioBlob) return null;
    return URL.createObjectURL(audioBlob.blob);
  }, [audioBlob]);
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  // Cleanup on unmount — stop any active recording.
  useEffect(() => {
    return () => {
      if (tickRef.current !== null) window.clearInterval(tickRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (mediaRecRef.current && mediaRecRef.current.state !== "inactive") {
        mediaRecRef.current.stop();
      }
    };
  }, []);

  const fileTooLarge = audioBlob !== null && audioBlob.blob.size > MAX_FILE_BYTES;
  const canTranscribe = !transcribing && audioBlob !== null && !fileTooLarge;

  const onPickFile = useCallback((f: File | null) => {
    if (!f) return;
    setAudioBlob({ blob: f, name: f.name });
    setResult(null);
    setTranscribeError(null);
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

  const startRecording = useCallback(async () => {
    setRecError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mime = pickMimeType();
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const type = rec.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
        const stamp = new Date().toISOString().slice(11, 19).replace(/:/g, "-");
        setAudioBlob({ blob, name: `recording-${stamp}.${ext}` });
        setRecState("stopped");
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };
      rec.start();
      mediaRecRef.current = rec;
      setRecState("recording");
      setRecSeconds(0);
      tickRef.current = window.setInterval(() => setRecSeconds((s) => s + 1), 1000);
      setAudioBlob(null);
      setResult(null);
      setTranscribeError(null);
    } catch (e) {
      const name = (e as { name?: string }).name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        setRecError("Microphone access was denied. Check the browser permission prompt and try again.");
      } else if (name === "NotFoundError") {
        setRecError("No microphone found — plug one in and reload.");
      } else {
        setRecError((e as Error).message || "Couldn't start recording.");
      }
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    const rec = mediaRecRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  const resetRecording = useCallback(() => {
    setRecState("idle");
    setRecSeconds(0);
    setAudioBlob(null);
    setResult(null);
  }, []);

  const onTranscribe = useCallback(async () => {
    if (!canTranscribe || !audioBlob) return;
    setTranscribing(true);
    setTranscribeError(null);
    setResult(null);
    const largeUpload = audioBlob.blob.size > 5 * 1024 * 1024;
    setUploadPct(largeUpload ? 0 : null);
    try {
      const form = new FormData();
      form.append("file", audioBlob.blob, audioBlob.name);
      if (language) form.append("language", language);
      if (hotwords.trim()) form.append("hotwords", hotwords.trim());
      form.append("itn", itn ? "true" : "false");
      form.append("punctuate", punctuate ? "true" : "false");

      if (largeUpload) {
        // Poor-man's progress — XHR so we can see upload bytes.
        const resp = await xhrUpload<STTResponse>("/v1/stt", form, (pct) => setUploadPct(pct));
        setResult(resp);
      } else {
        const resp = await apiFetch<STTResponse>("/v1/stt", { method: "POST", body: form });
        setResult(resp);
      }
    } catch (e) {
      setTranscribeError(
        humanizeApiError(e, {
          413: "Audio file too large — trim or compress before uploading.",
          503: "STT worker is unavailable — try again in a moment.",
        }),
      );
    } finally {
      setTranscribing(false);
      setUploadPct(null);
    }
  }, [canTranscribe, audioBlob, language, hotwords, itn, punctuate]);

  return (
    <PageShell
      kicker="06 · Transcribe"
      title="Words from audio."
      intro="Drop a file or record the mic. Pick a language if you know it. Read the transcript."
    >
      <div className="flex flex-col gap-6">
        <ModeToggle mode={mode} setMode={setMode} disabled={transcribing || recState === "recording"} />

        {mode === "upload" ? (
          <UploadArea
            file={audioBlob}
            dragHover={dragHover}
            inputRef={inputRef}
            onPickFile={onPickFile}
            onDrop={onDrop}
            setDragHover={setDragHover}
            fileTooLarge={fileTooLarge}
          />
        ) : (
          <RecordArea
            recState={recState}
            recSeconds={recSeconds}
            recError={recError}
            audioUrl={audioUrl}
            audioName={audioBlob?.name ?? null}
            onStart={() => void startRecording()}
            onStop={stopRecording}
            onReset={resetRecording}
          />
        )}

        {mode === "upload" && audioUrl && (
          <audio src={audioUrl} controls className="w-full" />
        )}

        <Options
          language={language}
          setLanguage={setLanguage}
          hotwords={hotwords}
          setHotwords={setHotwords}
          itn={itn}
          setItn={setItn}
          punctuate={punctuate}
          setPunctuate={setPunctuate}
        />

        <ErrorPanel error={transcribeError} />

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

        {result && <ResultPanel result={result} />}
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function ModeToggle({
  mode,
  setMode,
  disabled,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
  disabled: boolean;
}) {
  return (
    <div className="inline-flex self-start rounded-md border border-border bg-card/40 p-0.5 font-mono text-[11px] uppercase tracking-widest">
      {(["upload", "record"] as const).map((m) => (
        <button
          key={m}
          type="button"
          disabled={disabled}
          onClick={() => setMode(m)}
          className={`h-8 min-w-24 rounded-sm px-3 transition-colors ${
            mode === m
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground disabled:opacity-50"
          }`}
        >
          {m === "upload" ? "Upload audio" : "Record"}
        </button>
      ))}
    </div>
  );
}

function UploadArea({
  file,
  dragHover,
  inputRef,
  onPickFile,
  onDrop,
  setDragHover,
  fileTooLarge,
}: {
  file: { blob: Blob; name: string } | null;
  dragHover: boolean;
  inputRef: React.MutableRefObject<HTMLInputElement | null>;
  onPickFile: (f: File | null) => void;
  onDrop: (e: React.DragEvent<HTMLLabelElement>) => void;
  setDragHover: (v: boolean) => void;
  fileTooLarge: boolean;
}) {
  const sizeKB = file ? (file.blob.size / 1024).toFixed(1) : "";
  const sizeMB = file ? (file.blob.size / (1024 * 1024)).toFixed(2) : "";
  const label = file ? (file.blob.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`) : "";
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
          {file ? "Replace audio" : "Drop audio"}
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
      {file && (
        <div
          className={`rounded-md border p-3 text-xs font-mono ${
            fileTooLarge
              ? "border-destructive/40 bg-destructive/10 text-destructive"
              : "border-border bg-muted/30"
          }`}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="truncate">{file.name}</span>
            <span className="shrink-0 text-muted-foreground">{label}</span>
          </div>
          {fileTooLarge && (
            <p className="mt-2">File exceeds 100 MB — split or compress before uploading.</p>
          )}
        </div>
      )}
    </>
  );
}

function RecordArea({
  recState,
  recSeconds,
  recError,
  audioUrl,
  audioName,
  onStart,
  onStop,
  onReset,
}: {
  recState: RecState;
  recSeconds: number;
  recError: string | null;
  audioUrl: string | null;
  audioName: string | null;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}) {
  const mm = Math.floor(recSeconds / 60)
    .toString()
    .padStart(2, "0");
  const ss = (recSeconds % 60).toString().padStart(2, "0");
  return (
    <div className="rounded-md border border-border bg-card/40 p-8 text-center">
      <div className="flex flex-col items-center gap-4">
        <button
          type="button"
          onClick={recState === "recording" ? onStop : onStart}
          className={`flex h-24 w-24 items-center justify-center rounded-full border-2 transition-all ${
            recState === "recording"
              ? "border-destructive bg-destructive/10 text-destructive animate-pulse"
              : "border-primary bg-primary/5 text-primary hover:bg-primary/10"
          }`}
          aria-label={recState === "recording" ? "Stop recording" : "Start recording"}
        >
          <span
            className={`inline-block ${
              recState === "recording" ? "h-6 w-6 rounded-sm bg-destructive" : "h-10 w-10 rounded-full bg-primary"
            }`}
          />
        </button>
        <div className="font-mono text-sm tracking-widest text-muted-foreground">
          {recState === "recording" && (
            <span className="text-destructive">● REC {mm}:{ss}</span>
          )}
          {recState === "idle" && <span>Press to record</span>}
          {recState === "stopped" && <span>Recorded {mm}:{ss}</span>}
        </div>
        {recError && (
          <p className="text-sm text-destructive" role="alert">{recError}</p>
        )}
        {recState === "stopped" && audioUrl && (
          <div className="w-full max-w-md">
            <audio src={audioUrl} controls className="w-full" />
            <div className="mt-3 flex items-center justify-center gap-2">
              <Button variant="ghost" size="sm" onClick={onReset}>
                Re-record
              </Button>
              {audioName && (
                <span className="font-mono text-[11px] text-muted-foreground truncate max-w-48">
                  {audioName}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Options({
  language,
  setLanguage,
  hotwords,
  setHotwords,
  itn,
  setItn,
  punctuate,
  setPunctuate,
}: {
  language: string;
  setLanguage: (v: string) => void;
  hotwords: string;
  setHotwords: (v: string) => void;
  itn: boolean;
  setItn: (v: boolean) => void;
  punctuate: boolean;
  setPunctuate: (v: boolean) => void;
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
          Leave on auto for English/Chinese/Japanese. Other languages route through MLT.
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
      <div className="flex flex-col gap-2 md:col-span-2 md:flex-row md:gap-6">
        <Toggle label="Inverse text normalisation" value={itn} onChange={setItn} />
        <Toggle label="Add punctuation" value={punctuate} onChange={setPunctuate} />
      </div>
    </div>
  );
}

function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="inline-flex items-center gap-2 text-sm text-muted-foreground">
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 accent-primary"
      />
      {label}
    </label>
  );
}

function ResultPanel({ result }: { result: STTResponse }) {
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

function pickMimeType(): string | null {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const m of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) {
      return m;
    }
  }
  return null;
}

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
