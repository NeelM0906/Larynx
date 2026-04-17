"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { PageShell } from "@/components/page-shell";
import { apiFetch, ApiError } from "@/lib/api-client";
import { getToken } from "@/lib/token";

const BASE_URL = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "";

type PhaseAIssue = { code: string; detail: string };
type PhaseAReport = {
  ok: boolean;
  num_clips: number;
  total_duration_s: number;
  sample_rates: Record<string, number>;
  issues: PhaseAIssue[];
};

type DatasetUploadResponse = { dataset_id: string; report: PhaseAReport };
type JobCreateResponse = { job_id: string };

type Step = "upload" | "validate" | "configure" | "watch";

type StateEvent = {
  step: number;
  loss_diff?: number;
  loss_stop?: number;
  lr?: number;
  epoch?: number;
};

type TerminalEvent = {
  state: "SUCCEEDED" | "FAILED" | "CANCELLED";
  voice_id: string | null;
  error_code: string | null;
  error_detail: string | null;
};

export default function FinetunePage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("upload");

  // Upload state.
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [report, setReport] = useState<PhaseAReport | null>(null);

  // Configure state.
  const [voiceName, setVoiceName] = useState("");
  const [loraRank, setLoraRank] = useState(32);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [numIters, setNumIters] = useState(1000);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Watch state.
  const [jobId, setJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [lastState, setLastState] = useState<StateEvent | null>(null);
  const [terminal, setTerminal] = useState<TerminalEvent | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // --- Upload -----------------------------------------------------------

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    try {
      const resp = await apiFetch<DatasetUploadResponse>("/v1/finetune/datasets", {
        method: "POST",
        body: form,
      });
      setDatasetId(resp.dataset_id);
      setReport(resp.report);
      setStep("validate");
    } catch (e) {
      if (e instanceof ApiError && e.body && typeof e.body === "object") {
        const body = e.body as { detail?: { code?: string; issues?: PhaseAIssue[] } };
        if (body.detail?.code === "dataset_invalid" && body.detail.issues) {
          setReport({
            ok: false,
            num_clips: 0,
            total_duration_s: 0,
            sample_rates: {},
            issues: body.detail.issues,
          });
          setStep("validate");
          return;
        }
        setUploadError(JSON.stringify(body.detail ?? body));
      } else {
        setUploadError(String(e));
      }
    } finally {
      setUploading(false);
    }
  }, [files]);

  // --- Submit -----------------------------------------------------------

  const handleSubmit = useCallback(async () => {
    if (!datasetId || !voiceName) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const resp = await apiFetch<JobCreateResponse>("/v1/finetune/jobs", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: datasetId,
          name: voiceName,
          config_overrides: {
            num_iters: numIters,
            max_steps: numIters,
            lora: { r: loraRank, alpha: loraAlpha },
          },
        }),
      });
      setJobId(resp.job_id);
      setStep("watch");
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  }, [datasetId, voiceName, loraRank, loraAlpha, numIters]);

  // --- Watch (SSE) ------------------------------------------------------

  useEffect(() => {
    if (step !== "watch" || !jobId) return;
    const controller = new AbortController();
    abortRef.current = controller;
    const token = getToken();
    const url = `${BASE_URL}/v1/finetune/jobs/${jobId}/logs`;

    (async () => {
      try {
        const res = await fetch(url, {
          signal: controller.signal,
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        });
        if (!res.ok || !res.body) {
          setTerminal({
            state: "FAILED",
            voice_id: null,
            error_code: "stream_connect_failed",
            error_detail: `HTTP ${res.status}`,
          });
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const parsed = parseSseBlock(rawEvent);
            if (!parsed) continue;
            if (parsed.event === "log") {
              setLogs((prev) => [...prev.slice(-499), parsed.data]);
            } else if (parsed.event === "state") {
              try {
                setLastState(JSON.parse(parsed.data) as StateEvent);
              } catch {
                /* ignore */
              }
            } else if (parsed.event === "terminal") {
              try {
                setTerminal(JSON.parse(parsed.data) as TerminalEvent);
              } catch {
                /* ignore */
              }
            }
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setTerminal({
            state: "FAILED",
            voice_id: null,
            error_code: "stream_error",
            error_detail: String(e),
          });
        }
      }
    })();

    return () => controller.abort();
  }, [step, jobId]);

  // Redirect to /library on success.
  useEffect(() => {
    if (terminal?.state === "SUCCEEDED" && terminal.voice_id) {
      const id = terminal.voice_id;
      const timeout = setTimeout(() => {
        router.push(`/library?voice=${encodeURIComponent(id)}`);
      }, 1500);
      return () => clearTimeout(timeout);
    }
  }, [terminal, router]);

  // --- Cancel ----------------------------------------------------------

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await apiFetch(`/v1/finetune/jobs/${jobId}`, { method: "DELETE" });
    } catch {
      /* swallow — the SSE stream will report the terminal state */
    }
  }, [jobId]);

  // --- Rendering -------------------------------------------------------

  return (
    <PageShell
      kicker="07 · Fine-tune"
      title="Give a voice your own sound."
      intro="Drop a ~5-minute clip and a transcript (or just the audio — we'll transcribe). Pick a name and a rank. Watch it train. Voice lands in your library."
    >
      <div className="flex flex-col gap-6">
        <Progress step={step} />

        {step === "upload" && (
          <UploadStep
            files={files}
            setFiles={setFiles}
            uploading={uploading}
            uploadError={uploadError}
            onUpload={handleUpload}
          />
        )}

        {step === "validate" && report && (
          <ValidateStep
            report={report}
            onBack={() => {
              setDatasetId(null);
              setReport(null);
              setStep("upload");
            }}
            onNext={() => setStep("configure")}
          />
        )}

        {step === "configure" && (
          <ConfigureStep
            voiceName={voiceName}
            setVoiceName={setVoiceName}
            loraRank={loraRank}
            setLoraRank={setLoraRank}
            loraAlpha={loraAlpha}
            setLoraAlpha={setLoraAlpha}
            numIters={numIters}
            setNumIters={setNumIters}
            submitting={submitting}
            submitError={submitError}
            onBack={() => setStep("validate")}
            onSubmit={handleSubmit}
          />
        )}

        {step === "watch" && jobId && (
          <WatchStep
            jobId={jobId}
            logs={logs}
            lastState={lastState}
            terminal={terminal}
            maxSteps={numIters}
            onCancel={handleCancel}
          />
        )}
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function parseSseBlock(block: string): { event: string; data: string; id?: string } | null {
  const lines = block.split("\n");
  const result: { event?: string; data?: string; id?: string } = {};
  const dataLines: string[] = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    const [key, ...rest] = line.split(":");
    const value = rest.join(":").replace(/^ /, "");
    if (key === "data") dataLines.push(value);
    else if (key === "event") result.event = value;
    else if (key === "id") result.id = value;
  }
  if (dataLines.length > 0) result.data = dataLines.join("\n");
  if (!result.event || result.data === undefined) return null;
  return { event: result.event, data: result.data, id: result.id };
}

// ---------------------------------------------------------------------------
// Step components — kept local because they share state + stay small.
// ---------------------------------------------------------------------------

function Progress({ step }: { step: Step }) {
  const steps: { id: Step; label: string }[] = [
    { id: "upload", label: "Upload" },
    { id: "validate", label: "Validate" },
    { id: "configure", label: "Configure" },
    { id: "watch", label: "Watch" },
  ];
  const currentIdx = steps.findIndex((s) => s.id === step);
  return (
    <ol className="flex gap-3 text-xs font-mono uppercase tracking-widest">
      {steps.map((s, i) => (
        <li
          key={s.id}
          className={`flex items-center gap-2 ${
            i <= currentIdx ? "text-foreground" : "text-muted-foreground"
          }`}
        >
          <span
            className={`inline-flex h-6 w-6 items-center justify-center rounded-full border ${
              i === currentIdx
                ? "border-primary bg-primary text-primary-foreground"
                : i < currentIdx
                ? "border-primary text-primary"
                : "border-border"
            }`}
          >
            {i + 1}
          </span>
          {s.label}
        </li>
      ))}
    </ol>
  );
}

function UploadStep(props: {
  files: File[];
  setFiles: (f: File[]) => void;
  uploading: boolean;
  uploadError: string | null;
  onUpload: () => void;
}) {
  const { files, setFiles, uploading, uploadError, onUpload } = props;
  return (
    <section className="flex flex-col gap-4">
      <label
        htmlFor="ft-files"
        className="cursor-pointer rounded-md border border-dashed border-primary/40 p-10 text-center transition hover:bg-muted/30"
      >
        <p className="font-display text-2xl text-primary">Drop audio + transcripts.jsonl</p>
        <p className="mt-2 text-sm text-muted-foreground">
          .wav / .flac / .mp3 files + an optional ``transcripts.jsonl``. If you skip the
          manifest we&rsquo;ll transcribe every clip for you during preparation.
        </p>
        <input
          id="ft-files"
          type="file"
          multiple
          accept=".wav,.flac,.mp3,.jsonl,application/json"
          className="hidden"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </label>
      {files.length > 0 && (
        <ul className="max-h-40 overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs font-mono">
          {files.map((f) => (
            <li key={`${f.name}-${f.size}`} className="flex justify-between gap-2">
              <span>{f.name}</span>
              <span className="text-muted-foreground">{(f.size / 1024).toFixed(1)} KB</span>
            </li>
          ))}
        </ul>
      )}
      {uploadError && (
        <p className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {uploadError}
        </p>
      )}
      <button
        type="button"
        disabled={files.length === 0 || uploading}
        onClick={onUpload}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
      >
        {uploading ? "Uploading…" : `Upload ${files.length || ""} file(s)`}
      </button>
    </section>
  );
}

function ValidateStep(props: {
  report: PhaseAReport;
  onBack: () => void;
  onNext: () => void;
}) {
  const { report, onBack, onNext } = props;
  return (
    <section className="flex flex-col gap-4">
      <div
        className={`rounded-md border p-4 text-sm ${
          report.ok
            ? "border-primary/40 bg-primary/5 text-foreground"
            : "border-destructive/40 bg-destructive/5 text-destructive"
        }`}
      >
        <p className="font-display text-lg">
          {report.ok ? "Dataset looks good." : "Dataset isn't ready yet."}
        </p>
        <dl className="mt-3 grid grid-cols-3 gap-3 text-xs font-mono uppercase tracking-widest">
          <div>
            <dt className="text-muted-foreground">Clips</dt>
            <dd className="text-2xl font-semibold">{report.num_clips}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Total duration</dt>
            <dd className="text-2xl font-semibold">
              {Math.round(report.total_duration_s)}s
            </dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Sample rates</dt>
            <dd className="text-sm font-semibold">
              {Object.entries(report.sample_rates)
                .map(([sr, n]) => `${n}×${sr}`)
                .join("  ")}
            </dd>
          </div>
        </dl>
      </div>
      {report.issues.length > 0 && (
        <ul className="max-h-60 overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs font-mono">
          {report.issues.map((i, idx) => (
            <li key={idx} className="py-1">
              <span className="mr-2 font-semibold text-destructive">{i.code}</span>
              {i.detail}
            </li>
          ))}
        </ul>
      )}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onBack}
          className="rounded-md border border-border px-4 py-2 text-sm"
        >
          Back
        </button>
        <button
          type="button"
          disabled={!report.ok}
          onClick={onNext}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          Configure training
        </button>
      </div>
    </section>
  );
}

function ConfigureStep(props: {
  voiceName: string;
  setVoiceName: (s: string) => void;
  loraRank: number;
  setLoraRank: (n: number) => void;
  loraAlpha: number;
  setLoraAlpha: (n: number) => void;
  numIters: number;
  setNumIters: (n: number) => void;
  submitting: boolean;
  submitError: string | null;
  onBack: () => void;
  onSubmit: () => void;
}) {
  const {
    voiceName,
    setVoiceName,
    loraRank,
    setLoraRank,
    loraAlpha,
    setLoraAlpha,
    numIters,
    setNumIters,
    submitting,
    submitError,
    onBack,
    onSubmit,
  } = props;
  return (
    <section className="flex flex-col gap-5">
      <label className="flex flex-col gap-2 text-sm">
        <span className="font-mono uppercase tracking-widest text-muted-foreground">
          Voice name
        </span>
        <input
          type="text"
          value={voiceName}
          onChange={(e) => setVoiceName(e.target.value)}
          placeholder="e.g. amy-natural"
          className="rounded-md border border-border bg-background px-3 py-2"
        />
      </label>
      <Slider
        label="LoRA rank"
        value={loraRank}
        min={8}
        max={64}
        step={8}
        onChange={setLoraRank}
        hint="Higher = more capacity but larger weights. 32 is the upstream default."
      />
      <Slider
        label="LoRA alpha"
        value={loraAlpha}
        min={8}
        max={64}
        step={8}
        onChange={setLoraAlpha}
        hint="Typically equal to rank. Scaling factor on the low-rank update."
      />
      <Slider
        label="Training iterations"
        value={numIters}
        min={100}
        max={2000}
        step={100}
        onChange={setNumIters}
        hint="Upstream default is 1000. Larger datasets want more iterations."
      />
      {submitError && (
        <p className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {submitError}
        </p>
      )}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onBack}
          className="rounded-md border border-border px-4 py-2 text-sm"
        >
          Back
        </button>
        <button
          type="button"
          disabled={submitting || !voiceName.trim()}
          onClick={onSubmit}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
        >
          {submitting ? "Starting…" : "Start training"}
        </button>
      </div>
    </section>
  );
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (n: number) => void;
  hint?: string;
}) {
  const { label, value, min, max, step, onChange, hint } = props;
  return (
    <label className="flex flex-col gap-2 text-sm">
      <div className="flex items-baseline justify-between">
        <span className="font-mono uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-base text-foreground">{value}</span>
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
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

function WatchStep(props: {
  jobId: string;
  logs: string[];
  lastState: StateEvent | null;
  terminal: TerminalEvent | null;
  maxSteps: number;
  onCancel: () => void;
}) {
  const { jobId, logs, lastState, terminal, maxSteps, onCancel } = props;
  const progress = lastState
    ? Math.min(1, Math.max(0, lastState.step / Math.max(1, maxSteps)))
    : 0;
  const logsRef = useRef<HTMLPreElement | null>(null);
  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs.length]);
  return (
    <section className="flex flex-col gap-4">
      <div className="rounded-md border border-border bg-muted/30 p-4">
        <div className="flex items-baseline justify-between text-xs font-mono uppercase tracking-widest text-muted-foreground">
          <span>Job {jobId.slice(0, 8)}</span>
          <span>
            step {lastState?.step ?? 0} / {maxSteps}
          </span>
        </div>
        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-border">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${(progress * 100).toFixed(1)}%` }}
          />
        </div>
        {lastState && (
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            loss/diff={lastState.loss_diff?.toFixed(3) ?? "—"} · loss/stop=
            {lastState.loss_stop?.toFixed(3) ?? "—"} · lr=
            {lastState.lr?.toExponential(2) ?? "—"}
          </p>
        )}
      </div>
      <pre
        ref={logsRef}
        className="h-72 overflow-auto rounded-md border border-border bg-black/90 p-3 text-xs text-emerald-200"
      >
        {logs.join("\n")}
      </pre>
      {terminal ? (
        <Terminal terminal={terminal} />
      ) : (
        <button
          type="button"
          onClick={onCancel}
          className="self-start rounded-md border border-destructive/40 px-4 py-2 text-sm text-destructive"
        >
          Cancel job
        </button>
      )}
    </section>
  );
}

function Terminal({ terminal }: { terminal: TerminalEvent }) {
  const cls =
    terminal.state === "SUCCEEDED"
      ? "border-primary/40 bg-primary/5 text-foreground"
      : "border-destructive/40 bg-destructive/10 text-destructive";
  return (
    <div className={`rounded-md border p-4 text-sm ${cls}`}>
      <p className="font-display text-lg">
        {terminal.state === "SUCCEEDED" && "Training complete."}
        {terminal.state === "FAILED" && "Training failed."}
        {terminal.state === "CANCELLED" && "Training cancelled."}
      </p>
      {terminal.state === "SUCCEEDED" && terminal.voice_id && (
        <p className="mt-2 font-mono text-xs">
          Voice id <span className="font-semibold">{terminal.voice_id}</span> is loaded and
          ready. Jumping to the library…
        </p>
      )}
      {(terminal.error_code || terminal.error_detail) && (
        <p className="mt-2 font-mono text-xs">
          {terminal.error_code && <strong>{terminal.error_code}: </strong>}
          {terminal.error_detail}
        </p>
      )}
    </div>
  );
}
