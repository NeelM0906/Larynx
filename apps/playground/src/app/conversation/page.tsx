"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/error-panel";
import { apiFetch } from "@/lib/api-client";
import { humanizeApiError, type HumanizedError } from "@/lib/errors";
import { getToken, invalidateToken } from "@/lib/token";
import { startCapture, type AudioCaptureHandle } from "@/lib/audio-capture";
import { AudioPlayback } from "@/lib/conversation/audio-playback";

const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const RECONNECT_MAX = 5;
const RECONNECT_BASE_MS = 500;

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

type SessionState = "idle" | "user_speaking" | "llm_generating" | "tts_speaking";

type TurnRole = "user" | "assistant";
interface Turn {
  id: string;
  role: TurnRole;
  text: string;
  partial?: boolean;
  interrupted?: boolean;
  sessionEpoch: number; // bumped each reconnect so prior turns are labelled
}

type ConnState = "disconnected" | "connecting" | "open" | "reconnecting" | "error";

function resolveGatewayWs(): string {
  const http = process.env.NEXT_PUBLIC_GATEWAY_URL;
  if (http) return http.replace(/^http/, "ws");
  if (typeof window === "undefined") return "";
  return window.location.origin.replace(/^http/, "ws");
}

export default function ConversationPage() {
  const [voices, setVoices] = useState<Voice[] | null>(null);
  const [voicesError, setVoicesError] = useState<HumanizedError | null>(null);
  const [voiceId, setVoiceId] = useState<string>("");

  const [connState, setConnState] = useState<ConnState>("disconnected");
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [micOn, setMicOn] = useState(false);
  const [permDenied, setPermDenied] = useState(false);
  const [error, setError] = useState<HumanizedError | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<AudioCaptureHandle | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const sessionEpochRef = useRef<number>(0);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const lastConfigRef = useRef<{ voiceId: string } | null>(null);

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

  useEffect(() => {
    return () => {
      stopCapture();
      stopPlayback();
      closeSocket();
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopCapture = useCallback(() => {
    const cap = captureRef.current;
    captureRef.current = null;
    if (cap) void cap.stop();
    setMicOn(false);
  }, []);

  const stopPlayback = useCallback(() => {
    const pb = playbackRef.current;
    playbackRef.current = null;
    if (pb) void pb.dispose();
  }, []);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      try {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "session.end" }));
        }
      } catch {
        // ignore
      }
      try {
        ws.close();
      } catch {
        // ignore
      }
    }
  }, []);

  const upsertTurn = useCallback((turn: Turn) => {
    setTurns((prev) => {
      const idx = prev.findIndex(
        (t) => t.id === turn.id && t.sessionEpoch === turn.sessionEpoch,
      );
      if (idx === -1) return [...prev, turn];
      const next = prev.slice();
      next[idx] = { ...prev[idx], ...turn };
      return next;
    });
  }, []);

  const appendAssistantDelta = useCallback((delta: string) => {
    setTurns((prev) => {
      const epoch = sessionEpochRef.current;
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.partial && last.sessionEpoch === epoch) {
        const next = prev.slice();
        next[next.length - 1] = { ...last, text: last.text + delta };
        return next;
      }
      return [
        ...prev,
        {
          id: `assistant-${epoch}-${Date.now()}`,
          role: "assistant",
          text: delta,
          partial: true,
          sessionEpoch: epoch,
        },
      ];
    });
  }, []);

  const finaliseAssistant = useCallback(() => {
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.partial) {
        const next = prev.slice();
        next[next.length - 1] = { ...last, partial: false };
        return next;
      }
      return prev;
    });
  }, []);

  const markAssistantInterrupted = useCallback(() => {
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.partial) {
        const next = prev.slice();
        next[next.length - 1] = { ...last, partial: false, interrupted: true };
        return next;
      }
      return prev;
    });
  }, []);

  const handleEvent = useCallback(
    (ev: Record<string, unknown>) => {
      const type = ev.type as string | undefined;
      if (!type) return;

      if (type === "session.status") {
        setSessionState((ev.state as SessionState) ?? "idle");
        return;
      }
      if (type === "input.speech_start" || type === "input.speech_end") {
        // Display is driven by session.status; nothing else to do here.
        return;
      }
      if (type === "transcript.partial") {
        const text = (ev.text as string) ?? "";
        const ordinal = (ev.utterance_ordinal as number) ?? 0;
        upsertTurn({
          id: `user-${ordinal}`,
          role: "user",
          text,
          partial: true,
          sessionEpoch: sessionEpochRef.current,
        });
        return;
      }
      if (type === "transcript.final") {
        const text = (ev.text as string) ?? "";
        const ordinal = (ev.utterance_ordinal as number) ?? 0;
        upsertTurn({
          id: `user-${ordinal}`,
          role: "user",
          text,
          partial: false,
          sessionEpoch: sessionEpochRef.current,
        });
        return;
      }
      if (type === "response.text_delta") {
        const delta = (ev.delta as string) ?? "";
        appendAssistantDelta(delta);
        return;
      }
      if (type === "response.done") {
        finaliseAssistant();
        return;
      }
      if (type === "interrupt") {
        void playbackRef.current?.interrupt();
        markAssistantInterrupted();
        return;
      }
      if (type === "error") {
        const code = (ev.code as string) ?? "error";
        const message = (ev.message as string) ?? "Unknown error from gateway.";
        setError({
          headline: `Gateway: ${code}`,
          detail: message,
          raw: JSON.stringify(ev, null, 2),
        });
        return;
      }
    },
    [upsertTurn, appendAssistantDelta, finaliseAssistant, markAssistantInterrupted],
  );

  const openSession = useCallback(
    async (opts: { reconnect?: boolean } = {}) => {
      const token = getToken();
      if (!token) {
        setError({
          headline: "No token — sign in first.",
          raw: "No larynx.token in localStorage.",
        });
        return;
      }
      const url = `${resolveGatewayWs()}/v1/conversation?token=${encodeURIComponent(token)}`;
      setConnState(opts.reconnect ? "reconnecting" : "connecting");
      setError(null);

      // Fresh playback context on every (re)connect.
      stopPlayback();
      playbackRef.current = new AudioPlayback(OUTPUT_RATE);
      await playbackRef.current.init();

      let opened = false;
      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        setConnState("error");
        setError({
          headline: "Couldn't construct the conversation WebSocket",
          detail: (e as Error).message,
          raw: `URL: ${url}\n${String(e)}`,
        });
        return;
      }
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        opened = true;
        setConnState("open");
        reconnectAttemptsRef.current = 0;
        const cfg = {
          type: "config",
          voice_id: voiceId || undefined,
          // Force English — the STT auto-detect misclassifies short
          // English utterances and filler noise as Chinese tokens.
          language: "en",
          input_sample_rate: INPUT_RATE,
          output_sample_rate: OUTPUT_RATE,
          speech_end_silence_ms: 300,
          partial_interval_ms: 720,
        };
        lastConfigRef.current = { voiceId };
        ws.send(JSON.stringify(cfg));
      };

      ws.onmessage = (msg) => {
        if (typeof msg.data === "string") {
          try {
            const parsed = JSON.parse(msg.data) as Record<string, unknown>;
            handleEvent(parsed);
          } catch {
            // ignore — non-JSON text is not part of the protocol
          }
        } else if (msg.data instanceof ArrayBuffer) {
          playbackRef.current?.enqueue(msg.data);
        }
      };

      ws.onerror = (ev) => {
        setConnState("error");
        // Native `error` events on WebSocket carry no useful detail; the
        // immediately-following `close` is what exposes `code` + `reason`.
        // We leave the humanised surface for onclose to build.
        if (typeof console !== "undefined") {
          console.error("[conversation] WebSocket error", { url, event: ev });
        }
      };

      ws.onclose = (ev) => {
        if (wsRef.current !== ws) {
          return; // superseded — ignore
        }
        wsRef.current = null;
        const neverOpened = !opened;
        // Clean close initiated by us (session.end) — no reconnect.
        if (ev.code === 1000 || ev.code === 1001) {
          setConnState("disconnected");
          return;
        }
        // Policy-violation close = gateway rejected the bearer token.
        // Clear stored token so AuthGate reopens rather than spinning on
        // reconnects that will all fail.
        if (ev.code === 1008) {
          invalidateToken("ws-rejected");
          setConnState("disconnected");
          reconnectAttemptsRef.current = RECONNECT_MAX;
          const cap = captureRef.current;
          captureRef.current = null;
          if (cap) void cap.stop();
          setMicOn(false);
          return;
        }
        // If the socket never reached OPEN, the handshake itself failed —
        // surface a visible error with the URL + close code so testers
        // can tell why instead of seeing a mute "Connection error" pill.
        if (neverOpened) {
          setConnState("error");
          setError({
            headline: "Couldn't open the conversation WebSocket.",
            detail: `Closed before handshake — code=${ev.code}${ev.reason ? ` · ${ev.reason}` : ""}.`,
            raw: `URL: ${url}\ncode: ${ev.code}\nreason: ${ev.reason || "(none)"}\nwasClean: ${ev.wasClean}`,
          });
          reconnectAttemptsRef.current = RECONNECT_MAX;
          const cap = captureRef.current;
          captureRef.current = null;
          if (cap) void cap.stop();
          setMicOn(false);
          return;
        }
        // Auto-reconnect with backoff while we still want to be talking.
        if (micOnRef.current && reconnectAttemptsRef.current < RECONNECT_MAX) {
          reconnectAttemptsRef.current += 1;
          setConnState("reconnecting");
          sessionEpochRef.current += 1;
          const delay = RECONNECT_BASE_MS * 2 ** (reconnectAttemptsRef.current - 1);
          reconnectTimerRef.current = window.setTimeout(() => {
            void openSession({ reconnect: true });
          }, delay);
        } else {
          setConnState("disconnected");
        }
      };
    },
    [voiceId, handleEvent, stopPlayback],
  );

  // Mirror micOn into a ref so onclose can read current intent without a stale closure.
  const micOnRef = useRef(false);
  useEffect(() => {
    micOnRef.current = micOn;
  }, [micOn]);

  const startConversation = useCallback(async () => {
    setError(null);
    setPermDenied(false);
    try {
      const cap = await startCapture((pcm) => {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm);
      });
      captureRef.current = cap;
      setMicOn(true);
      await openSession();
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
          headline: "Couldn't start conversation",
          detail: (e as Error).message,
          raw: String(e),
        });
      }
      stopCapture();
      stopPlayback();
    }
  }, [openSession, stopCapture, stopPlayback]);

  const endConversation = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectAttemptsRef.current = RECONNECT_MAX; // block auto-reconnect
    closeSocket();
    stopCapture();
    stopPlayback();
    setSessionState("idle");
  }, [closeSocket, stopCapture, stopPlayback]);

  const stopAiPlayback = useCallback(() => {
    void playbackRef.current?.interrupt();
  }, []);

  const pillLabel = useMemo(() => {
    if (connState === "connecting") return "Connecting…";
    if (connState === "reconnecting") return "Reconnecting…";
    if (connState === "disconnected") return "Not connected";
    if (connState === "error") return "Connection error";
    switch (sessionState) {
      case "idle":
        return "Listening";
      case "user_speaking":
        return "Listening · you";
      case "llm_generating":
        return "Thinking";
      case "tts_speaking":
        return "Speaking";
    }
  }, [connState, sessionState]);

  const pillTone = useMemo(() => {
    if (connState === "reconnecting" || connState === "error") return "warn";
    if (connState !== "open") return "muted";
    if (sessionState === "llm_generating") return "primary";
    if (sessionState === "tts_speaking") return "accent";
    if (sessionState === "user_speaking") return "accent";
    return "muted";
  }, [connState, sessionState]);

  if (permDenied) {
    return (
      <PageShell
        kicker="05 · Conversation"
        title="Need mic access."
        intro="Allow microphone access for this origin and try again."
      >
        <div className="rounded-md border border-border bg-card/40 p-8 text-center">
          <p className="text-sm text-muted-foreground">
            The browser declined mic access. Open the site settings, flip mic to
            Allow, then retry.
          </p>
          <div className="mt-4">
            <Button onClick={() => void startConversation()}>Request again</Button>
          </div>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell
      kicker="05 · Conversation"
      title="Talk to the gateway."
      intro="Full-duplex mic → STT → LLM → TTS. Starting the session opens the WebSocket; speaking over the AI triggers a server-side barge-in."
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <VoicePicker
            voices={voices}
            error={voicesError}
            value={voiceId}
            onChange={setVoiceId}
            disabled={micOn}
          />
          <StatePill label={pillLabel} tone={pillTone} />
        </div>

        {connState === "reconnecting" && (
          <div className="rounded-md border border-primary/40 bg-primary/5 p-3 text-sm text-primary">
            Connection lost, reconnecting… ({reconnectAttemptsRef.current}/{RECONNECT_MAX})
          </div>
        )}

        <Transcript turns={turns} currentEpoch={sessionEpochRef.current} />

        <ErrorPanel error={error} />

        <Controls
          micOn={micOn}
          sessionState={sessionState}
          connState={connState}
          onStart={() => void startConversation()}
          onEnd={endConversation}
          onStopAi={stopAiPlayback}
        />

        <p className="text-[11px] text-muted-foreground">
          Mic audio streams continuously while the mic is on — the server
          segments utterances via VAD. &ldquo;Stop AI&rdquo; silences local
          playback only; to force the AI to yield, just start speaking.
        </p>
      </div>
    </PageShell>
  );
}

// ---------------------------------------------------------------------------

function VoicePicker({
  voices,
  error,
  value,
  onChange,
  disabled,
}: {
  voices: Voice[] | null;
  error: HumanizedError | null;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
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
  return (
    <div className="flex min-w-60 flex-col gap-2">
      <label
        htmlFor="conv-voice"
        className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
      >
        Voice
      </label>
      <select
        id="conv-voice"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || !voices}
        className="h-9 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/50 disabled:opacity-60"
      >
        <option value="">Default voice</option>
        {voices?.map((v) => (
          <option key={v.id} value={v.id}>
            {v.name} · {v.source}
          </option>
        ))}
      </select>
      <span className="text-[11px] text-muted-foreground">
        Leave on default to use the gateway&rsquo;s built-in voice.
      </span>
    </div>
  );
}

function StatePill({ label, tone }: { label: string; tone: string }) {
  const classes: Record<string, string> = {
    muted: "border-border bg-muted/40 text-muted-foreground",
    primary: "border-primary/60 bg-primary/10 text-primary",
    accent: "border-primary/60 bg-primary text-primary-foreground",
    warn: "border-destructive/40 bg-destructive/10 text-destructive",
  };
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px] uppercase tracking-widest ${
        classes[tone] ?? classes.muted
      }`}
    >
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      {label}
    </div>
  );
}

function Transcript({
  turns,
  currentEpoch,
}: {
  turns: Turn[];
  currentEpoch: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [turns]);

  if (turns.length === 0) {
    return (
      <div className="flex min-h-48 items-center justify-center rounded-md border border-dashed border-border p-8 text-sm text-muted-foreground">
        Start the conversation and the transcript will fill in here.
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className="flex max-h-96 flex-col gap-4 overflow-y-auto rounded-md border border-border bg-card/40 p-5"
    >
      {turns.map((t) => (
        <div
          key={`${t.sessionEpoch}-${t.id}`}
          className={`flex gap-3 ${t.role === "user" ? "" : "flex-row"}`}
        >
          <span
            className={`shrink-0 font-mono text-[10px] uppercase tracking-widest ${
              t.role === "user" ? "text-primary" : "text-muted-foreground"
            }`}
            style={{ width: 88 }}
          >
            {t.role === "user" ? "you" : "gateway"}
            {t.sessionEpoch !== currentEpoch && (
              <span className="ml-1 text-[9px] text-muted-foreground/60">·prev</span>
            )}
          </span>
          <p
            className={`text-sm leading-relaxed ${
              t.role === "user" ? "text-foreground" : "text-muted-foreground"
            }`}
          >
            {t.text || <span className="italic text-muted-foreground/60">…</span>}
            {t.partial && <span className="ml-1 animate-pulse text-primary">▏</span>}
            {t.interrupted && (
              <span className="ml-2 font-mono text-[10px] uppercase tracking-widest text-destructive/80">
                interrupted
              </span>
            )}
          </p>
        </div>
      ))}
    </div>
  );
}

function Controls({
  micOn,
  sessionState,
  connState,
  onStart,
  onEnd,
  onStopAi,
}: {
  micOn: boolean;
  sessionState: SessionState;
  connState: ConnState;
  onStart: () => void;
  onEnd: () => void;
  onStopAi: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      {!micOn ? (
        <Button size="lg" onClick={onStart}>
          Start conversation
        </Button>
      ) : (
        <Button size="lg" variant="destructive" onClick={onEnd}>
          End conversation
        </Button>
      )}
      {micOn && sessionState === "tts_speaking" && (
        <Button variant="outline" size="lg" onClick={onStopAi}>
          Stop AI
        </Button>
      )}
      {micOn && connState === "open" && (
        <span className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-muted-foreground">
          <span className="inline-block h-2 w-2 animate-ping rounded-full bg-primary" />
          mic live
        </span>
      )}
    </div>
  );
}
