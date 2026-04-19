"use client";

import { getToken, invalidateToken } from "./token";
import { startCapture, type AudioCaptureHandle } from "./audio-capture";

/**
 * Client for `WS /v1/stt/stream` — live transcription.
 *
 * Flow:
 *   1. Open WS → send `{type: "config", sample_rate: 16000, language, hotwords}`.
 *   2. Pump mic → PCM16 @ 16 kHz → WS binary frames.
 *   3. Receive JSON events: partial, final, speech_start, speech_end, error,
 *      heartbeat. Finals accumulate in the UI; partials replace the last
 *      in-flight segment.
 *   4. `stop()` sends `{type: "stop"}` and drains before closing.
 */

const INPUT_RATE = 16000;

function resolveGatewayWs(): string {
  const http = process.env.NEXT_PUBLIC_GATEWAY_URL;
  if (http) return http.replace(/^http/, "ws");
  if (typeof window === "undefined") return "";
  return window.location.origin.replace(/^http/, "ws");
}

export type STTStreamEvent =
  | { type: "speech_start"; session_ms?: number }
  | { type: "speech_end"; session_ms?: number }
  | {
      type: "partial";
      text: string;
      interval_ms?: number;
      decode_ms?: number;
    }
  | {
      type: "final";
      text: string;
      punctuated_text?: string;
      language?: string;
    }
  | { type: "error"; code: string; message: string }
  | { type: "heartbeat"; vad_state?: string; session_ms?: number };

export interface STTStreamOptions {
  language?: string;
  hotwords?: string[];
  onEvent: (ev: STTStreamEvent) => void;
  onOpen?: () => void;
  onClose?: (info: { code: number; reason: string; wasAccepted: boolean }) => void;
  onError?: (detail: { message: string; url: string }) => void;
}

export interface STTStreamHandle {
  stop(): Promise<void>;
}

export async function startSTTStream(opts: STTStreamOptions): Promise<STTStreamHandle> {
  const token = getToken();
  if (!token) {
    throw new Error("No bearer token — sign in first.");
  }

  const url = `${resolveGatewayWs()}/v1/stt/stream?token=${encodeURIComponent(token)}`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  let configured = false;
  let capture: AudioCaptureHandle | null = null;
  let stopping = false;

  const waitForOpen = new Promise<void>((resolve, reject) => {
    const onOpen = () => {
      ws.removeEventListener("open", onOpen);
      ws.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      ws.removeEventListener("open", onOpen);
      ws.removeEventListener("error", onError);
      reject(new Error(`WebSocket failed to open (${url})`));
    };
    ws.addEventListener("open", onOpen);
    ws.addEventListener("error", onError);
  });

  ws.addEventListener("message", (msg) => {
    if (typeof msg.data !== "string") return;
    try {
      const parsed = JSON.parse(msg.data) as STTStreamEvent;
      opts.onEvent(parsed);
    } catch {
      // non-JSON text is not part of the protocol
    }
  });

  ws.addEventListener("close", (ev) => {
    if (ev.code === 1008) invalidateToken("ws-rejected");
    opts.onClose?.({ code: ev.code, reason: ev.reason, wasAccepted: configured });
  });

  ws.addEventListener("error", () => {
    opts.onError?.({ message: "WebSocket error", url });
  });

  await waitForOpen;

  ws.send(
    JSON.stringify({
      type: "config",
      sample_rate: INPUT_RATE,
      language: opts.language || null,
      hotwords: opts.hotwords ?? [],
    }),
  );
  configured = true;
  opts.onOpen?.();

  capture = await startCapture((pcm) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(pcm);
  });

  const stop = async () => {
    if (stopping) return;
    stopping = true;
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
      }
    } catch {
      // ignore
    }
    if (capture) {
      await capture.stop();
      capture = null;
    }
    // Small grace so the final event lands before close fires.
    await new Promise<void>((r) => window.setTimeout(r, 300));
    try {
      ws.close();
    } catch {
      // ignore
    }
  };

  return { stop };
}
