"use client";

import { useEffect, useRef, useState } from "react";
import { getToken } from "./token";

export type WsStatus = "idle" | "connecting" | "open" | "closed" | "error";

export interface UseWebSocketOptions {
  /** Path relative to the gateway, e.g. "/v1/conversation". */
  path: string;
  /** If false, the hook won't auto-connect on mount. */
  enabled?: boolean;
  /** Called for every incoming message. */
  onMessage?: (event: MessageEvent) => void;
  /** Called once on open. */
  onOpen?: (ws: WebSocket) => void;
  /** Called on close (any reason). */
  onClose?: (event: CloseEvent) => void;
}

function resolveGatewayWs(): string {
  const http = process.env.NEXT_PUBLIC_GATEWAY_URL;
  if (http) return http.replace(/^http/, "ws");
  if (typeof window === "undefined") return "";
  return window.location.origin.replace(/^http/, "ws");
}

/**
 * Reusable WebSocket hook. Opens a single socket to the gateway with the
 * bearer token appended as `?token=…` (the gateway accepts it in either the
 * `Authorization` header or this query param). Exposes status + send.
 */
export function useWebSocket({
  path,
  enabled = true,
  onMessage,
  onOpen,
  onClose,
}: UseWebSocketOptions) {
  const [status, setStatus] = useState<WsStatus>("idle");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const token = getToken();
    const url = `${resolveGatewayWs()}${path}${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("open");
      onOpen?.(ws);
    };
    ws.onmessage = (ev) => onMessage?.(ev);
    ws.onerror = () => setStatus("error");
    ws.onclose = (ev) => {
      setStatus("closed");
      onClose?.(ev);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, enabled]);

  const send = (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(data);
    return true;
  };

  return { status, send, socket: wsRef } as const;
}
