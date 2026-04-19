import type { NextConfig } from "next";

const GATEWAY_ORIGIN = process.env.LARYNX_GATEWAY_ORIGIN ?? "http://localhost:8000";

// Reverse-proxy the gateway's HTTP endpoints under the playground's own
// origin so CORS drops out of the picture for local dev. The Next.js
// rewrite proxy is HTTP-only — it does not forward the WebSocket
// `Upgrade` handshake cleanly (Next.js 15 / turbopack drops the
// Upgrade + Connection headers). The /v1/conversation WebSocket must
// therefore either
//   (a) connect directly to the gateway (set NEXT_PUBLIC_GATEWAY_URL),
//   (b) go through a dedicated WS-aware reverse proxy in front of both
//       services — see `apps/playground/deploy/Caddyfile` + the
//       README next to it for the setup we use when fronting the
//       playground with an ngrok tunnel.
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/v1/:path*", destination: `${GATEWAY_ORIGIN}/v1/:path*` },
      { source: "/health", destination: `${GATEWAY_ORIGIN}/health` },
      { source: "/ready", destination: `${GATEWAY_ORIGIN}/ready` },
      { source: "/openapi.json", destination: `${GATEWAY_ORIGIN}/openapi.json` },
      { source: "/docs", destination: `${GATEWAY_ORIGIN}/docs` },
      { source: "/metrics", destination: `${GATEWAY_ORIGIN}/metrics` },
    ];
  },
};

export default nextConfig;
