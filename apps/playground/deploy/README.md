# Public exposure — playground via ngrok

Single-tunnel setup for putting the playground and the gateway behind
one public URL (`https://larynx.ngrok.app`). The layout:

```
browser ──https──▶ ngrok edge ──http──▶ caddy :3001 ─┬─▶ gateway :8000  (/v1/*, /health, /ready, /docs, /metrics)
                                                     └─▶ next dev :3000 (everything else, including turbopack HMR)
```

Caddy is the layer that makes WebSocket work. Next.js 15's built-in
rewrite proxy drops the `Upgrade` + `Connection` headers, so
`/v1/conversation` would 404 if ngrok pointed directly at the dev
server. Caddy's `reverse_proxy` forwards the upgrade handshake
correctly.

## One-time setup

```bash
# ngrok binary + your reserved domain's auth token
curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar -xz
mv ngrok ~/bin/ngrok && chmod +x ~/bin/ngrok
~/bin/ngrok config add-authtoken <TOKEN>

# caddy single binary
curl -sSL https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz \
  | tar -xz caddy
mv caddy ~/bin/caddy && chmod +x ~/bin/caddy
```

## Running

```bash
# 1. Gateway (real or mock, your choice). See the repo root README.
LARYNX_API_TOKEN=<token> uv run uvicorn larynx_gateway.main:app --host 0.0.0.0 --port 8000

# 2. Playground dev server. Important: NEXT_PUBLIC_GATEWAY_URL must be
#    empty so the UI fetches same-origin through the proxy.
cd apps/playground
echo "NEXT_PUBLIC_GATEWAY_URL=" > .env.local
npm run dev    # serves on :3000

# 3. Caddy reverse proxy (HTTP + WS). Holds :3001 open.
~/bin/caddy run --config apps/playground/deploy/Caddyfile --adapter caddyfile

# 4. ngrok tunnel to the caddy port.
~/bin/ngrok http 3001 --url=https://larynx.ngrok.app
```

Once those four processes are up, `https://larynx.ngrok.app/` serves
the playground, `https://larynx.ngrok.app/v1/voices` hits the gateway,
and `wss://larynx.ngrok.app/v1/conversation` tunnels the duplex WS to
the gateway on :8000.

## Giving out access

Hand the tester:

- URL: `https://larynx.ngrok.app`
- Bearer token: whatever `LARYNX_API_TOKEN` the gateway was started
  with. The AuthGate dialog pastes it once into localStorage; a rotated
  token auto-invalidates the browser's copy on the next request (see
  `apps/playground/src/lib/api-client.ts`).

## Reverting to local-only

Just set `NEXT_PUBLIC_GATEWAY_URL=http://localhost:8000` in
`.env.local` and skip caddy + ngrok. The UI will hit the gateway
directly; CORS already allows `http://localhost:3000`.
