# apps/playground-test

Hand-written HTML + vanilla JS pages for manual testing of the streaming
WebSocket endpoints. These are _not_ part of the real playground UI — they
are intentionally minimal (no framework, no bundler) so a developer can
point a browser at `pages/*.html` and hear TTS or speak into STT without
spinning up Next.js.

The real playground (Next.js + shadcn) lives under `apps/playground/`.

## Usage

Start the gateway:

```bash
uv run larynx-gateway  # or `docker compose up gateway`
```

Serve the pages from any static server on the same host (the pages assume
the gateway is reachable at `ws://<host>:8000`):

```bash
python -m http.server --directory apps/playground-test 7070
# then open http://localhost:7070/tts_stream.html
```

Paste your bearer token (the one in `.env`) into the token field and hit
**Play** / **Record**.

## Files

- `tts_stream.html` — streams `/v1/tts/stream` into a WebAudio `AudioContext`
  and shows browser-measured TTFB.
- `stt_stream.html` — captures the mic via `getUserMedia`, streams 20ms
  PCM16 chunks over `/v1/stt/stream`, and displays partials (replaced in
  place) + finals (appended).
