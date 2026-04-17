"""Per-route body-size caps.

Enforced in a Starlette middleware so a misbehaving client hitting
``/v1/tts`` with a 10GB payload never leaves a single
:class:`UploadFile` in memory. The cap is route-template-scoped — a
multipart TTS call gets a larger budget than a JSON TTS call.

Exceeding the limit returns 413 immediately, without reading the full
body. The check reads ``Content-Length`` where the client supplied
it; for chunked transfers we defer to the per-route body readers
(which have their own bounds).

See ORCHESTRATION-M8.md §3.6.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

# Byte limits per matched route template. Keys are the route
# patterns registered with FastAPI (``.path`` on the matched Route
# object). Unmatched requests fall back to DEFAULT_LIMIT.
#
# Values come straight from ORCHESTRATION-M8.md §3.6. If you add a
# new route or change its body shape, add it here too — absent
# routes silently fall back to the 1MB default.
DEFAULT_LIMIT = 1 * 1024 * 1024  # 1 MB

ROUTE_LIMITS: dict[str, int] = {
    "/v1/tts": 50 * 1024 * 1024,  # multipart TTS reference audio
    "/v1/stt": 100 * 1024 * 1024,
    "/v1/voices": 100 * 1024 * 1024,
    "/v1/batch": 2 * 1024 * 1024,  # 500 items * ~4KB/item fits inside 2MB
    "/v1/audio/speech": 64 * 1024,
    "/v1/audio/transcriptions": 100 * 1024 * 1024,
    "/v1/finetune/datasets": 500 * 1024 * 1024,
}


def _limit_for(request: Request) -> int:
    route = request.scope.get("route")
    path = getattr(route, "path", None) if route is not None else None
    if path is not None and path in ROUTE_LIMITS:
        return ROUTE_LIMITS[path]
    # Prefix-match so sub-paths (e.g. /v1/batch/{id}) inherit their
    # parent's limit. Routes are registered in the order above, so a
    # simple startswith pass is correct + O(n_routes) per request.
    url_path = request.url.path
    for prefix, limit in ROUTE_LIMITS.items():
        if url_path == prefix or url_path.startswith(prefix + "/"):
            return limit
    return DEFAULT_LIMIT


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Short-circuits oversized requests with 413 based on Content-Length."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only POST/PUT/PATCH carry bodies we care about. GET/DELETE
        # with a body is a protocol violation we won't guard against.
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        limit = _limit_for(request)
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": {
                                "code": "body_too_large",
                                "limit_bytes": limit,
                                "got_bytes": int(content_length),
                            }
                        },
                    )
            except ValueError:
                # Malformed header — let the handler reject it naturally.
                pass

        return await call_next(request)


def make_middleware(app: ASGIApp) -> BodySizeLimitMiddleware:
    """Convenience factory used by main.py's app builder."""
    return BodySizeLimitMiddleware(app)


__all__ = ["BodySizeLimitMiddleware", "ROUTE_LIMITS", "make_middleware"]
