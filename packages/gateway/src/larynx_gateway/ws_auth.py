"""WebSocket bearer-token authentication.

FastAPI's ``Depends(require_bearer_token)`` is HTTP-only (it relies on
``HTTPException`` to short-circuit). WebSockets need to reject by calling
``ws.close(code=...)``, so we expose a small helper that accepts the token
via either ``Authorization: Bearer <token>`` or ``?token=<value>`` — the
latter is the WS-native escape hatch for browsers that can't set WS
request headers.
"""

from __future__ import annotations

from fastapi import WebSocket, status

from larynx_gateway.config import get_settings


async def require_ws_bearer_token(ws: WebSocket) -> bool:
    """Validate bearer token on a WebSocket. Returns True if accepted.

    On rejection, closes the socket with policy-violation code and returns
    False. Callers should simply ``return`` if the result is False.
    """
    settings = get_settings()
    expected = settings.larynx_api_token

    supplied: str | None = None
    auth = ws.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        supplied = auth.split(" ", 1)[1].strip() or None
    if supplied is None:
        # Query-param fallback for browsers / SDKs that can't set headers.
        supplied = ws.query_params.get("token")

    if supplied is None:
        await ws.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="missing bearer token"
        )
        return False

    if len(supplied) != len(expected):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid token")
        return False
    mismatch = 0
    for a, b in zip(expected, supplied, strict=False):
        mismatch |= ord(a) ^ ord(b)
    if mismatch != 0:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid token")
        return False
    return True
