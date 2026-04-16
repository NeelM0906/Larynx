"""Bearer token auth.

Single static token in v1 (see PRD §7). When we move to per-user auth the
dependency signature stays the same — only the implementation swaps.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from larynx_gateway.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def require_bearer_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Constant-time comparison isn't strictly necessary for a single static
    # token against trusted internal clients, but it's cheap.
    expected = settings.larynx_api_token
    supplied = credentials.credentials
    if len(expected) != len(supplied):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    mismatch = 0
    for a, b in zip(expected, supplied, strict=False):
        mismatch |= ord(a) ^ ord(b)
    if mismatch != 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    request.state.authenticated = True
