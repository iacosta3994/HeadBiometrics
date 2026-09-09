"""Optional API key authentication via env ``API_KEY``.

When ``API_KEY`` is set (non-empty), measure endpoints require either:
  - ``X-API-Key: <key>`` header, or
  - ``Authorization: Bearer <key>``

``/health``, ``/version``, ``/metrics``, and ``/v1/scale-modes`` stay open.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def expected_api_key() -> str:
    return os.environ.get("API_KEY", "").strip()


def extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("X-API-Key")
    if header_key is not None and header_key.strip():
        return header_key.strip()
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce API key when configured."""
    expected = expected_api_key()
    if not expected:
        return
    provided = extract_api_key(request)
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
