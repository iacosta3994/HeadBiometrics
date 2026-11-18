#!/usr/bin/env python3
"""Minimal HeadBiometrics client (stdlib urllib; httpx if installed).

Examples:
  python examples/measure_client.py Video_Tests/Self.mp4
  API_KEY=secret BASE_URL=http://127.0.0.1:8000 \\
    python examples/measure_client.py clip.mp4 --scale-mode mm_per_pixel --mm-per-pixel 0.25
  python examples/measure_client.py clip.mp4 --async
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _post_multipart(
    url: str,
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    boundary = "----HeadBiometricsBoundary7MA4YWxkTrZu0gW"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{os.path.basename(filename)}"\r\n'
        ).encode()
    )
    body.extend(b"Content-Type: video/mp4\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req_headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if headers:
        req_headers.update(headers)

    # Prefer httpx when available (already a test dep).
    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=120.0) as client:
            files = {"video": (os.path.basename(filename), file_bytes, "video/mp4")}
            resp = client.post(url, data=fields, files=files, headers=headers or {})
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001
                payload = resp.text
            return resp.status_code, payload
    except ImportError:
        pass

    req = urllib.request.Request(url, data=bytes(body), headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def _get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Any]:
    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers or {})
            return resp.status_code, resp.json()
    except ImportError:
        pass

    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def main() -> int:
    parser = argparse.ArgumentParser(description="HeadBiometrics measure client")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--scale-mode", default="mm_per_pixel")
    parser.add_argument("--mm-per-pixel", type=float, default=0.25)
    parser.add_argument("--clockwise", action="store_true")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use job endpoints")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args()

    headers: Dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    with open(args.video, "rb") as fh:
        file_bytes = fh.read()

    fields = {
        "clockwise": "true" if args.clockwise else "false",
        "scale_mode": args.scale_mode,
    }
    if args.scale_mode == "mm_per_pixel":
        fields["mm_per_pixel"] = str(args.mm_per_pixel)

    if not args.use_async:
        status, body = _post_multipart(
            f"{args.base_url.rstrip('/')}/v1/measure",
            fields,
            "video",
            args.video,
            file_bytes,
            headers=headers,
        )
        print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
        return 0 if status == 200 else 1

    status, body = _post_multipart(
        f"{args.base_url.rstrip('/')}/v1/measure/jobs",
        fields,
        "video",
        args.video,
        file_bytes,
        headers=headers,
    )
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    if status != 200 or not isinstance(body, dict) or "job_id" not in body:
        return 1

    job_id = body["job_id"]
    while True:
        st, poll = _get_json(
            f"{args.base_url.rstrip('/')}/v1/measure/jobs/{job_id}",
            headers=headers,
        )
        print(json.dumps(poll, indent=2) if isinstance(poll, dict) else poll)
        if st != 200 or not isinstance(poll, dict):
            return 1
        if poll.get("status") in {"succeeded", "failed"}:
            return 0 if poll.get("status") == "succeeded" else 1
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
