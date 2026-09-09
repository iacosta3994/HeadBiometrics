# Changelog

All notable changes to the HeadBiometrics HTTP API (`head_biometrics/`) are
documented here. The legacy `src/` measurement pipeline remains available
alongside the service. Versions refer to `__version__` in
`head_biometrics/__init__.py`.

## [0.6.0] — 2026-09-09

### Ops / auth / metrics

- Optional `API_KEY`: when set, measure endpoints require `X-API-Key` or
  `Authorization: Bearer` (`/health`, `/version`, `/metrics`, `/v1/scale-modes`
  stay open; missing/wrong key → 401).
- `GET /metrics` — in-process JSON counters (`requests_total`,
  `measure_success`, `measure_failure`, `jobs_created`); structured access logs
  on measure paths.
- Durable async job store: `JOB_STORE=memory|sqlite` (default `memory`) with
  `JOB_STORE_PATH` (default `./data/jobs.sqlite3`). SQLite persists
  status/result/error; orphaned `queued`/`running` rows marked `failed` on
  open. Still **single-process** / one uvicorn worker for the MVP.
- Client helpers under `examples/` (`measure.sh`, `measure_async.sh`,
  `measure_client.py`).
- CI workflow **template** at `ci/github-actions.yml` (Python 3.12, pytest, no
  TensorFlow). Copy to `.github/workflows/ci.yml` manually — pushing workflow
  files needs a token with the `workflow` scope.

## [0.5.0] — 2026-09-09

### Scale modes

- `scale_mode=iris` — adult iris-diameter prior (default `iris_mm=11.7`) vs
  crude eyelid-landmark aperture (`37–40` / `43–46`; eye-width × 0.45
  fallback). Approximate; confidence capped like IPD; not medical-grade.

### Async jobs

- `POST /v1/measure/jobs` + `GET /v1/measure/jobs/{job_id}` — in-memory
  thread-pool jobs for long videos (`queued` → `running` → `succeeded` /
  `failed`). Sync `POST /v1/measure` unchanged. Documented as MVP / not
  multi-worker safe.

## [0.4.0] — 2026-09-09

### Scale modes

- `scale_mode=aruco` via `cv2.aruco` (`aruco_marker_length_mm` required;
  optional `aruco_dict`, default `4x4_50`). Prefer
  `opencv-contrib-python-headless`.
- Stronger ISO ID-1 card detect: ordered corner side lengths + relaxed second
  pass.

### API / quality

- Heuristic `meta.confidence`, `meta.warnings`, `meta.scale_frames_used`
  (non-blocking; not calibrated / not medical-grade).
- `GET /version`; `version` field on `/health`.
- Optional `CORS_ORIGINS` env.

## [0.1.0] — 2026-09-08

Initial FastAPI overhaul (versions 0.2 / 0.3 were not tagged; features below
shipped under `0.1.0` before the jump to `0.4.0`).

### API

- `head_biometrics/` FastAPI package wrapping the existing OpenCV/TF `src/`
  pipeline.
- `GET /health`, `POST /v1/measure` (multipart video) returning five mm
  metrics: circumference, front_to_nape, ear_to_ear, head_width, length.
- `GET /v1/scale-modes` catalog + upload size guard (413 / 100 MB default;
  `MAX_UPLOAD_BYTES`).
- Optional `DEMO_MODE=1` — labeled mock response only when the CV stack cannot
  import (never silent fakes).
- Lowercase renames under `src/` for Linux case-sensitive imports; guard
  `src/main.py` demo with `if __name__ == "__main__"`.
- Pytest suite with mocked pipeline (no TensorFlow required for CI).

### Scale modes

- Default `magstripe` plus alternates: `mm_per_pixel`, `reference_mm`, `ipd`
  (optional `ipd_mm`, default 63).
- `scale_mode=card` / `id1_card` — ISO ID-1 auto-detect
  (85.60 × 53.98 mm) via OpenCV quads; median longest-side mm/pixel;
  `meta.mm_per_pixel` on success.

### Headless pipeline

- Side metrics via landmarks + silhouette (no OpenCV GUI / mouse picks for the
  API). `interactive=True` retained for local debugging.

### Docker / docs

- Lean `Dockerfile` / `.dockerignore` (large weights excluded; `DEMO_MODE` by
  default; optional `INSTALL_TF` build-arg; mount model dirs for full CV).
- README: install, uvicorn, scale-mode table, curl examples, known limitations.

## Unreleased / notes for reviewers

- Package version remains **0.6.0**.
- Known limitations are listed in the README (scale approximations, single-
  process jobs, shared-secret auth, heuristic confidence, TF/model friction,
  not medical-grade).
- Enable GitHub Actions by copying `ci/github-actions.yml` →
  `.github/workflows/ci.yml` with a token that has the `workflow` scope.
