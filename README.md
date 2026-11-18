# HeadBiometrics

HTTP API (FastAPI) over a ~2020 OpenCV / TensorFlow pipeline that estimates head
dimensions (circumference, front-to-nape, ear-to-ear, width, length) from a
cellphone video. Scale can come from a **magnetic stripe**, a full **ISO ID-1
credit/ID card**, a printed **ArUco marker**, an explicit mm/pixel value, a
known reference object, an IPD population prior, or an iris-diameter prior.

The original scripts live under `src/` and remain available. The new service
package is `head_biometrics/`.

## Features

- `GET /health` — liveness check (includes `version`; works without TensorFlow)
- `GET /version` — package version
- `GET /metrics` — JSON counters (`requests_total`, `measure_success`,
  `measure_failure`, `jobs_created`)
- `GET /v1/scale-modes` — catalog of scale modes + required form fields
- `POST /v1/measure` — multipart video upload → JSON millimeter measurements
- `POST /v1/measure/jobs` + `GET /v1/measure/jobs/{job_id}` — async jobs for long videos
- **Headless end-to-end** — side metrics auto-select points via landmarks +
  silhouette (no OpenCV GUI / mouse picks required for the API)
- **Scale modes** — `magstripe` (default), `card` / `id1_card`, `aruco`,
  `mm_per_pixel`, `reference_mm`, `ipd`, `iris`
- **Quality meta** — heuristic `meta.confidence` (0–1) + `meta.warnings`
  (non-blocking; not calibrated / not medical-grade)
- Upload size guard — uploads over **100 MB** → HTTP **413**
- Optional `API_KEY` — require `X-API-Key` or `Authorization: Bearer` on measure
  endpoints (`/health`, `/version`, `/metrics`, `/v1/scale-modes` stay open)
- Optional `JOB_STORE=memory|sqlite` — durable job status/result via SQLite
- Optional `CORS_ORIGINS` env (e.g. `*` or `https://app.example.com`)
- Optional `DEMO_MODE=1` — labeled mock response when the CV stack cannot import
- Client helpers under `examples/` (curl + minimal Python)
- Optional `Dockerfile` for a lean API image
- Legacy CLI path: `python -m src.main` (from repo root)
- **Current API package version:** `0.6.0`

## Requirements

| Layer | Packages |
|-------|----------|
| API   | `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` |
| CV    | `numpy`, **`opencv-contrib-python-headless`** (includes `cv2.aruco`) |
| ML    | `tensorflow` (facial landmarks via `src/Proctoring_AI`) — **optional for /health & DEMO_MODE** |

> **OpenCV / ArUco:** Prefer `opencv-contrib-python-headless` so `cv2.aruco` is
> always present. Do not install `opencv-python` / `opencv-python-headless` at
> the same time (they conflict). Some recent headless builds already ship
> aruco; if `import cv2.aruco` fails, switch to the contrib package.

> **TensorFlow note:** The landmark model was built against TF ~2.x circa 2020.
> On modern Python (3.11+) try `pip install "tensorflow>=2.15,<3"`. Older
> environments may need a matching older wheel. TF is intentionally not a hard
> dependency of the API so the health endpoint and tests can run without it.

Model artifacts under `src/HED_model/` and `src/Proctoring_AI/` must remain in
place for the full pipeline; do not delete them.

## Install

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# for the real measurement pipeline (not needed for mocked tests / DEMO_MODE):
pip install "tensorflow>=2.15,<3"
```

Run the API **from the repository root** so relative model paths such as
`src/Proctoring_AI/models/...` resolve correctly.

## Run

```bash
uvicorn head_biometrics.app:app --host 0.0.0.0 --port 8000 --reload
```

Open interactive docs at http://127.0.0.1:8000/docs

### Demo mode (no CV/TF)

```bash
DEMO_MODE=1 uvicorn head_biometrics.app:app --port 8000
```

Mock measurements are returned only when the CV stack cannot be imported **and**
`DEMO_MODE=1`. Responses set `meta.demo_mode: true` and include an explanatory
`meta.note`. Fake numbers are never returned silently.

### CORS

```bash
# allow all (dev)
CORS_ORIGINS='*' uvicorn head_biometrics.app:app --port 8000

# allow specific origins
CORS_ORIGINS='https://app.example.com,https://staging.example.com' \
  uvicorn head_biometrics.app:app --port 8000
```

### Optional API key

```bash
API_KEY='your-secret' uvicorn head_biometrics.app:app --port 8000
```

When `API_KEY` is set, measure endpoints (`POST /v1/measure`,
`POST /v1/measure/jobs`, `GET /v1/measure/jobs/{id}`) require either:

```bash
curl -H "X-API-Key: your-secret" ...
# or
curl -H "Authorization: Bearer your-secret" ...
```

`/health`, `/version`, `/metrics`, and `/v1/scale-modes` stay open. Missing or
wrong key → HTTP **401**.

### Durable job store (SQLite)

```bash
JOB_STORE=sqlite JOB_STORE_PATH=./data/jobs.sqlite3 \
  uvicorn head_biometrics.app:app --port 8000 --workers 1
```

Default remains `JOB_STORE=memory`.

### Metrics

```bash
curl -s http://127.0.0.1:8000/metrics | jq
# {"requests_total":..., "measure_success":..., "measure_failure":..., "jobs_created":...}
```

In-process counters (reset on restart). Measure endpoints also emit a structured
access log line (`head_biometrics.access`).

### Client examples

```bash
./examples/measure.sh Video_Tests/Self.mp4 mm_per_pixel
./examples/measure_async.sh Video_Tests/Self.mp4 mm_per_pixel
python examples/measure_client.py Video_Tests/Self.mp4 --scale-mode mm_per_pixel
API_KEY=secret python examples/measure_client.py clip.mp4 --async
```

### Docker

```bash
# lean image (API + DEMO_MODE; large *.caffemodel / pose weights excluded)
docker build -t headbiometrics .
docker run --rm -p 8000:8000 -e DEMO_MODE=1 headbiometrics

# optional: install TensorFlow in the image
docker build --build-arg INSTALL_TF=1 -t headbiometrics:tf .

# full CV pipeline: mount model directories from the host
docker run --rm -p 8000:8000 \
  -v "$PWD/src/HED_model:/app/src/HED_model:ro" \
  -v "$PWD/src/Proctoring_AI:/app/src/Proctoring_AI:ro" \
  headbiometrics:tf
```

Override the upload limit with `-e MAX_UPLOAD_BYTES=...` (default 104857600 = 100 MB).
Optional: `-e CORS_ORIGINS='*'`, `-e API_KEY=...`, `-e JOB_STORE=sqlite -e JOB_STORE_PATH=/data/jobs.sqlite3`.

## Scale modes

| `scale_mode`     | Required form fields                         | Notes |
|------------------|----------------------------------------------|-------|
| `magstripe`      | _(none)_                                     | Default. Detects credit-card magstripe aspect ratios → mm/pixel. |
| `card` (`id1_card`) | _(none)_                                  | Auto-detect full **ISO ID-1** card **85.60 × 53.98 mm**. Scale uses the **longest side** (ordered corner distances when available): `mm_per_pixel = 85.60 / longest_side_px`, median across frames. |
| `aruco`          | `aruco_marker_length_mm`                     | Detect printed ArUco marker (`cv2.aruco`). Optional `aruco_dict` (default **`4x4_50`**; also try **`5x5_100`**). `mm_per_pixel = length_mm / side_px` (median). |
| `mm_per_pixel`   | `mm_per_pixel` (float > 0)                   | Caller supplies millimeters per pixel; skips auto detection. |
| `reference_mm`   | `reference_width_mm`, `reference_width_px`   | Known object width in mm and its measured width in px. Prefer `card` / `aruco` for automatic detection. |
| `ipd`            | optional `ipd_mm` (default **63**)           | Interpupillary distance from eye landmarks vs adult mean prior. **Approximate** — see `meta.scale_note` / `warnings`. |
| `iris`           | optional `iris_mm` (default **11.7**)        | Adult iris-diameter prior vs crude eyelid-landmark aperture (`37–40` / `43–46`; fallback eye-width × 0.45). **Approximate** — lower confidence; not medical-grade. |

List modes programmatically:

```bash
curl -s http://127.0.0.1:8000/v1/scale-modes | jq
```

All modes produce a `pixel_mm` value compatible with the legacy quantify helpers
(`pixel_mm[0]` = mm per pixel). Successful responses include `meta.mm_per_pixel`
(the scalar actually used), plus heuristic `meta.confidence` and `meta.warnings`.

### Card mode details

Implementation lives in `src/card_scale.py` (wired through `src/scale_modes.py`
and `head_biometrics/pipeline.py`):

1. Grayscale → blur → Canny + adaptive thresholds
2. `findContours` + `approxPolyDP` for convex quads
3. Score by aspect ≈ 85.60/53.98 (±12%), area fraction, rectangularity
4. Longest side from **ordered corner distances** (perspective-aware); second
   relaxed pass if the primary finds nothing (wider aspect/area gates)
5. Median-aggregate across frames

On failure the API returns **422** with a clear message asking you to show a
full credit/ID card flat in frame, or use another scale mode.

### ArUco mode details

Implementation: `src/aruco_scale.py`.

1. Resolve dictionary (`4x4_50` → `DICT_4X4_50`, etc.)
2. Detect markers per frame via `cv2.aruco.ArucoDetector` (legacy path supported)
3. Side length = mean of the four corner edges in px
4. `mm_per_pixel = aruco_marker_length_mm / side_px`; median across detections

Print a marker, measure its physical side in mm, and pass that as
`aruco_marker_length_mm`. No marker → **422**.


### Iris mode details

Implementation: `src/scale_modes.py` (`estimate_iris_px` / `from_iris`).

Adult horizontal iris diameter ≈ **11.7 mm** is a **population prior**
(configurable via `iris_mm`). Pixel size is a **crude landmark proxy**, not iris
segmentation:

1. Primary: mean vertical eyelid aperture `||landmark[37]−[40]||` (left) and
   `||[43]−[46]||` (right)
2. Fallback: mean outer→inner eye-corner width × **0.45** when the aperture is
   degenerate
3. `mm_per_pixel = iris_mm / iris_px` (best/max across frames)

Confidence is capped like IPD; prefer magstripe / card / aruco / a measured
reference when accuracy matters. **Not medical-grade.**

### Async measurement jobs

Long videos can block `POST /v1/measure`. Use the job endpoints instead:

1. `POST /v1/measure/jobs` — same multipart form fields as `/v1/measure`;
   returns `{ "job_id": "...", "status": "queued" }` immediately
2. `GET /v1/measure/jobs/{job_id}` — poll until `status` is `succeeded`
   (`result` present) or `failed` (`error` present). Intermediate values:
   `queued` | `running`

**Store backends** (still **single-process** — thread pool is in-process; not
safe across multiple uvicorn workers):

| Env | Default | Notes |
|-----|---------|-------|
| `JOB_STORE` | `memory` | In-process dict; lost on restart |
| `JOB_STORE=sqlite` | — | Persist status/result/error |
| `JOB_STORE_PATH` | `./data/jobs.sqlite3` | SQLite file path |

SQLite survives process restart for terminal jobs; orphaned `queued`/`running`
rows from a crash are marked `failed` on store open. Use a **single uvicorn
worker** for this MVP. Sync `POST /v1/measure` remains unchanged.

## Example curl

```bash
# health (includes version)
curl -s http://127.0.0.1:8000/health | jq

# version
curl -s http://127.0.0.1:8000/version | jq

# list scale modes
curl -s http://127.0.0.1:8000/v1/scale-modes | jq

# measure with ArUco marker scale (marker side = 40 mm, DICT_4X4_50)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=aruco" \
  -F "aruco_marker_length_mm=40" \
  -F "aruco_dict=4x4_50" | jq

# measure with ISO ID-1 card auto-detect scale
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=card" | jq

# measure with magstripe scale (Android-style rotation: clockwise=false)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=magstripe" | jq

# measure with explicit mm per pixel (no card/magstripe needed)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=mm_per_pixel" \
  -F "mm_per_pixel=0.25" | jq

# measure with IPD population prior (approximate)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=ipd" \
  -F "ipd_mm=63" | jq

# measure with iris-diameter prior (approximate; default 11.7 mm)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=iris" \
  -F "iris_mm=11.7" | jq

# async job (long videos) — enqueue then poll
JOB=$(curl -s -X POST http://127.0.0.1:8000/v1/measure/jobs \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=mm_per_pixel" \
  -F "mm_per_pixel=0.25")
echo "$JOB" | jq
JOB_ID=$(echo "$JOB" | jq -r .job_id)
curl -s "http://127.0.0.1:8000/v1/measure/jobs/$JOB_ID" | jq

# measure with known reference object (client supplies px width)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@/path/to/video.mp4;type=video/mp4" \
  -F "scale_mode=reference_mm" \
  -F "reference_width_mm=85.60" \
  -F "reference_width_px=428" | jq

# measure (iOS-style rotation: clockwise=true)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@/path/to/video.mp4;type=video/mp4" \
  -F "clockwise=true" | jq
```

Example success payload:

```json
{
  "measurements_mm": {
    "circumference": 560,
    "front_to_nape": 340,
    "ear_to_ear": 150,
    "head_width": 155,
    "length": 195
  },
  "meta": {
    "clockwise": false,
    "filename": "Self.mp4",
    "demo_mode": false,
    "note": null,
    "scale_mode": "aruco",
    "scale_note": "Scale from ArUco marker (dict=DICT_4X4_50, marker_length_mm=40.0, median over 8 detection(s)).",
    "mm_per_pixel": 0.2,
    "confidence": 0.92,
    "warnings": [
      "confidence is a heuristic (not calibrated) — not medical-grade."
    ],
    "scale_frames_used": 8
  }
}
```

## Pipeline overview

Same logic as `src/main.py` `run()`, plus scale-mode resolution:

1. Split video frames (`key_frame_extraction`) — optional 90° rotate for portrait
2. Resolve scale → mm per pixel (`magstripe` / `card` / `aruco` / `mm_per_pixel` / `reference_mm` / `ipd` / `iris`)
3. Pick narrow (front) / wide (side) frames via face + head-pose models
4. Front metrics (`front_quantify`) → ear-to-ear, head width
5. Side metrics (`side_quantify`, **headless**) → front-to-nape, length
6. Circumference ≈ `((width*2)+(length*2)) * 0.834626841674`
7. Attach heuristic confidence / warnings (does not block the response)

### Headless side metrics

`side_mm_metrics_postmtrp(..., interactive=False)` (API default):

1. Detect face + 68 landmarks on the wide (side) frame
2. Infer facing direction (nose tip vs face-box center)
3. Auto-pick length: glabella/mid-brow → farthest silhouette point opposite the face
4. Auto-pick front-to-nape: nose tip → back/down silhouette extreme (nape)
5. Reuse `img_head_contour_side` for the arc length, then apply the legacy
   `* pixel_mm[0]` and `* 0.5` scaling

Pass `interactive=True` only for local OpenCV mouse debugging.

The HTTP wrapper returns all **five** values (the original `run()` only returned
three).

## Known limitations

- **Magstripe optional, not gone.** Default `scale_mode=magstripe` still needs a
  visible stripe; use `card`, `aruco`, or another mode when none is present.
- **Card detection** needs a mostly flat, fully visible ISO ID-1 rectangle with
  enough contrast. Heavy perspective, glare, or partial occlusion → 422. A
  relaxed second pass helps mild cases but still gates on aspect + area.
- **ArUco** needs `cv2.aruco` (contrib/headless) and a fully visible printed
  marker whose physical side matches `aruco_marker_length_mm`.
- **IPD scale is approximate.** Adult mean ≈ 63 mm is a population prior;
  confidence is capped and warnings always mention this. Prefer magstripe,
  card, aruco, or a measured reference when accuracy matters.
- **Iris scale is approximate.** Adult iris ≈ 11.7 mm prior + crude eyelid
  landmark aperture (not iris segmentation). Confidence capped; same preference
  for physical references.
- **Async jobs are single-process MVP.** Thread pool + `memory` or `sqlite`
  store — not multi-worker safe. SQLite persists terminal results across
  restarts; use one uvicorn worker. Sync `/v1/measure` is unchanged.
- **API_KEY is shared-secret only.** No per-user accounts / rotation helpers.
- **Confidence is heuristic.** `meta.confidence` / `warnings` inform clients;
  they are **not** calibrated and **not** medical-grade.
- **Reference object.** `reference_mm` still requires the client to supply
  `reference_width_px`. Use `scale_mode=card` or `aruco` for automatic detection.
- **Case-sensitive imports.** Module files under `src/` were renamed to lowercase
  (`front_quantify.py`, etc.) so Linux imports match the historical
  `from src.front_quantify import ...` style.
- **Silhouette quality.** Auto side points depend on canny/sobel head edges;
  poor lighting / busy backgrounds can yield **422**.
- **Heavy models.** HED (~56MB caffemodel) and Proctoring_AI pose/face models
  must stay on disk; paths are hard-coded relative to the repo root. The Docker
  image excludes large weights by default — mount them or use `DEMO_MODE=1`.
- **Python / TF friction.** Installing an old TF stack on current Python can be
  painful — use `DEMO_MODE=1` or mocked tests when you only need the HTTP shell.
- **Not medical-grade.** Research / prototype accuracy only.

## Tests

```bash
pytest -q
```

Tests mock the pipeline / inject synthetic landmarks so CI does not need
TensorFlow. Card-scale unit tests draw synthetic ID-1 rectangles (including
rotated quads). ArUco tests use `cv2.aruco.generateImageMarker`. Iris tests
mock eyelid landmarks. Async job tests mock `measure_upload` and poll status.
SQLite store tests use a temp `JOB_STORE_PATH`. Auth/metrics tests cover
`API_KEY` and `/metrics` counters.

A lean GitHub Actions workflow lives at `ci/github-actions.yml` (Python 3.12,
`pip install -r requirements.txt`, `pytest` — no TensorFlow). Copy it to
`.github/workflows/ci.yml` to enable (creating workflow files needs a token
with the `workflow` scope).

## Project layout

```
head_biometrics/     # FastAPI app + pipeline adapter + async jobs + auth/metrics
src/                 # Original CV/TF measurement modules + model weights
  aruco_scale.py     # ArUco marker auto-detect
  card_scale.py      # ISO ID-1 card auto-detect
  scale_modes.py     # magstripe / card / aruco / mm_per_pixel / reference / ipd / iris
examples/            # curl + Python measure clients
tests/               # pytest (API/auth/metrics/jobs/sqlite + scale/card/aruco/iris)
ci/github-actions.yml  # CI template → copy to .github/workflows/ci.yml
Video_Tests/         # Sample video(s)
Dockerfile           # Lean API image (models volume-mounted or DEMO_MODE)
.dockerignore
requirements.txt
```

## License

See `LICENSE`.
