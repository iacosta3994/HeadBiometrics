# HeadBiometrics

HTTP API (FastAPI) over a ~2020 OpenCV / TensorFlow pipeline that estimates head
dimensions (circumference, front-to-nape, ear-to-ear, width, length) from a
cellphone video. A **magnetic stripe card** can provide real-world scale, or you
can supply an alternate scale mode (explicit mm/pixel, reference object, or IPD
prior).

The original scripts live under `src/` and remain available. The new service
package is `head_biometrics/`.

## Features

- `GET /health` — liveness check (works without TensorFlow / OpenCV)
- `POST /v1/measure` — multipart video upload → JSON millimeter measurements
- **Headless end-to-end** — side metrics auto-select points via landmarks +
  silhouette (no OpenCV GUI / mouse picks required for the API)
- **Scale modes** — magstripe (default), `mm_per_pixel`, `reference_mm`, `ipd`
- Optional `DEMO_MODE=1` — labeled mock response when the CV stack cannot import
- Legacy CLI path: `python -m src.main` (from repo root)

## Requirements

| Layer | Packages |
|-------|----------|
| API   | `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` |
| CV    | `numpy`, `opencv-python-headless` |
| ML    | `tensorflow` (facial landmarks via `src/Proctoring_AI`) — **optional for /health & DEMO_MODE** |

> **TensorFlow note:** The landmark model was built against TF ~2.x circa 2020.
> On modern Python (3.11+) try `pip install "tensorflow>=2.15,<3"`. Older
> environments may need a matching older wheel. TF is intentionally not a hard
> dependency of the API so the health endpoint and tests can run without it.

Model artifacts under `src/HED_model/` and `src/Proctoring_AI/` must remain in
place; do not delete them.

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

## Scale modes

| `scale_mode`     | Required form fields                         | Notes |
|------------------|----------------------------------------------|-------|
| `magstripe`      | _(none)_                                     | Default. Detects credit-card magstripe aspect ratios → mm/pixel. |
| `mm_per_pixel`   | `mm_per_pixel` (float > 0)                   | Caller supplies millimeters per pixel; skips magstripe. |
| `reference_mm`   | `reference_width_mm`, `reference_width_px`   | Known object width in mm and its measured width in px. ISO ID-1 auto-detect is a follow-up. |
| `ipd`            | optional `ipd_mm` (default **63**)           | Interpupillary distance from eye landmarks vs adult mean prior. **Approximate** — see `meta.scale_note`. |

All modes produce a `pixel_mm` value compatible with the legacy quantify helpers
(`pixel_mm[0]` = mm per pixel).

## Example curl

```bash
# health
curl -s http://127.0.0.1:8000/health | jq

# measure with magstripe scale (Android-style rotation: clockwise=false)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=magstripe" | jq

# measure with explicit mm per pixel (no magstripe needed)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=mm_per_pixel" \
  -F "mm_per_pixel=0.25" | jq

# measure with IPD population prior (approximate)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "scale_mode=ipd" \
  -F "ipd_mm=63" | jq

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
    "scale_mode": "magstripe",
    "scale_note": null
  }
}
```

## Pipeline overview

Same logic as `src/main.py` `run()`, plus scale-mode resolution:

1. Split video frames (`key_frame_extraction`) — optional 90° rotate for portrait
2. Resolve scale → mm per pixel (`magstripe` / `mm_per_pixel` / `reference_mm` / `ipd`)
3. Pick narrow (front) / wide (side) frames via face + head-pose models
4. Front metrics (`front_quantify`) → ear-to-ear, head width
5. Side metrics (`side_quantify`, **headless**) → front-to-nape, length
6. Circumference ≈ `((width*2)+(length*2)) * 0.834626841674`

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
  visible stripe; use another mode when none is present.
- **IPD scale is approximate.** Adult mean ≈ 63 mm is a population prior; responses
  include `meta.scale_note` explaining this. Prefer magstripe or a measured
  reference when accuracy matters.
- **Reference object.** `reference_mm` currently requires the client to supply
  `reference_width_px`. Automatic ISO ID-1 (85.60×53.98 mm) contour detection is
  documented as a follow-up and not implemented yet.
- **Case-sensitive imports.** Module files under `src/` were renamed to lowercase
  (`front_quantify.py`, etc.) so Linux imports match the historical
  `from src.front_quantify import ...` style.
- **Silhouette quality.** Auto side points depend on canny/sobel head edges;
  poor lighting / busy backgrounds can yield **422**.
- **Heavy models.** HED (~56MB caffemodel) and Proctoring_AI pose/face models
  must stay on disk; paths are hard-coded relative to the repo root.
- **Python / TF friction.** Installing an old TF stack on current Python can be
  painful — use `DEMO_MODE=1` or mocked tests when you only need the HTTP shell.
- **Not medical-grade.** Research / prototype accuracy only.

## Tests

```bash
pytest -q
```

Tests mock the pipeline / inject synthetic landmarks so CI does not need
TensorFlow. OpenCV is used lightly for array shapes in side-unit tests.

## Project layout

```
head_biometrics/     # FastAPI app + pipeline adapter
src/                 # Original CV/TF measurement modules + model weights
tests/               # pytest (health + mocked /v1/measure + side/scale units)
Video_Tests/         # Sample video(s)
requirements.txt
```

## License

See `LICENSE`.
