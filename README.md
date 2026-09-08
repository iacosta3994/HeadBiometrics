# HeadBiometrics

HTTP API (FastAPI) over a ~2020 OpenCV / TensorFlow pipeline that estimates head
dimensions (circumference, front-to-nape, ear-to-ear, width, length) from a
cellphone video that includes a **magnetic stripe card** as a real-world scale
reference.

The original scripts live under `src/` and remain available. The new service
package is `head_biometrics/`.

## Features

- `GET /health` — liveness check (works without TensorFlow / OpenCV)
- `POST /v1/measure` — multipart video upload → JSON millimeter measurements
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

## Example curl

```bash
# health
curl -s http://127.0.0.1:8000/health | jq

# measure (Android-style rotation: clockwise=false)
curl -s -X POST http://127.0.0.1:8000/v1/measure \
  -F "video=@Video_Tests/Self.mp4;type=video/mp4" \
  -F "clockwise=false" | jq

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
    "note": null
  }
}
```

## Pipeline overview

Same logic as `src/main.py` `run()`:

1. Split video frames (`key_frame_extraction`) — optional 90° rotate for portrait
2. Detect magstripe → pixels-per-mm scale (`video_to_mm`)
3. Pick narrow (front) / wide (side) frames via face + head-pose models
4. Front metrics (`front_quantify`) → ear-to-ear, head width
5. Side metrics (`side_quantify`) → front-to-nape, length
6. Circumference ≈ `((width*2)+(length*2)) * 0.834626841674`

The HTTP wrapper returns all **five** values (the original `run()` only returned
three).

## Known limitations

- **Magstripe required.** A credit-card-style magnetic stripe must be visible so
  the pipeline can compute a pixel→mm scale. Without it, `/v1/measure` returns
  **422**.
- **Case-sensitive imports.** Module files under `src/` were renamed to lowercase
  (`front_quantify.py`, etc.) so Linux imports match the historical
  `from src.front_quantify import ...` style.
- **Interactive side points.** Legacy `side_quantify.side_mm_metrics_postmtrp`
  uses OpenCV GUI mouse clicks to pick length / front-to-nape points. That does
  **not** work in a headless server. Expect **422** (or a GUI prompt) until that
  step is automated. Front / magstripe stages are automated.
- **Heavy models.** HED (~56MB caffemodel) and Proctoring_AI pose/face models
  must stay on disk; paths are hard-coded relative to the repo root.
- **Python / TF friction.** Installing an old TF stack on current Python can be
  painful — use `DEMO_MODE=1` or mocked tests when you only need the HTTP shell.
- **Not medical-grade.** Research / prototype accuracy only.

## Tests

```bash
pytest -q
```

Tests mock the pipeline so CI does not need TensorFlow or OpenCV.

## Project layout

```
head_biometrics/     # FastAPI app + pipeline adapter
src/                 # Original CV/TF measurement modules + model weights
tests/               # pytest (health + mocked /v1/measure)
Video_Tests/         # Sample video(s)
requirements.txt
```

## License

See `LICENSE`.
