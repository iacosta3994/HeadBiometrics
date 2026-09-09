# Lean API image (DEMO_MODE-friendly). Large caffemodel / pose weights are
# excluded via .dockerignore — mount them at runtime for the full CV pipeline,
# or run with DEMO_MODE=1.
#
# Build:
#   docker build -t headbiometrics .
# Run (demo / HTTP shell):
#   docker run --rm -p 8000:8000 -e DEMO_MODE=1 headbiometrics
# Run with models bind-mounted (full pipeline; also install TF — see below):
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/src/HED_model:/app/src/HED_model:ro" \
#     -v "$PWD/src/Proctoring_AI:/app/src/Proctoring_AI:ro" \
#     headbiometrics
#
# TensorFlow is NOT installed by default (keeps the image lean). For real
# landmark inference:
#   docker build --build-arg INSTALL_TF=1 -t headbiometrics:tf .

FROM python:3.12-slim

ARG INSTALL_TF=0

WORKDIR /app

# System libs helpful for opencv-headless / video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_TF" = "1" ]; then pip install --no-cache-dir "tensorflow>=2.15,<3"; fi

# Application code (large model blobs intentionally omitted — see .dockerignore)
COPY head_biometrics/ head_biometrics/
COPY src/ src/
COPY README.md LICENSE pytest.ini ./

ENV PYTHONUNBUFFERED=1 \
    DEMO_MODE=0 \
    MAX_UPLOAD_BYTES=104857600

EXPOSE 8000

CMD ["uvicorn", "head_biometrics.app:app", "--host", "0.0.0.0", "--port", "8000"]
