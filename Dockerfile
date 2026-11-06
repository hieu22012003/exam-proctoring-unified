FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EXAMWATCH_DATA=/data \
    EXAMWATCH_MODELS=/data/models \
    EXAMWATCH_WHISPER=zipformer-vi-30m \
    EXAMWATCH_THREADS=2 \
    EXAMWATCH_COOKIE_SECURE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt ./requirements-base.txt
RUN python -m pip install --no-cache-dir -r requirements-base.txt \
    && python -m pip install --no-cache-dir 'ultralytics>=8.3.0,<9'

COPY examwatch ./examwatch
COPY scripts ./scripts
COPY models ./models
COPY licenses ./licenses
COPY web ./web

RUN mkdir -p /data/models /data/evidence /app/web /app/examwatch/web
EXPOSE 8000

CMD ["sh", "-c", "python scripts/prepare_models.py --whisper \"$EXAMWATCH_WHISPER\" && exec python -m uvicorn examwatch.main:create_app --factory --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers"]
