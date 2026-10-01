FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000 \
    DATABASE_URL=sqlite:////data/heart.db

# XGBoost needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached until requirements change.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Run as a non-root user. /data holds the SQLite file and is meant to be a volume.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir /data \
    && chown appuser /data
COPY --chown=appuser . .
USER appuser

VOLUME /data
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT', '5000'))"

# `exec` makes gunicorn PID 1 so `docker stop` shuts it down cleanly.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 --access-logfile - app:app"]
