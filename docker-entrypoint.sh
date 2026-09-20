#!/bin/sh
set -e

echo "[entrypoint] installing dependencies..."
pip install --no-cache-dir -r /app/requirements.txt

echo "[entrypoint] starting API on port ${PORT:-8000}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
