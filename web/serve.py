"""Vertex production server entrypoint (Waitress)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waitress import serve
from app import app, _ensure_model

HOST = os.environ.get("VERTEX_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("VERTEX_PORT", 5000)))
THREADS = int(os.environ.get("VERTEX_THREADS", 4))
CHANNEL_TIMEOUT = int(os.environ.get("VERTEX_CHANNEL_TIMEOUT", 900))

if __name__ == "__main__":
    print("[Vertex] predictor loading...")
    _ensure_model()
    print(f"[Vertex] server: http://{HOST}:{PORT} | threads={THREADS} | timeout={CHANNEL_TIMEOUT}s")
    serve(
        app,
        host=HOST,
        port=PORT,
        threads=THREADS,
        channel_timeout=CHANNEL_TIMEOUT,
    )
