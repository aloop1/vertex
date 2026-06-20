"""Vertex — 프로덕션 서버 진입점 (waitress)

사용법:
    python web/serve.py                 # 기본 0.0.0.0:5000
    PORT=8080 python web/serve.py       # 포트 변경
    VERTEX_THREADS=16 python web/serve.py

개발 시에는 `python web/app.py`(Flask dev server)를 사용하세요.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waitress import serve

from app import app, _ensure_model

HOST    = os.environ.get("VERTEX_HOST", "0.0.0.0")
PORT    = int(os.environ.get("PORT", os.environ.get("VERTEX_PORT", 5000)))
THREADS = int(os.environ.get("VERTEX_THREADS", 8))

if __name__ == "__main__":
    print(f"[Vertex] 모델 로딩 중...")
    _ensure_model()
    print(f"[Vertex] 프로덕션 서버 시작: http://{HOST}:{PORT} (threads={THREADS})")
    serve(app, host=HOST, port=PORT, threads=THREADS,
          channel_timeout=300)  # sweep 재계산이 길어질 수 있어 타임아웃 여유
