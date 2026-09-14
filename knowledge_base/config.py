"""로컬 지식기반(RAG) 설정.

Ollama를 임베딩·생성 백엔드로 사용한다. 모델 이름은 `ollama list`로 이미 받아둔
모델과 맞춰져 있으니, 다른 모델을 쓰려면 환경변수로 덮어쓰면 된다.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# ── Ollama 연결 ──────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# bge-m3는 다국어(한국어+영어 혼합) 문서에 강함 — 회의록(한국어)과 코드(영어)가
# 섞여 있는 이 프로젝트에 적합. nomic-embed-text로 바꾸려면 환경변수로 지정.
EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "bge-m3")

# 답변 생성 모델. 더 큰 모델(qwen3:14b, gemma3:12b)로 바꾸려면 환경변수로 지정.
CHAT_MODEL = os.environ.get("KB_CHAT_MODEL", "qwen2.5:7b-instruct")

# ── 저장 위치 ────────────────────────────────────────────────────────────────
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "vertex_kb"
MANIFEST_PATH = BASE_DIR / "manifest.json"

# ── 청킹 ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1200      # 문자 수 기준
CHUNK_OVERLAP = 200

# ── 소스 1: 이 저장소의 코드/문서 (glob 패턴, PROJECT_ROOT 기준) ──────────────
REPO_INCLUDE_GLOBS = [
    "README.md",
    "CLAUDE.md",
    "data/readme.txt",
    "documents/*.md",
    "*.py",
    "data_preprocessing.py",
    "데이터전처리.py",
    "models/*.py",
    "ga/*.py",
    "tools/*.py",
    "web/*.py",
    "web/templates/*.html",
    "web/static/*.js",
    "web/static/*.css",
]
REPO_EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", "knowledge_base", "web/uploads"}

# ── 소스 2: 사용자가 직접 넣는 크립/합금 도메인 자료 ──────────────────────────
# .md, .txt, .pdf 파일을 이 폴더에 넣고 ingest.py를 다시 실행하면 반영된다.
DOMAIN_DOCS_DIR = BASE_DIR / "domain_docs"
DOMAIN_INCLUDE_EXT = {".md", ".txt", ".pdf"}
