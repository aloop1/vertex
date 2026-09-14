"""vertex 프로젝트 코드/문서 + 도메인 자료를 로컬 벡터 DB(Chroma)에 색인한다.

사용법:
    python knowledge_base/ingest.py            # 변경분만 반영 (증분)
    python knowledge_base/ingest.py --rebuild   # 전체 재색인

전제조건:
    1) Ollama가 실행 중이어야 함 (Windows 트레이에 떠 있으면 됨)
    2) pip install -r knowledge_base/requirements.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import chromadb
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def _iter_repo_files() -> Iterable[tuple[Path, str, str]]:
    """(절대경로, 저장소 기준 상대경로, 카테고리) 튜플을 반환."""
    seen: set[Path] = set()
    for pattern in cfg.REPO_INCLUDE_GLOBS:
        for path in cfg.PROJECT_ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            rel_parts = path.relative_to(cfg.PROJECT_ROOT).parts
            if any(part in cfg.REPO_EXCLUDE_DIRS for part in rel_parts):
                continue
            seen.add(path)
            category = "doc" if path.suffix.lower() in {".md", ".txt"} else "code"
            yield path, str(path.relative_to(cfg.PROJECT_ROOT)).replace("\\", "/"), category


def _iter_domain_files() -> Iterable[tuple[Path, str, str]]:
    cfg.DOMAIN_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(cfg.DOMAIN_DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in cfg.DOMAIN_INCLUDE_EXT:
            rel = str(path.relative_to(cfg.PROJECT_ROOT)).replace("\\", "/")
            yield path, rel, "domain"


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _load_manifest() -> dict:
    if cfg.MANIFEST_PATH.exists():
        return json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}}


def _save_manifest(manifest: dict) -> None:
    cfg.MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _embed_batch(chunks: list[str]) -> list[list[float]]:
    if not chunks:
        return []
    resp = ollama.embed(model=cfg.EMBED_MODEL, input=chunks)
    return resp["embeddings"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="전체 재색인 (기존 컬렉션 삭제 후 새로 생성)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))

    if args.rebuild:
        try:
            client.delete_collection(cfg.COLLECTION_NAME)
        except Exception:
            pass
        manifest = {"files": {}}
    else:
        manifest = _load_manifest()

    collection = client.get_or_create_collection(cfg.COLLECTION_NAME)

    all_files = list(_iter_repo_files()) + list(_iter_domain_files())
    current_rels = {rel for _, rel, _ in all_files}

    added, updated, skipped, removed = 0, 0, 0, 0

    # 삭제된 파일 정리
    for rel in list(manifest["files"].keys()):
        if rel not in current_rels:
            old_ids = manifest["files"][rel].get("chunk_ids", [])
            if old_ids:
                collection.delete(ids=old_ids)
            del manifest["files"][rel]
            removed += 1

    for abs_path, rel, category in all_files:
        try:
            text = _read_text(abs_path)
        except Exception as exc:
            print(f"  [skip] {rel}: 읽기 실패 ({exc})")
            continue

        content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        prior = manifest["files"].get(rel)
        if prior and prior.get("hash") == content_hash:
            skipped += 1
            continue

        if prior and prior.get("chunk_ids"):
            collection.delete(ids=prior["chunk_ids"])

        chunks = _chunk_text(text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        if not chunks:
            manifest["files"][rel] = {"hash": content_hash, "chunk_ids": []}
            continue

        embeddings = _embed_batch(chunks)
        chunk_ids = [f"{rel}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": rel, "category": category, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(ids=chunk_ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        manifest["files"][rel] = {"hash": content_hash, "chunk_ids": chunk_ids}

        if prior:
            updated += 1
            print(f"  [update] {rel} ({len(chunks)} chunks)")
        else:
            added += 1
            print(f"  [add]    {rel} ({len(chunks)} chunks)")

    _save_manifest(manifest)

    print()
    print(f"완료 — 추가 {added} · 업데이트 {updated} · 변경없음(skip) {skipped} · 삭제 {removed}")
    print(f"총 색인 파일 수: {len(manifest['files'])}, 총 청크 수: {collection.count()}")
    print(f"저장 위치: {cfg.CHROMA_DIR}")


if __name__ == "__main__":
    main()
