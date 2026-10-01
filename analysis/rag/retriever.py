"""Knowledge 검색: 질문 임베딩 1회 → Chroma 문장 검색 → 저장된 문맥.

Router의 multilingual-e5-small과 기존 pipeline/explainer 인터페이스는 유지한다.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import chromadb
import ollama

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DATA_DIR = PROJECT_ROOT / "data" / "assistant" / "rag_data"
CHROMA_DIR = RAG_DATA_DIR / "chroma_db_bge_m3"
COLLECTION_NAME = "vertex_knowledge_bge_m3"
SCHEMA_VERSION = 2
BGE_MODEL_NAME = os.getenv("KB_EMBED_MODEL", "bge-m3").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
ROUTER_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
_router_embedding_model: SentenceTransformer | None = None
_chroma_client = None
_collection = None
_ollama_client = ollama.Client(host=OLLAMA_HOST)


def get_embedding_model() -> SentenceTransformer:
    """pipeline Router 전용 E5. Knowledge 검색에서는 호출하지 않는다."""
    global _router_embedding_model
    if _router_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _router_embedding_model = SentenceTransformer(ROUTER_EMBEDDING_MODEL_NAME)
    return _router_embedding_model


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("BGE-M3 DB가 없습니다. python -m analysis.rag.indexer --rebuild 를 실행하세요.")
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = _chroma_client.get_collection(COLLECTION_NAME, embedding_function=None)
    except chromadb.errors.NotFoundError as exc:
        raise RuntimeError("Knowledge collection이 없습니다. python -m analysis.rag.indexer --rebuild 를 실행하세요.") from exc
    metadata = collection.metadata or {}
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("embedding_model") != BGE_MODEL_NAME:
        raise RuntimeError("문장 단위 색인이 필요합니다. python -m analysis.rag.indexer --rebuild 를 실행하세요.")
    if collection.count() == 0:
        raise RuntimeError("Chroma collection이 비어 있습니다. python -m analysis.rag.indexer 를 실행하세요.")
    _collection = collection
    return collection


def load_index():
    return _get_collection()


def clear_index_cache() -> None:
    global _chroma_client, _collection
    _collection = None
    _chroma_client = None


def warmup_retriever() -> None:
    get_embedding_model()
    load_index()


def _embed_query(question: str) -> list[float]:
    _ollama_client.generate(model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct").strip(), keep_alive=0)
    try:
        response = _ollama_client.embed(model=BGE_MODEL_NAME, input=[question], keep_alive=0)
        vectors = response.get("embeddings") if isinstance(response, dict) else response.embeddings
        if not vectors or len(vectors) != 1:
            raise RuntimeError("BGE-M3가 query embedding 하나를 반환하지 않았습니다.")
        return list(map(float, vectors[0]))
    except (AttributeError, TypeError):
        # 기존 ollama-python 인터페이스도 질문 하나만 임베딩한다.
        response = _ollama_client.embeddings(model=BGE_MODEL_NAME, prompt=question, keep_alive=0)
        vector = response.get("embedding") if isinstance(response, dict) else response.embedding
        if not vector:
            raise RuntimeError("BGE-M3가 빈 query embedding을 반환했습니다.")
        return list(map(float, vector))


def _evidence_key(text: str) -> str:
    # 대소문자/공백/구두점 차이만 무시한다. 의미나 숫자가 다른 근거는 합치지 않는다.
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def retrieve(question: str, top_k: int = 5, *, source_type: str | None = None) -> list[dict]:
    question = " ".join(question.strip().split())
    if not question or top_k <= 0:
        return []
    collection = _get_collection()
    kwargs: dict[str, Any] = {
        "query_embeddings": [_embed_query(question)],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if source_type:
        kwargs["where"] = {"source_type": source_type}
    raw = collection.query(**kwargs)
    results = []
    seen: set[str] = set()
    for document, metadata, distance in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        item = dict(metadata or {})
        context = item.get("context_text")
        if not context or not item.get("search_text"):
            raise RuntimeError("저장된 문장/문맥이 없습니다. python -m analysis.rag.indexer --rebuild 를 실행하세요.")
        key = _evidence_key(context)
        if key in seen:
            continue
        seen.add(key)
        score = 1.0 - float(distance)
        # 기존 호출부가 읽는 text는 답변용 문맥. 문장 검색 점수와 metadata는 보존.
        item.update(text=context, score=score, semantic_score=score)
        results.append(item)
    return results


def compress_results_for_llm(
    question: str, results: list[dict], *, max_total_chars: int = 1100, **_: object,
) -> list[dict]:
    """기존 explainer의 최상위 근거 하나를 유지하고 저장된 문맥을 그대로 전달한다.

    max_total_chars는 호출 호환용 soft limit. 검색 문장/이웃을 도중에 잘라
    비교 조건을 잃지 않도록 인덱싱 때 완성한 최대 세 문장을 보존한다.
    """
    if not question.strip() or not results or max_total_chars <= 0:
        return []
    best = dict(results[0])
    best["text"] = best.get("context_text", best.get("text", ""))
    return [best] if best["text"] else []


def _print_results(results: list[dict]) -> None:
    if not results:
        print("[RAG] 검색 결과 없음")
    for index, result in enumerate(results, 1):
        page = f" / p.{result['page']}" if result.get("page") else ""
        print(f"\n[{index}] score={result['score']:.4f} {result['source']}{page}")
        print(f"검색 문장: {result['search_text']}")
        print(f"답변 문맥: {result['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="생략하면 질문을 입력받습니다.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    question = args.question if args.question is not None else input("질문을 입력하세요: ")
    _print_results(retrieve(question, top_k=args.top_k))
