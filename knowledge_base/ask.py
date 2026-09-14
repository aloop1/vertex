"""로컬 지식기반에 질문하기 (RAG).

사용법:
    python knowledge_base/ask.py "GA에서 설계 변수로 쓰는 원소가 뭐야?"
    python knowledge_base/ask.py            # 인자 없이 실행하면 대화형 모드
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

TOP_K = 5

SYSTEM_PROMPT = (
    "너는 Vertex 프로젝트(크립 수명 예측 + 합금 설계 GA)의 코드/문서/도메인 자료를 "
    "참고해 답하는 어시스턴트다. 아래 제공되는 '참고 자료'에 근거해서만 답하고, "
    "자료에 없는 내용은 모른다고 말해라. 답변 끝에 참고한 출처 파일명을 나열해라."
)


def _get_collection():
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    try:
        return client.get_collection(cfg.COLLECTION_NAME)
    except Exception:
        print(
            "지식기반이 아직 없습니다. 먼저 다음을 실행하세요:\n"
            "  python knowledge_base/ingest.py"
        )
        sys.exit(1)


def retrieve(collection, question: str, top_k: int = TOP_K):
    q_emb = ollama.embed(model=cfg.EMBED_MODEL, input=[question])["embeddings"][0]
    result = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return list(zip(docs, metas))


def ask(question: str, top_k: int = TOP_K, stream: bool = True) -> str:
    collection = _get_collection()
    hits = retrieve(collection, question, top_k)

    if not hits:
        print("관련 자료를 찾지 못했습니다.")
        return ""

    context_block = "\n\n".join(
        f"[{i+1}] 출처: {meta['source']}\n{doc}" for i, (doc, meta) in enumerate(hits)
    )
    user_msg = f"참고 자료:\n{context_block}\n\n질문: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    sources = sorted({meta["source"] for _, meta in hits})
    print(f"(참고 후보: {', '.join(sources)})\n")

    full_text = []
    if stream:
        for part in ollama.chat(model=cfg.CHAT_MODEL, messages=messages, stream=True):
            token = part["message"]["content"]
            print(token, end="", flush=True)
            full_text.append(token)
        print()
    else:
        resp = ollama.chat(model=cfg.CHAT_MODEL, messages=messages)
        full_text.append(resp["message"]["content"])
        print(full_text[0])

    return "".join(full_text)


def main() -> None:
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print("Vertex 로컬 지식기반 — 질문을 입력하세요 (종료: exit/quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        ask(question)


if __name__ == "__main__":
    main()
