""" 문서 전처리·Semantic Chunking·임베딩·RAG 인덱스 구축 """

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any

import chromadb
import ollama

try:
    import pymupdf as fitz
except ImportError:
    import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DATA_DIR = PROJECT_ROOT / "data" / "assistant" / "rag_data"
CHROMA_DIR = RAG_DATA_DIR / "chroma_db_bge_m3"
MANIFEST_PATH = RAG_DATA_DIR / "manifest_bge_m3.json"
COLLECTION_NAME = "vertex_knowledge_bge_m3"
SCHEMA_VERSION = 2
BGE_MODEL_NAME = os.getenv("KB_EMBED_MODEL", "bge-m3").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
EMBED_BATCH_SIZE = max(1, int(os.getenv("KB_EMBED_BATCH_SIZE", "8")))
_ollama_client = ollama.Client(host=OLLAMA_HOST)


def _normalize_text(text: str) -> str:
    text = str(text).replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = text.translate(str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff"}))
    # 줄 끝에서 분리된 소문자 단어만 결합한다. 합금명/숫자 하이픈은 유지.
    text = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    dot = "\ue000"
    # 추가 NLP 모델 없이 논문 약어, 저자 이니셜, 소수점의 마침표를 보호.
    abbreviations = (
        r"\b(?:e\.g\.|i\.e\.|et\s+al\.|Fig[s]?\.|Eq[s]?\.|Ref[s]?\.|"
        r"No[s]?\.|Vol\.|pp?\.|Dr\.|Prof\.|vs\.|approx\.|cf\.|wt\.|at\.)"
    )
    text = re.sub(abbreviations, lambda m: m[0].replace(".", dot), text, flags=re.I)
    text = re.sub(r"(?<=\d)\.(?=\d)", dot, text)
    text = re.sub(r"\b(?:[A-Z]\.){2,}", lambda m: m[0].replace(".", dot), text)
    text = re.sub(r"(?<![°◦º\w])[A-Z]\.(?=\s+[A-Z](?:\.|[a-z]))", lambda m: m[0].replace(".", dot), text)
    # PDF가 위첨자 citation을 'period.4 Next'로 추출하는 경우에도 경계 보존.
    text = re.sub(r"(?<=[a-z)])\.([0-9]+(?:[,–-][0-9]+)*)(?=\s+[A-Z])", r". [\1]", text)
    boundary = re.compile(r'''[.!?。](?:["'”’\)\]]*)(?:\s*\[\d[\d,;\s–-]*\])*(?=\s|$)''')
    parts = []
    start = 0
    for match in boundary.finditer(text):
        parts.append(text[start:match.end()].strip().replace(dot, "."))
        start = match.end()
    if text[start:].strip():
        parts.append(text[start:].strip().replace(dot, "."))
    return [part for part in parts if part]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_type(path: Path) -> str:
    parts = {part.lower() for part in path.relative_to(RAG_DATA_DIR).parts[:-1]}
    return "project" if "project_docs" in parts else "domain"


def _discover_files() -> list[Path]:
    if not RAG_DATA_DIR.exists():
        raise FileNotFoundError(f"RAG 자료 폴더가 없습니다: {RAG_DATA_DIR}")
    return sorted(
        path for path in RAG_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        and CHROMA_DIR not in path.parents
        and RAG_DATA_DIR / "index" not in path.parents
    )


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": SCHEMA_VERSION, "embedding_model": BGE_MODEL_NAME, "files": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(MANIFEST_PATH)


def _get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=None,
        metadata={"hnsw:space": "cosine", "schema_version": SCHEMA_VERSION,
                  "embedding_model": BGE_MODEL_NAME},
    )


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", _normalize_text(text).casefold())


def _is_noise_line(text: str) -> bool:
    # 숫자의 양이 아니라 문장이 아닌 독립 라벨/수열만 제거한다.
    return bool(
        not text or not re.search(r"[^\W\d_]", text)
        or re.fullmatch(r"[ivxlcdm]+", text, re.I)
        or re.fullmatch(r"(?:Fig(?:ure)?\.?|Table)\s*[\d.]+[a-z]?[.:]?", text, re.I)
        or re.fullmatch(r"[A-Za-zα-ωΑ-Ω]", text)
    )


def _reference_entry(text: str) -> bool:
    """저자/연도/서지 형식을 함께 확인한다. 본문 citation 번호만으로 지우지 않는다."""
    numbered = re.match(r"^(?:\[\d+\]|\d+[.)]?)\s+", text)
    author = re.match(r"^(?:(?:\[\d+\]|\d+[.)]?)\s+)?[A-ZÀ-ÖØ-Ý][\w’'-]+,?\s+[A-Z](?:\.|[ -]?[A-Z])?[,; .]", text)
    year = re.search(r"\b(?:18|19|20)\d{2}[a-z]?\b", text)
    bibliographic = re.search(r"\b(?:doi|ISBN|ISSN)\b|https?://|\d+\s*[,;:]\s*\d+[–-]\d+", text, re.I)
    named_entry = re.match(r"^[A-ZÀ-ÖØ-Ý][\w’'-]+,\s+[A-Z]\.", text)
    return bool((numbered and author and (bibliographic or year)) or (named_entry and year))


def _ordered_blocks(blocks: list[dict], width: float) -> list[dict]:
    """좌우 본문 열이 구분될 때 각 열을 위에서 아래로 읽는다."""
    middle = width / 2
    left = [b for b in blocks if b["bbox"][2] < middle + width * .03]
    right = [b for b in blocks if b not in left and b["bbox"][0] > middle - width * .03]
    if len(left) < 3 or len(right) < 3:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    # 전폭 제목/그림을 기준으로 영역을 나누어 열 사이에 섞이지 않게 한다.
    wide = [b for b in blocks if b not in left and b not in right]
    result = []
    pending = left + right
    for divider in sorted(wide, key=lambda b: b["bbox"][1]):
        band = [b for b in pending if b["bbox"][1] < divider["bbox"][1]]
        result.extend(sorted(band, key=lambda b: (b["bbox"][0] >= middle - width * .03, b["bbox"][1])))
        pending = [b for b in pending if b not in band]
        result.append(divider)
    result.extend(sorted(pending, key=lambda b: (b["bbox"][0] >= middle - width * .03, b["bbox"][1])))
    return result


def _font_glyph_metrics(data: bytes) -> dict[int, tuple]:
    """TrueType의 실제 glyph 경계/폭을 읽는다. 문자나 문헌 내용은 추측하지 않는다."""
    try:
        count = struct.unpack_from(">H", data, 4)[0]
        tables = {}
        for i in range(count):
            tag, _, offset, length = struct.unpack_from(">4sIII", data, 12 + 16 * i)
            tables[tag] = data[offset:offset + length]
        units = struct.unpack_from(">H", tables[b"head"], 18)[0]
        long_offsets = struct.unpack_from(">h", tables[b"head"], 50)[0]
        glyph_count = struct.unpack_from(">H", tables[b"maxp"], 4)[0]
        metric_count = struct.unpack_from(">H", tables[b"hhea"], 34)[0]
        fmt, stride, scale = (">I", 4, 1) if long_offsets else (">H", 2, 2)
        offsets = [struct.unpack_from(fmt, tables[b"loca"], i * stride)[0] * scale
                   for i in range(min(glyph_count, 256) + 1)]
        metrics = {}
        for gid in range(len(offsets) - 1):
            if offsets[gid + 1] - offsets[gid] < 10:
                continue
            bbox = struct.unpack_from(">hhhh", tables[b"glyf"], offsets[gid] + 2)
            width = struct.unpack_from(">H", tables[b"hmtx"], min(gid, metric_count - 1) * 4)[0]
            metrics[gid] = (units, width, *bbox)
        return metrics
    except (KeyError, struct.error):
        return {}


def _missing_unicode_map(document, xref: int) -> dict[int, str]:
    """Unicode 표 없는 CID font만 동일 glyph를 가진 설치 글꼴로 복원한다."""
    _, extension, _, data = document.extract_font(xref)
    if extension != "ttf" or not data or fitz.Font(fontbuffer=data).valid_codepoints():
        return {}
    embedded = _font_glyph_metrics(data)
    if len(embedded) < 20:
        return {}
    return _matching_font_map(tuple(sorted(embedded.items())))


@lru_cache(maxsize=32)
def _matching_font_map(signature: tuple) -> dict[int, str]:
    embedded = dict(signature)
    font_dir = Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts"
    best = None
    best_score = second_score = 0
    for font_path in sorted(font_dir.glob("*.ttf")):
        candidate = font_path.read_bytes()
        metrics = _font_glyph_metrics(candidate)
        score = sum(metrics.get(gid) == value for gid, value in embedded.items())
        if score > best_score:
            second_score, best_score, best = best_score, score, candidate
        else:
            second_score = max(second_score, score)
        if score == len(embedded):
            break
    # 글꼴 버전 차이는 허용하되, 충분한 glyph가 일치하고 후보가 명확할 때만 복원.
    if best is None or best_score < len(embedded) * .90 or best_score == second_score:
        return {}
    font = fitz.Font(fontbuffer=best)
    mapping = {}
    for codepoint in font.valid_codepoints():
        gid = font.has_glyph(codepoint)
        if gid:
            mapping.setdefault(gid, chr(codepoint))
    return mapping


def _read_pdf(path: Path) -> list[dict]:
    """위치·반복 패턴으로 정제한다. References 이후를 통째로 삭제하지 않는다."""
    pages = []
    margin_pages: dict[str, set[int]] = defaultdict(set)
    font_counts: Counter = Counter()
    font_maps = {}
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, 1):
            repairs = {}
            for xref, _, _, name, _, encoding, *_ in page.get_fonts(full=True):
                if encoding != "Identity-H":
                    continue
                if xref not in font_maps:
                    font_maps[xref] = _missing_unicode_map(document, xref)
                if font_maps[xref]:
                    repairs[name.split("+")[-1]] = font_maps[xref]
            flags = fitz.TEXTFLAGS_DICT
            if repairs:
                flags |= fitz.TEXT_INHIBIT_SPACES
            blocks = []
            for block in page.get_text("dict", sort=True, flags=flags)["blocks"]:
                if block["type"] != 0:
                    continue
                lines = []
                for line in block["lines"]:
                    text = "".join(
                        "".join(repairs[span["font"]].get(ord(char), char) for char in span["text"])
                        if span["font"] in repairs else span["text"]
                        for span in line["spans"]
                    ).strip()
                    if not text:
                        continue
                    size = max(span["size"] for span in line["spans"])
                    font_counts[round(size, 1)] += len(text)
                    y0, y1 = line["bbox"][1], line["bbox"][3]
                    margin = y1 < page.rect.height * .12 or y0 > page.rect.height * .90
                    if margin and len(text) < 160:
                        margin_pages[_margin_key(text)].add(page_number)
                    lines.append({"text": text, "margin": margin, "size": size})
                if lines:
                    blocks.append({"bbox": block["bbox"], "lines": lines})
            pages.append((page_number, _ordered_blocks(blocks, page.rect.width)))
    repeated = {key for key, numbers in margin_pages.items() if len(numbers) >= 3}
    body_size = font_counts.most_common(1)[0][0] if font_counts else 10
    records = []
    section = 0
    in_references = False
    for page_number, blocks in pages:
        # 참고문헌이 계속되는 페이지에서만 상태를 유지한다.
        has_entries = any(
            _reference_entry(_normalize_text(" ".join(line["text"] for line in block["lines"][i:i + 5])))
            for block in blocks for i in range(len(block["lines"]))
        )
        if not has_entries:
            in_references = False
        for block in blocks:
            raw_lines = [line for line in block["lines"] if not (
                line["margin"] and _margin_key(line["text"]) in repeated
            )]
            raw_text = _normalize_text("\n".join(line["text"] for line in raw_lines))
            if not raw_text:
                continue
            if re.fullmatch(r"(?:\d+(?:\.\d+)*\.?\s*)?(?:References|Bibliography|Works cited)", raw_text, re.I):
                in_references = True
                section += 1
                continue
            is_heading = (
                len(raw_text) < 160 and not re.search(r"[.!?]\s*$", raw_text)
                and (max(line["size"] for line in raw_lines) > body_size * 1.13
                     or re.match(r"^\d+(?:\.\d+)+\s+[A-Z]", raw_text))
            )
            if is_heading:
                in_references = False
                section += 1
                continue
            if _reference_entry(raw_text):
                in_references = True
                section += 1
                continue
            if in_references:
                # 서지 항목과 함께 이어지는 줄만 제거하며 새 절 제목은 위에서 해제한다.
                if has_entries:
                    continue
                in_references = False
            lines = [line["text"] for line in raw_lines if not _is_noise_line(line["text"])]
            if not lines:
                continue
            # 축/표의 짧은 라벨을 한 문장으로 합치지 않는다.
            prose_lines = [line for line in lines if len(re.findall(r"\b[^\W\d_]{2,}\b", line)) >= 4]
            text = _normalize_text("\n".join(lines))
            if not prose_lines and not re.search(r"[.!?。](?:[\])0-9]*)$", text):
                continue
            records.append({"page": page_number, "section": section, "text": text})
    return records


def _read_text_file(path: Path) -> list[dict]:
    text = _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    return [{"page": 0, "section": 0, "text": text}] if text else []


def _read_document(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    if path.suffix.lower() in {".md", ".txt"}:
        return _read_text_file(path)
    raise ValueError(f"지원하지 않는 문서 형식: {path}")


def _sentence_records(page_records: list[dict]) -> list[dict]:
    sentences = []
    for record in page_records:
        for text in _split_sentences(record["text"]):
            if _is_noise_line(text) or _reference_entry(text):
                continue
            if len(re.findall(r"\b[^\W\d_]{2,}\b", text)) < 2:
                continue
            sentences.append({"page": int(record["page"]), "section": record.get("section", 0), "text": text})
    return sentences


def _build_chunks(path: Path) -> list[dict]:
    """기존 함수명 유지. 한 레코드는 한 검색 문장과 같은 페이지/절의 인접 문맥."""
    sentences = _sentence_records(_read_document(path))
    relative_path = path.relative_to(RAG_DATA_DIR).as_posix()
    chunks = []
    for index, sentence in enumerate(sentences):
        neighbors = [item for item in sentences[max(0, index - 1):index + 2]
                     if (item["page"], item["section"]) == (sentence["page"], sentence["section"])]
        context = " ".join(item["text"] for item in neighbors)
        metadata = {
            "source": path.name, "source_path": relative_path,
            "source_type": _source_type(path), "file_type": path.suffix.lower().lstrip("."),
            "title": path.stem, "page": sentence["page"],
            "page_start": sentence["page"], "page_end": sentence["page"],
            "sentence_index": index, "search_text": sentence["text"], "context_text": context,
        }
        chunks.append({"text": sentence["text"], "metadata": metadata})
    return chunks


def _extract_embeddings(response: Any) -> list[list[float]]:
    embeddings = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Ollama embed 응답에 embeddings가 없습니다.")
    return [list(map(float, vector)) for vector in embeddings]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = _ollama_client.embed(model=BGE_MODEL_NAME, input=texts, keep_alive="30m", truncate=False)
        vectors = _extract_embeddings(response)
    except (AttributeError, TypeError):
        vectors = []
        for text in texts:
            response = _ollama_client.embeddings(model=BGE_MODEL_NAME, prompt=text, keep_alive="30m")
            vector = response.get("embedding") if isinstance(response, dict) else response.embedding
            vectors.append(list(map(float, vector)))
    if len(vectors) != len(texts):
        raise RuntimeError("임베딩 수와 문장 수가 일치하지 않습니다.")
    return vectors


def _unload_generation_model() -> None:
    _ollama_client.generate(model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct").strip(), keep_alive=0)


def _unload_bge() -> None:
    try:
        _ollama_client.embed(model=BGE_MODEL_NAME, input=[], keep_alive=0)
    except Exception as exc:
        print(f"[경고] BGE-M3 unload 실패: {exc}")


def _chunk_ids(relative_path: str, file_hash: str, chunks: list[dict]) -> list[str]:
    return [hashlib.sha256(f"{SCHEMA_VERSION}|{relative_path}|{file_hash}|{index}".encode()).hexdigest()
            for index in range(len(chunks))]


def build_index(*, rebuild: bool = False) -> None:
    files = _discover_files()
    client = _get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
    collection = _get_collection(client)
    manifest = _load_manifest()
    if (manifest.get("version") != SCHEMA_VERSION
            or manifest.get("embedding_model") != BGE_MODEL_NAME
            or (collection.metadata or {}).get("schema_version") != SCHEMA_VERSION
            or (collection.metadata or {}).get("embedding_model") != BGE_MODEL_NAME):
        raise RuntimeError("색인 schema/model이 다릅니다. python -m analysis.rag.indexer --rebuild 를 실행하세요.")
    current_paths = {path.relative_to(RAG_DATA_DIR).as_posix() for path in files}
    deleted_paths = sorted(set(manifest["files"]) - current_paths)
    for relative_path in deleted_paths:
        collection.delete(where={"source_path": relative_path})
        manifest["files"].pop(relative_path)
        _save_manifest(manifest)
        print(f"[삭제 반영] {relative_path}")
    changed_count = skipped_count = 0
    embedding_started = False
    try:
        for path in files:
            relative_path = path.relative_to(RAG_DATA_DIR).as_posix()
            file_hash = _sha256_file(path)
            previous = manifest["files"].get(relative_path)
            stored_ids = set(collection.get(where={"source_path": relative_path}, include=[])["ids"])
            if previous and previous.get("sha256") == file_hash and stored_ids == set(previous.get("chunk_ids", [])):
                skipped_count += 1
                print(f"[변경 없음] {relative_path}", flush=True)
                continue
            print(f"[색인] {relative_path}", flush=True)
            chunks = _build_chunks(path)
            new_ids = _chunk_ids(relative_path, file_hash, chunks)
            new_id_set = set(new_ids)
            if chunks and not embedding_started:
                _unload_generation_model()
                embedding_started = True
            # 8GB RAM: 문헌 전체 벡터를 Python 메모리에 쌓지 않는다.
            try:
                for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                    batch = chunks[start:start + EMBED_BATCH_SIZE]
                    embeddings = _embed_batch([chunk["text"] for chunk in batch])
                    collection.upsert(
                        ids=new_ids[start:start + len(batch)], embeddings=embeddings,
                        documents=[chunk["text"] for chunk in batch],
                        metadatas=[chunk["metadata"] for chunk in batch],
                    )
                    print(f"[BGE-M3] 문장 {start + len(batch)}/{len(chunks)}", flush=True)
            except BaseException:
                # 실패하면 구버전을 유지한다. 중단 후에도 같은 ID로 upsert 가능.
                added = list(new_id_set - stored_ids)
                for start in range(0, len(added), 256):
                    collection.delete(ids=added[start:start + 256])
                raise
            stale_ids = list(stored_ids - new_id_set)
            for start in range(0, len(stale_ids), 256):
                collection.delete(ids=stale_ids[start:start + 256])
            manifest["files"][relative_path] = {
                "sha256": file_hash, "file_type": path.suffix.lower().lstrip("."),
                "source_type": _source_type(path), "chunk_ids": new_ids, "chunk_count": len(new_ids),
            }
            _save_manifest(manifest)
            changed_count += 1
            print(f"[완료] {relative_path} / {len(new_ids)} sentences", flush=True)
    finally:
        if embedding_started:
            _unload_bge()
    _save_manifest(manifest)
    print(f"[색인 완료] 문서={len(files)} 변경={changed_count} 유지={skipped_count} 삭제={len(deleted_paths)}")
    print(f"Chroma 문장 수: {collection.count()}\nDB: {CHROMA_DIR}\nManifest: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Knowledge BGE-M3 collection만 문장 단위로 재구축")
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()