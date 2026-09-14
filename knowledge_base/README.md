# Vertex 로컬 지식기반 (Ollama RAG)

이 프로젝트의 코드/문서와 사용자가 추가하는 크립/합금 도메인 자료를
로컬 LLM(Ollama)이 검색해서 답변에 활용하도록 하는 간단한 RAG 파이프라인입니다.
외부로 데이터가 나가지 않고 전부 로컬에서 동작합니다.

## 준비물

- Ollama가 실행 중이어야 합니다 (Windows 트레이 아이콘으로 떠 있으면 OK).
- 아래 모델이 필요합니다 (이미 받아둔 상태라면 생략):
  ```bash
  ollama pull bge-m3              # 임베딩 (다국어 — 한국어 회의록 + 영어 코드에 적합)
  ollama pull qwen2.5:7b-instruct # 답변 생성
  ```
- Python 패키지 설치:
  ```bash
  pip install -r knowledge_base/requirements.txt
  ```

## 사용법

### 1. 도메인 자료 추가 (선택)

크립/합금 관련 논문·자료를 `knowledge_base/domain_docs/` 폴더에 `.md`, `.txt`,
`.pdf` 형식으로 넣으세요. 하위 폴더로 정리해도 됩니다.

### 2. 색인 생성

```bash
python knowledge_base/ingest.py
```

- 프로젝트 코드(`web/`, `ga/`, `models/`, 루트 `*.py`)와 문서(`README.md`,
  `CLAUDE.md`, `documents/*.md`), 그리고 `domain_docs/`의 자료를 모두 읽어
  청크로 나누고 임베딩해서 `knowledge_base/chroma_db/`에 저장합니다.
- 파일 내용 해시를 기록해두므로, 다시 실행하면 **변경된 파일만** 재색인합니다.
- 전체를 처음부터 다시 만들고 싶으면 `--rebuild` 옵션을 추가하세요.

### 3. 질문하기

```bash
python knowledge_base/ask.py "GA에서 설계 변수로 쓰는 원소가 뭐야?"
```

인자 없이 실행하면 대화형 모드로 들어갑니다:

```bash
python knowledge_base/ask.py
```

## 다른 모델 쓰기

기본값은 임베딩 `bge-m3`, 답변 생성 `qwen2.5:7b-instruct`입니다.
환경변수로 바꿀 수 있습니다 (예: 더 큰 모델로):

```bash
set KB_CHAT_MODEL=qwen3:14b
python knowledge_base/ask.py "..."
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 모델 이름, 청킹 크기, 색인 대상 경로 설정 |
| `ingest.py` | 코드/문서/도메인 자료 → 청크 → 임베딩 → Chroma 저장 (증분 지원) |
| `ask.py` | 질문 임베딩 → 유사 청크 검색 → 컨텍스트와 함께 LLM에 질의 |
| `domain_docs/` | 사용자가 추가하는 도메인 자료 (git에는 커밋되지 않음) |
| `chroma_db/` | 벡터 DB 저장소 (git에는 커밋되지 않음, 로컬 생성물) |
| `manifest.json` | 파일별 해시 기록 (증분 색인용, git에는 커밋되지 않음) |
