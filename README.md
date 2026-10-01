# 크립 수명 예측을 위한 AI 솔루션

> 고온·고압 환경 핵심 소재의 크립(Creep) 파단 수명을 예측하고, 의사결정을 돕는 AI 챗봇 시스템

---

## 📁 프로젝트 구조

```text
vertex/
├── data/
│   ├── taka.xlsx               # 원본 데이터 (2066행 × 31열)
│   ├── creep.csv               # 추가 크립 데이터 (1024행 × 42열)
│   ├── creep_data.csv          # 추가 크립 데이터 (265행 × 25열)
│   ├── preprocessor.pkl        # 저장된 StandardScaler + 피처 메타정보
│   ├── correlation_heatmap.png # 전처리 결과 변수 간 상관관계 히트맵
│   └── assistant/
│       └── rag_data/           # RAG 문헌 및 Chroma 벡터 DB
│       └── what_if__data/      # API 호출 캐시
│
├── documents/
│   └── 회의록.md                # 팀 프로젝트 진행 기록
│
├── models/
│   ├── transformer_and_tree_ensemble.py # 커스텀 Transformer + Tree 앙상블 모델
│   ├── custom_histogram.py     # Histogram + RBF 기반 앙상블 학습
│   └── LMP_데이터증강.py        # 두 모델이 재사용하는 LMP 기반 데이터 증강
│
├── analysis/
│   ├── pipeline.py             # 챗봇 의도 분류 및 전체 AI 기능 통합
│   ├── rag/
│   │   ├── indexer.py          # 문헌 전처리·문장 임베딩·Chroma 인덱싱
│   │   └── retriever.py        # BGE-M3 기반 관련 문헌 검색
│   ├── local_llm/
│   │   ├── model.py            # Ollama Local LLM 실행
│   │   └── explainer.py        # 검색 문헌 기반 답변 생성
│   └── what_if/
│       └── service.py          # AI API 기반 What-if 변경 후보 생성
│
├── web/
│   ├── app.py                  # Flask 웹 애플리케이션 및 AI Pipeline 연동
│   ├── serve.py                # Waitress 프로덕션 서버 실행
│   │
│   ├── static/
│   │   ├── vertex.css          # 공통 UI 스타일
│   │   └── theme.js            # 다크/라이트 테마 및 UI 스크립트
│   │
│   └── templates/
│       ├── index.html          # 메인 입력 화면
│       ├── result.html         # 크립 수명 예측 및 AI 분석 결과 화면
│       └── _loading_overlay.html
│
├── data_preprocessing.py       # 이전 데이터 전처리 및 피처 엔지니어링
├── 데이터전처리.py              # 추가 데이터셋 병합 및 데이터 전처리
├── requirements.txt            # 프로젝트 라이브러리 의존성 목록
└── README.md                   # 본 문서
```


### 1. 데이터 전처리 및 피처 엔지니어링 (`데이터전처리.py`)

- 원본 데이터(taka.xlsx) 로드: **2066행 × 31열**
- 데이터 정제: 결측치 처리(합금 성분 NaN → 0) 및 물리적 무결성 검사 (음수 수명/온도 필터링)
- 이상치 정책: 응력(Stress) 변수의 통계적 이상치(14%)는 실제 실험 인풋 조건(5~450MPa)으로 확인되어 제거 없이 도메인 지식을 반영하여 유지
- 물리 기반 피처 엔지니어링:
  - Severity Index (가혹도 지수) 3종 추가: `N/T/A_severity`
  - 소재 도메인 지식(Hollomon-Jaffe 파라미터)을 응용하여 온도-시간 비선형 관계 수치화
- 피처 최적화:
  - 오스테나이트계 합금 특성상 수명 영향력이 미미한 냉각 방식(Cooling1/2/3) 변수 제거
  - 무의미한 화학 성분 및 노이즈 컬럼 제거를 통한 모델 경량화
- 제품군 단위 데이터 분할 (Group-based Split):
  - 문제 해결: 단순 무작위 분할 시 발생하는 데이터 누수(Data Leakage) 문제를 차단하기 위해 합금 조성비 기준 Group ID 생성
  - 검증 방식: 모델 학습 코드에서 조성 그룹을 섞고 목표 행 수에 도달할 때까지 그룹을 선택하는 직접 구현 분할 사용. 외부 테스트 목표 비율 20%, 외부 학습 세트 내부 검증 목표 비율 15%. 같은 조성 그룹은 학습·평가 양쪽에 들어가지 않음
- **최종 피처 수: 30개**


### 2. 커스텀 Transformer + 트리 앙상블 모델 (`models/transformer_and_tree_ensemble.py`)

- Transformer 인코더 기반 변수 간 상호작용 학습
  - 각 수치 피처를 토큰으로 변환
  - 다중 헤드 자기어텐션을 직접 구현하여 조성, 운전 조건, 열처리, 물리 파생 변수 간 관계 학습
- 트리 기반 앙상블 보정
  - CART 방식 회귀트리를 직접 구현
  - 부트스트랩 앙상블을 구성하여 Transformer 예측 잔차를 보정
- 물리 기반 파생 변수 사용
  - 운전 가혹도 지수
  - 응력-온도 상호작용
  - 역온도
  - 총 열처리 가혹도
- LMP는 수명 타깃을 포함하므로 일반 입력 피처에는 사용하지 않음. 학습 세트의 제한적 데이터 증강과 예측 후 물리 검증에 사용

- **모델 성능:**

| 스케일 | RMSE | R² |
|--------|------|----|
| log10 | 0.6517 | 0.5520 |
| 시간(hours) | 11,231.6 | - |

- **물리 반응 검증:**

| 검증 항목 | 결과 |
|----------|------|
| LMP R² | 0.9775 |
| 운전 가혹도-예측 수명 Spearman 상관 | -0.8256 |
| 온도 sweep 기울기 | -0.001104 |
| 고온 조건 응력 sweep 기울기 | -0.001646 |

- 검증 결과:
  - 온도 증가 시 예측 수명이 감소하는 경향 확인
  - 고온 조건에서 응력 증가 시 예측 수명이 감소하는 경향 확인
  - 운전 가혹도 지수가 증가할수록 예측 수명이 감소하는 음의 상관 확인


### 3. Histogram 기반 앙상블 모델 (`models/custom_histogram.py`)

기존 Transformer 모델과 별도로 추가한 모델입니다. 핵심 학습 알고리즘은 NumPy로 직접 구현하고, 데이터프레임 처리는 pandas를 사용합니다.

- Histogram Gradient Boosting: 실제 학습 관측값의 분위수로 최대 128개 bin을 만들고 가중 CART 트리로 잔차를 순차 학습
- 기본 앙상블: Histogram과 Gaussian RBF ridge 회귀 2개를 각각 0.68175, 0.27075, 0.04750 비율로 결합
- 최종 앙상블: 기본 앙상블 외에 LMP-RBF, 수명 RBF, 무작위 비선형 특징 회귀, 이웃 회귀 후보를 내부 검증에서 평가하고 결합 가중치를 선택
- 입력: 공통 30개 피처와 물리 파생 피처 4개, 총 34개
- 타깃: `log10(lifetime)` 학습, `10 ** prediction`으로 hours 복원
- 분할: 조성 그룹 기반 외부 train/test와 내부 train/validation 분리
- LMP 증강: 학습 세트에만 추가. 비율 0.5, 그룹당 최대 40행, 원본 가중치 1.0, 합성 가중치 0.35
- Histogram bin 경계와 거리 기반 모델의 입력 통계는 해당 분할의 실제 학습 행에서 계산
- 내부 검증에서 결합 가중치와 선택적 slope·intercept 보정 결정 후 외부 테스트 평가

#### 제공 아티팩트의 기록된 성능

| 지표 | 결과 |
|---|---:|
| RMSE (log10) | 0.392946 |
| MAE (log10) | 0.273061 |
| R² (log10) | 0.837122 |
| RMSE (hours) | 10,260.676 |
| MAE (hours) | 2,901.502 |

- 외부 원본 학습 2,682행, 최종 학습 합성 618행, 외부 테스트 673행
- Seed: 외부 분할 42, 내부 분할 59, 내부 증강 73, 최종 증강 95

### 4. AI 기반 크립 수명 분석 및 의사결정 지원 (`analysis/`)

사용자의 질문을 의미 기반으로 분류한 뒤,  
질문의 목적에 따라 크립 수명 예측, What-if 분석, 문헌 검색 기반 답변을 수행함

```text
                       사용자 질문
                            ↓
                      질문 의도 분류
                            ↓
┌────────────┬──────────────┬──────────────┬──────────────┐
│ Prediction │   What-if    │  Knowledge   │ General Chat │
└─────┬──────┴──────┬───────┴──────┬───────┴──────┬───────┘
      ↓             ↓              ↓              ↓
수명예측 모델       API          BGE-M3        Local LLM
      ↓         후보 생성      + Chroma          (경량)
  예상 수명          ↓              ↓
               수명예측 모델   관련 문헌 검색
                    ↓              ↓
               Before/After    Local LLM
                    ↓              ↓
                결과 해석      근거 기반 답변

```

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| ML / Data | Pandas, NumPy, Scikit-learn, PyTorch, Joblib |
| 크립 수명 예측 | Custom Transformer + Tree Ensemble / Custom Histogram + RBF 기반 앙상블 |
| 질문 의도 분류 | multilingual-e5-small, One-vs-Rest Logistic Regression |
| What-if 분석 | Gemini API |
| RAG 임베딩 | BGE-M3 |
| Vector DB | ChromaDB |
| Local LLM | Ollama 기반 Local LLM |
| Backend | Flask, Waitress |
| Frontend | HTML, CSS, JavaScript |
| 데이터 시각화 | Plotly |
| 모델 저장 | Transformer: `.pkl`, `.pt` / Histogram: `.pkl` |
| 버전 관리 | Git, GitHub |
