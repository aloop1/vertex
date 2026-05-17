# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크리프(Creep) 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

---

## 📁 프로젝트 구조

```
vertex/
├── data/
│   ├── taka.xlsx              # 원본 데이터 (2066행 × 31열)
│   ├── creep.csv              # 추가 크립 데이터 (1024행 × 42열)
│   ├── creep_data.csv         # 추가 크립 데이터 (265행 × 25열)
│   ├── preprocessor.pkl       # 저장된 StandardScaler + 피처 메타정보
│   └── correlation_heatmap.png # 전처리 결과 변수 간 상관관계 히트맵
├── documents/
│   └── 회의록.md               # 팀 프로젝트 진행 기록
├── models/
│   ├── train.py               # XGBoost 모델 학습/평가
│   ├── transformer_and_tree_ensemble.py # 커스텀 모델 학습/평가
│   └── compare_models.py      # MLP, RF 모델 학습/ 평가
├── ga/                        
│   ├── __init__.py            
│   ├── config.py              # 전체 설정 관리
│   ├── physics.py             # 물리 제약 및 penalty 계산
│   ├── llm.py                 # 초기 seed 생성
│   └── engine.py              # GA 최적화 실행
├── plots/
│   └── 성능 비교 그래프        # 다양한 알고리즘 간의 수명 예측 정확도 비교
├── data_preprocessing.py      # 이전 데이터 전처리 및 피처 엔지니어링
├── 데이터전처리.py             # 새롭게 추가된 데이터셋 병합 및 데이터 전처리
├── streamlit_app.py           # Streamlit 기반 웹 애플리케이션 실행 파일
├── requirements.txt           # 프로젝트 라이브러리 의존성 목록
└── README.md                  # 본 문서
```

---

### 1. 데이터 전처리 및 피처 엔지니어링 (`data_preprocessing.py`)
- 원본 데이터(taka.xlsx) 로드: **2066행 × 31열**
- 데이터 정제: 결측치 처리(합금 성분 NaN → 0) 및 물리적 무결성 검사 (음수 수명/온도 필터링)
- 이상치 정책: 응력(Stress) 변수의 통계적 이상치(14%)는 실제 실험 인풋 조건(5~450MPa)으로 확인되어 제거 없이 도메인 지식을 반영하여 유지
- 물리 기반 피처 엔지니어링:
  * Severity Index (가혹도 지수) 3종 추가: ```N/T/A_severity```
  * 소재 도메인 지식(Hollomon-Jaffe 파라미터)을 응용하여 온도-시간 비선형 관계 수치화
- 피처 최적화
  * 오스테나이트계 합금 특성상 수명 영향력이 미미한 냉각 방식(Cooling1/2/3) 변수 제거
  * 무의미한 화학 성분 및 노이즈 컬럼 제거를 통한 모델 경량화
- 제품군 단위 데이터 분할 (Group-based Split):
  * 문제 해결: 단순 무작위 분할 시 발생하는 데이터 누수(Data Leakage) 문제를 차단하기 위해 합금 조성비 기준 Group ID 생성
  * 검증 방식: GroupShuffleSplit을 활용, 학습 시 보지 못한 완전히 새로운 신규 합금 제품군에 대한 예측 성능을 평가함 (총 154개 제품군 중 20%를 테스트셋으로 격리)
- **최종 피처 수: 30개**

### 2. XGBoost 베이스라인 모델 (`models/train.py`)
- XGBoost 회귀 모델 학습 (500 estimators, max_depth=6, lr=0.05)
- **모델 성능:**

| 스케일 | RMSE | R² |
|--------|------|----|
| log10 | 0.2793 | **0.9168** |
| 시간(hours) | 8,741.4 | 0.7533 |

- 모델 저장: `xgb_baseline.json`
- 시각화: 예측 vs 실제 산점도, 피처 중요도 그래프

### 3. 커스텀 Transformer + 트리 앙상블 모델 (`models/transformer_and_tree_ensemble.py`)
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
- LMP는 수명 타깃을 포함하므로 학습 입력에는 사용하지 않고, 예측 후 물리 검증 지표로만 사용

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
---

### 2. Physics-Informed GA 최적화 (`ga/`)
- 예측 모델 연동:
  * `preprocessor.pkl`, `selected_features.json`, `resnet_best.pt`를 불러와 GA 후보 조성에 대한 수명 예측 수행

- 탐색 변수 정의:
  * GA가 직접 조절할 원소 11개 선정: `C, Si, Mn, Cr, Mo, W, Ni, V, Nb, N, B`
  * 불순물·고가·데이터 부족 원소는 고정값 처리: `P, S, O, Co, Re, Al, Cu, Ta`

- 물리 기반 제약 조건 반영:
  * KN 계수, Ms temperature, Laves phase 위험도, Z-phase 위험도, CEQ, MX balance등 penalty로 반영
  * 총합금량 15 wt% 제한 및 원소 단가 기반 비용 계산 추가

- 다목적 최적화:
  * DEAP 기반 NSGA-II 적용
  * 최적화 목표: 예측 수명 최대화, 재료 비용 최소화, (물리적 위험도 최소화)

- 합금 조성 보정 장치:
  * 교차/변이 이후 합금 총합이 15 wt%를 초과하지 않도록 repair 로직 추가
  * 개별 원소 범위와 전체 합금량 제한을 동시에 만족하도록 보정

- LLM 기반 초기 seed 생성:
  * Gemini API로 초기 합금 조성 seed 생성 구조 설계
  * 개발 중에는 local/cached seed를 사용하고, 최종 검증 시에만 API를 호출한 뒤 저장된 JSON seed 재사용

- 결과 저장:
  * 최종 추천 조성 저장: `best_alloy.json`
  * Pareto 후보 Top 10 저장: `pareto_top10.json`, `pareto_top10.csv`
---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| ML/Data | Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Joblib |
| 최적화 | DEAP (유전 알고리즘) |
| 백엔드 | FastAPI |
| 프론트엔드 | Streamlit |
| 시각화 | Matplotlib |
