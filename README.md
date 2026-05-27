# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크립(Creep) 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

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
│   ├── transformer_and_tree_ensemble.py # 커스텀 모델 학습/평가
│   └── LMP_데이터증강.py       # LMP 기반 데이터 증강
├── ga/                        
│   ├── __init__.py            
│   ├── config.py              # 전체 설정 관리
│   ├── physics.py             # 물리 제약 및 penalty 계산
│   ├── llm.py                 # 초기 seed 생성
│   └── engine.py              # GA 최적화 실행
├── data_preprocessing.py      # 이전 데이터 전처리 및 피처 엔지니어링
├── 데이터전처리.py             # 추가된 데이터셋 병합 및 데이터 전처리
├── app.py           # Streamlit 기반 웹 애플리케이션 실행 파일
├── requirements.txt           # 프로젝트 라이브러리 의존성 목록
└── README.md                  # 본 문서
```

---

### 1. 데이터 전처리 및 피처 엔지니어링 (`데이터전처리.py`)
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

### 3. GA 최적화 (`ga/`)

- 예측 모델 연동:
  * `models/transformer_tree_ensemble.pkl`로 GA 후보 조성의 예측 수명 계산
  * 모델 입력 feature는 학습 artifact에 저장된 feature schema를 기준으로 구성

- 탐색 변수 정의:
  * GA가 직접 조절할 주요 합금 원소 11개 선정: C, Si, Mn, Cr, Mo, W, Ni, V, Nb, N, B
  * 불순물 또는 현재 설계 대상에서 제외한 원소는 고정값 처리: `P, S, O, Al`
  * `Co, Ta, Re, Cu` 등 데이터 희박성·비용·설계 범위 문제를 가진 원소는 기본적으로 0 또는 제한값으로 관리
  * Fe는 직접 최적화하지 않고, 전체 조성 합에서 balance로 계산

- 물리 기반 제약 조건 반영:
  * 9–12Cr ferritic/martensitic steel 설계 범위를 기준으로 KN 계수, Ms temperature, Laves phase 위험도, Z-phase 위험도, CEQ, MX balance를 계산
  * 각 지표가 기준값을 초과하거나 미달할 경우 연속 penalty를 부여하여 비현실적 조성 억제
  * 총합금량 제한 및 원소 단가 기반 재료비 index를 함께 계산

- Mahalanobis OOD penalty:
  * GA가 학습 데이터 분포 밖의 조성을 선택하는 문제를 줄이기 위함
  * OOD reference는 `data/ood_reference.pkl`에서 로드하며,
   reference가 없을 경우 OOD penalty가 조용히 꺼지지 않도록 실행 단계에서 확인
  * 현재 OOD 기준은 GA 최종 설계 대상인 Fe계 9–12Cr 조성 영역과 대응되는 `taka.xlsx` 기반 조성 분포를 사용
  * 사용 feature: C, Si, Mn, Cr, Mo, W, Ni, V, Nb, N, B

- CALPHAD 기반 상 안정성 검증:
  * pycalphad와 Fe계 TDB 파일(`data/thermo/fe_thermo.tdb`)을 사용하여 후보 조성의 제한적 상평형 계산 수행
  * 계산 대상 phase: `BCC_A2`, `FCC_A1`, `LAVES_PHASE`, `SIGMA`, `M23C6`, `M6C`
  * Laves phase, Sigma phase, FCC_A1, M6C가 기준 이상 형성될 경우 CALPHAD penalty를 부여
  * BCC_A2와 M23C6는 Fe계 내열강에서 주요 기지상/탄화물로 해석 가능하므로 주로 기록 및 해석용으로 사용

- LLM 기반 초기 seed 생성:
  * Gemini API를 사용하여 Fe계 9–12Cr 내열강 설계 범위에 맞는 초기 후보 조성 seed 생성
  * API 호출 결과는 `data/seed_cache.json`에 저장하여 재사용 가능
  * LLM seed는 그대로 사용하지 않고, 조성 범위·총합금량·물리 penalty 기준을 통과한 seed만 GA 초기 population에 반영
  * seed 부족 상황에 대비하여 strict mode 및 cache 기반 fallback 구조를 사용

- 다목적 최적화:
  * DEAP 기반 NSGA-II 적용
  * 최적화 목표:
    1. 물리 기반 위험도 최소화
    2. 예측 수명 최대화
    3. 재료 비용 최소화
  * 물리 기반 위험도에는 heuristic metallurgy penalty, Mahalanobis OOD penalty, elemental OOD penalty가 포함됨
  * CALPHAD 검증은 계산 비용을 고려하여 최종 Pareto Top 10 후보에 대해 후처리 검증으로 수행

- 합금 조성 보정 장치:
  * 교차/변이 이후 개별 원소 범위와 전체 합금량 제한을 만족하도록 repair 로직 적용
  * Fe balance가 음수가 되거나 총합금량 제한을 초과하는 후보를 방지
  * GA 탐색 중 물리적으로 불가능한 조성이 누적되지 않도록 후보 조성을 지속적으로 보정

- 결과 저장:
  * 최종 추천 조성 저장: `ga/best_alloy.json`
  * Pareto 후보 Top 10 저장: `ga/pareto_top10.json`, `ga/pareto_top10.csv`
  * 저장 결과에는 예측 수명, 재료비, 물리 penalty, OOD distance, CALPHAD phase fraction, 최종 조성 wt%가 포함됨
---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| ML/Data | Pandas, NumPy, Scikit-learn, Seaborn, Joblib |
| 최적화 | DEAP (유전 알고리즘) |
| 프론트엔드 | Streamlit |
