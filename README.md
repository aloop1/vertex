# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크립(Creep) 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

---

## 📁 프로젝트 구조

```
vertex/
├── data/
│   ├── taka.xlsx               # 원본 데이터 (2066행 × 31열)
│   ├── creep.csv               # 추가 크립 데이터 (1024행 × 42열)
│   ├── creep_data.csv          # 추가 크립 데이터 (265행 × 25열)
│   ├── preprocessor.pkl        # 저장된 StandardScaler + 피처 메타정보
│   └── correlation_heatmap.png # 전처리 결과 변수 간 상관관계 히트맵
│   ├── ood_reference.pkl       # Mahalanobis OOD 검증 기준 파일
│   ├── seed_cache.json         # LLM 초기 조성 seed cache
│   └── thermo/
│       └── fe_thermo.tdb       # CALPHAD phase validation용 Fe계 열역학 DB
├── documents/
│   └── 회의록.md               # 팀 프로젝트 진행 기록
├── models/
│   ├── transformer_and_tree_ensemble.py # 커스텀 모델 학습/평가
│   └── LMP_데이터증강.py       # LMP 기반 데이터 증강
├── ga/                        
│   ├── __init__.py            
│   ├── config.py               # GA 탐색 범위, 비용, OOD, CALPHAD 등 전체 설정
│   ├── physics.py              # 물리야금학적 제약 및 penalty 계산
│   ├── llm.py                  # LLM seed / cache seed 기반 초기 후보 생성
│   ├── ood_analysis.py         # OOD 거리 계산 및 기여 원소 분석
│   ├── engine.py               # GA 최적화 실행 및 결과 저장
├── tools/
│   ├── build_ood_reference.py # OOD reference 생성 스크립트
│   └── build_ood.py           # OOD 관련 보조 생성 스크립트
├── data_preprocessing.py      # 이전 데이터 전처리 및 피처 엔지니어링
├── 데이터전처리.py             # 추가된 데이터셋 병합 및 데이터 전처리
├── web/
│   ├── app.py                 # 웹 애플리케이션 실행 파일
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

### 3. GA 최적화 (`ga/`)

GA 모듈은 수명 예측 모델을 기반으로 후보 합금 조성을 탐색하고, 물리 제약·OOD 검증·CALPHAD 검증을 거쳐 최종 신합금 후보를 선정한다.

| 구분               | 내용                                                                                           |
| :--------------- | :------------------------------------------------------------------------------------------- |
| 모델 연동            | `models/transformer_tree_ensemble.pkl`을 사용해 후보 조성의 예측 수명 계산                                  |
| 탐색 원소            | `C`, `Si`, `Mn`, `Cr`, `Mo`, `W`, `Ni`, `V`, `Nb`, `N`, `B`                                  |
| 고정 원소            | `P`, `S`, `O`, `Al`은 고정값으로 처리                                                                |
| 제한/제외 원소         | `Co`, `Ta`, `Re`, `Cu`는 데이터 희박성, 비용, 설계 범위 문제로 0 또는 제한값으로 관리                                 |
| Fe 처리            | `Fe`는 직접 최적화하지 않고 전체 조성의 balance로 계산                                                         |
| 물리 제약            | `KN`, `Ms temperature`, `Laves risk`, `Z-phase risk`, `CEQ`, `MX balance` 등을 계산하여 비현실적 조성 억제 |
| OOD 검증           | Mahalanobis distance를 이용해 학습 데이터 분포 밖 후보 여부 확인                                               |
| OOD 기준 파일        | `data/ood_reference.pkl`                                                                     |
| OOD 사용 원소        | `C`, `Si`, `Mn`, `Cr`, `Mo`, `W`, `Ni`, `V`, `Nb`, `N`, `B`                                  |
| CALPHAD 검증       | `pycalphad`와 `data/thermo/fe_thermo.tdb`를 이용해 후보 조성의 phase fraction 확인                       |
| CALPHAD 확인 phase | `BCC_A2`, `FCC_A1`, `LAVES_PHASE`, `SIGMA`, `M23C6`, `M6C`                                   |
| LLM seed         | Gemini API 또는 `data/seed_cache.json`을 이용해 초기 후보 조성 seed 생성                                   |
| Seed 검증          | 조성 범위, 총합금량, 물리 penalty 기준을 통과한 seed만 초기 population에 반영                                      |
| 최적화 방식           | DEAP 기반 NSGA-II 사용                                                                           |
| 최적화 기준           | 물리야금학적 타당성 확보 → 예측 수명 최대화 → 재료 비용 최소화                                                        |
| 조성 보정            | 교차·변이 후 원소 범위, 총합금량, `Fe balance`를 만족하도록 repair 수행                                           |
| 결과 저장            | `ga/best_alloy.json`, `ga/best_alloy.csv`, `ga/pareto_top10.json`, `ga/pareto_top10.csv` 생성  |
| 결과 포함 항목         | 최종 조성 wt%, 예측 수명, 재료비 index, 물리 penalty, OOD distance, CALPHAD phase fraction                |
 

---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| ML/Data | Pandas, NumPy, Scikit-learn, Seaborn, Joblib |
| 최적화 | DEAP (유전 알고리즘) |
| 프론트엔드 | Streamlit |
