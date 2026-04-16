# AI 기반 크립 수명 예측 및 합금 설계 시스템

> 고온·고압 환경 핵심 소재의 크리프(Creep) 파단 수명을 예측하고, 수명을 최대화하는 최적 합금 조성을 AI로 도출하는 웹 애플리케이션

---

## 📁 프로젝트 구조

```
vertex/
├── data/
│   ├── taka.xlsx              # 원본 데이터 (2066행 × 31열)
│   ├── preprocessor.pkl       # 저장된 StandardScaler + 피처 메타정보
│   └── correlation_heatmap.png # 전처리 결과 변수 간 상관관계 히트맵
├── documents/
│   └── 회의록.md               # 팀 프로젝트 진행 기록
├── models/
│   ├── train.py               # XGBoost 모델 학습/평가
│   └── compare_models.py       # MLP, RF 모델 학습/ 평가
├── data_preprocessing.py      # 데이터 전처리 및 피처 엔지니어링
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
