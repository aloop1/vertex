import os
from pathlib import Path

# ============================================================
# [1] 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SCALER_PATH = DATA_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "resnet_best.pt"

# ============================================================
# [2] 합금 원소 목록
# ============================================================
FULL_COMPOSITION_FEATURES = [
    "C", "Si", "Mn", "P", "S", "Cr", "Mo", "W", "Ni", "Cu",
    "V", "Nb", "N", "Al", "B", "Co", "Ta", "O", "Re"
]

DESIGN_VARIABLES = [
    "C", "Si", "Mn", "Cr", "Mo", "W",
    "Ni", "V", "Nb", "N", "B"
]

# ============================================================
# [3] 불순물 원소 분리
# 실제 alloy-design variable이 아니라 impurity constraint로 관리
# ============================================================
IMPURITY_ELEMENTS = ["O", "P", "S"]
FIXED_ELEMENTS = {
    "P": 0.005,
    "S": 0.002,
    "O": 0.001,
    "Co": 0.0,
    "Re": 0.0,
    "Al": 0.003,
    "Cu": 0.0,
    "Ta": 0.0
}

# ============================================================
# [4] 열처리 기본값
# ============================================================
DEFAULT_HEAT_TREATMENT = {
    "Ntemp": 1323.15,
    "Ntime": 1.0,
    "Ttemp": 1023.15,
    "Ttime": 2.0,
    "Atemp": 0.0,
    "Atime": 0.0
}

# ============================================================
# [5] 페라이트계 내열강 설계 범위
# ============================================================
FERRITIC_SYSTEM_LIMITS = {
    'C': (0.08, 0.15),
    'Si': (0.10, 0.40),
    'Mn': (0.30, 0.60),

    # major strengthening elements
    'Cr': (8.5, 11.5),
    'Mo': (0.30, 1.20),
    'W': (1.00, 2.50),

    # ferrite destabilizer
    'Ni': (0.0, 0.40),

    # precipitation strengthening
    'V': (0.18, 0.25),
    'Nb': (0.04, 0.08),
    'N': (0.03, 0.06),
    'B': (0.001, 0.010),
}

# ============================================================
# [6] 합금 총합 제한
# ============================================================
MAX_TOTAL_ALLOY_WT = 15.0
MIN_FE_BALANCE = 80.0

# ============================================================
# [7] 원소별 원료 단가
# ============================================================
ELEMENT_COST = {
    'Fe': 0.6,
    'Cr': 4.0,
    'C': 0.2,
    'Mo': 35.0,
    'W': 45.0,
    'V': 40.0,
    'Nb': 60.0,
    'Ni': 25.0,
    'Mn': 2.5,
    'Si': 2.0,
    'N': 1.0,
    'B': 15.0,
    'Al': 3.0,
    'Cu': 10.0,
    'P': 0.2,
    'S': 0.2,
    'Co': 60.0,
    'Ta': 200.0,
    'Re': 2500.0,
    'O': 0.0
}

# ============================================================
# [8] OOD 위험 원소 가중치
# ============================================================
OOD_SENSITIVE_ELEMENTS = {
    'Re': 1.0,
    'Ta': 0.7,
    'Co': 0.5
}

# ============================================================
# [9] coupling constraint
# Independent bounds만으로는 방어 불가능한 영역 제어
# ============================================================
METALLURGICAL_CONSTRAINTS = {
    "MAX_KN": 8.5,
    "MIN_MS_TEMP": 250,
    "MAX_LAVES_INDEX": 1.8,
    "MAX_Z_PHASE_RATIO": 8.0,
    "MAX_CEQ": 2.8,
    "MAX_MX_BALANCE": 2.5,
    "MAX_TOTAL_ALLOY_WT": 15.0,
}

# ============================================================
# [10] OOD
# ============================================================
OOD_CONFIG = {
    "ENABLE_OOD_PENALTY": True,
    "USE_MAHALANOBIS": True,

    "REQUIRE_OOD_REFERENCE": True,

    "REFERENCE_PATH": str(DATA_DIR / "ood_reference.pkl"),

    "REFERENCE_FEATURES": DESIGN_VARIABLES,

    "OOD_DISTANCE_THRESHOLD": 3.0,
    "OOD_PENALTY_WEIGHT": 0.25,

    "OOD_QUANTILE": 0.95,
    "COV_REGULARIZATION": 1e-6,
}

# ============================================================
# [11] LLM seed generation configuration
# ============================================================
LLM_CONFIG = {
    'MODEL_NAME': 'gemini-2.5-flash',

    'SEED_COUNT': 30,
    'BATCH_SIZE': 10,
    'TEMPERATURE': 0.2,

    'SYSTEM_PROMPT_TEMPLATE': """
    Role:
    Expert in 9-12% Cr Ferritic-Martensitic Steel Metallurgy.

    Objective:
    Generate physically realistic alloy seed compositions
    for high-temperature creep-resistant ferritic steel.

    Requirements:
    - Respect all elemental bounds.
    - Avoid unrealistic refractory overload.
    - Maintain ferritic-martensitic stability.
    - Avoid excessive Laves-forming tendency.
    - Output ONLY JSON.

    Constraints:
    {bounds_string}

    Generate {seed_count} diverse alloy seeds.
    """
}

# ============================================================
# [12] 탐색 정책
# ============================================================
SEARCH_POLICY = {

    # initialization
    'INITIAL_POPULATION_SIZE': 300,
    'LLM_SEED_RATIO': 0.3,

    # diversity control
    'MIN_COMPOSITION_DISTANCE': 0.15,

    # mutation safety
    'MAX_MUTATION_STEP': 0.20,

    # elitism
    'ELITE_RATIO': 0.10,
}

# ============================================================
# [13] Thermodynamic CALPHAD configuration
# ============================================================

THERMO_CONFIG = {
    "ENABLE_THERMO_CALC": True,
    "TDB_PATH": str(DATA_DIR / "thermo" / "fe_thermo.tdb"),

    "CALPHAD_ELEMENTS": [
    "Fe",
    "C",
    "Cr",
    "Mo",
    "W",
    "V",
    "Nb",
    "N",
    "Mn",
    "Si",
    "Ni",
    ],

    "TARGET_PHASES": [
        "BCC_A2",
        "FCC_A1",
        "LAVES_PHASE",
        "SIGMA",
        "M23C6",
        "M6C",
    ],

    # 위험상 기준
    "MAX_LAVES_FRACTION": 0.03,
    "MAX_SIGMA_FRACTION": 0.01,

    # 페라이트/마르텐사이트계에서 FCC가 지나치게 안정하면 위험 신호로 봄
    "MAX_FCC_FRACTION": 0.05,

    # M6C는 Mo/W-rich carbide로 과다하면 장기 안정성 위험으로 약하게 감점
    "MAX_M6C_FRACTION": 0.05,

    "THERMO_PENALTY_WEIGHT": 10.0,

    # 계산 실패 시 0점 처리하면 오히려 실패 후보가 이득을 봄
    "THERMO_FAILURE_PENALTY": 100.0,

    # 발표 전 개발 중에는 False 권장.
    # 완전히 검증한 뒤 True로 바꿔도 됨.
    "STRICT_THERMO_MODE": False,

    # GA 중 같은 조성이 반복될 때 계산 재사용
    "USE_THERMO_CACHE": True,

        # GA 탐색 중에는 CALPHAD를 끄고, 최종 후보 검증에서만 사용
    "APPLY_THERMO_DURING_GA": False,
    "APPLY_THERMO_TO_TOP_CANDIDATES": True,
}