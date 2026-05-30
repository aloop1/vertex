import os
import json
import pandas as pd
import numpy as np
import joblib
import random
import warnings
import math
from typing import Any
from pathlib import Path
from deap import base, creator, tools, algorithms
from dotenv import load_dotenv
from models.transformer_and_tree_ensemble import load_transformer_tree_predictor

# ============================================================
# [0] 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

warnings.filterwarnings("ignore")

from .config import (
    DESIGN_VARIABLES,
    FULL_COMPOSITION_FEATURES,
    FERRITIC_SYSTEM_LIMITS,
    DEFAULT_HEAT_TREATMENT,
    MAX_TOTAL_ALLOY_WT,
    SEARCH_POLICY,
    OOD_CONFIG,
    SCALER_PATH,
)

from .physics import (
    as_full_composition,
    calculate_material_cost,
    calculate_metallurgical_scores,
    evaluate_physics,
)

from .llm import get_expert_seeds
from .ood_analysis import analyze_ood_candidate

# ============================================================
# [1] 수명 예측 모델 로드
# ============================================================

try:
    artifact_path_env = os.environ.get("ALLOY_PREDICTOR_ARTIFACT_PATH", "").strip()
    artifact_path = artifact_path_env if artifact_path_env else None

    life_predictor = load_transformer_tree_predictor(
        artifact_path=artifact_path,
        allow_smoke_fallback=False,
    )

    if life_predictor.is_smoke_model:
        raise RuntimeError(
            "현재 로드된 모델이 smoke 테스트용 모델입니다. "
            "최종 GA에는 transformer_tree_ensemble.pkl 또는 .pt를 사용해야 합니다."
        )

    print("[System] Transformer + Tree Ensemble 수명 예측 모델 로드 성공")
    print(f"[System] model artifact: {life_predictor.artifact_path}")
    print(f"[System] feature count: {len(life_predictor.feature_names)}")

except Exception as e:
    raise RuntimeError(
        "[FATAL] 수명 예측 모델 로드 실패. "
        "models/transformer_tree_ensemble.pkl 또는 "
        "models/transformer_tree_ensemble.pt 파일이 있는지 확인하십시오."
    ) from e


# ============================================================
# [3] DEAP Fitness / Individual 정의
# ============================================================

if not hasattr(creator, "FitnessAlloyMulti"):
    creator.create("FitnessAlloyMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    # 목적:
    # 1. pred_log_life 최대화
    # 2. material_cost 최소화
    # 3. physics_risk 최소화

if not hasattr(creator, "AlloyIndividual"):
    creator.create("AlloyIndividual", list, fitness=creator.FitnessAlloyMulti)


# ============================================================
# [4] Helper Functions
# ============================================================


def _design_individual_to_comp(individual):
    """
    GA individual은 DESIGN_VARIABLES 11개만 가짐.
    평가 전 as_full_composition()으로 19개 full composition 복원.
    """

    design_comp = {
        elem: float(individual[i]) for i, elem in enumerate(DESIGN_VARIABLES)
    }

    return as_full_composition(design_comp)


def _total_alloy_from_individual(individual):
    """
    FIXED_ELEMENTS까지 포함한 총합 계산.
    """

    full_comp = _design_individual_to_comp(individual)
    return sum(full_comp.values())


def _is_valid_total_alloy(individual):
    return _total_alloy_from_individual(individual) <= MAX_TOTAL_ALLOY_WT

def _fixed_elements_total():
    """
    FIXED_ELEMENTS까지 포함한 full composition에서
    DESIGN_VARIABLES가 아닌 고정 원소들의 총합을 계산.
    """

    empty_design = {}
    full_comp = as_full_composition(empty_design)

    return sum(full_comp.values())


def _repair_individual(individual):
    """
    교차/변이 이후 개체를 물리적으로 가능한 조성으로 교정.

    수행:
    1. 각 gene을 FERRITIC_SYSTEM_LIMITS 범위 안으로 clip
    2. FIXED_ELEMENTS를 고려하여 DESIGN_VARIABLES 총합 허용치를 계산
    3. 총합이 초과되면 lower bound 위의 여유분을 비례 축소
    """

    # --------------------------------------------------------
    # [1] 원소별 bounds 강제
    # --------------------------------------------------------

    lows = np.array(
        [
            FERRITIC_SYSTEM_LIMITS[elem][0]
            for elem in DESIGN_VARIABLES
        ],
        dtype=float
    )

    ups = np.array(
        [
            FERRITIC_SYSTEM_LIMITS[elem][1]
            for elem in DESIGN_VARIABLES
        ],
        dtype=float
    )

    values = np.array(
        individual,
        dtype=float
    )

    values = np.clip(
        values,
        lows,
        ups
    )

    # --------------------------------------------------------
    # [2] 총합 제한 계산
    # --------------------------------------------------------

    fixed_total = _fixed_elements_total()

    max_design_total = (
        MAX_TOTAL_ALLOY_WT
        - fixed_total
    )

    min_design_total = float(
        np.sum(lows)
    )

    if max_design_total < min_design_total:
        raise ValueError(
            "설계 공간이 불가능합니다. "
            f"MAX_TOTAL_ALLOY_WT={MAX_TOTAL_ALLOY_WT}, "
            f"fixed_total={fixed_total:.6f}, "
            f"min_design_total={min_design_total:.6f}"
        )

    # --------------------------------------------------------
    # [3] 총합 초과 시 repair
    # --------------------------------------------------------

    design_total = float(
        np.sum(values)
    )

    if design_total > max_design_total:

        excess = (
            design_total
            - max_design_total
        )

        reducible = (
            values
            - lows
        )

        reducible_sum = float(
            np.sum(reducible)
        )

        if reducible_sum <= 1e-12:
            values = lows.copy()

        else:
            reduction = (
                excess
                * reducible
                / reducible_sum
            )

            values = (
                values
                - reduction
            )

            values = np.clip(
                values,
                lows,
                ups
            )

        # 부동소수점 오차 보정용 2차 repair
        design_total = float(
            np.sum(values)
        )

        if design_total > max_design_total:

            excess = (
                design_total
                - max_design_total
            )

            reducible = (
                values
                - lows
            )

            reducible_sum = float(
                np.sum(reducible)
            )

            if reducible_sum > 1e-12:
                values = (
                    values
                    - excess
                    * reducible
                    / reducible_sum
                )

                values = np.clip(
                    values,
                    lows,
                    ups
                )

    # --------------------------------------------------------
    # [4] repaired values를 individual에 다시 기록
    # --------------------------------------------------------

    for i in range(len(individual)):
        individual[i] = float(values[i])

    return individual


def _repair_operator(operator_func):
    """
    DEAP mate/mutate operator에 붙이는 repair decorator.

    교차/변이가 끝난 직후 offspring을 repair합니다.
    """

    def wrapper(*args, **kwargs):
        offspring = operator_func(*args, **kwargs)

        for child in offspring:
            _repair_individual(child)

        return offspring

    return wrapper


def _make_random_individual():
    """
    DESIGN_VARIABLES 기준 무작위 개체 생성.
    생성 직후 repair를 적용하여 원소별 bounds와 총합 제한을 동시에 보정.
    """

    values = [
        random.uniform(
            FERRITIC_SYSTEM_LIMITS[elem][0],
            FERRITIC_SYSTEM_LIMITS[elem][1]
        )
        for elem in DESIGN_VARIABLES
    ]

    ind = creator.AlloyIndividual(values)

    _repair_individual(ind)

    return ind


def _make_individual_from_seed(seed):
    """
    llm.py의 seed는 full composition을 반환할 수 있음.
    GA individual에는 DESIGN_VARIABLES 11개만 추출.
    추출 후 repair를 적용하여 bounds와 총합 제한을 보정합니다.
    """

    comp = seed.get("composition", {})

    values = [
        float(comp.get(elem, 0.0))
        for elem in DESIGN_VARIABLES
    ]

    ind = creator.AlloyIndividual(values)

    _repair_individual(ind)

    return ind


def _is_diverse(candidate, population, min_distance):
    """
    DESIGN_VARIABLES 11차원 기준 유클리드 거리 다양성 검사.
    """

    if not population:
        return True

    cand_arr = np.array(candidate, dtype=float)

    for p in population:
        dist = np.linalg.norm(cand_arr - np.array(p, dtype=float))

        if dist < min_distance:
            return False

    return True


def _predict_log_life(comp_dict, temp_k, stress):
    """
    Transformer + Tree Ensemble 기반 크리프 수명 예측 모델을 사용합니다.

    반환값:
    - log10(creep rupture life)
    """

    try:
        result = life_predictor.predict_one(
            stress=float(stress),
            temp=float(temp_k),
            composition=comp_dict,
            heat_treatment=DEFAULT_HEAT_TREATMENT,
        )

        pred_log_life = float(result.log_lifetime)

    except Exception as e:
        raise RuntimeError(
            "[Predictor-FATAL] Transformer + Tree Ensemble 예측 실패. "
            "입력 조성, 온도/응력, 열처리 조건, 모델 artifact를 확인하십시오."
        ) from e

    if np.isnan(pred_log_life) or np.isinf(pred_log_life):
        raise RuntimeError(
            "[Predictor-FATAL] 수명 예측 모델이 NaN 또는 Inf를 반환했습니다."
        )

    return pred_log_life


def _extract_ood_reference_payload(obj):
    """
    다양한 저장 형식에서 OOD reference를 추출한다.
    지원:
    - data/ood_reference.pkl 형태의 dict
    - preprocessor.pkl 안에 들어간 dict
    - attribute 기반 객체
    """

    if obj is None:
        return None

    def get_value(keys):
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]

        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)

        return None

    mean = get_value([
        "ood_reference_mean",
        "reference_mean",
        "composition_mean",
        "mean",
    ])

    cov_inv = get_value([
        "ood_reference_cov_inv",
        "reference_cov_inv",
        "composition_cov_inv",
        "cov_inv",
        "inverse_covariance",
    ])

    features = get_value([
        "ood_reference_features",
        "reference_features",
        "composition_features",
        "features",
    ])

    threshold = get_value([
        "ood_distance_threshold",
        "reference_threshold",
        "mahalanobis_threshold",
    ])

    if mean is None or cov_inv is None:
        return None

    return {
        "reference_mean": mean,
        "reference_cov_inv": cov_inv,
        "reference_features": features,
        "reference_threshold": threshold,
    }


def _build_ood_reference_if_available():
    """
    Mahalanobis OOD reference를 로드한다.

    우선순위:
    1. OOD_CONFIG["REFERENCE_PATH"] = data/ood_reference.pkl
    2. SCALER_PATH = data/preprocessor.pkl 내부 dict/attribute

    중요:
    - OOD_CONFIG["REQUIRE_OOD_REFERENCE"]가 True면 reference 없을 때 실행 중단.
    - 더 이상 조용히 OOD를 비활성화하지 않는다.
    """

    if not OOD_CONFIG.get("ENABLE_OOD_PENALTY", True):
        print("[OOD] OOD_CONFIG에 의해 OOD penalty가 비활성화되어 있습니다.")
        return None, None, None, None

    candidate_paths = []

    reference_path = OOD_CONFIG.get("REFERENCE_PATH", None)

    if reference_path:
        candidate_paths.append(Path(reference_path))

    candidate_paths.append(Path(SCALER_PATH))

    load_errors = []

    for path in candidate_paths:
        try:
            if not path.exists():
                load_errors.append(f"{path}: file not found")
                continue

            obj = joblib.load(path)
            payload = _extract_ood_reference_payload(obj)

            if payload is None:
                load_errors.append(f"{path}: OOD keys not found")
                continue

            mean = payload["reference_mean"]
            cov_inv = payload["reference_cov_inv"]
            features = payload.get("reference_features", None)
            threshold = payload.get("reference_threshold", None)

            if features is None:
                features = OOD_CONFIG.get("REFERENCE_FEATURES", FULL_COMPOSITION_FEATURES)

            if threshold is None:
                threshold = OOD_CONFIG.get("OOD_DISTANCE_THRESHOLD", 3.0)

            print(f"[OOD] Mahalanobis OOD reference 로드 성공: {path}")
            print(f"[OOD] features: {features}")
            print(f"[OOD] threshold: {float(threshold):.4f}")

            return mean, cov_inv, features, threshold

        except Exception as e:
            load_errors.append(f"{path}: {e}")

    msg = (
        "[FATAL] OOD penalty가 ENABLE 상태이지만 Mahalanobis reference를 찾지 못했습니다.\n"
        "해결 방법: python tools\\build_ood_reference.py --input data\\taka.xlsx 실행 후 "
        "data\\ood_reference.pkl을 생성하십시오.\n"
        "[load errors]\n"
        + "\n".join(load_errors)
    )

    if OOD_CONFIG.get("REQUIRE_OOD_REFERENCE", True):
        raise RuntimeError(msg)

    print("[OOD-WARNING] " + msg)
    return None, None, None, None


# ============================================================
# [4-1] Result Export Functions
# 입력값:
# - temp_c
# - stress_mpa
# - cost_limit

# 출력 파일:
# - best_alloy.json
# - pareto_top10.json
# - pareto_top30.csv
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent


def _summarize_candidate(
    individual,
    temp_k,
    stress,
    reference_mean=None,
    reference_cov_inv=None,
    reference_features=None,
    reference_threshold=None,
    use_thermo=True,
):
    """
    Pareto 후보 1개 요약.
    GA 평가에 사용한 OOD reference를 그대로 받아 저장 결과와 평가 결과의 일관성을 유지.
    """

    # 반드시 제일 먼저 comp_dict를 만들어야 합니다.
    comp_dict = _design_individual_to_comp(individual)

    pred_log_life = _predict_log_life(
        comp_dict,
        temp_k,
        stress
    )

    material_cost = calculate_material_cost(
        comp_dict
    )

    metallurgy = calculate_metallurgical_scores(
        comp_dict
    )

    physics = evaluate_physics(
        comp_dict=comp_dict,
        temp_k=temp_k,
        reference_mean=reference_mean,
        reference_cov_inv=reference_cov_inv,
        reference_features=reference_features,
        reference_threshold=reference_threshold,
        use_thermo=use_thermo,
    )

    try:
        ood_explain = analyze_ood_candidate(
            comp_dict=comp_dict,
            top_k=5,
        )
    except Exception as e:
        ood_explain = {
            "ood_percentile": None,
            "ood_is_out_of_distribution": None,
            "ood_top_contributors": [],
            "nearest_neighbors": [],
            "ood_explain_error": str(e),
        }

    row = {
    # ========================================================
    # [1] 최종 목적함수/핵심 결과
    # ========================================================
    "predicted_log_life": pred_log_life,
    "material_cost": material_cost,
    "physics_risk": physics.get("final_penalty", 9999.0),

    # ========================================================
    # [2] Heuristic metallurgy 지표
    # ========================================================
    "KN": metallurgy.get("kn", None),
    "Ms_temp": metallurgy.get("ms_temp", None),
    "laves_risk_heuristic": metallurgy.get("laves_risk", None),
    "Z_phase_risk_heuristic": metallurgy.get("z_phase_risk", None),
    "CEQ": metallurgy.get("ceq", None),
    "MX_balance_heuristic": metallurgy.get("mx_balance", None),
    "total_alloy_wt": metallurgy.get("total_alloy", None),

    # ========================================================
    # [3] penalty breakdown
    # ========================================================
    "metallurgy_penalty": physics.get("metallurgy_penalty", 0.0),
    "thermo_penalty": physics.get("thermo_penalty", 0.0),
    "ood_penalty": physics.get("ood_penalty", 0.0),
    "elemental_ood_penalty": physics.get("elemental_ood_penalty", 0.0),

    # ========================================================
    # [4] CALPHAD 실행 상태
    # ========================================================
    "thermo_success": physics.get("thermo_success", False),
    "thermo_error": physics.get("thermo_error", None),

    # ========================================================
    # [5] CALPHAD phase fraction
    # ========================================================
    "laves_fraction": physics.get("laves_fraction", 0.0),
    "sigma_fraction": physics.get("sigma_fraction", 0.0),
    "m23c6_fraction": physics.get("m23c6_fraction", 0.0),
    "m6c_fraction": physics.get("m6c_fraction", 0.0),
    "bcc_fraction": physics.get("bcc_fraction", 0.0),
    "fcc_fraction": physics.get("fcc_fraction", 0.0),

    # ========================================================
    # [6] CALPHAD penalty breakdown
    # ========================================================
    "laves_penalty_calphad": physics.get("laves_penalty_calphad", 0.0),
    "sigma_penalty_calphad": physics.get("sigma_penalty_calphad", 0.0),
    "fcc_penalty_calphad": physics.get("fcc_penalty_calphad", 0.0),
    "m6c_penalty_calphad": physics.get("m6c_penalty_calphad", 0.0),

    # ========================================================
    # [7] 상세 phase 정보
    # ========================================================
    "phase_info": physics.get("phase_info", {}),

    # ========================================================
    # [8] OOD 상세
    # ========================================================
    "ood_distance": physics.get("ood_distance", 0.0),
    "ood_distance_threshold": physics.get("ood_distance_threshold", None),
    "ood_reference_features": physics.get("ood_reference_features", []),
    "ood_error": physics.get("ood_error", None),
    "ood_percentile": ood_explain.get("ood_percentile", None),
    "ood_is_out_of_distribution": ood_explain.get("ood_is_out_of_distribution", None),
    "ood_top_contributors": ood_explain.get("ood_top_contributors", []),
    "nearest_neighbors": ood_explain.get("nearest_neighbors", []),
    "ood_explain_error": ood_explain.get("ood_explain_error", None),
}

    for elem in FULL_COMPOSITION_FEATURES:
        row[elem] = comp_dict.get(elem, 0.0)

    row["Fe_balance"] = max(
        0.0,
        100.0 - sum(comp_dict.values())
    )

    return row

# ============================================================
# Best alloy selection policy
# 우선순위:
# 1. 물리야금학적 타당성
# 2. 예측 수명 최대화
# 3. 목표 비용 충족
# ============================================================

LIFE_TIE_TOL_LOG = 0.03
# log10 기준 0.03 차이는 약 7% 수명 차이입니다.
# 이 정도 차이는 거의 비슷한 수명으로 보고 비용을 비교합니다.


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default

    if math.isnan(value) or math.isinf(value):
        return default

    return value


def _is_cost_ok(row: dict, cost_limit: float | None) -> bool:
    if cost_limit is None:
        return True

    material_cost = _to_float(row.get("material_cost"), default=float("inf"))
    return material_cost <= float(cost_limit)


def _physical_violation_score(
    row: dict,
    require_thermo: bool = False,
) -> tuple[float, list[str]]:
    """
    물리야금학적 타당성 평가.

    점수가 낮을수록 좋습니다.
    0이면 물리 제약을 모두 통과한 후보로 봅니다.
    """

    violations: list[str] = []
    score = 0.0

    physics_risk = _to_float(row.get("physics_risk"), 0.0)
    metallurgy_penalty = _to_float(row.get("metallurgy_penalty"), 0.0)
    thermo_penalty = _to_float(row.get("thermo_penalty"), 0.0)
    ood_penalty = _to_float(row.get("ood_penalty"), 0.0)
    elemental_ood_penalty = _to_float(row.get("elemental_ood_penalty"), 0.0)

    score += physics_risk
    score += metallurgy_penalty
    score += thermo_penalty
    score += ood_penalty
    score += elemental_ood_penalty

    if physics_risk > 1e-9:
        violations.append(f"physics_risk={physics_risk:.4f}")

    if metallurgy_penalty > 1e-9:
        violations.append(f"metallurgy_penalty={metallurgy_penalty:.4f}")

    if thermo_penalty > 1e-9:
        violations.append(f"thermo_penalty={thermo_penalty:.4f}")

    if ood_penalty > 1e-9:
        violations.append(f"ood_penalty={ood_penalty:.4f}")

    if elemental_ood_penalty > 1e-9:
        violations.append(f"elemental_ood_penalty={elemental_ood_penalty:.4f}")

    # OOD threshold 초과 여부
    ood_is_out = row.get("ood_is_out_of_distribution", None)
    if ood_is_out is True:
        violations.append("ood_is_out_of_distribution=True")
        score += 100.0

    # CALPHAD 성공 여부
    if require_thermo:
        thermo_success = row.get("thermo_success", None)

        if thermo_success is not True:
            violations.append("thermo_success=False")
            score += 100.0

    # CALPHAD phase fraction 기반 위험상 확인
    laves_fraction = _to_float(row.get("laves_fraction"), 0.0)
    sigma_fraction = _to_float(row.get("sigma_fraction"), 0.0)
    fcc_fraction = _to_float(row.get("fcc_fraction"), 0.0)
    m6c_fraction = _to_float(row.get("m6c_fraction"), 0.0)

    if laves_fraction > 0.02:
        violations.append(f"laves_fraction={laves_fraction:.6f}")
        score += 10.0 * (laves_fraction - 0.02)

    if sigma_fraction > 1e-6:
        violations.append(f"sigma_fraction={sigma_fraction:.6f}")
        score += 20.0 * sigma_fraction

    # ferritic/martensitic 후보에서 FCC가 과도하면 위험
    if fcc_fraction > 0.05:
        violations.append(f"fcc_fraction={fcc_fraction:.6f}")
        score += 10.0 * fcc_fraction

    if m6c_fraction > 1e-6:
        violations.append(f"m6c_fraction={m6c_fraction:.6f}")
        score += 10.0 * m6c_fraction

    return score, violations


def _is_physically_valid(
    row: dict,
    require_thermo: bool = False,
) -> bool:
    score, violations = _physical_violation_score(
        row=row,
        require_thermo=require_thermo,
    )

    return score <= 1e-9 and len(violations) == 0


def rank_candidates_by_priority(
    candidate_rows: list[dict],
    cost_limit: float | None,
    require_thermo: bool = False,
) -> list[dict]:
    """
    우선순위:
    1. 물리야금학적 타당성
    2. predicted_log_life 최대화
    3. 목표 비용 충족
    4. 비용 최소화

    require_thermo=False:
        CALPHAD 전 후보 정렬용

    require_thermo=True:
        CALPHAD 검증 이후 best_alloy 선정용
    """

    enriched_rows: list[dict] = []

    for row in candidate_rows:
        row = dict(row)

        physical_score, violations = _physical_violation_score(
            row=row,
            require_thermo=require_thermo,
        )

        predicted_log_life = _to_float(
            row.get("predicted_log_life"),
            default=-1e9,
        )

        material_cost = _to_float(
            row.get("material_cost"),
            default=float("inf"),
        )

        cost_ok = _is_cost_ok(
            row=row,
            cost_limit=cost_limit,
        )

        cost_excess = (
            max(0.0, material_cost - float(cost_limit))
            if cost_limit is not None
            else 0.0
        )

        row["selection_physical_score"] = physical_score
        row["selection_physical_violations"] = violations
        row["selection_cost_ok"] = cost_ok
        row["selection_cost_excess"] = cost_excess

        enriched_rows.append(row)

    # 정렬 기준
    # 1. physical_score 낮을수록 우선
    # 2. predicted_log_life 높을수록 우선
    # 3. cost_ok True 우선
    # 4. cost_excess 낮을수록 우선
    # 5. material_cost 낮을수록 우선
    ranked = sorted(
        enriched_rows,
        key=lambda r: (
            _to_float(r.get("selection_physical_score"), 1e9),
            -_to_float(r.get("predicted_log_life"), -1e9),
            not bool(r.get("selection_cost_ok", False)),
            _to_float(r.get("selection_cost_excess"), 1e9),
            _to_float(r.get("material_cost"), 1e9),
        ),
    )

    return ranked


def select_best_alloy_by_priority(
    candidate_rows: list[dict],
    cost_limit: float | None,
    require_thermo: bool = True,
) -> dict:
    """
    최종 best_alloy 선정.

    핵심:
    - 물리적으로 타당한 후보를 먼저 고릅니다.
    - 그 안에서 수명이 가장 긴 후보를 고릅니다.
    - 수명 차이가 매우 작으면 비용 조건을 봅니다.
    """

    if not candidate_rows:
        raise ValueError("best alloy를 선택할 후보가 없습니다.")

    ranked = rank_candidates_by_priority(
        candidate_rows=candidate_rows,
        cost_limit=cost_limit,
        require_thermo=require_thermo,
    )

    physically_valid_rows = [
        row for row in ranked
        if _is_physically_valid(row, require_thermo=require_thermo)
    ]

    # CALPHAD까지 요구했는데 모두 실패한 경우:
    # 발표/실행이 깨지는 것보다, fallback을 명확히 기록하고 선택합니다.
    if not physically_valid_rows and require_thermo:
        print(
            "[Warning] CALPHAD까지 완전히 통과한 후보가 없습니다. "
            "thermo 조건을 완화하여 best alloy를 선택합니다."
        )

        return select_best_alloy_by_priority(
            candidate_rows=candidate_rows,
            cost_limit=cost_limit,
            require_thermo=False,
        )

    # 물리 통과 후보가 아예 없으면 physical_score가 가장 낮은 후보 중 수명 최대를 선택
    if not physically_valid_rows:
        best = ranked[0]
        best["selection_policy"] = (
            "fallback: minimum physical violation score > maximum life > cost"
        )
        return best

    max_life = max(
        _to_float(row.get("predicted_log_life"), -1e9)
        for row in physically_valid_rows
    )

    # 수명이 거의 같은 후보들은 같은 그룹으로 보고 비용 비교
    life_band_rows = [
        row for row in physically_valid_rows
        if max_life - _to_float(row.get("predicted_log_life"), -1e9)
        <= LIFE_TIE_TOL_LOG
    ]

    cost_ok_rows = [
        row for row in life_band_rows
        if _is_cost_ok(row, cost_limit)
    ]

    final_pool = cost_ok_rows if cost_ok_rows else life_band_rows

    best = sorted(
        final_pool,
        key=lambda r: (
            not bool(r.get("selection_cost_ok", False)),
            _to_float(r.get("selection_cost_excess"), 1e9),
            _to_float(r.get("material_cost"), 1e9),
            -_to_float(r.get("predicted_log_life"), -1e9),
        ),
    )[0]

    best["selection_policy"] = (
        "physical validity >= predicted life > target cost satisfaction"
    )

    best["selection_note"] = (
        "물리야금학적 타당성을 먼저 통과한 후보군 중에서 "
        "예측 수명이 가장 높은 후보를 우선 선택하고, "
        "수명이 거의 같은 후보 사이에서는 목표 비용 충족 여부와 비용을 비교함."
    )

    return best


def _export_pareto_top_candidates(
    candidates,
    temp_k,
    stress,
    reference_mean=None,
    reference_cov_inv=None,
    reference_features=None,
    reference_threshold=None,
    cost_limit=None,
    top_n=30
):
    """
    Pareto 후보 Top 30을 CSV/JSON으로 저장.

    우선순위:
    1. 물리야금학적 타당성
    2. 예측 수명 최대화
    3. 목표 비용 충족

    구조:
    - GA 평가 중에는 CALPHAD를 사용하지 않음
    - fitness 기준으로 물리 타당성 우선 후보를 넉넉히 선별
    - 선별 후보에 대해서만 pycalphad CALPHAD 검증 수행
    - CALPHAD 결과까지 반영한 뒤 최종 우선순위로 다시 정렬
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # [1] CALPHAD 전 후보 선별
    # fitness.values:
    # 0: pred_log_life 최대화
    # 1: material_cost 최소화
    # 2: physics_risk 최소화
    #
    # 여기서는 물리 타당성을 먼저 보고,
    # 그 다음 수명, 마지막으로 비용을 봅니다.
    # --------------------------------------------------------

    ranked_candidates = sorted(
        candidates,
        key=lambda ind: (
            float(ind.fitness.values[2]),    # physics risk 최소
            -float(ind.fitness.values[0]),   # life 최대
            float(ind.fitness.values[1]),    # cost 최소
        )
    )

    selected_candidates = ranked_candidates[:top_n]

    print(
        f"[CALPHAD] Pareto 후보 {len(candidates)}개 중 "
        f"Top {len(selected_candidates)}개에 대해서만 CALPHAD 검증을 수행합니다."
    )

    # --------------------------------------------------------
    # [2] Top N 후보에 대해서만 CALPHAD 수행
    # --------------------------------------------------------

    rows = []

    for idx, ind in enumerate(selected_candidates, start=1):
        print(f"[CALPHAD] Top candidate {idx}/{len(selected_candidates)} 검증 중...")

        rows.append(
            _summarize_candidate(
                individual=ind,
                temp_k=temp_k,
                stress=stress,
                reference_mean=reference_mean,
                reference_cov_inv=reference_cov_inv,
                reference_features=reference_features,
                reference_threshold=reference_threshold,
                use_thermo=True,
            )
        )

    # --------------------------------------------------------
    # [3] CALPHAD 결과까지 포함한 뒤 최종 우선순위로 다시 정렬
    # --------------------------------------------------------

    rows = rank_candidates_by_priority(
        candidate_rows=rows,
        cost_limit=cost_limit,
        require_thermo=True,
    )

    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "pareto_top30.csv"
    json_path = RESULTS_DIR / "pareto_top30.json"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"[Result] Pareto Top {len(rows)} CSV 저장: {csv_path}")
    print(f"[Result] Pareto Top {len(rows)} JSON 저장: {json_path}")

    return rows


# ============================================================
# [5] GA 평가 및 진화 엔진
# ============================================================


def run_alloy_optimization(u_temp_c, u_stress, u_cost_limit):

    temp_k = u_temp_c + 273.15
    toolbox = base.Toolbox()

    pop_size = int(SEARCH_POLICY.get("INITIAL_POPULATION_SIZE", 100))

    min_distance = float(SEARCH_POLICY.get("MIN_COMPOSITION_DISTANCE", 0.15))

    reference_mean, reference_cov_inv, reference_features, reference_threshold = _build_ood_reference_if_available()

    # --------------------------------------------------------
    # Fitness Evaluation
    # --------------------------------------------------------

    def evaluate(individual):

        try:
            comp_dict = _design_individual_to_comp(individual)

            physics_result = evaluate_physics(
                comp_dict=comp_dict,
                temp_k=temp_k,
                reference_mean=reference_mean,
                reference_cov_inv=reference_cov_inv,
                reference_features=reference_features,
                reference_threshold=reference_threshold,
                use_thermo=False,
            )

            material_cost = calculate_material_cost(comp_dict)

            cost_excess = max(0.0, material_cost - u_cost_limit)

            pred_log_life = _predict_log_life(comp_dict, temp_k, u_stress)

            physics_risk = physics_result.get("final_penalty", 9999.0) + cost_excess

            return (float(pred_log_life), float(material_cost), float(physics_risk))

        except Exception as e:
            print(f"[EVALUATION ERROR] {e}")

            return (-9999.0, 9999.0, 9999.0)

    toolbox.register("evaluate", evaluate)

    low_bounds = [FERRITIC_SYSTEM_LIMITS[elem][0] for elem in DESIGN_VARIABLES]

    up_bounds = [FERRITIC_SYSTEM_LIMITS[elem][1] for elem in DESIGN_VARIABLES]

    toolbox.register(
        "mate", 
        tools.cxSimulatedBinaryBounded, 
        low=low_bounds, 
        up=up_bounds, 
        eta=20.0
    )

    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=low_bounds,
        up=up_bounds,
        eta=20.0,
        indpb=0.1,
    )
    
    toolbox.decorate(
        "mate",
        _repair_operator
    )

    toolbox.decorate(
        "mutate",
        _repair_operator
    )

    toolbox.register(
        "select",
        tools.selNSGA2
    )

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])

    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    # --------------------------------------------------------
    # 초기 population 구성
    # --------------------------------------------------------

    population = []

    print("[System] seed 수집 및 인구 다양성 검증 중...")

    try:
        seeds = get_expert_seeds()
    except Exception as e:
        raise RuntimeError(
            "[FATAL] 초기 seed 수집 실패로 최적화를 중단합니다. "
            "Gemini API, cache seed, local seed 설정을 확인하십시오. "
            "API 모드에서는 실패한 seed를 무작위 개체로 대체하지 않습니다."
        ) from e

    if not isinstance(seeds, list):
        raise RuntimeError(
            "[FATAL] get_expert_seeds()가 list가 아닌 값을 반환했습니다. "
            f"반환 타입: {type(seeds).__name__}"
        )

    if len(seeds) == 0:
        raise RuntimeError(
            "[FATAL] get_expert_seeds()가 빈 seed list를 반환했습니다. "
            "초기 population을 무작위 개체만으로 생성하지 않기 위해 최적화를 중단합니다."
        )

    seed_convert_errors = []

    for idx, s in enumerate(seeds, start=1):
        try:
            ind = _make_individual_from_seed(s)

            if not _is_valid_total_alloy(ind):
                seed_convert_errors.append(
                    f"seed {idx}: 총합 제한 초과"
                )
                continue

            if _is_diverse(ind, population, min_distance):
                population.append(ind)

                if len(population) >= pop_size:
                    break

            else:
                seed_convert_errors.append(
                    f"seed {idx}: 다양성 조건 미달"
                )

        except Exception as e:
            seed_convert_errors.append(
                f"seed {idx}: {e}"
            )

    if len(population) == 0:
        error_preview = "\n".join(seed_convert_errors[:10])
        raise RuntimeError(
            "[FATAL] 수집된 seed는 존재하지만 GA에 투입 가능한 seed가 0개입니다. "
            "LLM seed 형식, 원소 bounds, 총합 제한, 다양성 조건을 확인하십시오.\n"
            f"[Seed conversion errors]\n{error_preview}"
        )

    if seed_convert_errors:
        print(f"[Seed Warning] 일부 seed 변환/검증 실패: {len(seed_convert_errors)}개")
        for msg in seed_convert_errors[:10]:
            print(f"  - {msg}")

    remaining_random_count = max(0, pop_size - len(population))

    print(
        f"[System] 검증 통과 seed {len(population)}개 확보. "
        f"나머지 {remaining_random_count}개는 bounds 기반 무작위 개체로 보충합니다."
    )

    # 무작위 개체
    max_attempts = pop_size * 200
    attempts = 0

    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1

        candidate = _make_random_individual()

        if not _is_valid_total_alloy(candidate):
            continue

        if not _is_diverse(candidate, population, min_distance):
            continue

        population.append(candidate)

    if len(population) < pop_size:
        raise RuntimeError(
            f"초기 population 생성 실패: " f"{len(population)}/{pop_size}개만 생성됨. "
        )

    # NSGA-II가 초기 population fitness를 먼저 평가함
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = list(map(toolbox.evaluate, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(population, len(population))

    print(f"[최적화 시작] " f"{u_temp_c}℃ / {u_stress}MPa / 비용한도 {u_cost_limit}")

    pareto_front = tools.ParetoFront()

    pop, logbook = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=150,
        lambda_=300,
        cxpb=0.7,
        mutpb=0.3,
        ngen=80,
        stats=stats,
        halloffame=pareto_front,
        verbose=False,
    )

    # --------------------------------------------------------
    # Pareto front 기반 best 선택
    # --------------------------------------------------------

    if len(pareto_front) > 0:
        candidates = list(pareto_front)
    else:
        candidates = pop

    # 웹 Pareto Top 30 저장
    top_candidates = _export_pareto_top_candidates(
        candidates=candidates,
        temp_k=temp_k,
        stress=u_stress,
        reference_mean=reference_mean,
        reference_cov_inv=reference_cov_inv,
        reference_features=reference_features,
        reference_threshold=reference_threshold,
        cost_limit=u_cost_limit,
        top_n=30
    )

    # --------------------------------------------------------
    # 최종 best_alloy 선정
    # 우선순위:
    # 1. 물리야금학적 타당성
    # 2. 예측 수명 최대화
    # 3. 목표 비용 충족
    # --------------------------------------------------------

    best_summary = select_best_alloy_by_priority(
        candidate_rows=top_candidates,
        cost_limit=u_cost_limit,
        require_thermo=True,
    )

    best_recipe = {
        elem: float(best_summary.get(elem, 0.0))
        for elem in FULL_COMPOSITION_FEATURES
    }

    # 최종 best 후보 JSON, CSV 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_json_path = RESULTS_DIR / "best_alloy.json"
    best_csv_path = RESULTS_DIR / "best_alloy.csv"

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([best_summary]).to_csv(
        best_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[Result] Best alloy JSON 저장: {best_json_path}")
    print(f"[Result] Best alloy CSV 저장: {best_csv_path}")

    return best_recipe, logbook


# ============================================================
# [6] 실행 인터페이스
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("신합금 설계")
    print("=" * 60)

    try:
        T = float(input("▶ 설계 목표 온도(℃): "))
        S = float(input("▶ 가해지는 응력(MPa): "))
        C = float(input("▶ 재료비 한도 Index: "))

        best_recipe, logbook = run_alloy_optimization(T, S, C)

        # ----------------------------------------------------
        # 결과 분석
        # ----------------------------------------------------

        m = calculate_metallurgical_scores(best_recipe)
        total_cost = calculate_material_cost(best_recipe)

        temp_k = T + 273.15

        pred_log_life = _predict_log_life(best_recipe, temp_k, S)


        # ----------------------------------------------------
        # 최종 출력
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("[최종 신합금 설계 결과]")
        print("=" * 60)

        print(f"조건: {T:.1f}℃ / {S:.1f}MPa")
        print(f"예측 Log Life: {pred_log_life:.4f}")
        print(f"원재료 Cost Index: {total_cost:.4f}")

        print("-" * 60)
        print("조성 wt%")

        for elem in FULL_COMPOSITION_FEATURES:
            val = best_recipe.get(elem, 0.0)

            if abs(val) > 1e-6:
                print(f"  - {elem:>2}: {val:.6f}")

        fe_balance = max(0.0, 100.0 - sum(best_recipe.values()))

        print(f"  - Fe: {fe_balance:.6f} (Balance)")

        print("-" * 60)
        print("야금학 지표")
        print(f"  - KN              : {m['kn']:.4f}")
        print(f"  - Ms Temp         : {m['ms_temp']:.4f} °C")
        print(f"  - Laves Risk      : {m['laves_risk']:.4f}")
        print(f"  - Z-phase Risk    : {m['z_phase_risk']:.4f}")
        print(f"  - CEQ             : {m['ceq']:.4f}")
        print(f"  - MX Balance      : {m['mx_balance']:.4f}")
        print(f"  - Total Alloy wt% : {m['total_alloy']:.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"[오류] 실행 실패: {e}")