import numpy as np

from .config import (
    ELEMENT_COST,
    FULL_COMPOSITION_FEATURES,
    FIXED_ELEMENTS,
    METALLURGICAL_CONSTRAINTS,
    OOD_CONFIG,
    OOD_SENSITIVE_ELEMENTS,
    MAX_TOTAL_ALLOY_WT,
)


# ============================================================
# [1] Utility
# ============================================================

def as_full_composition(comp_dict):
    """
    GA가 DESIGN_VARIABLES만 넘겨도,
    P/S/O 등 FIXED_ELEMENTS를 포함한 19개 full composition으로 복원합니다.

    핵심:
    - physics.py와 ML predictor는 항상 FULL_COMPOSITION_FEATURES 기준으로 작동해야 함
    - FIXED_ELEMENTS는 사용자가 잘못 넘겨도 고정값으로 강제 override
    """

    full_comp = {elem: 0.0 for elem in FULL_COMPOSITION_FEATURES}

    for elem, val in comp_dict.items():
        if elem in full_comp:
            full_comp[elem] = float(val)

    for elem, val in FIXED_ELEMENTS.items():
        if elem in full_comp:
            full_comp[elem] = float(val)

    return full_comp


def excess_penalty(
    value,
    threshold,
    direction="upper",
    scale=1.0,
    normalizer=1.0,
    power=2.0
):
    """
    기준값을 넘었을 때만 증가하는 연속 penalty 함수.

    기존 sigmoid penalty는 threshold에서 이미 0.5 * scale penalty가 발생합니다.
    이 함수는 안전 영역에서는 0, 위험 영역에서는 부드럽게 증가합니다.

    direction:
    - "upper": value > threshold 이면 penalty
    - "lower": value < threshold 이면 penalty
    """

    if normalizer <= 0:
        normalizer = 1.0

    if direction == "upper":
        violation = max(0.0, float(value) - float(threshold))
    elif direction == "lower":
        violation = max(0.0, float(threshold) - float(value))
    else:
        raise ValueError("direction must be either 'upper' or 'lower'")

    return float(scale * (violation / normalizer) ** power)


# ============================================================
# [2] 경제성 평가
# ============================================================

def calculate_material_cost(comp_dict):
    """
    Fe balance 포함 원재료 cost index 계산
    """

    full_comp = as_full_composition(comp_dict)

    fe_content = max(0.0, 100.0 - sum(full_comp.values()))

    total_cost = fe_content * ELEMENT_COST.get("Fe", 0.6)

    for elem, wt in full_comp.items():
        total_cost += wt * ELEMENT_COST.get(elem, 0.0)

    return float(total_cost / 100.0)


# ============================================================
# [3] Metallurgy Heuristic Layer
# ============================================================

def calculate_metallurgical_scores(comp_dict):
    """
    9-12% Cr ferritic-martensitic steel용 heuristic metallurgy 평가
    """

    full_comp = as_full_composition(comp_dict)

    # --------------------------------------------------------
    # 원소 추출
    # --------------------------------------------------------

    c  = full_comp.get("C", 0.0)
    n  = full_comp.get("N", 0.0)
    mn = full_comp.get("Mn", 0.0)
    ni = full_comp.get("Ni", 0.0)
    cu = full_comp.get("Cu", 0.0)

    si = full_comp.get("Si", 0.0)
    cr = full_comp.get("Cr", 0.0)
    mo = full_comp.get("Mo", 0.0)
    w  = full_comp.get("W", 0.0)

    v  = full_comp.get("V", 0.0)
    nb = full_comp.get("Nb", 0.0)

    al = full_comp.get("Al", 0.0)
    co = full_comp.get("Co", 0.0)

    # --------------------------------------------------------
    # 기준값 로드
    # --------------------------------------------------------

    max_kn = METALLURGICAL_CONSTRAINTS["MAX_KN"]
    min_ms_temp = METALLURGICAL_CONSTRAINTS["MIN_MS_TEMP"]
    max_laves = METALLURGICAL_CONSTRAINTS["MAX_LAVES_INDEX"]
    max_z = METALLURGICAL_CONSTRAINTS["MAX_Z_PHASE_RATIO"]
    max_ceq = METALLURGICAL_CONSTRAINTS["MAX_CEQ"]
    max_mx = METALLURGICAL_CONSTRAINTS["MAX_MX_BALANCE"]
    max_total_alloy = METALLURGICAL_CONSTRAINTS.get(
        "MAX_TOTAL_ALLOY_WT",
        MAX_TOTAL_ALLOY_WT
    )

    # --------------------------------------------------------
    # [1] Delta Ferrite Risk: KN
    # --------------------------------------------------------

    kn = (
        cr
        + 6.0 * si
        + 4.0 * mo
        + 2.0 * al
        + 4.0 * nb
        + 1.5 * v
        + 1.5 * w
    ) - (
        40.0 * c
        + 30.0 * n
        + 2.0 * mn
        + 4.0 * ni
        + 2.0 * cu
        + co
    )

    kn_penalty = excess_penalty(
        value=kn,
        threshold=max_kn,
        direction="upper",
        scale=1.5,
        normalizer=1.0,
        power=2.0
    )

    # --------------------------------------------------------
    # [2] Martensite Stability: Ms temperature
    # --------------------------------------------------------

    ms_temp = (
        561.0
        - 474.0 * c
        - 33.0 * mn
        - 17.0 * cr
        - 17.0 * ni
        - 21.0 * mo
    )

    ms_penalty = excess_penalty(
        value=ms_temp,
        threshold=min_ms_temp,
        direction="lower",
        scale=1.2,
        normalizer=100.0,
        power=2.0
    )

    # --------------------------------------------------------
    # [3] Laves Phase Risk
    # --------------------------------------------------------

    laves_risk = mo + 0.5 * w

    laves_penalty = excess_penalty(
        value=laves_risk,
        threshold=max_laves,
        direction="upper",
        scale=1.5,
        normalizer=0.5,
        power=2.0
    )

    # --------------------------------------------------------
    # [4] Z-phase Risk
    # --------------------------------------------------------

    z_phase_risk = (v + nb) / (n + 1e-6)

    z_phase_penalty = excess_penalty(
        value=z_phase_risk,
        threshold=max_z,
        direction="upper",
        scale=1.2,
        normalizer=2.0,
        power=2.0
    )

    # --------------------------------------------------------
    # [5] Weldability: Carbon Equivalent
    # --------------------------------------------------------

    ceq = (
        c
        + mn / 6.0
        + (cr + mo + v) / 5.0
        + (ni + cu) / 15.0
    )

    ceq_penalty = excess_penalty(
        value=ceq,
        threshold=max_ceq,
        direction="upper",
        scale=1.0,
        normalizer=0.5,
        power=2.0
    )

    # --------------------------------------------------------
    # [6] MX Stoichiometry
    # --------------------------------------------------------

    mx_balance = (v + nb) / (c + n + 1e-6)

    mx_penalty = excess_penalty(
        value=mx_balance,
        threshold=max_mx,
        direction="upper",
        scale=0.8,
        normalizer=1.0,
        power=2.0
    )

    # --------------------------------------------------------
    # [7] Total Alloy Boundary
    # --------------------------------------------------------

    total_alloy = sum(full_comp.values())
    fe_balance = max(0.0, 100.0 - total_alloy)

    total_alloy_penalty = excess_penalty(
        value=total_alloy,
        threshold=max_total_alloy,
        direction="upper",
        scale=2.0,
        normalizer=1.0,
        power=2.0
    )

    # --------------------------------------------------------
    # [8] Fe balance penalty
    # --------------------------------------------------------

    metallurgy_penalty = (
        kn_penalty
        + ms_penalty
        + laves_penalty
        + z_phase_penalty
        + ceq_penalty
        + mx_penalty
        + total_alloy_penalty
    )

    return {
        # raw metrics
        "kn": float(kn),
        "ms_temp": float(ms_temp),
        "laves_risk": float(laves_risk),
        "z_phase_risk": float(z_phase_risk),
        "ceq": float(ceq),
        "mx_balance": float(mx_balance),
        "total_alloy": float(total_alloy),
        "fe_balance": float(fe_balance),

        # penalties
        "kn_penalty": float(kn_penalty),
        "ms_penalty": float(ms_penalty),
        "laves_penalty": float(laves_penalty),
        "z_phase_penalty": float(z_phase_penalty),
        "ceq_penalty": float(ceq_penalty),
        "mx_penalty": float(mx_penalty),
        "total_alloy_penalty": float(total_alloy_penalty),

        "metallurgy_penalty": float(metallurgy_penalty)
    }


# ============================================================
# [4] OOD Layer
# ============================================================

def calculate_multivariate_ood_penalty(
    comp_dict,
    reference_mean,
    reference_cov_inv
):
    """
    Mahalanobis distance 기반 multivariate OOD 탐지.

    주의:
    - reference_mean과 reference_cov_inv는 반드시 FULL_COMPOSITION_FEATURES 순서 기준이어야 함
    - 실제 학습 데이터 분포에서 계산해야 의미 있음
    """

    full_comp = as_full_composition(comp_dict)

    x = np.array([
        full_comp.get(elem, 0.0)
        for elem in FULL_COMPOSITION_FEATURES
    ], dtype=float)

    mean_vec = np.array(reference_mean, dtype=float)
    cov_inv = np.array(reference_cov_inv, dtype=float)

    if mean_vec.shape[0] != x.shape[0]:
        return {
            "ood_distance": 9999.0,
            "ood_penalty": 9999.0,
            "ood_error": (
                f"reference_mean dimension mismatch: "
                f"expected {x.shape[0]}, got {mean_vec.shape[0]}"
            )
        }

    if cov_inv.shape != (x.shape[0], x.shape[0]):
        return {
            "ood_distance": 9999.0,
            "ood_penalty": 9999.0,
            "ood_error": (
                f"reference_cov_inv dimension mismatch: "
                f"expected {(x.shape[0], x.shape[0])}, got {cov_inv.shape}"
            )
        }

    diff = x - mean_vec
    distance_sq = float(diff.T @ cov_inv @ diff)
    distance_sq = max(0.0, distance_sq)
    distance = float(np.sqrt(distance_sq))

    threshold = OOD_CONFIG.get("OOD_DISTANCE_THRESHOLD", 3.0)
    weight = OOD_CONFIG.get("OOD_PENALTY_WEIGHT", 0.25)

    ood_penalty = excess_penalty(
        value=distance,
        threshold=threshold,
        direction="upper",
        scale=weight,
        normalizer=1.0,
        power=2.0
    )

    return {
        "ood_distance": float(distance),
        "ood_penalty": float(ood_penalty),
        "ood_error": None
    }


def calculate_elemental_ood_penalty(comp_dict):
    """
    데이터 부족 가능성이 높고 비용이 큰 원소에 대한 보수적 penalty.

    Re, Ta, Co 등은 학습 데이터가 희박하거나 비용/제조 난이도가 높을 수 있으므로
    Mahalanobis OOD와 별도로 약한 안전장치로 사용합니다.
    """

    full_comp = as_full_composition(comp_dict)

    penalty = 0.0

    for elem, weight in OOD_SENSITIVE_ELEMENTS.items():
        val = full_comp.get(elem, 0.0)
        penalty += float(weight) * float(val)

    return {
        "elemental_ood_penalty": float(penalty)
    }


def calculate_uncertainty_penalty(comp_dict, reference_ranges):

    full_comp = as_full_composition(comp_dict)

    penalty = 0.0

    for elem, val in full_comp.items():
        if elem in reference_ranges:
            low, high = reference_ranges[elem]

            if val < low:
                penalty += abs(low - val)
            elif val > high:
                penalty += abs(val - high)

    return float(penalty)


# ============================================================
# [5] Thermodynamic Hook Layer
# ============================================================

def calculate_thermo_penalty(
    comp_dict,
    temp_k
):
    """
    Future pycalphad integration hook.

    현재는 구조만 열어둔 placeholder입니다.
    이후 pycalphad 연결 시:
    - phase equilibrium
    - Gibbs minimization
    - Laves fraction
    - delta ferrite fraction
    - sigma phase stability
    를 여기서 계산하게 됩니다.
    """

    _ = as_full_composition(comp_dict)

    thermo_penalty = 0.0

    phase_info = {
        "BCC_A2": None,
        "FCC_A1": None,
        "LAVES_PHASE": None,
        "SIGMA": None
    }

    return {
        "thermo_penalty": float(thermo_penalty),
        "phase_info": phase_info,
        "thermo_success": True,
        "thermo_error": None
    }


# ============================================================
# [6] Heat Treatment Severity
# ============================================================

def calculate_severity(
    temp_k,
    time_h
):
    """
    Larson-Miller inspired severity metric.
    """

    if temp_k <= 0 or time_h <= 0:
        return 0.0

    return float(
        temp_k * (
            20.0 + np.log10(
                max(time_h, 1e-6)
            )
        )
    )


# ============================================================
# [7] Full Feature Builder
# ============================================================

def build_full_feature_dict(comp_dict, temp_k, stress, ht_dict):
    """
    ML predictor input feature builder
    """

    full_comp = as_full_composition(comp_dict)

    full_data = {
        **full_comp,
        "temp": temp_k,
        "stress": stress,
        **ht_dict
    }

    full_data["N_severity"] = calculate_severity(
        ht_dict.get("Ntemp", 0.0),
        ht_dict.get("Ntime", 0.0)
    )

    full_data["T_severity"] = calculate_severity(
        ht_dict.get("Ttemp", 0.0),
        ht_dict.get("Ttime", 0.0)
    )

    full_data["A_severity"] = calculate_severity(
        ht_dict.get("Atemp", 0.0),
        ht_dict.get("Atime", 0.0)
    )

    return full_data


# ============================================================
# [8] Unified Physics Evaluation
# ============================================================

def evaluate_physics(
    comp_dict,
    temp_k,
    reference_mean=None,
    reference_cov_inv=None
):
    """
    모든 physics-informed layer 통합 평가.

    구성:
    - metallurgy heuristic penalty
    - OOD penalty
    - elemental OOD-sensitive penalty
    - future thermodynamic penalty
    """

    metallurgy = calculate_metallurgical_scores(
        comp_dict
    )

    thermo = calculate_thermo_penalty(
        comp_dict,
        temp_k
    )

    elemental_ood = calculate_elemental_ood_penalty(
        comp_dict
    )

    result = {
        **metallurgy,
        **thermo,
        **elemental_ood
    }

    # --------------------------------------------------------
    # OOD layer
    # --------------------------------------------------------

    if (
        OOD_CONFIG.get("ENABLE_OOD_PENALTY", True)
        and reference_mean is not None
        and reference_cov_inv is not None
    ):
        ood = calculate_multivariate_ood_penalty(
            comp_dict,
            reference_mean,
            reference_cov_inv
        )

        result.update(ood)

    else:
        result["ood_distance"] = 0.0
        result["ood_penalty"] = 0.0
        result["ood_error"] = None

    # --------------------------------------------------------
    # Unified Final Penalty
    # --------------------------------------------------------

    result["final_penalty"] = float(
        result["metallurgy_penalty"]
        + result["ood_penalty"]
        + result["elemental_ood_penalty"]
        + result["thermo_penalty"]
    )

    return result