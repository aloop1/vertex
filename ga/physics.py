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

try:
    from .config import THERMO_CONFIG
except ImportError:
    THERMO_CONFIG = {}


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
    reference_cov_inv,
    reference_features=None,
    reference_threshold=None,
):
    """
    Mahalanobis distance 기반 multivariate OOD 탐지.

    핵심:
    - reference_features 기준으로 candidate vector를 구성한다.
    - reference_mean/reference_cov_inv와 feature order가 반드시 일치해야 한다.
    - reference_threshold가 있으면 그것을 우선 사용한다.
    """

    full_comp = as_full_composition(comp_dict)

    features = list(reference_features) if reference_features else list(FULL_COMPOSITION_FEATURES)

    x = np.array([
        full_comp.get(elem, 0.0)
        for elem in features
    ], dtype=float)

    mean_vec = np.array(reference_mean, dtype=float)
    cov_inv = np.array(reference_cov_inv, dtype=float)

    if mean_vec.shape[0] != x.shape[0]:
        return {
            "ood_distance": 9999.0,
            "ood_penalty": 9999.0,
            "ood_error": (
                f"reference_mean dimension mismatch: "
                f"expected {x.shape[0]}, got {mean_vec.shape[0]}, "
                f"features={features}"
            ),
            "ood_reference_features": features,
        }

    if cov_inv.shape != (x.shape[0], x.shape[0]):
        return {
            "ood_distance": 9999.0,
            "ood_penalty": 9999.0,
            "ood_error": (
                f"reference_cov_inv dimension mismatch: "
                f"expected {(x.shape[0], x.shape[0])}, got {cov_inv.shape}, "
                f"features={features}"
            ),
            "ood_reference_features": features,
        }

    diff = x - mean_vec
    distance_sq = float(diff.T @ cov_inv @ diff)
    distance_sq = max(0.0, distance_sq)
    distance = float(np.sqrt(distance_sq))

    threshold = (
        float(reference_threshold)
        if reference_threshold is not None
        else float(OOD_CONFIG.get("OOD_DISTANCE_THRESHOLD", 3.0))
    )

    weight = float(OOD_CONFIG.get("OOD_PENALTY_WEIGHT", 0.25))

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
        "ood_error": None,
        "ood_reference_features": features,
        "ood_distance_threshold": float(threshold),
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
# [5] Thermodynamic CALPHAD Layer
# ============================================================

ATOMIC_WEIGHTS = {
    "Fe": 55.845,
    "C": 12.011,
    "Si": 28.085,
    "Mn": 54.938,
    "P": 30.974,
    "S": 32.06,
    "Cr": 51.996,
    "Mo": 95.95,
    "W": 183.84,
    "Ni": 58.693,
    "Cu": 63.546,
    "V": 50.942,
    "Nb": 92.906,
    "N": 14.007,
    "Al": 26.982,
    "B": 10.81,
    "Co": 58.933,
    "Ta": 180.948,
    "O": 15.999,
    "Re": 186.207,
}

DEFAULT_CALPHAD_ELEMENTS = [
    "Fe", "C", "Cr", "Mo", "W", "V", "Nb", "N", "Mn", "Si", "Ni"
]

DEFAULT_CALPHAD_PHASES = [
    "BCC_A2",
    "FCC_A1",
    "LAVES_PHASE",
    "SIGMA",
    "M23C6",
    "M6C",
]

_THERMO_DB_CACHE = {}
_THERMO_RESULT_CACHE = {}


def _thermo_config_value(key, default):
    """
    THERMO_CONFIG가 없거나 일부 key가 빠져도 physics.py가 죽지 않게 하는 안전 getter.
    """
    if isinstance(THERMO_CONFIG, dict):
        return THERMO_CONFIG.get(key, default)
    return default


def _empty_thermo_result(
    thermo_penalty=0.0,
    thermo_success=False,
    thermo_error=None,
):
    """
    calculate_thermo_penalty()의 반환 schema를 항상 일정하게 유지한다.
    engine.py/result export에서 KeyError가 나지 않게 하는 목적.
    """
    return {
        "thermo_penalty": float(thermo_penalty),
        "phase_info": {},
        "thermo_success": bool(thermo_success),
        "thermo_error": thermo_error,

        "laves_fraction": 0.0,
        "sigma_fraction": 0.0,
        "m23c6_fraction": 0.0,
        "m6c_fraction": 0.0,
        "bcc_fraction": 0.0,
        "fcc_fraction": 0.0,

        "laves_penalty_calphad": 0.0,
        "sigma_penalty_calphad": 0.0,
        "fcc_penalty_calphad": 0.0,
        "m6c_penalty_calphad": 0.0,
    }


def _load_thermo_database(tdb_path):
    """
    pycalphad Database 로드는 비용이 크므로 경로별로 캐싱한다.
    """
    if tdb_path in _THERMO_DB_CACHE:
        return _THERMO_DB_CACHE[tdb_path]

    from pycalphad import Database

    dbf = Database(tdb_path)
    _THERMO_DB_CACHE[tdb_path] = dbf
    return dbf


def _get_calphad_elements(available_elements):
    """
    CALPHAD 계산에 투입할 원소를 제한한다.

    중요한 이유:
    - P/S/O/Al/B 같은 미량 원소를 무조건 넣으면,
      현재 target phase set에 해당 원소를 수용할 phase가 부족해서 equilibrium이 매우 느려지거나 실패할 수 있다.
    - 본 모듈의 목표는 완전 상평형 계산이 아니라 Laves/Sigma/M23C6/M6C 위험상 penalty이다.
    """
    configured = _thermo_config_value("CALPHAD_ELEMENTS", DEFAULT_CALPHAD_ELEMENTS)

    selected = []
    for elem in configured:
        elem_title = str(elem).strip()
        if not elem_title:
            continue

        elem_upper = elem_title.upper()

        if elem_upper == "FE":
            selected.append("Fe")
            continue

        if elem_upper in available_elements and elem_title in ATOMIC_WEIGHTS:
            selected.append(elem_title)

    # Fe와 vacancy는 pycalphad components에 반드시 필요하다.
    if "Fe" not in selected:
        selected.insert(0, "Fe")

    # 순서 보존 중복 제거
    return list(dict.fromkeys(selected))


def _wt_percent_to_mole_fraction(full_comp, available_elements):
    """
    GA 조성 wt%를 pycalphad X(component) 조건용 mole fraction으로 변환한다.

    원칙:
    - Fe는 balance로 계산한다.
    - CALPHAD 계산에는 THERMO_CONFIG["CALPHAD_ELEMENTS"]에 명시한 원소만 사용한다.
    - pycalphad equilibrium에서는 Fe를 dependent component로 두고,
      Fe가 아닌 원소의 X만 condition에 넣는다.
    """

    calphad_elements = _get_calphad_elements(available_elements)

    alloy_sum = sum(float(v) for v in full_comp.values())
    fe_wt = max(0.0, 100.0 - alloy_sum)

    wt_map = {"Fe": fe_wt}

    for elem in calphad_elements:
        if elem == "Fe":
            continue

        wt = float(full_comp.get(elem, 0.0))

        if wt <= 0:
            continue

        elem_upper = elem.upper()

        if elem_upper not in available_elements:
            continue

        if elem not in ATOMIC_WEIGHTS:
            continue

        wt_map[elem] = wt

    mole_map = {}

    for elem, wt in wt_map.items():
        aw = ATOMIC_WEIGHTS.get(elem)

        if aw is None or aw <= 0:
            continue

        mole_map[elem] = wt / aw

    total_moles = sum(mole_map.values())

    if total_moles <= 0:
        raise ValueError("Total mole amount is zero. Cannot convert wt% to mole fraction.")

    return {
        elem: mol / total_moles
        for elem, mol in mole_map.items()
    }


def _get_target_phases(available_phases):
    """
    실제 TDB에 존재하는 phase만 target으로 사용한다.
    config가 오래되어 MX가 들어가 있거나 M6C가 빠져 있어도,
    검증된 기본 phase set은 자동으로 보강한다.
    """
    configured = _thermo_config_value("TARGET_PHASES", [])
    merged = list(configured) + DEFAULT_CALPHAD_PHASES

    target_phases = []
    for phase in merged:
        phase_name = str(phase).strip()
        if phase_name and phase_name in available_phases:
            target_phases.append(phase_name)

    # 순서 보존 중복 제거
    return list(dict.fromkeys(target_phases))

def _empty_thermo_result(reason="CALPHAD thermo calculation skipped"):
    """
    GA 탐색 중 CALPHAD를 끌 때 사용하는 안전한 thermo 결과.
    final_penalty 계산에서 thermo_penalty는 0으로 처리된다.
    """

    return {
        "thermo_penalty": 0.0,
        "phase_info": {},
        "thermo_success": False,
        "thermo_error": reason,

        "laves_fraction": 0.0,
        "sigma_fraction": 0.0,
        "m23c6_fraction": 0.0,
        "m6c_fraction": 0.0,
        "bcc_fraction": 0.0,
        "fcc_fraction": 0.0,

        "laves_penalty_calphad": 0.0,
        "sigma_penalty_calphad": 0.0,
        "fcc_penalty_calphad": 0.0,
        "m6c_penalty_calphad": 0.0,
    }


def calculate_thermo_penalty(comp_dict, temp_k):
    """
    pycalphad 기반 제한적 CALPHAD penalty 계산.

    목적:
    - LAVES_PHASE, SIGMA, M6C 등 위험상 과다 형성 감점
    - FCC_A1 과다 안정성 감점
    - M23C6, BCC_A2는 기록용으로 사용

    핵심 판단:
    - 9-12%Cr ferritic/martensitic steel에서는 BCC계 기지가 정상이다.
      따라서 BCC_A2가 높다는 이유만으로 감점하지 않는다.
    - MX는 현재 TDB에서 phase 이름으로 직접 잡히지 않았으므로,
      기존 heuristic MX_balance로 관리한다.
    """

    import os
    import warnings

    if not _thermo_config_value("ENABLE_THERMO_CALC", False):
        return _empty_thermo_result(
            thermo_penalty=0.0,
            thermo_success=False,
            thermo_error="THERMO_CONFIG['ENABLE_THERMO_CALC'] is False"
        )

    tdb_path = _thermo_config_value("TDB_PATH", "")

    if not tdb_path or not os.path.exists(tdb_path):
        error_msg = f"TDB file not found: {tdb_path}"
        if _thermo_config_value("STRICT_THERMO_MODE", False):
            raise RuntimeError(error_msg)

        return _empty_thermo_result(
            thermo_penalty=_thermo_config_value("THERMO_FAILURE_PENALTY", 100.0),
            thermo_success=False,
            thermo_error=error_msg
        )

    try:
        from pycalphad import equilibrium, variables as v
    except Exception as e:
        error_msg = f"pycalphad import failed: {e}"
        if _thermo_config_value("STRICT_THERMO_MODE", False):
            raise RuntimeError(error_msg) from e

        return _empty_thermo_result(
            thermo_penalty=_thermo_config_value("THERMO_FAILURE_PENALTY", 100.0),
            thermo_success=False,
            thermo_error=error_msg
        )

    full_comp = as_full_composition(comp_dict)

    cache_key = None
    if _thermo_config_value("USE_THERMO_CACHE", True):
        cache_key = (
            str(tdb_path),
            round(float(temp_k), 2),
            tuple(
                round(float(full_comp.get(elem, 0.0)), 6)
                for elem in FULL_COMPOSITION_FEATURES
            )
        )

        if cache_key in _THERMO_RESULT_CACHE:
            return _THERMO_RESULT_CACHE[cache_key]

    try:
        with warnings.catch_warnings():
            # 해당 TDB의 TYPE_DEFINITION warning은 사전 검증했으므로 실행 중 반복 출력만 억제한다.
            warnings.simplefilter("ignore")
            dbf = _load_thermo_database(tdb_path)

        available_elements = set(str(c).upper() for c in dbf.elements)
        available_phases = set(dbf.phases.keys())

        x_map = _wt_percent_to_mole_fraction(
            full_comp=full_comp,
            available_elements=available_elements
        )

        comps = ["FE", "VA"]
        conditions = {
            v.T: float(temp_k),
            v.P: 101325,
            v.N: 1,
        }

        calphad_elements_upper = {
            elem.upper()
            for elem in _get_calphad_elements(available_elements)
        }

        minor_x_sum = 0.0

        for elem, x_val in x_map.items():
            elem_upper = elem.upper()

            if elem_upper == "FE":
                continue

            if elem_upper not in calphad_elements_upper:
                continue

            if elem_upper not in available_elements:
                continue

            if float(x_val) <= 0:
                continue

            comps.append(elem_upper)
            conditions[v.X(elem_upper)] = float(x_val)
            minor_x_sum += float(x_val)

        comps = list(dict.fromkeys(comps))

        if minor_x_sum >= 0.95:
            raise RuntimeError(
                f"Invalid mole fractions: non-Fe sum too high = {minor_x_sum:.6f}"
            )

        target_phases = _get_target_phases(available_phases)

        if not target_phases:
            raise RuntimeError("No target phases exist in the TDB file.")

        required_phases = [
            phase for phase in DEFAULT_CALPHAD_PHASES
            if phase not in available_phases
        ]

        if required_phases:
            raise RuntimeError(
                f"Required CALPHAD phases missing from TDB: {required_phases}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq = equilibrium(
                dbf,
                comps,
                target_phases,
                conditions
            )

        phase_values = eq.Phase.values.flatten()
        np_values = eq.NP.values.flatten()

        phase_info = {}

        for phase, amount in zip(phase_values, np_values):
            phase = str(phase)

            if phase == "" or phase.upper() == "NAN":
                continue

            try:
                amount = float(amount)
            except Exception:
                continue

            if np.isnan(amount) or amount <= 1e-12:
                continue

            phase_info[phase] = float(phase_info.get(phase, 0.0) + amount)

        total_np = sum(phase_info.values())

        if total_np > 0:
            phase_fraction = {
                phase: float(val / total_np)
                for phase, val in phase_info.items()
            }
        else:
            phase_fraction = {}

        laves_fraction = phase_fraction.get("LAVES_PHASE", 0.0)
        sigma_fraction = phase_fraction.get("SIGMA", 0.0)
        m23c6_fraction = phase_fraction.get("M23C6", 0.0)
        m6c_fraction = phase_fraction.get("M6C", 0.0)
        bcc_fraction = phase_fraction.get("BCC_A2", 0.0)
        fcc_fraction = phase_fraction.get("FCC_A1", 0.0)

        weight = float(_thermo_config_value("THERMO_PENALTY_WEIGHT", 10.0))

        max_laves = float(_thermo_config_value("MAX_LAVES_FRACTION", 0.03))
        max_sigma = float(_thermo_config_value("MAX_SIGMA_FRACTION", 0.01))
        max_fcc = float(_thermo_config_value("MAX_FCC_FRACTION", 0.05))
        max_m6c = float(_thermo_config_value("MAX_M6C_FRACTION", 0.05))

        laves_penalty = excess_penalty(
            value=laves_fraction,
            threshold=max_laves,
            direction="upper",
            scale=weight,
            normalizer=0.01,
            power=2.0
        )

        sigma_penalty = excess_penalty(
            value=sigma_fraction,
            threshold=max_sigma,
            direction="upper",
            scale=weight,
            normalizer=0.01,
            power=2.0
        )

        fcc_penalty = excess_penalty(
            value=fcc_fraction,
            threshold=max_fcc,
            direction="upper",
            scale=weight * 0.5,
            normalizer=0.02,
            power=2.0
        )

        m6c_penalty = excess_penalty(
            value=m6c_fraction,
            threshold=max_m6c,
            direction="upper",
            scale=weight * 0.3,
            normalizer=0.02,
            power=2.0
        )

        thermo_penalty = (
            laves_penalty
            + sigma_penalty
            + fcc_penalty
            + m6c_penalty
        )

        result = {
            "thermo_penalty": float(thermo_penalty),
            "phase_info": phase_fraction,
            "thermo_success": True,
            "thermo_error": None,

            "laves_fraction": float(laves_fraction),
            "sigma_fraction": float(sigma_fraction),
            "m23c6_fraction": float(m23c6_fraction),
            "m6c_fraction": float(m6c_fraction),
            "bcc_fraction": float(bcc_fraction),
            "fcc_fraction": float(fcc_fraction),

            "laves_penalty_calphad": float(laves_penalty),
            "sigma_penalty_calphad": float(sigma_penalty),
            "fcc_penalty_calphad": float(fcc_penalty),
            "m6c_penalty_calphad": float(m6c_penalty),
        }

        if cache_key is not None:
            _THERMO_RESULT_CACHE[cache_key] = result

        return result

    except Exception as e:
        error_msg = f"pycalphad equilibrium failed: {e}"

        if _thermo_config_value("STRICT_THERMO_MODE", False):
            raise RuntimeError(error_msg) from e

        result = _empty_thermo_result(
            thermo_penalty=_thermo_config_value("THERMO_FAILURE_PENALTY", 100.0),
            thermo_success=False,
            thermo_error=error_msg
        )

        if cache_key is not None:
            _THERMO_RESULT_CACHE[cache_key] = result

        return result


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
    reference_cov_inv=None,
    reference_features=None,
    reference_threshold=None,
    use_thermo=None
):
    """
    모든 physics-informed layer 통합 평가.

    구성:
    - metallurgy heuristic penalty
    - OOD penalty
    - elemental OOD-sensitive penalty
    - optional CALPHAD thermodynamic penalty

    use_thermo:
    - True  : pycalphad CALPHAD 계산 수행
    - False : CALPHAD 계산 생략
    - None  : THERMO_CONFIG["APPLY_THERMO_DURING_GA"] 값을 따름
    """

    metallurgy = calculate_metallurgical_scores(
        comp_dict
    )

    # --------------------------------------------------------
    # Thermodynamic layer
    # --------------------------------------------------------

    if use_thermo is None:
        use_thermo = bool(
            THERMO_CONFIG.get("APPLY_THERMO_DURING_GA", False)
        )

    if use_thermo:
        thermo = calculate_thermo_penalty(
            comp_dict,
            temp_k
        )
    else:
        thermo = _empty_thermo_result(
            reason="CALPHAD skipped during GA evaluation"
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
            reference_cov_inv,
            reference_features=reference_features,
            reference_threshold=reference_threshold,
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