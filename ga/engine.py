import os
import json
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import random
import warnings
from pathlib import Path
from deap import base, creator, tools, algorithms
from dotenv import load_dotenv

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
    MODELS_DIR,
    SCALER_PATH,
    MODEL_PATH,
    DEFAULT_HEAT_TREATMENT,
    MAX_TOTAL_ALLOY_WT,
    SEARCH_POLICY,
)

from .physics import (
    as_full_composition,
    calculate_material_cost,
    calculate_metallurgical_scores,
    build_full_feature_dict,
    evaluate_physics,
)

from .llm import get_expert_seeds


# ============================================================
# [1] AI 모델 구조 정의
# ============================================================


class ResBlock(nn.Module):
    """Pre-activation ResNet block"""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """
    Tabular ResNet:
      stem  : Linear(in_features → hidden_size)
      blocks: N × ResBlock(hidden_size)
      head  : Linear(hidden_size → 1)
    """

    def __init__(
        self, in_features: int, hidden_size: int, num_blocks: int, dropout: float
    ):
        super().__init__()

        self.stem = nn.Linear(in_features, hidden_size)

        self.blocks = nn.Sequential(
            *[ResBlock(hidden_size, dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x))).squeeze(1)


# ============================================================
# [2] 모델 및 전처리 스케일러 로드
# ============================================================

DEVICE = torch.device("cpu")

try:
    with open(MODELS_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    pre_data = joblib.load(SCALER_PATH)
    scaler = pre_data["scaler"]
    scaler_features = pre_data["feature_names"]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_params = checkpoint.get("params", {})

    predictor = TabularResNet(
        in_features=checkpoint.get("in_features", len(selected_features)),
        hidden_size=model_params.get("hidden_size", 24),
        num_blocks=model_params.get("num_blocks", 5),
        dropout=model_params.get("dropout", 0.2),
    ).to(DEVICE)

    predictor.load_state_dict(checkpoint.get("model_state", checkpoint))

    predictor.eval()

    print("[System] AI 모델 로드 성공")
    print(f"[System] selected_features: {selected_features}")

except Exception as e:
    print(f"[에러] 초기화 실패: {e}")
    exit(1)


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
    예측 모델 추론.
    """

    full_feat = build_full_feature_dict(
        comp_dict, temp_k, stress, DEFAULT_HEAT_TREATMENT
    )

    # scaler가 학습한 전체 feature 순서 기준으로 DataFrame 구성
    X_raw = pd.DataFrame([{f: full_feat.get(f, 0.0) for f in scaler_features}])

    X_scaled = scaler.transform(X_raw)

    X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_features)

    # selected_features.json 기준 slicing
    X_final = X_scaled_df[selected_features].values.astype(np.float32)

    X_tensor = torch.tensor(X_final, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_log_life = predictor(X_tensor).item()

    if np.isnan(pred_log_life) or np.isinf(pred_log_life):
        return -9999.0

    return float(pred_log_life)


def _build_ood_reference_if_available():
    """
    실제 학습 데이터 기반 OOD 통계량이 있을 때만 사용.
    """

    possible_mean_keys = [
        "ood_reference_mean",
        "reference_mean",
        "composition_mean",
    ]

    possible_cov_inv_keys = [
        "ood_reference_cov_inv",
        "reference_cov_inv",
        "composition_cov_inv",
    ]

    reference_mean = None
    reference_cov_inv = None

    for key in possible_mean_keys:
        if key in pre_data:
            reference_mean = pre_data[key]
            break

    for key in possible_cov_inv_keys:
        if key in pre_data:
            reference_cov_inv = pre_data[key]
            break

    if reference_mean is None or reference_cov_inv is None:
        print("[OOD] 실제 학습 데이터 기반 OOD 통계량이 없습니다.")
        return None, None

    reference_mean = np.array(reference_mean, dtype=float)
    reference_cov_inv = np.array(reference_cov_inv, dtype=float)

    expected_dim = len(FULL_COMPOSITION_FEATURES)

    if reference_mean.shape[0] != expected_dim:
        print(
            f"[OOD] reference_mean 차원 불일치: "
            f"{reference_mean.shape[0]} != {expected_dim}. OOD 비활성화."
        )
        return None, None

    if reference_cov_inv.shape != (expected_dim, expected_dim):
        print(
            f"[OOD] reference_cov_inv 차원 불일치: "
            f"{reference_cov_inv.shape} != {(expected_dim, expected_dim)}. OOD 비활성화."
        )
        return None, None

    print("[OOD] 학습 데이터 기반 OOD 통계량 로드 완료.")
    return reference_mean, reference_cov_inv


# ============================================================
# [4-1] Result Export Functions
# 입력값:
# - temp_c
# - stress_mpa
# - cost_limit

# 출력 파일:
# - best_alloy.json
# - pareto_top10.json
# ============================================================

RESULTS_DIR = BASE_DIR / "results"


def _summarize_candidate(
    individual,
    temp_k,
    stress,
    reference_mean=None,
    reference_cov_inv=None,
):
    """
    Pareto 후보 1개 요약.
    GA 평가에 사용한 OOD reference를 그대로 받아 저장 결과와 평가 결과의 일관성을 유지.
    """

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
        reference_cov_inv=reference_cov_inv
    )

    row = {
        "predicted_log_life": pred_log_life,
        "material_cost": material_cost,
        "physics_risk": physics.get("final_penalty", 9999.0),

        "KN": metallurgy.get("kn", None),
        "Ms_temp": metallurgy.get("ms_temp", None),
        "Laves_risk": metallurgy.get("laves_risk", None),
        "Z_phase_risk": metallurgy.get("z_phase_risk", None),
        "CEQ": metallurgy.get("ceq", None),
        "MX_balance": metallurgy.get("mx_balance", None),
        "total_alloy_wt": metallurgy.get("total_alloy", None),

        "ood_distance": physics.get("ood_distance", 0.0),
        "ood_penalty": physics.get("ood_penalty", 0.0),
        "metallurgy_penalty": physics.get("metallurgy_penalty", 0.0),
        "thermo_penalty": physics.get("thermo_penalty", 0.0),
    }

    for elem in FULL_COMPOSITION_FEATURES:
        row[elem] = comp_dict.get(elem, 0.0)

    row["Fe_balance"] = max(
        0.0,
        100.0 - sum(comp_dict.values())
    )

    return row


def _export_pareto_top_candidates(
    candidates,
    temp_k,
    stress,
    reference_mean=None,
    reference_cov_inv=None,
    top_n=10
):
    """
    Pareto 후보 Top 10을 CSV/JSON으로 저장.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for ind in candidates:
        rows.append(
            _summarize_candidate(
                individual=ind,
                temp_k=temp_k,
                stress=stress,
                reference_mean=reference_mean,
                reference_cov_inv=reference_cov_inv
            )
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -r["predicted_log_life"],
            r["material_cost"],
            r["physics_risk"]
        )
    )

    top_rows = rows[:top_n]

    df = pd.DataFrame(top_rows)

    csv_path = RESULTS_DIR / "pareto_top10.csv"
    json_path = RESULTS_DIR / "pareto_top10.json"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            top_rows,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"[Result] Pareto Top {top_n} CSV 저장: {csv_path}")
    print(f"[Result] Pareto Top {top_n} JSON 저장: {json_path}")

    return top_rows


# ============================================================
# [5] GA 평가 및 진화 엔진
# ============================================================


def run_alloy_optimization(u_temp_c, u_stress, u_cost_limit):

    temp_k = u_temp_c + 273.15
    toolbox = base.Toolbox()

    pop_size = int(SEARCH_POLICY.get("INITIAL_POPULATION_SIZE", 100))

    min_distance = float(SEARCH_POLICY.get("MIN_COMPOSITION_DISTANCE", 0.15))

    reference_mean, reference_cov_inv = _build_ood_reference_if_available()

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
        print(f"[경고] LLM/local seed 수집 실패: {e}")
        seeds = []

    for s in seeds:
        try:
            ind = _make_individual_from_seed(s)

            if not _is_valid_total_alloy(ind):
                continue

            if _is_diverse(ind, population, min_distance):
                population.append(ind)

        except Exception as e:
            print(f"[경고] seed 변환 실패: {e}")

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
        mu=50,
        lambda_=100,
        cxpb=0.7,
        mutpb=0.3,
        ngen=50,
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

    # 웹 전달용 Pareto Top 10 저장
    top_candidates = _export_pareto_top_candidates(
        candidates=candidates,
        temp_k=temp_k,
        stress=u_stress,
        reference_mean=reference_mean,
        reference_cov_inv=reference_cov_inv,
        top_n=10
    )

    best_ind = sorted(
        candidates,
        key=lambda ind: (
            ind.fitness.values[2],  # physics risk 최소
            ind.fitness.values[1],  # cost 최소
            -ind.fitness.values[0],  # life 최대
        ),
    )[0]

    best_recipe = _design_individual_to_comp(best_ind)

    # 최종 best 후보 JSON 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_summary = _summarize_candidate(
        individual=best_ind,
        temp_k=temp_k,
        stress=u_stress,
        reference_mean=reference_mean,
        reference_cov_inv=reference_cov_inv
    )

    best_json_path = RESULTS_DIR / "best_alloy.json"

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)

    print(f"[Result] Best alloy JSON 저장: {best_json_path}")

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