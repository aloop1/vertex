import os
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from .config import (
    DATA_DIR,
    FERRITIC_SYSTEM_LIMITS,
    LLM_CONFIG,
    DESIGN_VARIABLES,
    FULL_COMPOSITION_FEATURES,
    FIXED_ELEMENTS,
    MAX_TOTAL_ALLOY_WT,
)

from .physics import (
    as_full_composition,
    evaluate_physics,
)

 
# ============================================================
# [1] 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


USE_LLM_API = _env_flag("USE_LLM_API", default=False)
FORCE_LLM_REFRESH = _env_flag("FORCE_LLM_REFRESH", default=False)
STRICT_LLM_SEED_MODE = _env_flag("STRICT_LLM_SEED_MODE", default=True) 

CACHE_SEED_PATH = DATA_DIR / "seed_cache.json"
SAMPLE_SEED_PATH = DATA_DIR / "sample_seeds.json"


# ============================================================
# [2] JSON Parser
# ============================================================

def _robust_json_parser(text: str) -> str:

    if not text:
        return "[]"

    text = text.strip()

    match = re.search(
        r"\[\s*\{.*\}\s*\]",
        text,
        re.DOTALL
    )

    clean_text = match.group(0) if match else text

    clean_text = re.sub(
        r"[\x00-\x1F\x7F]",
        "",
        clean_text
    )

    return clean_text


# ============================================================
# [3] Seed 정규화 및 검증
# ============================================================

def _normalize_seed_composition(raw_comp: dict) -> dict:
    """
    raw seed를 FULL_COMPOSITION_FEATURES 19개 기준으로 정규화함.
    원칙:
    - seed는 DESIGN_VARIABLES 11개만 가져도 됨
    - P/S/O/Co/Re/Al/Cu/Ta는 FIXED_ELEMENTS에서 자동 삽입
    - 최종 반환은 항상 full composition
    """

    design_comp = {}

    for elem in DESIGN_VARIABLES:
        val = raw_comp.get(elem, 0.0)

        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0

        design_comp[elem] = val

    return as_full_composition(design_comp)


def _is_within_element_bounds(full_comp: dict) -> tuple[bool, str]:
    """
    DESIGN_VARIABLES가 FERRITIC_SYSTEM_LIMITS 범위 안에 있는지 확인함.
    """

    for elem in DESIGN_VARIABLES:
        if elem not in FERRITIC_SYSTEM_LIMITS:
            return False, f"{elem}의 FERRITIC_SYSTEM_LIMITS가 없습니다."

        low, high = FERRITIC_SYSTEM_LIMITS[elem]
        val = full_comp.get(elem, 0.0)

        if val < low or val > high:
            return (
                False,
                f"{elem} 범위 위반: {val:.6f} not in [{low}, {high}]"
            )

    return True, "OK"


def _is_within_total_alloy_limit(full_comp: dict) -> tuple[bool, str]:
    """
    전체 합금 원소 총합 제한 확인.
    """

    total_alloy = sum(full_comp.values())

    if total_alloy > MAX_TOTAL_ALLOY_WT:
        return (
            False,
            f"합금 총합 초과: {total_alloy:.4f}% > {MAX_TOTAL_ALLOY_WT}%"
        )

    return True, "OK"


def _is_physics_reasonable(full_comp: dict) -> tuple[bool, str]:
    """
    physics.py의 evaluate_physics()로 seed 1차 검증 (위험한 seed만 제거)
    """

    physics_result = evaluate_physics(
        comp_dict=full_comp,
        temp_k=650.0 + 273.15,
        reference_mean=None,
        reference_cov_inv=None
    )

    final_penalty = physics_result.get("final_penalty", 9999.0)

    max_allowed_penalty = float(
        os.environ.get("LLM_SEED_MAX_PHYSICS_PENALTY", "20.0")
    )

    if final_penalty > max_allowed_penalty:
        return (
            False,
            f"physics penalty 과다: {final_penalty:.4f} > {max_allowed_penalty}"
        )

    return True, "OK"


def _deduplicate_seeds(seeds: list[dict], decimals: int = 4) -> list[dict]:
    """
    거의 같은 seed 제거.
    """

    seen = set()
    unique = []

    for seed in seeds:
        comp = seed["composition"]

        key = tuple(
            round(comp.get(elem, 0.0), decimals)
            for elem in FULL_COMPOSITION_FEATURES
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(seed)

    return unique


def _validate_and_filter_seeds(raw_seeds: list[dict]) -> list[dict]:
    """
    raw seed를 실제 GA에 투입 가능한 seed로 정제합니다.
    """

    final_seeds = []
    rejected = []

    for idx, seed in enumerate(raw_seeds, start=1):
        raw_comp = seed.get("composition", {})

        if not isinstance(raw_comp, dict):
            rejected.append((idx, "composition이 dict가 아님"))
            continue

        full_comp = _normalize_seed_composition(raw_comp)

        ok, reason = _is_within_element_bounds(full_comp)
        if not ok:
            rejected.append((idx, reason))
            continue

        ok, reason = _is_within_total_alloy_limit(full_comp)
        if not ok:
            rejected.append((idx, reason))
            continue

        ok, reason = _is_physics_reasonable(full_comp)
        if not ok:
            rejected.append((idx, reason))
            continue

        final_seeds.append({
            "composition": full_comp
        })

    final_seeds = _deduplicate_seeds(final_seeds)

    if rejected:
        print(f"[LLM] 폐기된 seed {len(rejected)}개:")
        for idx, reason in rejected[:10]:
            print(f"  - seed {idx}: {reason}")

    return final_seeds


# ============================================================
# [4] Local Seed 입출력
# ============================================================

def _extract_seed_list(raw_data):
    """
    seed 파일이 list 형식이든,
    {"seeds": [...]} 형식이든 모두 처리합니다.
    """

    if isinstance(raw_data, list):
        return raw_data

    if isinstance(raw_data, dict) and isinstance(raw_data.get("seeds"), list):
        return raw_data["seeds"]

    return []


def _load_seed_file(path: Path) -> list[dict]:
    """
    JSON seed 파일을 읽고 검증합니다.
    """

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        raw_seeds = _extract_seed_list(raw_data)

        if not raw_seeds:
            print(f"[Seed] {path.name}에서 seed list를 찾지 못했습니다.")
            return []

        final_seeds = _validate_and_filter_seeds(raw_seeds)

        if final_seeds:
            print(f"[Seed] {path.name}에서 {len(final_seeds)}개 seed 로드 완료.")

        return final_seeds

    except Exception as e:
        print(f"[Seed] {path.name} 로드 실패: {e}")
        return []


def _save_seed_file(path: Path, seeds: list[dict]) -> None:
    """
    검증된 seed를 JSON 파일로 저장합니다.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config_version": "ferritic_v2_design_variables_fixed_elements",
            "generated_by": LLM_CONFIG.get("MODEL_NAME", "unknown"),
            "seed_count": len(seeds),
            "design_variables": DESIGN_VARIABLES,
            "fixed_elements": FIXED_ELEMENTS,
            "max_total_alloy_wt": MAX_TOTAL_ALLOY_WT,
            "seeds": seeds,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"[Seed] {path.name} 저장 완료.")

    except Exception as e:
        print(f"[Seed] {path.name} 저장 실패: {e}")


# ============================================================
# [5] API 없이 사용하는 rule-based local seed
# ============================================================

def _generate_rule_based_local_seeds() -> list[dict]:
    """
    API 호출 없이 개발/디버깅용으로 사용하는 seed.

    목적:
    - Gemini 토큰을 쓰지 않고 engine.py / physics.py / predictor 연결 검증
    """

    templates = [
        {
            "C": 0.10,
            "Si": 0.25,
            "Mn": 0.45,
            "Cr": 9.20,
            "Mo": 0.50,
            "W": 1.80,
            "Ni": 0.10,
            "V": 0.21,
            "Nb": 0.060,
            "N": 0.045,
            "B": 0.005,
        },
        {
            "C": 0.11,
            "Si": 0.20,
            "Mn": 0.50,
            "Cr": 9.50,
            "Mo": 0.70,
            "W": 1.60,
            "Ni": 0.08,
            "V": 0.22,
            "Nb": 0.055,
            "N": 0.045,
            "B": 0.004,
        },
        {
            "C": 0.09,
            "Si": 0.30,
            "Mn": 0.40,
            "Cr": 8.90,
            "Mo": 0.90,
            "W": 2.00,
            "Ni": 0.05,
            "V": 0.20,
            "Nb": 0.050,
            "N": 0.040,
            "B": 0.006,
        },
        {
            "C": 0.12,
            "Si": 0.18,
            "Mn": 0.48,
            "Cr": 10.00,
            "Mo": 0.40,
            "W": 1.50,
            "Ni": 0.10,
            "V": 0.23,
            "Nb": 0.065,
            "N": 0.050,
            "B": 0.004,
        },
        {
            "C": 0.10,
            "Si": 0.22,
            "Mn": 0.42,
            "Cr": 10.30,
            "Mo": 0.35,
            "W": 1.30,
            "Ni": 0.12,
            "V": 0.21,
            "Nb": 0.050,
            "N": 0.040,
            "B": 0.005,
        },
        {
            "C": 0.13,
            "Si": 0.15,
            "Mn": 0.50,
            "Cr": 9.10,
            "Mo": 0.60,
            "W": 2.20,
            "Ni": 0.05,
            "V": 0.24,
            "Nb": 0.070,
            "N": 0.055,
            "B": 0.003,
        },
        {
            "C": 0.10,
            "Si": 0.28,
            "Mn": 0.38,
            "Cr": 9.70,
            "Mo": 0.80,
            "W": 1.70,
            "Ni": 0.00,
            "V": 0.20,
            "Nb": 0.060,
            "N": 0.050,
            "B": 0.006,
        },
        {
            "C": 0.11,
            "Si": 0.24,
            "Mn": 0.44,
            "Cr": 8.80,
            "Mo": 1.00,
            "W": 1.90,
            "Ni": 0.08,
            "V": 0.22,
            "Nb": 0.060,
            "N": 0.050,
            "B": 0.005,
        },
        {
            "C": 0.09,
            "Si": 0.26,
            "Mn": 0.52,
            "Cr": 10.80,
            "Mo": 0.30,
            "W": 1.20,
            "Ni": 0.15,
            "V": 0.19,
            "Nb": 0.045,
            "N": 0.035,
            "B": 0.004,
        },
        {
            "C": 0.12,
            "Si": 0.20,
            "Mn": 0.46,
            "Cr": 9.40,
            "Mo": 0.55,
            "W": 1.85,
            "Ni": 0.05,
            "V": 0.23,
            "Nb": 0.060,
            "N": 0.045,
            "B": 0.005,
        },
    ]

    raw_seeds = [
        {"composition": comp}
        for comp in templates
    ]

    final_seeds = _validate_and_filter_seeds(raw_seeds)

    print(f"[Seed] rule-based local seed {len(final_seeds)}개 생성 완료.")

    return final_seeds


def _load_local_or_rule_based_seeds() -> list[dict]:
    """
    API를 쓰지 않는 개발 모드용 seed 로딩 순서.

    1. cached_llm_seeds_ferritic_v2.json
    2. sample_seeds.json
    3. 코드 내부 rule-based seed
    """

    cached = _load_seed_file(CACHE_SEED_PATH)
    if cached:
        return cached

    sample = _load_seed_file(SAMPLE_SEED_PATH)
    if sample:
        return sample

    return _generate_rule_based_local_seeds()


# ============================================================
# [6] Gemini API 설정
# ============================================================

def _configure_gemini_api():
    """
    FORCE_LLM_REFRESH=true일 때만 호출됩니다.
    여기서만 google.generativeai를 import합니다.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            f"\nGEMINI_API_KEY가 비어있거나 설정되지 않았습니다.\n"
            f"   - 탐색 경로: {ENV_PATH}\n"
            f"   - 조치: .env 파일에 GEMINI_API_KEY를 설정하거나 "
            f"USE_LLM_API=false로 개발 모드를 사용하십시오."
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    return genai


# ============================================================
# [7] Gemini API Seed Generation
# ============================================================
def _generate_seeds_with_gemini() -> list[dict]:
    """
    Gemini API를 호출하여 seed를 생성함.
    """

    genai = _configure_gemini_api()

    target_count = int(LLM_CONFIG.get("SEED_COUNT", 10))
    batch_size = int(LLM_CONFIG.get("BATCH_SIZE", 10))

    if batch_size <= 0:
        batch_size = 10

    if batch_size > target_count:
        batch_size = target_count

    print(
        f"[API] {LLM_CONFIG['MODEL_NAME']} 통신 준비 중 "
        f"(FORCE_LLM_REFRESH=true)"
    )
    print(
        f"[API] 목표 LLM seed {target_count}개를 "
        f"{batch_size}개씩 나누어 생성합니다."
    )

    bounds_str = "\n".join([
        f"- {elem}: {FERRITIC_SYSTEM_LIMITS[elem][0]} to {FERRITIC_SYSTEM_LIMITS[elem][1]} wt%"
        for elem in DESIGN_VARIABLES
    ])

    fixed_str = "\n".join([
        f"- {elem}: fixed at {val} wt%"
        for elem, val in FIXED_ELEMENTS.items()
    ])

    system_instruction = (
        "You are a world-class computational materials scientist specializing in "
        "9-12% Cr ferritic-martensitic creep-resistant steels. "
        "Your task is to generate physically realistic initial alloy compositions "
        "for genetic algorithm seeding."
    )

    model = genai.GenerativeModel(
        model_name=LLM_CONFIG["MODEL_NAME"],
        system_instruction=system_instruction
    )

    max_retries = 3
    retry_delay = 10
    temperature = float(LLM_CONFIG.get("TEMPERATURE", 0.2))

    final_seeds = []
    seen_keys = set()

    def _seed_key(seed: dict) -> tuple:
        comp = seed.get("composition", {})
        return tuple(
            round(float(comp.get(elem, 0.0)), 6)
            for elem in DESIGN_VARIABLES
        )

    batch_index = 0

    while len(final_seeds) < target_count:
        batch_index += 1

        remaining = target_count - len(final_seeds)
        current_batch_count = min(batch_size, remaining)

        user_prompt = f"""
Generate exactly {current_batch_count} unique, physically realistic ferritic-martensitic steel seed compositions.

IMPORTANT DESIGN RULES:
1. Generate values ONLY for the following DESIGN_VARIABLES:
{', '.join(DESIGN_VARIABLES)}

2. Do NOT generate values for impurity/fixed elements:
{', '.join(FIXED_ELEMENTS.keys())}

These fixed elements will be inserted automatically:
{fixed_str}

3. All DESIGN_VARIABLES must strictly satisfy these bounds:
{bounds_str}

4. The sum of all elements including fixed elements must not exceed {MAX_TOTAL_ALLOY_WT} wt%.

5. Prefer plausible 9-12% Cr ferritic-martensitic creep steel logic:
- Cr must remain in the 9-12% ferritic heat-resistant steel range.
- Avoid excessive Mo+W loading because of Laves/TCP risk.
- Keep V/Nb/N balanced for MX strengthening.
- Maintain martensitic transformability and weldability.
- Do not attempt to optimize fixed elements such as P, S, O, Co, Re, Al, Cu, or Ta.

OUTPUT INSTRUCTIONS:
- Return ONLY a JSON array.
- Each object MUST have a "composition" key.
- Inside "composition", include ONLY these DESIGN_VARIABLES:
C, Si, Mn, Cr, Mo, W, Ni, V, Nb, N, B
- Do NOT include:
P, S, O, Co, Re, Al, Cu, Ta
- Do not include markdown, explanation, reasoning, comments, or code fences.

Required JSON shape:
[
  {{
    "composition": {{
      "C": 0.10,
      "Si": 0.20,
      "Mn": 0.45,
      "Cr": 9.20,
      "Mo": 0.50,
      "W": 1.80,
      "Ni": 0.10,
      "V": 0.21,
      "Nb": 0.06,
      "N": 0.045,
      "B": 0.005
    }}
  }}
]
"""

        batch_success = False
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"[API] batch {batch_index} 생성 요청 "
                    f"{attempt}/{max_retries} "
                    f"({current_batch_count}개 요청)"
                )

                try:
                    response = model.generate_content(
                        user_prompt,
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192,
                            temperature=temperature,
                        ),
                        request_options={
                            "timeout": 60
                        }
                    )

                except Exception as api_err:
                    print(f"[Warning] Strict JSON GenerationConfig 실패: {api_err}")
                    print("[Warning] 일반 generation_config로 재시도합니다.")

                    response = model.generate_content(
                        user_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=8192,
                            temperature=temperature,
                        ),
                        request_options={
                            "timeout": 60
                        }
                    )

                raw_text = response.text.strip()
                clean_text = _robust_json_parser(raw_text)

                try:
                    raw_data = json.loads(clean_text)
                except json.JSONDecodeError as json_err:
                    preview = clean_text[:1000]
                    raise RuntimeError(
                        "Gemini 응답 JSON 파싱 실패\n"
                        f"- batch: {batch_index}\n"
                        f"- attempt: {attempt}\n"
                        f"- error: {json_err}\n"
                        f"- preview:\n{preview}"
                    ) from json_err

                raw_seeds = _extract_seed_list(raw_data)

                if not raw_seeds:
                    raise RuntimeError(
                        f"batch {batch_index}에서 seed list를 찾지 못했습니다."
                    )

                valid_seeds = _validate_and_filter_seeds(raw_seeds)

                accepted_in_batch = 0

                for seed in valid_seeds:
                    key = _seed_key(seed)

                    if key in seen_keys:
                        continue

                    seen_keys.add(key)
                    final_seeds.append(seed)
                    accepted_in_batch += 1

                    if len(final_seeds) >= target_count:
                        break

                print(
                    f"[API] batch {batch_index} 완료: "
                    f"검증 통과 신규 seed {accepted_in_batch}개 / "
                    f"누적 {len(final_seeds)}/{target_count}개"
                )

                batch_success = True
                break

            except Exception as e:
                last_error = e
                print(f"[Warning] batch {batch_index} 실패: {e}")

                if attempt < max_retries:
                    print(f"[API] {retry_delay}초 후 재시도합니다.")
                    time.sleep(retry_delay)

        if not batch_success:
            raise RuntimeError(
                f"[LLM-FATAL] Gemini seed batch {batch_index} 생성 실패. "
                f"마지막 오류: {last_error}"
            )

    if len(final_seeds) < target_count:
        raise RuntimeError(
            f"[LLM-FATAL] 목표 seed {target_count}개 중 "
            f"{len(final_seeds)}개만 확보했습니다."
        )

    print(f"[LLM] 최종 Gemini seed {len(final_seeds)}개 확보 완료.")

    return final_seeds[:target_count]


# ============================================================
# [8] Public API
# ============================================================

def get_expert_seeds():
    """
    engine.py에서 호출하는 공개 함수.

    최종 정책:
    - USE_LLM_API=false:
      개발 모드입니다. Gemini API를 호출하지 않고 cache → sample → rule-based seed를 사용합니다.
      단, seed가 0개이면 즉시 중단합니다.

    - USE_LLM_API=true, FORCE_LLM_REFRESH=false:
      검증된 cache seed만 사용합니다.
      cache가 없으면 STRICT_LLM_SEED_MODE 값에 따라 동작합니다.
        * STRICT_LLM_SEED_MODE=true  : 즉시 중단
        * STRICT_LLM_SEED_MODE=false : local/rule-based seed fallback 허용

    - USE_LLM_API=true, FORCE_LLM_REFRESH=true:
      Gemini API를 실제 호출합니다.
      API 호출, JSON 파싱, seed 검증 중 하나라도 실패하면 즉시 중단합니다.
    """

    print(
        f"[LLM] USE_LLM_API={USE_LLM_API}, "
        f"FORCE_LLM_REFRESH={FORCE_LLM_REFRESH}, "
        f"STRICT_LLM_SEED_MODE={STRICT_LLM_SEED_MODE}"
    )

    # --------------------------------------------------------
    # [1] 개발 모드: 사용자가 명시적으로 API를 끈 경우에만 local seed 허용
    # --------------------------------------------------------
    if not USE_LLM_API:
        print("[LLM] 개발 모드: Gemini API를 호출하지 않고 local/rule-based seed를 사용합니다.")

        seeds = _load_local_or_rule_based_seeds()

        if not seeds:
            raise RuntimeError(
                "[LLM-FATAL] 개발 모드 seed 생성 실패: "
                "cache/sample/rule-based seed가 모두 비어 있습니다."
            )

        return seeds

    # --------------------------------------------------------
    # [2] API 사용 모드 + refresh 아님: cache만 우선 허용
    # --------------------------------------------------------
    if USE_LLM_API and not FORCE_LLM_REFRESH:
        cached = _load_seed_file(CACHE_SEED_PATH)

        if cached:
            print("[LLM] 검증된 cache seed를 사용합니다. Gemini API를 새로 호출하지 않습니다.")
            return cached

        if STRICT_LLM_SEED_MODE:
            raise RuntimeError(
                "[LLM-FATAL] USE_LLM_API=true이지만 검증된 cache seed가 없습니다. "
                "STRICT_LLM_SEED_MODE=true이므로 local/rule-based seed로 대체하지 않습니다. "
                "해결 방법: FORCE_LLM_REFRESH=true로 Gemini API를 실제 호출하거나, "
                "개발용 실행이면 USE_LLM_API=false로 명시하십시오."
            )

        print(
            "[LLM-WARNING] USE_LLM_API=true이지만 cache가 없습니다. "
            "STRICT_LLM_SEED_MODE=false이므로 local/rule-based seed로 fallback합니다. "
            "이 결과는 Gemini API 신규 호출 결과가 아닙니다."
        )

        seeds = _load_local_or_rule_based_seeds()

        if not seeds:
            raise RuntimeError(
                "[LLM-FATAL] fallback seed 생성 실패: "
                "cache/sample/rule-based seed가 모두 비어 있습니다."
            )

        return seeds

    # --------------------------------------------------------
    # [3] API 실제 호출 모드: 실패 시 _generate_seeds_with_gemini()가 RuntimeError 발생
    # --------------------------------------------------------
    final_seeds = _generate_seeds_with_gemini()

    if not final_seeds:
        raise RuntimeError(
            "[LLM-FATAL] Gemini API 호출은 완료됐지만 검증을 통과한 seed가 0개입니다."
        )

    _save_seed_file(
        CACHE_SEED_PATH,
        final_seeds
    )

    return final_seeds


# ============================================================
# [9] Standalone Test
# ============================================================

if __name__ == "__main__":
    try:
        test_seeds = get_expert_seeds()

        if test_seeds:
            print("\n[Sample Seed Check - 검증 통과 데이터]")
            print(
                json.dumps(
                    test_seeds[0],
                    indent=2,
                    ensure_ascii=False
                )
            )

    except Exception as e:
        print(f"\n[Fatal Error] {e}")