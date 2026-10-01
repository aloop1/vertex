"""사용자 What-if 질문 → Gemini 후보 생성 → 결과 정리"""

from __future__ import annotations

import json
from typing import Any

from .api import generate_json


SYSTEM_INSTRUCTION = """
You are an AI assistant supporting exploratory design of creep-resistant Fe-based alloys.

Your role is to answer a user's What-if request by proposing a revised scenario,
which may include:
- alloy composition changes
- operating condition changes
- heat-treatment changes
- or a combination of them

The proposal is exploratory.

The proposed candidate will be evaluated separately by:
1. the project's creep-life prediction model
2. a human expert

Do not claim that the proposed alloy is experimentally validated, safe,
manufacturable, certified, or optimal.

Return only the requested JSON object.
""".strip()


ALLOWED_ACTIONS = {
    "modify_composition",
    "modify_conditions",
    "modify_heat_treatment",
    "new_alloy",
    "combined_change",
}


def _as_numeric_dict(
    data: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, float]:
    """
    dict 값을 float로 변환한다.

    이것은 데이터 형식 검증이며
    물리적 제약 검증이 아니다.
    """

    result: dict[str, float] = {}

    for key, value in data.items():

        try:
            result[str(key)] = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}.{key} 값이 숫자가 아닙니다: {value!r}"
            ) from exc

    return result


def _reject_unknown_keys(
    proposed: dict[str, Any],
    current: dict[str, Any],
    *,
    field_name: str,
) -> None:
    """
    현재 시스템에 존재하지 않는 필드가
    Gemini 응답에 추가되었는지 검사한다.

    물리 제약이 아니라 schema validation이다.
    """

    unknown_keys = (
        set(proposed.keys())
        - set(current.keys())
    )

    if unknown_keys:
        unknown_text = ", ".join(
            sorted(unknown_keys)
        )

        raise ValueError(
            f"AI 응답의 {field_name}에 "
            f"알 수 없는 필드가 있습니다: "
            f"{unknown_text}"
        )


def _build_prompt(
    *,
    question: str,
    current_composition: dict[str, float],
    current_conditions: dict[str, float],
    current_heat_treatment: dict[str, float],
    current_prediction: float | None,
) -> str:

    composition_keys = list(
        current_composition.keys()
    )

    condition_keys = list(
        current_conditions.keys()
    )

    heat_treatment_keys = list(
        current_heat_treatment.keys()
    )

    context = {
        "current_composition": (
            current_composition
        ),
        "current_conditions": (
            current_conditions
        ),
        "current_heat_treatment": (
            current_heat_treatment
        ),
        "current_predicted_life": (
            current_prediction
        ),
        "user_what_if_question": (
            question
        ),
    }

    return f"""
Analyze the following What-if request and propose one revised candidate scenario.

CURRENT CONTEXT
{json.dumps(context, ensure_ascii=False, indent=2)}

OUTPUT RULES

1. Return exactly one JSON object.
   Do not use markdown or code fences.

2. "action" must be exactly one of:
   - "modify_composition"
   - "modify_conditions"
   - "modify_heat_treatment"
   - "new_alloy"
   - "combined_change"

3. "composition" must be a JSON object.

   The system currently supports ONLY these composition fields:
   {composition_keys}

   Do not create any additional composition fields.

4. "conditions" must be a JSON object.

   The system currently supports ONLY these condition fields:
   {condition_keys}

   Do not create any additional condition fields.

5. "heat_treatment" must be a JSON object.

   The system currently supports ONLY these heat-treatment fields:
   {heat_treatment_keys}

   Do not create any additional heat-treatment fields.

6. If a value is not intended to change,
   keep its current value.

7. You may freely propose changes that answer the user's What-if request.
   Do not apply project-internal GA constraints,
   physical penalty functions,
   Pareto optimization,
   seed-generation logic,
   cached alloy seeds,
   or hidden design bounds.

8. "design_rationale" must briefly explain
   why the proposed changes answer the user's request.

9. "assumptions" must be a JSON array
   containing short strings.

10. Do not calculate creep life yourself.
    The project's creep-life prediction model
    will calculate the new predicted life after your proposal.

REQUIRED JSON SHAPE

{{
  "action": "modify_composition",
  "composition": {{
    "C": 0.1
  }},
  "conditions": {{
    "stress": 100.0,
    "temp": 923.15
  }},
  "heat_treatment": {{
    "Ntemp": 1323.15,
    "Ntime": 1.0,
    "Ttemp": 1023.15,
    "Ttime": 2.0,
    "Atemp": 0.0,
    "Atime": 0.0
  }},
  "design_rationale": "...",
  "assumptions": ["..."]
}}
""".strip()


def _normalize_result(
    raw: dict[str, Any],
    *,
    current_composition: dict[str, float],
    current_conditions: dict[str, float],
    current_heat_treatment: dict[str, float],
) -> dict[str, Any]:
    """
    Gemini 응답을 predictor에 전달 가능한 형태로 정리한다.

    합금 물리 제약은 적용하지 않는다.
    schema/type validation만 수행한다.
    """

    # --------------------------------------------------------
    # action
    # --------------------------------------------------------

    action = str(
        raw.get(
            "action",
            "",
        )
    ).strip()

    if action not in ALLOWED_ACTIONS:
        raise ValueError(
            "AI 응답의 action이 올바르지 않습니다: "
            f"{action!r}"
        )

    # --------------------------------------------------------
    # 각 영역 확인
    # --------------------------------------------------------

    raw_composition = (
        raw.get("composition") or {}
    )

    raw_conditions = (
        raw.get("conditions") or {}
    )

    raw_heat_treatment = (
        raw.get("heat_treatment") or {}
    )

    if not isinstance(
        raw_composition,
        dict,
    ):
        raise ValueError(
            "AI 응답의 composition은 object여야 합니다."
        )

    if not isinstance(
        raw_conditions,
        dict,
    ):
        raise ValueError(
            "AI 응답의 conditions는 object여야 합니다."
        )

    if not isinstance(
        raw_heat_treatment,
        dict,
    ):
        raise ValueError(
            "AI 응답의 heat_treatment는 object여야 합니다."
        )

    # --------------------------------------------------------
    # 시스템에 없는 key 차단
    # --------------------------------------------------------

    _reject_unknown_keys(
        raw_composition,
        current_composition,
        field_name="composition",
    )

    _reject_unknown_keys(
        raw_conditions,
        current_conditions,
        field_name="conditions",
    )

    _reject_unknown_keys(
        raw_heat_treatment,
        current_heat_treatment,
        field_name="heat_treatment",
    )

    # --------------------------------------------------------
    # 변경하지 않은 값은 기존 값 유지
    # --------------------------------------------------------

    composition = dict(
        current_composition
    )

    composition.update(
        _as_numeric_dict(
            raw_composition,
            field_name="composition",
        )
    )

    conditions = dict(
        current_conditions
    )

    conditions.update(
        _as_numeric_dict(
            raw_conditions,
            field_name="conditions",
        )
    )

    heat_treatment = dict(
        current_heat_treatment
    )

    heat_treatment.update(
        _as_numeric_dict(
            raw_heat_treatment,
            field_name="heat_treatment",
        )
    )

    # --------------------------------------------------------
    # 설명
    # --------------------------------------------------------

    design_rationale = str(
        raw.get(
            "design_rationale",
            "",
        )
    ).strip()

    assumptions = raw.get(
        "assumptions",
        [],
    )

    if isinstance(
        assumptions,
        str,
    ):
        assumptions = [
            assumptions
        ]

    elif not isinstance(
        assumptions,
        list,
    ):
        assumptions = []

    assumptions = [
        str(item).strip()
        for item in assumptions
        if str(item).strip()
    ]

    # --------------------------------------------------------
    # 최종 결과
    # --------------------------------------------------------

    return {
        "action": action,
        "composition": composition,
        "conditions": conditions,
        "heat_treatment": heat_treatment,
        "design_rationale": design_rationale,
        "assumptions": assumptions,
    }


def run_what_if(
    *,
    question: str,
    current_composition: dict[str, float],
    current_conditions: dict[str, float],
    current_heat_treatment: dict[str, float],
    current_prediction: float | None = None,
) -> dict[str, Any]:
    """
    Gemini를 이용해 What-if 후보를 생성한다.

    여기서는 후보만 만든다.

    실제 creep life 계산은
    analysis.pipeline에서 동일한 예측 모델로
    다시 수행한다.
    """

    if (
        not question
        or not question.strip()
    ):
        raise ValueError(
            "What-if 질문이 비어 있습니다."
        )

    if not current_composition:
        raise ValueError(
            "현재 합금 조성이 필요합니다."
        )

    if not current_conditions:
        raise ValueError(
            "현재 사용 조건이 필요합니다."
        )

    if not current_heat_treatment:
        raise ValueError(
            "현재 열처리 조건이 필요합니다."
        )

    prompt = _build_prompt(
        question=question.strip(),
        current_composition=current_composition,
        current_conditions=current_conditions,
        current_heat_treatment=current_heat_treatment,
        current_prediction=current_prediction,
    )

    raw = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=prompt,
    )

    return _normalize_result(
        raw,
        current_composition=current_composition,
        current_conditions=current_conditions,
        current_heat_treatment=current_heat_treatment,
    )