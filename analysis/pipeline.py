"""질문 라우팅 / 기능 통합"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from models.transformer_and_tree_ensemble import load_transformer_tree_predictor
from .local_llm.explainer import answer_with_rag
from .local_llm.model import generate_general_chat, warmup_local_llm
from .rag.retriever import get_embedding_model, load_index, retrieve
from .what_if.service import run_what_if


# ============================================================
# Router
# ============================================================

ALLOWED_INTENTS = ("prediction", "what_if", "knowledge")
ROUTABLE_INTENTS = ("prediction", "what_if", "knowledge", "general_chat")
GENERAL_CHAT_THRESHOLD = 0.60
GENERAL_CHAT_FALLBACK_THRESHOLD = 0.50

# 언어 규칙/키워드 분기를 사용하지 않는다.
# multilingual-e5-small의 문장 의미 임베딩을 입력으로 사용하고,
# 독립적인 3개 이진 Logistic Regression을 One-vs-Rest로 학습한다.
ROUTER_LABEL_THRESHOLD = 0.45
ROUTER_CLASSIFIER_C = 4.0

# 각 항목: (질문, 정답 intents)
# 단일 의도뿐 아니라 실제 서비스에서 중요한 복합 의도 문장을 함께 학습한다.
# 조사/어미/유의어/구어체 변화에 특정 문자열 규칙으로 반응하지 않도록
# 문장 전체 의미 임베딩만 사용한다.
ROUTER_TRAINING_DATA: list[tuple[str, tuple[str, ...]]] = [
    # prediction -------------------------------------------------
    ("이 조건에서 예상 크립 수명을 계산해줘", ("prediction",)),
    ("현재 조성의 크립 수명은 몇 시간이야", ("prediction",)),
    ("이 합금의 수명을 예측해줘", ("prediction",)),
    ("650도 100MPa에서 예상 수명이 얼마나 돼", ("prediction",)),
    ("현재 입력값으로 수명 계산해줘", ("prediction",)),
    ("크립 파단 수명을 예측해줘", ("prediction",)),
    ("이 조건으로 몇 시간 버틸 수 있어", ("prediction",)),
    ("예상 수명을 알려줘", ("prediction",)),
    ("지금 입력한 합금의 수명이 궁금해", ("prediction",)),
    ("현재 상태 그대로 두면 크립 수명이 어느 정도야", ("prediction",)),
    ("이 조성 기준으로 파단까지 몇 시간 예상돼", ("prediction",)),
    ("현재 온도와 응력에서 life를 계산해줘", ("prediction",)),
    ("지금 조건의 log life와 시간을 보여줘", ("prediction",)),
    ("현재 합금으로 수명 예측 한번 해봐", ("prediction",)),
    ("이 입력값에서 모델이 예측하는 수명은", ("prediction",)),
    ("조건은 그대로고 예상 파단 수명만 알고 싶어", ("prediction",)),
    ("현재 데이터로 creep life prediction 해줘", ("prediction",)),
    ("이 재료가 이 조건에서 얼마나 버틸지 계산해줘", ("prediction",)),
    ("지금 설정 그대로 수명만 출력해줘", ("prediction",)),
    ("변경 없이 현재 수명부터 계산해줘", ("prediction",)),

    # what_if ----------------------------------------------------
    ("현재 W 1.8%를 2.2%로 바꾸면 수명이 어떻게 돼", ("what_if",)),
    ("Cr을 9.2%에서 10%로 올리면 결과가 어떻게 변해", ("what_if",)),
    ("Mo를 줄였을 때 수명을 다시 계산해줘", ("what_if",)),
    ("현재 조성에서 W만 높여줘", ("what_if",)),
    ("온도를 650도에서 700도로 바꾸면 수명이 어떻게 돼", ("what_if",)),
    ("응력을 낮추면 예상 수명이 얼마나 달라져", ("what_if",)),
    ("수명이 길어지도록 새로운 조성을 제안해줘", ("what_if",)),
    ("이 합금의 조성을 바꿔서 다시 예측해줘", ("what_if",)),
    ("B 함량을 0.008로 변경하면 어떻게 돼", ("what_if",)),
    ("다른 조성 후보를 만들어줘", ("what_if",)),
    ("지금 합금에서 Cr만 10으로 수정해서 결과 비교해줘", ("what_if",)),
    ("W를 조금 더 넣은 후보를 하나 만들어봐", ("what_if",)),
    ("현재 온도를 20도 높인 경우를 계산해줘", ("what_if",)),
    ("응력을 90MPa로 바꿔서 다시 예측해줘", ("what_if",)),
    ("열처리 온도를 바꿨을 때 새 수명을 보여줘", ("what_if",)),
    ("현재 조성을 기반으로 수명이 더 긴 후보를 제안해줘", ("what_if",)),
    ("이 조건에서 니켈을 0.2로 바꾸면 얼마나 달라질까", ("what_if",)),
    ("지금 값 중 W만 수정해서 다시 돌려줘", ("what_if",)),
    ("현재 합금에 변화를 줘서 새로운 시나리오를 만들어줘", ("what_if",)),
    ("이 조성 말고 다른 조성으로 예측해보고 싶어", ("what_if",)),

    # knowledge --------------------------------------------------
    ("9Cr강에서 W 함량이 증가하면 크립 특성에 어떤 영향을 미쳐", ("knowledge",)),
    ("W가 크립 수명에 미치는 영향을 설명해줘", ("knowledge",)),
    ("Cr의 역할이 뭐야", ("knowledge",)),
    ("B가 크립 강도에 왜 중요한가", ("knowledge",)),
    ("M23C6가 크립 저항성에 어떤 역할을 해", ("knowledge",)),
    ("Laves phase가 무엇인지 설명해줘", ("knowledge",)),
    ("W와 Mo의 차이를 문헌 근거로 설명해줘", ("knowledge",)),
    ("왜 장시간 크립에서 강도가 떨어져", ("knowledge",)),
    ("9Cr강의 크립 파단 메커니즘을 알려줘", ("knowledge",)),
    ("W 함량과 크립 강도의 관계가 어떻게 돼", ("knowledge",)),
    ("Cr을 많이 넣으면 일반적으로 크립 특성이 어떻게 변해", ("knowledge",)),
    ("W가 늘어날수록 어떤 미세조직 변화가 나타나는지 설명해줘", ("knowledge",)),
    ("9Cr강에서 보론의 효과를 알려줘", ("knowledge",)),
    ("장시간 사용 중 Laves phase가 왜 문제가 될 수 있어", ("knowledge",)),
    ("크립 강도와 크립 수명은 어떻게 다른 개념이야", ("knowledge",)),
    ("정규화와 템퍼링이 9Cr 내열강에 어떤 영향을 주는지 알려줘", ("knowledge",)),
    ("W 첨가가 크립 거동에 미치는 일반적인 경향을 설명해줘", ("knowledge",)),
    ("Cr 함량 증가 효과를 논문 근거로 알려줘", ("knowledge",)),
    ("Mo와 W가 석출상 형성에 어떤 영향을 주는지 궁금해", ("knowledge",)),
    ("9Cr 계열에서 장시간 크립 열화 원인을 설명해줘", ("knowledge",)),
    ("크롬이 많아지면 왜 특성이 달라지는지 설명해줘", ("knowledge",)),
    ("W를 더 넣는 것이 항상 좋은지 문헌 기준으로 알려줘", ("knowledge",)),

    # prediction + knowledge ------------------------------------
    ("현재 예상 수명은 얼마고 그 결과와 관련된 재료학적 근거도 설명해줘", ("prediction", "knowledge")),
    ("이 조건의 수명을 계산하고 왜 이런 경향이 나오는지 문헌으로 설명해줘", ("prediction", "knowledge")),
    ("현재 수명 예측값과 관련 메커니즘을 같이 알려줘", ("prediction", "knowledge")),
    ("지금 합금이 몇 시간 버티는지와 W의 역할도 설명해줘", ("prediction", "knowledge")),
    ("현재 조건에서 수명부터 계산하고 관련 연구 내용도 알려줘", ("prediction", "knowledge")),
    ("이 조성의 예측 수명과 Cr이 크립에 미치는 영향도 같이 설명해줘", ("prediction", "knowledge")),
    ("현재 life를 구하고 그 수치가 실험값이 아닌 모델 예측이라는 점과 관련 문헌도 보여줘", ("prediction", "knowledge")),
    ("지금 수명 예측 결과를 주고 장시간 크립 열화 원인도 알려줘", ("prediction", "knowledge")),

    # what_if + knowledge ---------------------------------------
    ("Cr을 10%로 올리면 수명이 어떻게 되고 왜 그런지도 논문 근거로 설명해줘", ("what_if", "knowledge")),
    ("W를 2.2%로 바꾸면 왜 수명이 달라지는지도 문헌으로 알려줘", ("what_if", "knowledge")),
    ("현재 B를 0.008로 수정해서 재예측하고 B의 역할도 설명해줘", ("what_if", "knowledge")),
    ("응력을 90MPa로 낮췄을 때 결과와 그 현상을 설명할 근거를 찾아줘", ("what_if", "knowledge")),
    ("온도를 700도로 바꿔서 수명을 보고 온도와 크립의 관계도 설명해줘", ("what_if", "knowledge")),
    ("W를 더 넣은 후보를 만들어보고 왜 그런 조성이 의미 있는지도 문헌으로 설명해줘", ("what_if", "knowledge")),
    ("현재 Cr을 높여서 다시 계산한 뒤 Cr의 일반적인 효과도 알려줘", ("what_if", "knowledge")),
    ("Mo를 줄인 시나리오의 결과와 Mo 관련 크립 메커니즘을 함께 설명해줘", ("what_if", "knowledge")),
    ("새 조성 후보를 제안해서 수명을 계산하고 관련 논문 근거도 붙여줘", ("what_if", "knowledge")),
    ("열처리 조건을 바꿔서 다시 예측하고 그 변화와 관련된 재료학적 설명도 해줘", ("what_if", "knowledge")),
    ("현재 W 값을 수정한 결과와 W가 9Cr강에서 하는 역할을 같이 알려줘", ("what_if", "knowledge")),
    ("Cr 10퍼센트 조건으로 돌려보고 결과가 왜 그럴 수 있는지 자료 기반으로 설명해줘", ("what_if", "knowledge")),

    # prediction + what_if --------------------------------------
    ("현재 수명도 알려주고 W를 2.2%로 바꿨을 때와 비교해줘", ("prediction", "what_if")),
    ("지금 수명과 Cr을 10으로 바꾼 뒤 수명을 둘 다 보여줘", ("prediction", "what_if")),
    ("현재 결과부터 계산하고 W를 높였을 때 얼마나 달라지는지도 알려줘", ("prediction", "what_if")),
    ("기존 수명과 온도를 700도로 바꾼 수명을 비교해줘", ("prediction", "what_if")),
    ("현재 합금의 수명을 먼저 보여주고 다른 조성 후보의 수명도 비교해줘", ("prediction", "what_if")),
    ("지금 예측값이랑 응력을 낮춘 뒤 예측값을 같이 보여줘", ("prediction", "what_if")),
    ("변경 전 수명과 B를 0.008로 수정한 뒤 수명을 비교해줘", ("prediction", "what_if")),
    ("현재 조건 결과와 열처리 변경 후 결과를 나란히 보여줘", ("prediction", "what_if")),

    # prediction + what_if + knowledge --------------------------
    ("현재 수명과 W 변경 후 수명을 비교하고 그 이유도 설명해줘", ("prediction", "what_if", "knowledge")),
    ("지금 수명을 계산한 다음 Cr을 10%로 바꿔 다시 계산하고 차이의 문헌 근거도 알려줘", ("prediction", "what_if", "knowledge")),
    ("현재 결과와 W 2.2% 변경 결과를 비교하면서 W의 재료학적 역할도 설명해줘", ("prediction", "what_if", "knowledge")),
    ("기존 수명, 응력을 낮춘 뒤 수명, 그리고 왜 차이가 나는지 근거까지 보여줘", ("prediction", "what_if", "knowledge")),
    ("지금 합금 수명과 새 후보 수명을 비교하고 후보 변경의 의미를 논문으로 설명해줘", ("prediction", "what_if", "knowledge")),
    ("현재 예측값을 보여주고 온도를 바꾼 결과와 크립 메커니즘까지 같이 알려줘", ("prediction", "what_if", "knowledge")),
    ("변경 전 수명부터 계산해서 B 수정 후와 비교하고 B의 효과도 설명해줘", ("prediction", "what_if", "knowledge")),
    ("현재 수명과 Cr 변경 수명을 둘 다 구한 뒤 관련 문헌에서 어떤 경향인지 알려줘", ("prediction", "what_if", "knowledge")),
]

# 일반 대화는 기능 intent와 배타적인 별도 semantic gate로 처리한다.
# 키워드/정규식으로 인사를 판정하지 않는다.
GENERAL_CHAT_TRAINING_DATA: list[str] = [
    "안녕",
    "안녕하세요",
    "반가워",
    "도움이 필요해",
    "나 좀 도와줄래",
    "고마워",
    "감사합니다",
    "너는 누구야",
    "너는 어떤 챗봇이야",
    "무슨 기능이 있어",
    "여기서 뭘 할 수 있어",
    "사용법 알려줘",
    "어떤 질문을 하면 돼",
    "처음 왔는데 어떻게 쓰면 돼",
    "잘 부탁해",
    "좋아 시작하자",
    "hello",
    "hi",
    "thanks",
    "what can you do",
]

_router_classifier: OneVsRestClassifier | None = None
_general_chat_classifier: LogisticRegression | None = None
_life_predictor = None



def _make_router_targets() -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    targets = np.zeros((len(ROUTER_TRAINING_DATA), len(ALLOWED_INTENTS)), dtype=np.int32)

    for row_index, (text, intents) in enumerate(ROUTER_TRAINING_DATA):
        texts.append(text)
        for intent in intents:
            targets[row_index, ALLOWED_INTENTS.index(intent)] = 1

    return texts, targets



def _get_router_classifier() -> OneVsRestClassifier:
    """E5 임베딩 기반 multi-label 분류기를 최초 1회 학습해 재사용한다."""

    global _router_classifier

    if _router_classifier is not None:
        return _router_classifier

    texts, targets = _make_router_targets()
    model = get_embedding_model()

    embeddings = model.encode(
        [f"query: {text}" for text in texts],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    base_classifier = LogisticRegression(
        C=ROUTER_CLASSIFIER_C,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier = OneVsRestClassifier(base_classifier)
    classifier.fit(embeddings, targets)

    _router_classifier = classifier
    return _router_classifier



def _get_general_chat_classifier() -> LogisticRegression:
    """E5 의미 임베딩으로 일반 대화 여부를 판정하는 별도 binary classifier."""

    global _general_chat_classifier

    if _general_chat_classifier is not None:
        return _general_chat_classifier

    # positive: 일반 대화
    # negative: 실제 기능 요청
    positive_texts = GENERAL_CHAT_TRAINING_DATA
    negative_texts = [
        text
        for text, _ in ROUTER_TRAINING_DATA
    ]

    texts = positive_texts + negative_texts
    labels = np.asarray(
        [1] * len(positive_texts)
        + [0] * len(negative_texts),
        dtype=np.int32,
    )

    model = get_embedding_model()

    embeddings = model.encode(
        [f"query: {text}" for text in texts],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    classifier = LogisticRegression(
        C=ROUTER_CLASSIFIER_C,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    classifier.fit(
        embeddings,
        labels,
    )

    _general_chat_classifier = classifier
    return _general_chat_classifier



def _router_probabilities(question: str) -> dict[str, float]:
    question = " ".join(question.strip().split())
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    embedding = get_embedding_model().encode(
        [f"query: {question}"],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    functional_probabilities = (
        _get_router_classifier()
        .predict_proba(embedding)[0]
    )

    general_probability = float(
        _get_general_chat_classifier()
        .predict_proba(embedding)[0, 1]
    )

    scores = {
        intent: float(functional_probabilities[index])
        for index, intent in enumerate(ALLOWED_INTENTS)
    }
    scores["general_chat"] = general_probability
    return scores



def _normalize_intents(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("intents는 list 또는 tuple이어야 합니다.")

    found: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise ValueError("intent 값은 문자열이어야 합니다.")

        intent = item.strip().lower()

        if intent not in ROUTABLE_INTENTS:
            raise ValueError(f"허용되지 않은 intent: {intent}")

        found.add(intent)

    if not found:
        raise ValueError("intent가 하나도 없습니다.")

    # general_chat은 기능 실행과 동시에 사용하지 않는다.
    functional = [
        intent
        for intent in ALLOWED_INTENTS
        if intent in found
    ]

    if functional:
        return functional

    return ["general_chat"]



def build_execution_plan(intents: list[str]) -> list[str]:
    """
    의도 조합을 실제 실행 모듈 단위로 정리한다.

    What-if 파이프라인이 Before/After를 모두 예측하므로
    prediction + what_if 조합에서 predictor를 중복 실행하지 않는다.
    """

    intents = _normalize_intents(intents)

    if intents == ["general_chat"]:
        return ["general_chat"]

    plan: list[str] = []

    if "what_if" in intents:
        plan.append("what_if")
    elif "prediction" in intents:
        plan.append("prediction")

    if "knowledge" in intents:
        plan.append("knowledge")

    return plan



def classify_question(question: str) -> dict[str, Any]:
    """
    E5 문장 의미 임베딩 기반 Router.

    1) general_chat binary semantic gate
    2) prediction / what_if / knowledge multi-label classifier

    특정 단어/어미/정규식으로 intent를 판정하지 않는다.
    """

    scores = _router_probabilities(question)

    max_functional_score = max(
        scores[intent]
        for intent in ALLOWED_INTENTS
    )

    # 명확한 일반 대화일 때만 general_chat으로 보낸다.
    if (
        scores["general_chat"] >= GENERAL_CHAT_THRESHOLD
        and max_functional_score < GENERAL_CHAT_THRESHOLD
    ):
        intents = ["general_chat"]

    else:
        intents = [
            intent
            for intent in ALLOWED_INTENTS
            if scores[intent] >= ROUTER_LABEL_THRESHOLD
        ]

        if not intents:
            if (
                scores["general_chat"]
                >= GENERAL_CHAT_FALLBACK_THRESHOLD
            ):
                intents = ["general_chat"]
            else:
                intents = [
                    max(
                        ALLOWED_INTENTS,
                        key=lambda intent: scores[intent],
                    )
                ]

    intents = _normalize_intents(intents)

    return {
        "intents": intents,
        "execution_plan": build_execution_plan(intents),
        "router_source": "e5_logistic_general_gate",
        "intent_scores": scores,
        "fallback_reasons": [],
        "gemma_error": None,
    }


# ============================================================
# Actual functions
# ============================================================



def _get_life_predictor():
    global _life_predictor

    if _life_predictor is None:
        predictor = load_transformer_tree_predictor(
            artifact_path=None,
            allow_smoke_fallback=False,
        )

        if getattr(predictor, "is_smoke_model", False):
            raise RuntimeError("실제 수명예측 모델이 아닌 smoke-test 모델이 로드되었습니다.")

        _life_predictor = predictor

    return _life_predictor



def _predict_creep_life(
    composition: dict[str, float],
    conditions: dict[str, float],
    heat_treatment: dict[str, float],
) -> dict[str, float]:
    if "stress" not in conditions or "temp" not in conditions:
        raise ValueError("conditions에는 stress와 temp(K)가 필요합니다.")

    result = _get_life_predictor().predict_one(
        stress=float(conditions["stress"]),
        temp=float(conditions["temp"]),
        composition=composition,
        heat_treatment=heat_treatment,
    )

    log_life = float(result.log_lifetime)
    if not math.isfinite(log_life):
        raise RuntimeError("수명예측 모델이 NaN 또는 Inf를 반환했습니다.")

    return {
        "log_life": log_life,
        "life_hours": 10.0 ** log_life,
    }



def run_what_if_pipeline(
    *,
    question: str,
    current_composition: dict[str, float],
    current_conditions: dict[str, float],
    current_heat_treatment: dict[str, float],
) -> dict[str, Any]:
    before = _predict_creep_life(
        current_composition,
        current_conditions,
        current_heat_treatment,
    )

    candidate = run_what_if(
        question=question,
        current_composition=current_composition,
        current_conditions=current_conditions,
        current_heat_treatment=current_heat_treatment,
        current_prediction=before["life_hours"],
    )

    after = _predict_creep_life(
        candidate["composition"],
        candidate["conditions"],
        candidate["heat_treatment"],
    )

    before_hours = before["life_hours"]
    after_hours = after["life_hours"]
    change_percent = (
        ((after_hours - before_hours) / before_hours) * 100.0
        if before_hours > 0
        else None
    )

    return {
        "before": {
            "composition": current_composition,
            "conditions": current_conditions,
            "heat_treatment": current_heat_treatment,
            "prediction": before,
        },
        "after": {
            "composition": candidate["composition"],
            "conditions": candidate["conditions"],
            "heat_treatment": candidate["heat_treatment"],
            "prediction": after,
        },
        "comparison": {
            "life_change_hours": after_hours - before_hours,
            "life_change_percent": change_percent,
        },
        "what_if": {
            "question": question,
            "action": candidate["action"],
            "design_rationale": candidate["design_rationale"],
            "assumptions": candidate["assumptions"],
        },
    }



def _require_current_state(
    composition: dict[str, float] | None,
    conditions: dict[str, float] | None,
    heat_treatment: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if not composition:
        raise ValueError("현재 합금 조성이 필요합니다.")
    if not conditions:
        raise ValueError("현재 사용 조건이 필요합니다.")
    if not heat_treatment:
        raise ValueError("현재 열처리 조건이 필요합니다.")

    return composition, conditions, heat_treatment


# ============================================================
# 복합 질문용 context
# ============================================================



def _changed_values(before: dict[str, float], after: dict[str, float]) -> dict[str, dict[str, float]]:
    changed: dict[str, dict[str, float]] = {}

    for key in before.keys() | after.keys():
        old = before.get(key)
        new = after.get(key)

        if old is None or new is None:
            continue

        old_float = float(old)
        new_float = float(new)

        if not math.isclose(old_float, new_float, rel_tol=1e-12, abs_tol=1e-12):
            changed[key] = {
                "before": old_float,
                "after": new_float,
            }

    return changed



def _build_prediction_context(prediction: dict[str, float]) -> str:
    payload = {
        "type": "project_prediction",
        "predicted_life_hours": prediction["life_hours"],
        "predicted_log10_life": prediction["log_life"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)



def _build_what_if_context(what_if_result: dict[str, Any]) -> str:
    before = what_if_result["before"]
    after = what_if_result["after"]
    comparison = what_if_result["comparison"]

    payload = {
        "type": "project_what_if_prediction",
        "baseline_predicted_life_hours": before["prediction"]["life_hours"],
        "changed_predicted_life_hours": after["prediction"]["life_hours"],
        "life_change_hours": comparison["life_change_hours"],
        "life_change_percent": comparison["life_change_percent"],
        "changed_composition": _changed_values(
            before["composition"],
            after["composition"],
        ),
        "changed_conditions": _changed_values(
            before["conditions"],
            after["conditions"],
        ),
        "changed_heat_treatment": _changed_values(
            before["heat_treatment"],
            after["heat_treatment"],
        ),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


# ============================================================
# Warm-up
# ============================================================



def warmup_pipeline(
    *,
    load_predictor: bool = True,
    load_local_llm: bool = True,
) -> dict[str, float]:
    """
    서버/발표 시작 전에 무거운 초기 로딩을 끝낸다.

    같은 프로세스에서는 이후 질문에서 E5 Router, router classifier,
    Chroma 연결, creep predictor를 다시 초기화하지 않는다.
    Local LLM도 Ollama 메모리에 미리 올려 첫 Knowledge 응답의 load 지연을 줄인다.
    """

    started = time.perf_counter()

    t0 = time.perf_counter()
    get_embedding_model()
    _get_router_classifier()
    _get_general_chat_classifier()
    router_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    load_index()
    rag_seconds = time.perf_counter() - t0

    predictor_seconds = 0.0
    if load_predictor:
        t0 = time.perf_counter()
        _get_life_predictor()
        predictor_seconds = time.perf_counter() - t0

    local_llm_seconds = 0.0
    if load_local_llm:
        local_llm_seconds = warmup_local_llm()

    return {
        "router_seconds": router_seconds,
        "rag_index_seconds": rag_seconds,
        "predictor_seconds": predictor_seconds,
        "local_llm_seconds": local_llm_seconds,
        "total_seconds": time.perf_counter() - started,
    }


# ============================================================
# Final integrated entry point
# ============================================================



def run_assistant_pipeline(
    *,
    question: str,
    current_composition: dict[str, float] | None = None,
    current_conditions: dict[str, float] | None = None,
    current_heat_treatment: dict[str, float] | None = None,
    knowledge_top_k: int = 5,
) -> dict[str, Any]:
    """UI가 호출할 최종 진입점."""

    total_started = time.perf_counter()

    question = " ".join(question.strip().split())
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    route_started = time.perf_counter()
    routing = classify_question(question)
    routing_seconds = time.perf_counter() - route_started

    plan = routing["execution_plan"]
    intents = routing["intents"]
    results: dict[str, Any] = {}

    if "prediction" in plan or "what_if" in plan:
        current_composition, current_conditions, current_heat_treatment = _require_current_state(
            current_composition,
            current_conditions,
            current_heat_treatment,
        )

    execution_started = time.perf_counter()

    # --------------------------------------------------------
    # 일반 대화
    # --------------------------------------------------------
    if plan == ["general_chat"]:
        results["general_chat"] = generate_general_chat(question)

        execution_seconds = time.perf_counter() - execution_started

        return {
            "question": question,
            "routing": routing,
            "results": results,
            "timings": {
                "routing_seconds": routing_seconds,
                "execution_seconds": execution_seconds,
                "total_seconds": time.perf_counter() - total_started,
            },
        }

    # --------------------------------------------------------
    # 복합 질문: What-if와 RAG 검색을 동시에 시작한다.
    # Gemini API 대기 시간 동안 문헌 검색을 끝내기 위한 구조다.
    # --------------------------------------------------------
    if "what_if" in plan and "knowledge" in plan:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vertex") as executor:
            what_if_future = executor.submit(
                run_what_if_pipeline,
                question=question,
                current_composition=current_composition,
                current_conditions=current_conditions,
                current_heat_treatment=current_heat_treatment,
            )
            rag_future = executor.submit(
                retrieve,
                question,
                knowledge_top_k,
            )

            what_if_result = what_if_future.result()
            rag_results = rag_future.result()

        results["what_if"] = what_if_result

        if "prediction" in intents:
            results["prediction"] = what_if_result["before"]["prediction"]

        results["knowledge"] = answer_with_rag(
            question,
            top_k=knowledge_top_k,
            retrieved_results=rag_results,
            project_context=_build_what_if_context(what_if_result),
        )

    # --------------------------------------------------------
    # Prediction + Knowledge도 계산과 문헌 검색을 병렬화한다.
    # --------------------------------------------------------
    elif "prediction" in plan and "knowledge" in plan:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vertex") as executor:
            prediction_future = executor.submit(
                _predict_creep_life,
                current_composition,
                current_conditions,
                current_heat_treatment,
            )
            rag_future = executor.submit(
                retrieve,
                question,
                knowledge_top_k,
            )

            prediction_result = prediction_future.result()
            rag_results = rag_future.result()

        results["prediction"] = prediction_result
        results["knowledge"] = answer_with_rag(
            question,
            top_k=knowledge_top_k,
            retrieved_results=rag_results,
            project_context=_build_prediction_context(prediction_result),
        )

    # --------------------------------------------------------
    # 단일/나머지 실행
    # --------------------------------------------------------
    else:
        if "what_if" in plan:
            what_if_result = run_what_if_pipeline(
                question=question,
                current_composition=current_composition,
                current_conditions=current_conditions,
                current_heat_treatment=current_heat_treatment,
            )
            results["what_if"] = what_if_result

            if "prediction" in intents:
                results["prediction"] = what_if_result["before"]["prediction"]

        elif "prediction" in plan:
            results["prediction"] = _predict_creep_life(
                current_composition,
                current_conditions,
                current_heat_treatment,
            )

        if "knowledge" in plan:
            results["knowledge"] = answer_with_rag(
                question,
                top_k=knowledge_top_k,
            )

    execution_seconds = time.perf_counter() - execution_started

    return {
        "question": question,
        "routing": routing,
        "results": results,
        "timings": {
            "routing_seconds": routing_seconds,
            "execution_seconds": execution_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
    }


# ============================================================
# UI 연결 전 임시 통합 테스트
# ============================================================



def _sample_state():
    composition = {
        "C": 0.10,
        "Si": 0.25,
        "Mn": 0.45,
        "P": 0.005,
        "S": 0.002,
        "Cr": 9.20,
        "Mo": 0.50,
        "W": 1.80,
        "Ni": 0.10,
        "Cu": 0.0,
        "V": 0.21,
        "Nb": 0.060,
        "N": 0.045,
        "Al": 0.0,
        "B": 0.005,
        "Co": 0.0,
        "Ta": 0.0,
        "O": 0.001,
        "Re": 0.0,
    }

    conditions = {
        "stress": 100.0,
        "temp": 923.15,
    }

    heat_treatment = {
        "Ntemp": 1323.15,
        "Ntime": 1.0,
        "Ttemp": 1023.15,
        "Ttime": 2.0,
        "Atemp": 0.0,
        "Atime": 0.0,
    }

    return composition, conditions, heat_treatment



def _print_test_result(result: dict[str, Any]) -> None:
    routing = result["routing"]
    outputs = result["results"]
    timings = result.get("timings", {})

    print("\n[의도]", routing["intents"])
    print("[라우터]", routing["router_source"])
    print("[실행]", routing["execution_plan"])
    print("[의도 점수]", {key: round(value, 4) for key, value in routing["intent_scores"].items()})

    if "prediction" in outputs:
        prediction = outputs["prediction"]
        print(
            f"\n[Prediction] {prediction['life_hours']:.4f} h "
            f"(log10={prediction['log_life']:.6f})"
        )

    if "what_if" in outputs:
        what_if_result = outputs["what_if"]
        before_hours = what_if_result["before"]["prediction"]["life_hours"]
        after_hours = what_if_result["after"]["prediction"]["life_hours"]
        comparison = what_if_result["comparison"]

        print(f"\n[What-if] Before={before_hours:.4f} h / After={after_hours:.4f} h")
        print(
            f"Change={comparison['life_change_hours']:.4f} h / "
            f"{comparison['life_change_percent']:.4f}%"
        )

    if "knowledge" in outputs:
        knowledge = outputs["knowledge"]
        print("\n[Knowledge]")
        print(
            knowledge["answer"]
            if isinstance(knowledge, dict) and "answer" in knowledge
            else knowledge
        )

    if "general_chat" in outputs:
        print("\n[General Chat]")
        print(outputs["general_chat"])

    if timings:
        print(
            "\n[Timing] "
            f"router={timings['routing_seconds']:.3f}s / "
            f"execution={timings['execution_seconds']:.3f}s / "
            f"total={timings['total_seconds']:.3f}s"
        )


if __name__ == "__main__":
    composition, conditions, heat_treatment = _sample_state()

    print("=" * 60)
    print("Vertex 통합 Pipeline 테스트")
    print("UI 연결 전이라 샘플 조성/조건을 사용합니다.")
    print("초기 모델을 미리 로드합니다.")
    print("종료: exit")
    print("=" * 60)

    try:
        warmup = warmup_pipeline(
            load_predictor=True,
            load_local_llm=True,
        )
        print(
            "[초기화 완료] "
            f"router={warmup['router_seconds']:.2f}s / "
            f"RAG={warmup['rag_index_seconds']:.2f}s / "
            f"predictor={warmup['predictor_seconds']:.2f}s / "
            f"local_llm={warmup['local_llm_seconds']:.2f}s / "
            f"total={warmup['total_seconds']:.2f}s"
        )
    except Exception as exc:
        print(f"[초기화 경고] {type(exc).__name__}: {exc}")

    while True:
        question = input("\n질문을 입력하세요: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        try:
            result = run_assistant_pipeline(
                question=question,
                current_composition=composition,
                current_conditions=conditions,
                current_heat_treatment=heat_treatment,
            )
            _print_test_result(result)

        except Exception as exc:
            print(f"\n[오류] {type(exc).__name__}: {exc}")