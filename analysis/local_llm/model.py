from __future__ import annotations

import os
import time
from typing import Any

import ollama


# ============================================================
# 설정
# ============================================================

MODEL_NAME = os.getenv(
    "LOCAL_LLM_MODEL",
    "qwen2.5:7b-instruct",
).strip()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
).strip()

NUM_CTX = int(
    os.getenv(
        "LOCAL_LLM_NUM_CTX",
        "4096",
    )
)

NUM_PREDICT = int(
    os.getenv(
        "LOCAL_LLM_NUM_PREDICT",
        "300",
    )
)

TEMPERATURE = float(
    os.getenv(
        "LOCAL_LLM_TEMPERATURE",
        "0",
    )
)

KEEP_ALIVE = os.getenv(
    "LOCAL_LLM_KEEP_ALIVE",
    "30m",
).strip()

DEBUG_TIMING = (
    os.getenv(
        "LOCAL_LLM_DEBUG_TIMING",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

_client = ollama.Client(
    host=OLLAMA_HOST
)

_last_generation_stats: dict[
    str,
    float | int | str,
] = {}


# ============================================================
# System prompts
# ============================================================

KNOWLEDGE_SYSTEM_PROMPT = (
    "당신은 크립 수명과 내열강 분석을 보조하는 AI입니다. "
    "제공된 문헌 근거 밖의 재료학적 사실을 만들지 마세요. "
    "크립 강도, 크립 파단 시간, 크립 수명, 인장강도 등 "
    "서로 다른 물성을 같은 의미로 바꾸지 마세요. "
    "프로젝트 모델 예측값은 실험값과 구분하세요. "
    "한국어로 명확하게 답하세요."
)

GENERAL_SYSTEM_PROMPT = (
    "당신은 Vertex 크립 수명 예측 시스템의 대화형 안내 AI입니다. "
    "일반적인 인사와 사용 안내에는 자연스럽고 간단하게 답하세요. "
    "Vertex는 크립 수명 예측, 조건/조성 변경 What-if 비교, "
    "관련 문헌 기반 설명을 지원한다고 안내할 수 있습니다. "
    "재료학적 사실을 묻는 질문에는 임의로 답하지 말고 "
    "Knowledge 기능을 이용해 질문하도록 유도하세요."
)


# ============================================================
# Prompt
# ============================================================


def _build_user_prompt(
    question: str,
    context: str | None,
) -> str:
    question = question.strip()

    if not context:
        return question

    return f"""
[문헌 근거]
{context}

[답변 지시]
{question}
""".strip()


# ============================================================
# 응답 메타데이터
# ============================================================


def _ns_to_seconds(
    value: Any,
) -> float:
    try:
        return (
            float(value or 0)
            / 1_000_000_000.0
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _record_stats(
    response: Any,
    wall_seconds: float,
) -> None:
    global _last_generation_stats

    def value(
        name: str,
        default: Any = 0,
    ) -> Any:
        if isinstance(
            response,
            dict,
        ):
            return response.get(
                name,
                default,
            )

        return getattr(
            response,
            name,
            default,
        )

    _last_generation_stats = {
        "model": MODEL_NAME,
        "wall_seconds":
            float(
                wall_seconds
            ),
        "load_seconds":
            _ns_to_seconds(
                value(
                    "load_duration"
                )
            ),
        "prompt_eval_seconds":
            _ns_to_seconds(
                value(
                    "prompt_eval_duration"
                )
            ),
        "generation_seconds":
            _ns_to_seconds(
                value(
                    "eval_duration"
                )
            ),
        "prompt_tokens":
            int(
                value(
                    "prompt_eval_count",
                    0,
                )
                or 0
            ),
        "generated_tokens":
            int(
                value(
                    "eval_count",
                    0,
                )
                or 0
            ),
    }

    generated_tokens = int(
        _last_generation_stats[
            "generated_tokens"
        ]
    )

    generation_seconds = float(
        _last_generation_stats[
            "generation_seconds"
        ]
    )

    if (
        generated_tokens > 0
        and generation_seconds > 0
    ):
        _last_generation_stats[
            "tokens_per_second"
        ] = (
            generated_tokens
            / generation_seconds
        )
    else:
        _last_generation_stats[
            "tokens_per_second"
        ] = 0.0

    if DEBUG_TIMING:
        print(
            "[Local LLM timing] "
            f"total="
            f"{_last_generation_stats['wall_seconds']:.2f}s / "
            f"load="
            f"{_last_generation_stats['load_seconds']:.2f}s / "
            f"prompt="
            f"{_last_generation_stats['prompt_eval_seconds']:.2f}s / "
            f"generate="
            f"{_last_generation_stats['generation_seconds']:.2f}s / "
            f"prompt_tokens="
            f"{_last_generation_stats['prompt_tokens']} / "
            f"generated_tokens="
            f"{_last_generation_stats['generated_tokens']} / "
            f"tok_s="
            f"{_last_generation_stats['tokens_per_second']:.2f}"
        )


def get_last_generation_stats() -> dict[
    str,
    float | int | str,
]:
    return dict(
        _last_generation_stats
    )


# ============================================================
# 공통 생성
# ============================================================


def _chat(
    *,
    system_prompt: str,
    user_prompt: str,
    num_predict: int,
) -> str:
    started = time.perf_counter()

    response = _client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content":
                    system_prompt,
            },
            {
                "role": "user",
                "content":
                    user_prompt,
            },
        ],
        stream=False,
        options={
            "temperature":
                TEMPERATURE,
            "num_ctx":
                NUM_CTX,
            "num_predict":
                num_predict,
        },
        keep_alive=KEEP_ALIVE,
    )

    wall_seconds = (
        time.perf_counter()
        - started
    )

    _record_stats(
        response,
        wall_seconds,
    )

    if isinstance(
        response,
        dict,
    ):
        content = (
            response
            .get(
                "message",
                {},
            )
            .get(
                "content",
                "",
            )
        )
    else:
        message = getattr(
            response,
            "message",
            None,
        )
        content = (
            getattr(
                message,
                "content",
                "",
            )
            if message is not None
            else ""
        )

    answer = str(
        content
    ).strip()

    if not answer:
        raise RuntimeError(
            "Local LLM이 빈 "
            "답변을 반환했습니다."
        )

    return answer


# ============================================================
# Warm-up
# ============================================================


def warmup_local_llm() -> float:
    started = time.perf_counter()

    _client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "준비",
            }
        ],
        stream=False,
        options={
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 1,
        },
        keep_alive=KEEP_ALIVE,
    )

    return (
        time.perf_counter()
        - started
    )


# ============================================================
# Knowledge / General chat
# ============================================================


def generate_response(
    question: str,
    context: str | None = None,
) -> str:
    if (
        not question
        or not question.strip()
    ):
        raise ValueError(
            "질문이 비어 있습니다."
        )

    user_prompt = (
        _build_user_prompt(
            question,
            context,
        )
    )

    return _chat(
        system_prompt=(
            KNOWLEDGE_SYSTEM_PROMPT
        ),
        user_prompt=user_prompt,
        num_predict=NUM_PREDICT,
    )


def generate_general_chat(
    question: str,
) -> str:
    if (
        not question
        or not question.strip()
    ):
        raise ValueError(
            "질문이 비어 있습니다."
        )

    return _chat(
        system_prompt=(
            GENERAL_SYSTEM_PROMPT
        ),
        user_prompt=question.strip(),
        num_predict=min(
            NUM_PREDICT,
            120,
        ),
    )


if __name__ == "__main__":
    print(
        f"모델: {MODEL_NAME}"
    )

    question = input(
        "질문을 입력하세요: "
    ).strip()

    print()
    print(
        generate_general_chat(
            question
        )
    )