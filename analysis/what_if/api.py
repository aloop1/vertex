""" AI API 호출 """

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .cache import (
    get_cached_result,
    make_cache_key,
    save_cached_result,
)


# ============================================================
# [1] 환경 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)


def _env_flag(
    name: str,
    default: bool = False,
) -> bool:

    raw = os.environ.get(
        name,
        str(default),
    ).strip().lower()

    return raw in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_int(
    name: str,
    default: int,
) -> int:

    try:
        return int(
            os.environ.get(
                name,
                default,
            )
        )

    except (TypeError, ValueError):
        return default


def _env_float(
    name: str,
    default: float,
) -> float:

    try:
        return float(
            os.environ.get(
                name,
                default,
            )
        )

    except (TypeError, ValueError):
        return default


GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()

GEMINI_TEMPERATURE = _env_float(
    "GEMINI_TEMPERATURE",
    0.3,
)

GEMINI_TIMEOUT = _env_int(
    "GEMINI_TIMEOUT",
    60,
)

GEMINI_MAX_RETRIES = _env_int(
    "GEMINI_MAX_RETRIES",
    3,
)

GEMINI_MAX_OUTPUT_TOKENS = _env_int(
    "GEMINI_MAX_OUTPUT_TOKENS",
    4096,
)

WHAT_IF_USE_CACHE = _env_flag(
    "WHAT_IF_USE_CACHE",
    True,
)


# ============================================================
# [2] Gemini API 설정
# ============================================================

def _configure_gemini_api():
    """Gemini API key 설정."""

    api_key = os.environ.get(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다.\n"
            f".env 경로: {ENV_PATH}"
        )

    import google.generativeai as genai

    genai.configure(
        api_key=api_key
    )

    return genai


# ============================================================
# [3] JSON Parser
# ============================================================

def _parse_json_response(
    text: str,
) -> dict[str, Any]:
    """Gemini 응답을 JSON object로 변환."""

    if not text:
        raise RuntimeError(
            "Gemini 응답이 비어 있습니다."
        )

    cleaned = text.strip()

    # ```json ... ``` 제거
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError:

        # 응답에 설명이 섞인 경우
        # 첫 JSON object를 찾아 재시도
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:
            raise RuntimeError(
                "Gemini 응답에서 JSON을 찾지 못했습니다."
            )

        try:
            result = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Gemini 응답 JSON 파싱 실패"
            ) from e

    if not isinstance(result, dict):
        raise RuntimeError(
            "Gemini 응답은 JSON object여야 합니다."
        )

    return result


# ============================================================
# [4] Gemini 호출
# ============================================================

def generate_json(
    *,
    system_instruction: str,
    prompt: str,
) -> dict[str, Any]:
    

    # --------------------------------------------------------
    # Cache key 생성
    # --------------------------------------------------------

    cache_key = make_cache_key(
        model_name=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        system_instruction=system_instruction,
        prompt=prompt,
    )

    # --------------------------------------------------------
    # Cache 사용
    # --------------------------------------------------------

    if WHAT_IF_USE_CACHE:

        cached_result = get_cached_result(
            cache_key
        )

        if cached_result is not None:

            print(
                "[What-if Cache] 저장된 결과 사용"
            )

            return cached_result

    # --------------------------------------------------------
    # Gemini 실제 호출
    # --------------------------------------------------------

    print(
        f"[What-if API] Gemini 호출: {GEMINI_MODEL}"
    )

    genai = _configure_gemini_api()

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        GEMINI_MAX_RETRIES + 1,
    ):

        try:

            print(
                f"[What-if API] "
                f"요청 {attempt}/{GEMINI_MAX_RETRIES}"
            )

            response = model.generate_content(
                prompt,
                generation_config=(
                    genai.types.GenerationConfig(
                        temperature=GEMINI_TEMPERATURE,
                        max_output_tokens=(
                            GEMINI_MAX_OUTPUT_TOKENS
                        ),
                    )
                ),
                request_options={
                    "timeout": GEMINI_TIMEOUT
                },
            )

            result = _parse_json_response(
                response.text
            )

            # -----------------------------------------------
            # API 성공 결과 캐시 저장
            # -----------------------------------------------

            if WHAT_IF_USE_CACHE:

                save_cached_result(
                    cache_key,
                    result,
                )

            return result

        except Exception as e:

            last_error = e

            print(
                f"[What-if API] 호출 실패: {e}"
            )

            if attempt < GEMINI_MAX_RETRIES:

                wait_seconds = min(
                    attempt * 2,
                    5,
                )

                print(
                    f"[What-if API] "
                    f"{wait_seconds}초 후 재시도"
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "Gemini API 호출에 실패했습니다. "
        f"마지막 오류: {last_error}"
    ) from last_error