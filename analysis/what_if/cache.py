""" AI API 호출 끈 상태에서 사용하는 캐시 """

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "assistant"
    / "what_if_data"
    / "what_if_cache.json"
)


def make_cache_key(
    *,
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    system_instruction: str,
    prompt: str,
) -> str:
    """
    Gemini 요청 내용이 완전히 같을 때
    동일한 cache key를 생성한다.
    """

    payload = {
        "model_name": model_name,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_instruction,
        "prompt": prompt,
    }

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _load_cache() -> dict[str, Any]:
    """캐시 파일 전체를 읽는다."""

    if not CACHE_PATH.exists():
        return {}

    try:
        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as e:
        print(f"[What-if Cache] 읽기 실패: {e}")

    return {}


def get_cached_result(
    cache_key: str,
) -> dict[str, Any] | None:
    """cache key에 해당하는 결과가 있으면 반환."""

    cache = _load_cache()

    result = cache.get(cache_key)

    if isinstance(result, dict):
        return result

    return None


def save_cached_result(
    cache_key: str,
    result: dict[str, Any],
) -> None:
    """Gemini 응답을 cache에 저장."""

    cache = _load_cache()

    cache[cache_key] = result

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = CACHE_PATH.with_suffix(".tmp")

    try:
        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                cache,
                f,
                ensure_ascii=False,
                indent=2,
            )

        temp_path.replace(CACHE_PATH)

        print("[What-if Cache] 저장 완료")

    except Exception as e:
        print(f"[What-if Cache] 저장 실패: {e}")