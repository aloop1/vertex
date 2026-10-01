from __future__ import annotations

import re
from typing import Any

from analysis.rag.retriever import retrieve
from .model import generate_response


# ============================================================
# 표시용 문헌 metadata
# ============================================================

SOURCE_METADATA: dict[
    str,
    dict[str, str],
] = {
    "00_Creep-Resistant_Steels.pdf": {
        "title":
            "Creep-resistant steels",
        "author":
            "Fujio Abe, Torsten-Ulf Kern & R. Viswanathan (eds.)",
        "year": "2008",
    },
    "01_Abe_2008_Precipitate_design_9Cr.pdf": {
        "title":
            "Precipitate design for creep strengthening of 9% Cr tempered martensitic steel for ultra-supercritical power plants",
        "author":
            "Fujio Abe",
        "year": "2008",
    },
    "02_Dudova_2022_9-12Cr_B_N_review.pdf": {
        "title":
            "9–12% Cr Heat-Resistant Martensitic Steels with Increased Boron and Decreased Nitrogen Contents",
        "author":
            "Nadezhda Dudova",
        "year": "2022",
    },
    "03_Long_term_microstructure_9-12Cr_2013.pdf": {
        "title":
            "Long Term Microstructural Evolution of 9-12%Cr Steel Grades for Steam Power Generation Plants",
        "author":
            "A. Di Gianfrancesco et al.",
        "year": "2013",
    },
    "04_P92_Long_term_service_2024.pdf": {
        "title":
            "Study on the microstructure evolution and effect on mechanical properties of P92 steel during long term service",
        "author":
            "Xiao Jin et al.",
        "year": "2024",
    },
    "05_M23C6_coarsening_2016.pdf": {
        "title":
            "Coarsening behaviour of M23C6 carbides in creep-resistant steel exposed to high temperatures",
        "author":
            "M. Godec & D. A. Skobir Balantič",
        "year": "2016",
    },
    "06_W_optimization_9Cr3W3Co_2018.pdf": {
        "title":
            "Study on the Microstructure Evolution and Tungsten Content Optimization of 9Cr-3W-3Co Steel",
        "author":
            "Longteng Ma, Yanfeng Wang & Guobiao Di",
        "year": "2018",
    },
}


def _metadata(
    result: dict,
) -> dict[str, str]:
    source = str(
        result.get(
            "source",
            "",
        )
    )

    known = (
        SOURCE_METADATA
        .get(source)
    )

    if known is not None:
        return known

    title = (
        str(
            result.get(
                "title",
                "",
            )
        ).strip()
        or source
        or "자료"
    )

    return {
        "title": title,
        "author":
            "저자 정보 미등록",
        "year": "",
    }


def _page_text(
    result: dict,
) -> str:
    start = result.get(
        "page_start",
        result.get("page"),
    )

    end = result.get(
        "page_end",
        start,
    )

    if start in (
        None,
        0,
        "0",
        "",
    ):
        return ""

    if start == end:
        return f"p.{start}"

    return (
        f"p.{start}-{end}"
    )


def _citation(
    result: dict,
) -> str:
    meta = _metadata(result)

    parts = [
        meta["title"],
        meta["author"],
    ]

    if meta["year"]:
        parts.append(
            meta["year"]
        )

    page = _page_text(
        result
    )

    if page:
        parts.append(page)

    return (
        "["
        + " — ".join(
            parts[:2]
        )
        + (
            ", "
            + ", ".join(
                parts[2:]
            )
            if len(parts) > 2
            else ""
        )
        + "]"
    )


# ============================================================
# Prompt
# ============================================================


def _build_project_context_block(
    project_context: str | None,
) -> str:
    if not project_context:
        return ""

    return f"""
PROJECT MODEL CONTEXT
아래 값은 이 프로젝트의 예측 모델 또는 What-if 계산 결과입니다.
문헌의 실험값이 아닙니다.

{project_context}
""".strip()


def _build_prompt(
    question: str,
    project_context: str | None,
) -> str:
    project_block = (
        _build_project_context_block(
            project_context
        )
    )

    return f"""
USER QUESTION
{question}

{project_block}

Answer the user's question in Korean using only the provided literature evidence.

RULES

1. Give the most useful evidence-supported answer first.
   Start directly with the conclusion supported by the evidence.

2. Consider ALL provided evidence items that are relevant.
   Do not rely only on EVIDENCE 1.
   Different evidence items may describe different alloys,
   test durations, stresses, temperatures, or creep regimes.

3. Preserve technical distinctions exactly.
   Do not treat creep strength, creep rupture strength,
   creep rupture time, creep life, creep rate,
   tensile strength, or other properties as interchangeable.

4. Preserve experimental conditions and comparisons.
   Do not change temperatures, stresses, compositions,
   time ranges, numerical values, comparison groups,
   or increase/decrease directions.

5. If different evidence describes short-term and long-term
   behavior differently, explain that distinction clearly.

6. Do not generalize beyond the evidence.
   A result for one alloy or condition must not be presented
   as a universal rule for all 9Cr steels or all heat-resistant steels.

7. Do not invent mechanisms, causes, relationships,
   numerical values, or conclusions that are not supported
   by the provided evidence.

8. Stay focused on the user's question.
   Omit unrelated alloying elements, composition values,
   background information, and experimental details
   unless they are necessary to answer the question.

9. Do NOT add generic closing disclaimers such as:
   "직접적인 근거는 없습니다",
   "직접적인 정보는 제공되지 않습니다",
   "현재 자료만으로는 단정하기 어렵습니다",
   "추가 자료가 필요합니다"
   when the provided evidence already supports
   a meaningful answer to the user's question.

10. If the evidence reports a closely related property
    rather than the exact property named by the user,
    state the property that was actually measured and explain
    the result using that terminology.
    Do not turn this into an unnecessary absence-of-evidence disclaimer.

11. Only say that the literature cannot answer the question
    when NONE of the provided evidence contains meaningful
    information relevant to the question.

12. When exceptions or different behavior among alloys exist,
    mention them briefly if they materially affect the conclusion.

13. If PROJECT MODEL CONTEXT is present,
    clearly distinguish project-model predictions
    from experimental literature results.
    Never describe a model prediction as a measured experimental value.

14. Do not reproduce internal labels such as
    EVIDENCE 1, EVIDENCE 2, etc. in the final answer.

15. Write a concise, professional Korean answer.
    Prefer 2-3 short paragraphs.
    Avoid unnecessary repetition.
    Keep the answer within 900 Korean characters.
""".strip()


# ============================================================
# 후처리
# ============================================================


def _normalize_answer(
    answer: str,
) -> str:
    text = answer.strip()

    text = re.sub(
        r"\[(?:S|자료|근거)\s*\d+\]",
        "",
        text,
    )

    text = re.sub(
        r"(?<=\d)\.\s+(?=\d)",
        ".",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _last_sentence_boundary(
    text: str,
    limit: int,
) -> int:
    last = -1

    for index, char in enumerate(
        text[:limit]
    ):
        if char not in ".!?。":
            continue

        if (
            char == "."
            and index > 0
            and index + 1
            < len(text)
            and text[
                index - 1
            ].isdigit()
            and text[
                index + 1
            ].isdigit()
        ):
            continue

        last = index + 1

    return last


def _validate_answer(
    answer: str,
    *,
    max_chars: int = 1100,
) -> tuple[
    str | None,
    dict[str, Any],
]:
    text = _normalize_answer(
        answer
    )

    if not text:
        return None, {
            "reason": "empty",
        }

    if (
        text[-1]
        not in ".!?。"
    ):
        boundary = (
            _last_sentence_boundary(
                text,
                len(text),
            )
        )

        if boundary > 0:
            text = (
                text[:boundary]
                .strip()
            )
        else:
            return None, {
                "reason":
                    "no_complete_sentence",
            }

    if len(text) > max_chars:
        boundary = (
            _last_sentence_boundary(
                text,
                max_chars,
            )
        )

        if boundary <= 0:
            return None, {
                "reason":
                    "no_sentence_within_limit",
            }

        text = (
            text[:boundary]
            .strip()
        )

    return text, {
        "reason": None,
        "answer_chars":
            len(text),
    }


# ============================================================
# 검색 결과 → AI 근거 문맥
# ============================================================


def _result_evidence_text(result: dict) -> str:
    """sentence-level 검색 결과에서 AI에 줄 주변 문맥을 선택한다."""
    return str(
        result.get("context_text")
        or result.get("context")
        or result.get("text")
        or ""
    ).strip()


def _build_evidence_context(
    results: list[dict],
    *,
    max_total_chars: int = 2800,
) -> tuple[str, list[dict]]:
    """
    검색된 Top-k를 추가 임베딩/재랭킹 없이 그대로 사용한다.

    검색 점수 계산은 sentence-level 결과를 그대로 유지하고, AI에는
    각 결과의 사전 저장 주변 문맥(context_text)을 제공한다. 전체 prompt가
    지나치게 커지지 않도록 문자 수만 제한한다.
    """
    parts: list[str] = []
    used: list[dict] = []
    total_chars = 0

    for rank, result in enumerate(results, start=1):
        evidence_text = _result_evidence_text(result)
        if not evidence_text:
            continue

        page = _page_text(result)
        header = f"[Evidence {rank}] {result.get('source', '')}"
        if page:
            header += f" / {page}"

        block = f"{header}\n{evidence_text}"

        if parts and total_chars + len(block) > max_total_chars:
            continue

        # 첫 근거 하나는 길더라도 버리지 않는다.
        if not parts and len(block) > max_total_chars:
            block = block[:max_total_chars].rstrip()

        parts.append(block)
        used_item = dict(result)
        used_item["llm_context"] = evidence_text
        used.append(used_item)
        total_chars += len(block) + 2

    return "\n\n".join(parts), used


def _result_search_text(
    result: dict,
) -> str:
    return str(
        result.get("search_text")
        or result.get("text")
        or ""
    ).strip()


def _result_context_text(
    result: dict,
) -> str:
    return str(
        result.get("context_text")
        or result.get("context")
        or result.get("text")
        or ""
    ).strip()


def _build_llm_evidence(
    retrieved: list[dict],
    *,
    full_context_count: int = 3,
    max_results: int = 5,
    max_total_chars: int = 2000,
) -> tuple[str, list[dict]]:

    parts: list[str] = []
    used_results: list[dict] = []
    used_chars = 0

    selected = retrieved[:max_results]

    for index, result in enumerate(
        selected,
        start=1,
    ):
        if index <= full_context_count:
            evidence_text = (
                _result_context_text(
                    result
                )
            )
        else:
            evidence_text = (
                _result_search_text(
                    result
                )
            )

        if not evidence_text:
            continue

        page_start = result.get(
            "page_start",
            result.get("page"),
        )

        page_end = result.get(
            "page_end",
            page_start,
        )

        if page_start == page_end:
            page_text = str(
                page_start
            )
        else:
            page_text = (
                f"{page_start}-{page_end}"
            )

        header = (
            f"EVIDENCE {index}\n"
            f"Source: "
            f"{result.get('source', '')}\n"
            f"Page: {page_text}\n"
        )

        block = (
            header
            + evidence_text
        ).strip()

        remaining = (
            max_total_chars
            - used_chars
        )

        if remaining <= 0:
            break

        if len(block) > remaining:
            # 중요한 검색 문장 자체는 가능한 한 남긴다.
            search_text = (
                _result_search_text(
                    result
                )
            )

            short_block = (
                header
                + search_text
            ).strip()

            if (
                search_text
                and len(short_block)
                <= remaining
            ):
                block = short_block
            else:
                continue

        parts.append(block)
        used_results.append(
            result
        )

        used_chars += (
            len(block) + 2
        )

    return (
        "\n\n".join(parts),
        used_results,
    )


# ============================================================
# Knowledge
# ============================================================


def answer_with_rag(
    question: str,
    top_k: int = 5,
    *,
    retrieved_results:
        list[dict] | None = None,
    project_context:
        str | None = None,
) -> dict:
    question = question.strip()

    if not question:
        raise ValueError(
            "질문이 비어 있습니다."
        )

    if retrieved_results is None:
        print()
        print(
            "[1] 관련 문헌 검색 중..."
        )

        retrieved = retrieve(
            question,
            top_k=top_k,
        )
    else:
        retrieved = (
            retrieved_results
        )

    if not retrieved:
        return {
            "answer":
                "현재 RAG 자료에서 관련 근거를 찾지 못했습니다.",
            "sources": [],
            "retrieved_sources": [],
            "validation": {
                "llm_used": False,
                "reason":
                    "no_retrieval_result",
            },
        }

    evidence_text, used_evidence = (
        _build_llm_evidence(
            retrieved,
            full_context_count=3,
            max_results=5,
            max_total_chars=2000,
        )
    )

    if not evidence_text:
        return {
            "answer":
                "현재 RAG 자료에서 관련 근거를 찾지 못했습니다.",
            "sources": [],
            "retrieved_sources":
                retrieved,
            "validation": {
                "llm_used": False,
                "reason":
                    "no_evidence",
            },
        }

    print(
        "[RAG 근거] "
        f"검색 {len(retrieved)}개 "
        f"→ "
        f"본문 {len(evidence_text)}자"
    )

    print()
    print(
        "[2] 답변 생성 중..."
    )
    print()

    raw_answer = (
        generate_response(
            question=_build_prompt(
                question,
                project_context,
            ),
            context=evidence_text,
        )
    )

    validated, validation = (
        _validate_answer(
            raw_answer,
            max_chars=1100,
        )
    )

    if validated is None:
        answer = (
            "문헌 근거는 검색됐지만 "
            "완결된 답변을 생성하지 못했습니다."
        )
    else:
        answer = validated

    source_items: list[dict] = []

    for index, evidence in enumerate(
        used_evidence,
        start=1,
    ):
        meta = _metadata(
            evidence
        )

        source_item = dict(
            evidence
        )

        if index <= 3:
            quote = (
                _result_context_text(
                    evidence
                )
            )
        else:
            quote = (
                _result_search_text(
                    evidence
                )
            )

        source_item.update(
            {
                "citation":
                    _citation(
                        evidence
                    ),
                "title":
                    meta["title"],
                "author":
                    meta["author"],
                "year":
                    meta["year"],
                "quote":
                    quote,
            }
        )

        source_items.append(
            source_item
        )

    validation.update(
        {
            "llm_used": True,
            "retrieval_score":
                used_evidence[0].get(
                    "score",
                    0,
                )
                if used_evidence
                else 0,
            "evidence_count":
                len(used_evidence),
            "evidence_chars":
                len(evidence_text),
        }
    )

    return {
        "answer": answer,
        "sources":
            source_items,
        "retrieved_sources":
            retrieved,
        "validation":
            validation,
    }


def print_answer(
    result: dict,
) -> None:
    print(
        "[3] 답변:"
    )
    print(
        result["answer"]
    )

    if result.get(
        "sources"
    ):
        print()
        print(
            "[4] 사용한 문헌 원문"
        )

        for source in (
            result["sources"]
        ):
            print(
                source[
                    "citation"
                ]
            )
            print(
                source["quote"]
            )
            print()


if __name__ == "__main__":
    question = input(
        "질문을 입력하세요: "
    ).strip()

    result = answer_with_rag(
        question,
        top_k=5,
    )

    print_answer(result)