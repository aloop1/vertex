from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import (
    DATA_DIR,
    OOD_CONFIG,
)

from .physics import as_full_composition


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None

    if np.isnan(value) or np.isinf(value):
        return None

    return value


@lru_cache(maxsize=1)
def load_ood_reference() -> dict:
    path = Path(
        OOD_CONFIG.get(
            "REFERENCE_PATH",
            str(DATA_DIR / "ood_reference.pkl"),
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"OOD reference file not found: {path}"
        )

    payload = joblib.load(path)

    if not isinstance(payload, dict):
        raise TypeError(
            f"OOD reference must be dict, got {type(payload).__name__}"
        )

    return payload


def _candidate_vector(
    comp_dict: dict,
    features: list[str],
) -> np.ndarray:
    full_comp = as_full_composition(comp_dict)

    return np.array(
        [
            float(full_comp.get(elem, 0.0))
            for elem in features
        ],
        dtype=float,
    )


def _mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    diff = x - mean
    distance_sq = float(diff.T @ cov_inv @ diff)
    distance_sq = max(0.0, distance_sq)
    return float(np.sqrt(distance_sq))


def _percentile_from_train_distances(
    distance: float,
    train_distances: np.ndarray,
) -> float | None:
    if train_distances.size == 0:
        return None

    return float(
        100.0
        * np.mean(train_distances <= distance)
    )


def _feature_contributions(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
    features: list[str],
) -> list[dict]:
    """
    Mahalanobis distance를 키우는 원소를 설명합니다.

    raw_contribution:
    - diff_i * (cov_inv @ diff)_i
    - 합은 distance^2와 같습니다.
    - 공분산 구조 때문에 음수가 나올 수 있습니다.

    abs_share:
    - 발표/해석용으로 어떤 원소가 크게 작용했는지 보는 값입니다.
    """

    diff = x - mean
    projected = cov_inv @ diff
    raw = diff * projected

    abs_raw = np.abs(raw)
    abs_sum = float(abs_raw.sum())

    rows = []

    for i, elem in enumerate(features):
        rows.append(
            {
                "element": elem,
                "candidate_value": _safe_float(x[i]),
                "reference_mean": _safe_float(mean[i]),
                "difference": _safe_float(diff[i]),
                "raw_contribution": _safe_float(raw[i]),
                "abs_share": (
                    _safe_float(abs_raw[i] / abs_sum)
                    if abs_sum > 1e-12
                    else 0.0
                ),
            }
        )

    rows.sort(
        key=lambda r: abs(float(r["raw_contribution"] or 0.0)),
        reverse=True,
    )

    return rows


def _nearest_neighbors(
    x: np.ndarray,
    payload: dict,
    features: list[str],
    top_k: int,
) -> list[dict]:
    reference_rows = payload.get("reference_rows", [])

    if not reference_rows:
        return []

    std = np.array(
        payload.get("std", np.ones(len(features))),
        dtype=float,
    )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    )

    ref_x = np.array(
        [
            [
                float(row.get(elem, 0.0) or 0.0)
                for elem in features
            ]
            for row in reference_rows
        ],
        dtype=float,
    )

    diff = (ref_x - x.reshape(1, -1)) / std.reshape(1, -1)

    distances = np.sqrt(
        np.sum(
            diff * diff,
            axis=1,
        )
    )

    order = np.argsort(distances)[:top_k]

    neighbors = []

    for rank, idx in enumerate(order, start=1):
        source = reference_rows[int(idx)]

        composition = {
            elem: _safe_float(source.get(elem, 0.0))
            for elem in features
        }

        delta = {
            elem: _safe_float(
                float(x[j])
                - float(source.get(elem, 0.0) or 0.0)
            )
            for j, elem in enumerate(features)
        }

        neighbors.append(
            {
                "rank": int(rank),
                "row_index": source.get("row_index", int(idx)),
                "standardized_distance": _safe_float(distances[idx]),
                "temp": _safe_float(source.get("temp")),
                "stress": _safe_float(source.get("stress")),
                "log_life": _safe_float(source.get("log_life")),
                "composition": composition,
                "delta_from_candidate": delta,
            }
        )

    return neighbors


def analyze_ood_candidate(
    comp_dict: dict,
    top_k: int = 5,
) -> dict:
    """
    후보 조성 1개에 대한 OOD 해석 결과를 반환합니다.

    반환:
    - ood_distance
    - ood_percentile
    - threshold
    - is_out_of_distribution
    - feature_contributions
    - nearest_neighbors
    """

    payload = load_ood_reference()

    features = list(
        payload.get(
            "features",
            payload.get(
                "reference_features",
                OOD_CONFIG.get("REFERENCE_FEATURES", []),
            ),
        )
    )

    if not features:
        raise ValueError(
            "OOD reference에 features가 없습니다."
        )

    mean = np.array(
        payload.get(
            "mean",
            payload.get("ood_reference_mean"),
        ),
        dtype=float,
    )

    cov_inv = np.array(
        payload.get(
            "cov_inv",
            payload.get("ood_reference_cov_inv"),
        ),
        dtype=float,
    )

    threshold = float(
        payload.get(
            "threshold",
            payload.get(
                "ood_distance_threshold",
                OOD_CONFIG.get("OOD_DISTANCE_THRESHOLD", 3.0),
            ),
        )
    )

    train_distances = np.array(
        payload.get(
            "train_distances",
            [],
        ),
        dtype=float,
    )

    x = _candidate_vector(
        comp_dict,
        features,
    )

    if mean.shape[0] != x.shape[0]:
        raise ValueError(
            f"OOD mean dimension mismatch: mean={mean.shape[0]}, x={x.shape[0]}"
        )

    if cov_inv.shape != (x.shape[0], x.shape[0]):
        raise ValueError(
            f"OOD cov_inv dimension mismatch: cov_inv={cov_inv.shape}, x={x.shape[0]}"
        )

    distance = _mahalanobis_distance(
        x=x,
        mean=mean,
        cov_inv=cov_inv,
    )

    percentile = _percentile_from_train_distances(
        distance=distance,
        train_distances=train_distances,
    )

    contributions = _feature_contributions(
        x=x,
        mean=mean,
        cov_inv=cov_inv,
        features=features,
    )

    neighbors = _nearest_neighbors(
        x=x,
        payload=payload,
        features=features,
        top_k=top_k,
    )

    return {
        "ood_distance": _safe_float(distance),
        "ood_distance_threshold": _safe_float(threshold),
        "ood_percentile": _safe_float(percentile),
        "ood_is_out_of_distribution": bool(distance > threshold),
        "ood_reference_features": features,
        "ood_top_contributors": contributions[:5],
        "nearest_neighbors": neighbors,
    }