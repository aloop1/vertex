from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from ga.config import DESIGN_VARIABLES, OOD_CONFIG
from 데이터전처리 import COMPOSITION_COLS, prepare_dataset
from models.transformer_and_tree_ensemble import group_holdout_split


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None

    if np.isnan(value) or np.isinf(value):
        return None

    return value


def _mahalanobis_batch(
    X: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    diff = X - mean
    distance_sq = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    distance_sq = np.maximum(distance_sq, 0.0)
    return np.sqrt(distance_sq)


def _composition_sum_and_fe_balance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    COMPOSITION_COLS 기준으로 총 합금량과 Fe balance를 계산합니다.
    Fe 컬럼이 있으면 Fe 컬럼을 우선 사용하고,
    없으면 100 - 합금원소총합으로 계산합니다.
    """

    comp_cols = [
        col for col in COMPOSITION_COLS
        if col in df.columns and col.upper() != "FE"
    ]

    total_alloy = df[comp_cols].astype(float).sum(axis=1)

    if "Fe" in df.columns:
        fe_balance = pd.to_numeric(df["Fe"], errors="coerce").fillna(100.0 - total_alloy)
    elif "FE" in df.columns:
        fe_balance = pd.to_numeric(df["FE"], errors="coerce").fillna(100.0 - total_alloy)
    else:
        fe_balance = 100.0 - total_alloy

    return total_alloy, fe_balance


def _filter_ferritic_martensitic_rows(
    X: pd.DataFrame,
    cr_min: float,
    cr_max: float,
    ni_max: float,
    fe_min: float,
    total_alloy_max: float,
) -> pd.Series:
    """
    9-12Cr ferritic/martensitic creep steel 계열만 OOD reference로 사용하기 위한 필터.

    핵심:
    - Ni-base / austenitic alloy 제거
    - Fe-rich 9Cr계 후보만 남김
    """

    required = ["Cr", "Ni"]

    missing = [
        col for col in required
        if col not in X.columns
    ]

    if missing:
        raise KeyError(f"Ferritic filter에 필요한 컬럼이 없습니다: {missing}")

    cr = pd.to_numeric(X["Cr"], errors="coerce").fillna(-9999.0)
    ni = pd.to_numeric(X["Ni"], errors="coerce").fillna(9999.0)

    total_alloy, fe_balance = _composition_sum_and_fe_balance(X)

    mask = (
        (cr >= cr_min)
        & (cr <= cr_max)
        & (ni >= 0.0)
        & (ni <= ni_max)
        & (fe_balance >= fe_min)
        & (total_alloy <= total_alloy_max)
    )

    return mask


def build_ood_reference(
    output_path: Path,
    test_size: float = 0.2,
    seed: int = 42,
    rounding: int = 3,
    quantile: float = 0.95,
    cov_regularization: float = 1e-6,
    ferritic_only: bool = True,
    cr_min: float = 8.0,
    cr_max: float = 12.0,
    ni_max: float = 2.0,
    fe_min: float = 80.0,
    total_alloy_max: float = 20.0,
    min_rows: int = 50,
) -> dict:
    """
    Mahalanobis OOD reference 생성.

    중요:
    - 모델 전체 학습 데이터가 아니라,
      GA가 탐색하는 ferritic/martensitic 영역만 기준으로 OOD reference를 만든다.
    - 이렇게 해야 Ni 평균이 23%처럼 Ni-base 데이터에 끌려가는 문제를 막을 수 있다.
    """

    dataset = prepare_dataset(
        rounding_decimals=rounding,
        use_scaler=False,
    )

    raw_outer = group_holdout_split(
        dataset.X,
        dataset.y,
        dataset.groups,
        test_size=test_size,
        seed=seed,
    )

    X_train = raw_outer.X_train.reset_index(drop=True).copy()
    y_train = raw_outer.y_train.reset_index(drop=True).astype(float)

    n_before_filter = int(len(X_train))

    if ferritic_only:
        mask = _filter_ferritic_martensitic_rows(
            X=X_train,
            cr_min=cr_min,
            cr_max=cr_max,
            ni_max=ni_max,
            fe_min=fe_min,
            total_alloy_max=total_alloy_max,
        )

        X_ref = X_train.loc[mask].reset_index(drop=True).copy()
        y_ref = y_train.loc[mask].reset_index(drop=True).copy()

    else:
        X_ref = X_train.reset_index(drop=True).copy()
        y_ref = y_train.reset_index(drop=True).copy()

    n_after_filter = int(len(X_ref))

    if n_after_filter < min_rows:
        raise RuntimeError(
            f"OOD reference row가 너무 적습니다: {n_after_filter} rows. "
            f"필터를 완화하십시오. 예: --ni-max 3 또는 --cr-min 7.5 --cr-max 12.5"
        )

    features = list(DESIGN_VARIABLES)

    missing = [
        col for col in features
        if col not in X_ref.columns
    ]

    if missing:
        raise KeyError(
            f"OOD reference 생성에 필요한 DESIGN_VARIABLES 컬럼이 없습니다: {missing}"
        )

    X_comp_df = X_ref[features].astype(float).copy()
    X_comp = X_comp_df.to_numpy(dtype=float)

    mean = X_comp.mean(axis=0)

    cov = np.cov(
        X_comp,
        rowvar=False,
    )

    cov_reg = cov + cov_regularization * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov_reg)

    std = X_comp.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    train_distances = _mahalanobis_batch(
        X=X_comp,
        mean=mean,
        cov_inv=cov_inv,
    )

    threshold = float(
        np.quantile(
            train_distances,
            quantile,
        )
    )

    total_alloy, fe_balance = _composition_sum_and_fe_balance(X_ref)

    reference_rows = []

    for idx, row in X_ref.iterrows():
        record = {
            "row_index": int(idx),
        }

        for col in features:
            record[col] = _safe_float(row.get(col, 0.0))

        if "temp" in X_ref.columns:
            record["temp"] = _safe_float(row.get("temp"))

        if "stress" in X_ref.columns:
            record["stress"] = _safe_float(row.get("stress"))

        record["log_life"] = _safe_float(y_ref.iloc[idx])
        record["total_alloy_wt"] = _safe_float(total_alloy.iloc[idx])
        record["fe_balance"] = _safe_float(fe_balance.iloc[idx])

        reference_rows.append(record)

    mean_by_feature = {
        elem: float(mean[i])
        for i, elem in enumerate(features)
    }

    payload = {
        "version": "ood_reference_v2_ferritic_only",
        "description": (
            "Ferritic/martensitic-only Mahalanobis OOD reference for GA alloy candidates."
        ),

        "features": features,
        "reference_features": features,
        "ood_reference_features": features,

        "mean": mean,
        "ood_reference_mean": mean,
        "mean_by_feature": mean_by_feature,

        "cov_inv": cov_inv,
        "ood_reference_cov_inv": cov_inv,

        "std": std,

        "threshold": threshold,
        "ood_distance_threshold": threshold,
        "ood_quantile": float(quantile),

        "train_distances": train_distances,

        "n_before_filter": n_before_filter,
        "n_reference_rows": n_after_filter,
        "reference_rows": reference_rows,

        "split": {
            "test_size": float(test_size),
            "seed": int(seed),
            "rounding": int(rounding),
            "source": "prepare_dataset() + group_holdout_split()",
            "uses_synthetic_lmp_rows": False,
        },

        "family_filter": {
            "enabled": bool(ferritic_only),
            "cr_min": float(cr_min),
            "cr_max": float(cr_max),
            "ni_max": float(ni_max),
            "fe_min": float(fe_min),
            "total_alloy_max": float(total_alloy_max),
            "purpose": "Restrict OOD reference to Fe-rich 9-12Cr ferritic/martensitic creep steels.",
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        payload,
        output_path,
    )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ferritic-only Mahalanobis OOD reference for alloy GA."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OOD_CONFIG.get(
            "REFERENCE_PATH",
            str(ROOT_DIR / "data" / "ood_reference.pkl"),
        ),
    )

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounding", type=int, default=3)
    parser.add_argument("--quantile", type=float, default=float(OOD_CONFIG.get("OOD_QUANTILE", 0.95)))
    parser.add_argument("--cov-regularization", type=float, default=float(OOD_CONFIG.get("COV_REGULARIZATION", 1e-6)))

    parser.add_argument("--all-data", action="store_true", help="Ferritic filter를 끄고 전체 데이터로 OOD reference 생성")

    parser.add_argument("--cr-min", type=float, default=8.0)
    parser.add_argument("--cr-max", type=float, default=12.0)
    parser.add_argument("--ni-max", type=float, default=2.0)
    parser.add_argument("--fe-min", type=float, default=80.0)
    parser.add_argument("--total-alloy-max", type=float, default=20.0)
    parser.add_argument("--min-rows", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = Path(args.output)

    payload = build_ood_reference(
        output_path=output_path,
        test_size=args.test_size,
        seed=args.seed,
        rounding=args.rounding,
        quantile=args.quantile,
        cov_regularization=args.cov_regularization,
        ferritic_only=not args.all_data,
        cr_min=args.cr_min,
        cr_max=args.cr_max,
        ni_max=args.ni_max,
        fe_min=args.fe_min,
        total_alloy_max=args.total_alloy_max,
        min_rows=args.min_rows,
    )

    print("[OOD] ferritic-only reference 생성 완료")
    print(f"[OOD] output: {output_path}")
    print(f"[OOD] before filter: {payload['n_before_filter']}")
    print(f"[OOD] after filter : {payload['n_reference_rows']}")
    print(f"[OOD] threshold    : {payload['threshold']:.6f}")
    print("[OOD] feature means:")

    for elem, mean_val in payload["mean_by_feature"].items():
        print(f"  - {elem}: {mean_val:.6f}")