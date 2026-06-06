from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODEL_DIR = ROOT_DIR / "models"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from 데이터전처리 import prepare_dataset


@dataclass(frozen=True)
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


@dataclass(frozen=True)
class LmpBaseline:
    c_value: float
    coefficients: np.ndarray
    train_rmse_log: float


def group_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    seed: int,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    groups_arr = groups.to_numpy()
    unique_groups = np.unique(groups_arr)
    rng.shuffle(unique_groups)

    target_count = max(1, int(round(len(X) * test_size)))
    selected: list[object] = []
    selected_count = 0
    for group in unique_groups:
        selected.append(group)
        selected_count += int(np.sum(groups_arr == group))
        if selected_count >= target_count:
            break

    test_mask = np.isin(groups_arr, np.array(selected, dtype=object))
    train_mask = ~test_mask

    return SplitResult(
        X_train=X.loc[train_mask].reset_index(drop=True),
        X_test=X.loc[test_mask].reset_index(drop=True),
        y_train=y.loc[train_mask].reset_index(drop=True),
        y_test=y.loc[test_mask].reset_index(drop=True),
        groups_train=groups.loc[train_mask].reset_index(drop=True),
        groups_test=groups.loc[test_mask].reset_index(drop=True),
    )


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def lmp_from_log_life(temp_k: np.ndarray, log_life: np.ndarray, c_value: float) -> np.ndarray:
    return temp_k * (c_value + log_life) / 1000.0


def predict_log_life(
    temp_k: np.ndarray,
    stress_mpa: np.ndarray,
    c_value: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    log_stress = np.log10(np.clip(stress_mpa, a_min=1e-12, a_max=None))
    lmp_pred = np.polyval(coefficients, log_stress)
    return (1000.0 * lmp_pred / np.clip(temp_k, a_min=1e-12, a_max=None)) - c_value


def valid_lmp_rows(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    stress = X["stress"].to_numpy(dtype=float)
    temp = X["temp"].to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    return (
        np.isfinite(stress)
        & np.isfinite(temp)
        & np.isfinite(target)
        & (stress > 0)
        & (temp > 0)
    )


def fit_lmp_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_values: Iterable[float],
    degree: int,
) -> LmpBaseline:
    mask = valid_lmp_rows(X_train, y_train)
    X_fit = X_train.loc[mask].reset_index(drop=True)
    y_fit = y_train.loc[mask].reset_index(drop=True)

    stress = X_fit["stress"].to_numpy(dtype=float)
    temp = X_fit["temp"].to_numpy(dtype=float)
    log_life = y_fit.to_numpy(dtype=float)
    log_stress = np.log10(stress)

    best: LmpBaseline | None = None
    for c_value in c_values:
        lmp = lmp_from_log_life(temp, log_life, float(c_value))
        coefficients = np.polyfit(log_stress, lmp, deg=degree)
        if degree == 1 and coefficients[0] >= 0:
            continue

        pred_log_life = predict_log_life(temp, stress, float(c_value), coefficients)
        train_rmse = rmse_np(log_life, pred_log_life)
        candidate = LmpBaseline(
            c_value=float(c_value),
            coefficients=coefficients.astype(float),
            train_rmse_log=train_rmse,
        )
        if best is None or candidate.train_rmse_log < best.train_rmse_log:
            best = candidate

    if best is None:
        raise RuntimeError("유효한 LMP baseline 피팅 결과가 없습니다.")
    return best


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true_hours = np.power(10.0, y_true_log)
    y_pred_hours = np.power(10.0, y_pred_log)
    return {
        "rmse_log": rmse_np(y_true_log, y_pred_log),
        "mae_log": mae_np(y_true_log, y_pred_log),
        "r2_log": r2_np(y_true_log, y_pred_log),
        "rmse_hours": rmse_np(y_true_hours, y_pred_hours),
        "mae_hours": mae_np(y_true_hours, y_pred_hours),
        "r2_hours": r2_np(y_true_hours, y_pred_hours),
    }


def parse_ai_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "rmse_log": r"최종 RMSE\(log10\):\s*([-+0-9.]+)",
        "mae_log": r"최종 MAE\(log10\):\s*([-+0-9.]+)",
        "r2_log": r"최종 R2\(log10\):\s*([-+0-9.]+)",
        "rmse_hours": r"최종 RMSE\(hours\):\s*([-+0-9.]+)",
        "mae_hours": r"최종 MAE\(hours\):\s*([-+0-9.]+)",
    }
    metrics: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def format_poly(coefficients: np.ndarray) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for idx, coef in enumerate(coefficients):
        power = degree - idx
        if power == 0:
            terms.append(f"{coef:.6f}")
        elif power == 1:
            terms.append(f"{coef:.6f}*log10(stress)")
        else:
            terms.append(f"{coef:.6f}*log10(stress)^{power}")
    return " + ".join(terms)


def write_report(
    output_path: Path,
    split: SplitResult,
    baseline: LmpBaseline,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    ai_metrics: dict[str, float],
    degree: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "========================================================================",
        "LMP 기반 기존 물리 경험식 baseline",
        "========================================================================",
        "모델: Larson-Miller Parameter master curve",
        "학습식: LMP = f(log10(stress))",
        "예측식: log10(lifetime) = 1000 * LMP_pred / T(K) - C",
        "테스트 세트에는 피팅하지 않음",
        "",
        "[데이터 분할]",
        f"학습 샘플: {len(split.X_train)} | 테스트 샘플: {len(split.X_test)}",
        f"학습 그룹: {split.groups_train.nunique()} | 테스트 그룹: {split.groups_test.nunique()}",
        "",
        "[피팅 결과]",
        f"다항 차수: {degree}",
        f"최적 C 값: {baseline.c_value:.3f}",
        f"LMP 곡선: {format_poly(baseline.coefficients)}",
        f"학습 RMSE(log10): {baseline.train_rmse_log:.6f}",
        "",
        "[LMP baseline 테스트 성능]",
        f"RMSE(log10): {test_metrics['rmse_log']:.6f}",
        f"MAE(log10): {test_metrics['mae_log']:.6f}",
        f"R2(log10): {test_metrics['r2_log']:.6f}",
        f"RMSE(hours): {test_metrics['rmse_hours']:.3f}",
        f"MAE(hours): {test_metrics['mae_hours']:.3f}",
        "",
        "[AI 모델과 비교]",
    ]

    if ai_metrics:
        lines.extend(
            [
                f"AI RMSE(log10): {ai_metrics.get('rmse_log', float('nan')):.6f}",
                f"AI MAE(log10): {ai_metrics.get('mae_log', float('nan')):.6f}",
                f"AI R2(log10): {ai_metrics.get('r2_log', float('nan')):.6f}",
                f"R2 개선폭: {ai_metrics.get('r2_log', float('nan')) - test_metrics['r2_log']:.6f}",
                f"RMSE(log10) 감소량: {test_metrics['rmse_log'] - ai_metrics.get('rmse_log', float('nan')):.6f}",
            ]
        )
    else:
        lines.append("AI 결과 파일을 찾지 못해 비교값을 생략함")

    lines.extend(
        [
            "",
            "[해석]",
            "LMP baseline은 온도-응력 기반 물리 경험식으로 수명을 예측한다.",
            "AI 모델은 동일 테스트셋에서 조성, 열처리, 운전 조건의 비선형 상호작용까지 함께 학습한다.",
            "========================================================================",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LMP 기반 기존 물리 경험식 baseline 평가")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--c-start", type=float, default=10.0)
    parser.add_argument("--c-stop", type=float, default=30.0)
    parser.add_argument("--c-step", type=float, default=0.25)
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(MODEL_DIR / "lmp_physics_baseline_output.txt"),
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=str(MODEL_DIR / "lmp_physics_baseline_predictions.csv"),
    )
    parser.add_argument(
        "--ai-output-path",
        type=str,
        default=str(MODEL_DIR / "transformer_and_tree_ensemble_output.txt"),
    )
    args = parser.parse_args()

    prepared = prepare_dataset()
    split = group_holdout_split(
        prepared.X,
        prepared.y,
        prepared.groups,
        test_size=args.test_size,
        seed=args.seed,
    )

    c_values = np.arange(args.c_start, args.c_stop + args.c_step * 0.5, args.c_step)
    baseline = fit_lmp_baseline(split.X_train, split.y_train, c_values, degree=args.degree)

    train_pred = predict_log_life(
        split.X_train["temp"].to_numpy(dtype=float),
        split.X_train["stress"].to_numpy(dtype=float),
        baseline.c_value,
        baseline.coefficients,
    )
    test_pred = predict_log_life(
        split.X_test["temp"].to_numpy(dtype=float),
        split.X_test["stress"].to_numpy(dtype=float),
        baseline.c_value,
        baseline.coefficients,
    )

    y_train_log = split.y_train.to_numpy(dtype=float)
    y_test_log = split.y_test.to_numpy(dtype=float)
    train_metrics = evaluate_predictions(y_train_log, train_pred)
    test_metrics = evaluate_predictions(y_test_log, test_pred)
    ai_metrics = parse_ai_metrics(Path(args.ai_output_path))

    predictions = pd.DataFrame(
        {
            "stress": split.X_test["stress"].to_numpy(dtype=float),
            "temp": split.X_test["temp"].to_numpy(dtype=float),
            "y_true_log": y_test_log,
            "y_pred_log": test_pred,
            "y_true_hours": np.power(10.0, y_test_log),
            "y_pred_hours": np.power(10.0, test_pred),
            "error_log": test_pred - y_test_log,
        }
    )
    predictions.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")

    write_report(
        Path(args.output_path),
        split,
        baseline,
        train_metrics,
        test_metrics,
        ai_metrics,
        degree=args.degree,
    )

    print(f"LMP baseline 결과 저장: {args.output_path}")
    print(f"LMP baseline 예측 저장: {args.predictions_path}")
    print(f"R2(log10): {test_metrics['r2_log']:.6f}")
    print(f"RMSE(log10): {test_metrics['rmse_log']:.6f}")


if __name__ == "__main__":
    main()
