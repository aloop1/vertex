from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from 데이터전처리 import ( 
    COMPOSITION_COLS,
    HEAT_TREATMENT_COLS,
    LOG_TARGET_COL,
)


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for conservative LMP interpolation."""

    min_group_size: int = 5
    group_by_heat_treatment: bool = True
    c_range: tuple[float, float] = (10.0, 30.0)
    c_step: float = 0.25
    max_group_rmse_log: float = 0.35
    synthetic_ratio: float = 1.0
    max_synthetic_per_group: int = 20
    random_state: int = 42
    group_rounding_decimals: int = 3
    target_margin_log: float = 0.25
    min_unique_temperatures: int = 2
    min_unique_stresses: int = 2
    max_sampling_attempts_multiplier: int = 25


@dataclass(frozen=True)
class LmpFit:
    """Accepted LMP fit for one material-state group."""

    group_id: str
    c_value: float
    intercept: float
    slope: float
    rmse_log: float
    n_rows: int
    temp_min: float
    temp_max: float
    stress_min: float
    stress_max: float
    log_life_min: float
    log_life_max: float


@dataclass
class AugmentationResult:
    """Augmented training matrix plus non-feature metadata."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    metadata: pd.DataFrame
    fit_summary: pd.DataFrame

    @property
    def synthetic_count(self) -> int:
        return int(self.metadata["is_synthetic"].sum())

    @property
    def original_count(self) -> int:
        return int((~self.metadata["is_synthetic"]).sum())


def larson_miller_parameter(
    temp_k: np.ndarray | pd.Series,
    log_lifetime: np.ndarray | pd.Series,
    c_value: float,
) -> np.ndarray:
    """Compute LMP = T * (C + log10(t_r))."""
    return np.asarray(temp_k, dtype=float) * (c_value + np.asarray(log_lifetime, dtype=float))


def _rounded_key(df: pd.DataFrame, columns: Iterable[str], decimals: int) -> pd.Series:
    rounded = df[list(columns)].astype(float).round(decimals)
    return rounded.apply(
        lambda row: "|".join(f"{value:.{decimals}f}" for value in row.to_numpy()),
        axis=1,
    )


def make_material_state_group_id(
    df: pd.DataFrame,
    config: AugmentationConfig | None = None,
    prefix: str = "state",
) -> pd.Series:
    """Build a material-state key from composition and, by default, heat treatment."""
    config = config or AugmentationConfig()
    group_cols = list(COMPOSITION_COLS)
    if config.group_by_heat_treatment:
        group_cols.extend(HEAT_TREATMENT_COLS)

    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise KeyError(f"LMP 그룹 생성에 필요한 컬럼이 없습니다: {missing}")

    key = _rounded_key(df, group_cols, config.group_rounding_decimals)
    return prefix + "_" + key


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept), float(slope)


def _predict_log_life(
    temp_k: np.ndarray,
    stress_mpa: np.ndarray,
    c_value: float,
    intercept: float,
    slope: float,
) -> np.ndarray:
    lmp_pred = intercept + slope * np.log10(stress_mpa)
    return lmp_pred / temp_k - c_value


def fit_lmp_group(
    group_id: str,
    group_df: pd.DataFrame,
    config: AugmentationConfig | None = None,
) -> tuple[LmpFit | None, dict]:
    """Fit an LMP stress curve for one group and return rejection diagnostics."""
    config = config or AugmentationConfig()
    required = ["stress", "temp", LOG_TARGET_COL]
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        raise KeyError(f"LMP 피팅에 필요한 컬럼이 없습니다: {missing}")

    clean = group_df.loc[
        (group_df["stress"] > 0)
        & (group_df["temp"] > 0)
        & np.isfinite(group_df[LOG_TARGET_COL])
    ].copy()

    diagnostics = {
        "group_id": group_id,
        "n_rows": int(len(clean)),
        "accepted": False,
        "reason": "",
        "c_value": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "rmse_log": np.nan,
        "synthetic_rows": 0,
    }

    if len(clean) < config.min_group_size:
        diagnostics["reason"] = "too_few_rows"
        return None, diagnostics
    if clean["temp"].nunique() < config.min_unique_temperatures:
        diagnostics["reason"] = "too_few_temperatures"
        return None, diagnostics
    if clean["stress"].nunique() < config.min_unique_stresses:
        diagnostics["reason"] = "too_few_stresses"
        return None, diagnostics

    stress = clean["stress"].to_numpy(dtype=float)
    temp = clean["temp"].to_numpy(dtype=float)
    log_life = clean[LOG_TARGET_COL].to_numpy(dtype=float)
    log_stress = np.log10(stress)

    best: tuple[float, float, float, float] | None = None
    c_start, c_stop = config.c_range
    for c_value in np.arange(c_start, c_stop + config.c_step * 0.5, config.c_step):
        lmp = larson_miller_parameter(temp, log_life, float(c_value))
        intercept, slope = _linear_fit(log_stress, lmp)
        if slope >= 0:
            continue
        pred_log_life = _predict_log_life(temp, stress, float(c_value), intercept, slope)
        rmse_log = float(np.sqrt(np.mean((pred_log_life - log_life) ** 2)))
        if best is None or rmse_log < best[0]:
            best = (rmse_log, float(c_value), intercept, slope)

    if best is None:
        diagnostics["reason"] = "non_monotonic_fit"
        return None, diagnostics

    rmse_log, c_value, intercept, slope = best
    diagnostics.update(
        {
            "c_value": c_value,
            "slope": slope,
            "intercept": intercept,
            "rmse_log": rmse_log,
        }
    )

    if rmse_log > config.max_group_rmse_log:
        diagnostics["reason"] = "rmse_too_high"
        return None, diagnostics

    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted"

    fit = LmpFit(
        group_id=group_id,
        c_value=c_value,
        intercept=intercept,
        slope=slope,
        rmse_log=rmse_log,
        n_rows=int(len(clean)),
        temp_min=float(clean["temp"].min()),
        temp_max=float(clean["temp"].max()),
        stress_min=float(clean["stress"].min()),
        stress_max=float(clean["stress"].max()),
        log_life_min=float(clean[LOG_TARGET_COL].min()),
        log_life_max=float(clean[LOG_TARGET_COL].max()),
    )
    return fit, diagnostics


def _target_synthetic_count(group_size: int, config: AugmentationConfig) -> int:
    if config.synthetic_ratio <= 0:
        return 0
    raw_count = int(round(group_size * config.synthetic_ratio))
    return max(1, min(config.max_synthetic_per_group, raw_count))


def _generate_group_samples(
    group_df: pd.DataFrame,
    fit: LmpFit,
    feature_columns: list[str],
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> tuple[list[pd.Series], list[float], list[dict]]:
    target_count = _target_synthetic_count(len(group_df), config)
    if target_count <= 0:
        return [], [], []

    synthetic_rows: list[pd.Series] = []
    synthetic_targets: list[float] = []
    synthetic_meta: list[dict] = []
    attempts = 0
    max_attempts = max(target_count, target_count * config.max_sampling_attempts_multiplier)
    log_stress_min = np.log10(fit.stress_min)
    log_stress_max = np.log10(fit.stress_max)

    while len(synthetic_rows) < target_count and attempts < max_attempts:
        attempts += 1
        template_idx = int(rng.integers(0, len(group_df)))
        template = group_df.iloc[template_idx][feature_columns].copy()

        temp_new = float(rng.uniform(fit.temp_min, fit.temp_max))
        log_stress_new = float(rng.uniform(log_stress_min, log_stress_max))
        stress_new = float(10 ** log_stress_new)
        log_life_new = float(
            _predict_log_life(
                np.array([temp_new]),
                np.array([stress_new]),
                fit.c_value,
                fit.intercept,
                fit.slope,
            )[0]
        )

        lower = fit.log_life_min - config.target_margin_log
        upper = fit.log_life_max + config.target_margin_log
        if not (lower <= log_life_new <= upper):
            continue

        template["temp"] = temp_new
        template["stress"] = stress_new

        synthetic_rows.append(template)
        synthetic_targets.append(log_life_new)
        synthetic_meta.append(
            {
                "is_synthetic": True,
                "source_group_id": fit.group_id,
                "source_composition_group": group_df["_composition_group"].iloc[template_idx],
                "lmp_c": fit.c_value,
                "lmp_fit_rmse": fit.rmse_log,
                "temp_min": fit.temp_min,
                "temp_max": fit.temp_max,
                "stress_min": fit.stress_min,
                "stress_max": fit.stress_max,
            }
        )

    return synthetic_rows, synthetic_targets, synthetic_meta


def augment_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None = None,
    config: AugmentationConfig | None = None,
) -> AugmentationResult:
    """Create conservative train-only LMP synthetic samples.

    The returned ``X`` contains only model features. LMP metadata is stored in
    ``metadata`` so downstream code cannot accidentally train on it.
    """
    config = config or AugmentationConfig()
    feature_columns = list(X_train.columns)
    base = X_train.reset_index(drop=True).copy()
    y_base = y_train.reset_index(drop=True).astype(float)
    base[LOG_TARGET_COL] = y_base

    if groups_train is None:
        base["_composition_group"] = make_material_state_group_id(base, config)
        original_groups: pd.Series | None = None
    else:
        original_groups = groups_train.reset_index(drop=True).astype(str)
        base["_composition_group"] = original_groups

    base["_material_state_group"] = make_material_state_group_id(base, config)
    rng = np.random.default_rng(config.random_state)

    metadata = pd.DataFrame(
        {
            "is_synthetic": False,
            "source_group_id": base["_material_state_group"].to_numpy(),
            "source_composition_group": base["_composition_group"].to_numpy(),
            "lmp_c": np.nan,
            "lmp_fit_rmse": np.nan,
            "temp_min": np.nan,
            "temp_max": np.nan,
            "stress_min": np.nan,
            "stress_max": np.nan,
        }
    )

    synthetic_rows: list[pd.Series] = []
    synthetic_targets: list[float] = []
    synthetic_meta: list[dict] = []
    fit_records: list[dict] = []

    for group_id, group_df in base.groupby("_material_state_group", sort=False):
        fit, diagnostics = fit_lmp_group(str(group_id), group_df, config)
        if fit is not None:
            rows, targets, meta = _generate_group_samples(
                group_df,
                fit,
                feature_columns,
                config,
                rng,
            )
            synthetic_rows.extend(rows)
            synthetic_targets.extend(targets)
            synthetic_meta.extend(meta)
            diagnostics["synthetic_rows"] = len(rows)
        fit_records.append(diagnostics)

    if synthetic_rows:
        X_synthetic = pd.DataFrame(synthetic_rows, columns=feature_columns).reset_index(drop=True)
        y_synthetic = pd.Series(synthetic_targets, name=y_train.name or LOG_TARGET_COL)
        synthetic_metadata = pd.DataFrame(synthetic_meta)

        X_aug = pd.concat([X_train.reset_index(drop=True), X_synthetic], ignore_index=True)
        y_aug = pd.concat([y_base, y_synthetic], ignore_index=True)
        metadata = pd.concat([metadata, synthetic_metadata], ignore_index=True)

        if original_groups is None:
            synthetic_groups = synthetic_metadata["source_group_id"].astype(str)
            groups_aug: pd.Series | None = pd.concat(
                [base["_material_state_group"].astype(str), synthetic_groups],
                ignore_index=True,
            )
        else:
            synthetic_groups = synthetic_metadata["source_composition_group"].astype(str)
            groups_aug = pd.concat([original_groups, synthetic_groups], ignore_index=True)
    else:
        X_aug = X_train.reset_index(drop=True).copy()
        y_aug = y_base.copy()
        groups_aug = None if groups_train is None else groups_train.reset_index(drop=True).astype(str)

    fit_summary = pd.DataFrame(fit_records)
    return AugmentationResult(
        X=X_aug[feature_columns].reset_index(drop=True),
        y=y_aug.reset_index(drop=True),
        groups=groups_aug.reset_index(drop=True) if groups_aug is not None else None,
        metadata=metadata.reset_index(drop=True),
        fit_summary=fit_summary,
    )


def summarize_augmentation(result: AugmentationResult) -> dict[str, float]:
    """Return compact augmentation diagnostics for logs and artifacts."""
    fit_summary = result.fit_summary
    accepted = int(fit_summary["accepted"].sum()) if not fit_summary.empty else 0
    total = int(len(fit_summary))
    return {
        "original_rows": float(result.original_count),
        "synthetic_rows": float(result.synthetic_count),
        "total_rows": float(len(result.X)),
        "fit_groups_total": float(total),
        "fit_groups_accepted": float(accepted),
        "fit_acceptance_rate": float(accepted / total) if total else 0.0,
    }


def assert_synthetic_within_source_bounds(result: AugmentationResult, X: pd.DataFrame) -> None:
    """Validate that generated samples are interpolation-only."""
    synthetic_meta = result.metadata[result.metadata["is_synthetic"]].reset_index(drop=True)
    if synthetic_meta.empty:
        return
    synthetic_X = X.loc[result.metadata["is_synthetic"].to_numpy()].reset_index(drop=True)
    temp_ok = (
        (synthetic_X["temp"] >= synthetic_meta["temp_min"])
        & (synthetic_X["temp"] <= synthetic_meta["temp_max"])
    )
    stress_ok = (
        (synthetic_X["stress"] >= synthetic_meta["stress_min"])
        & (synthetic_X["stress"] <= synthetic_meta["stress_max"])
    )
    if not bool(temp_ok.all() and stress_ok.all()):
        raise AssertionError("합성 샘플이 원본 그룹의 온도/응력 범위를 벗어났습니다.")
