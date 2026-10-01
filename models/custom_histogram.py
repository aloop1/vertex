"""
Histogram boosting과 radial basis 회귀를 결합한 크립 수명 예측 모델

데이터 전처리와 LMP 데이터 증강 파일을 가져오며
분할, 피처 생성, 트리, RBF, 비선형 특징망, 앙상블을 구현함
"""
import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "데이터전처리.py").exists() else SCRIPT_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
sys.path.insert(0, str(ROOT_DIR))

from 데이터전처리 import (
    COMPOSITION_COLS,
    HEAT_TREATMENT_COLS,
    prepare_dataset,
)
from models.LMP_데이터증강 import (
    AugmentationConfig,
    assert_synthetic_within_source_bounds,
    augment_training_data,
)


BASE_WEIGHTS = np.array([0.68175, 0.27075, 0.0475])
TREE_DEPTHS = (3, 5, 7)
TREE_ROUNDS = (650, 514, 383)
TREE_BLEND = np.array([0.10, 0.55, 0.35])


# ---------------------------------------------------------------------------
# 1. 공통 피처 생성, 그룹 분할, 평가
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """조성 그룹이 겹치지 않는 학습/평가 분할."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def add_physics_features(frame):
    """기존 30개 입력에 네 개의 물리 파생 피처를 추가한다."""
    output = frame.copy()
    stress = pd.to_numeric(output["stress"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    temp = pd.to_numeric(output["temp"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    output["operating_severity"] = temp / 1000.0 * np.log10(stress + 1.0)
    output["stress_temperature_interaction"] = stress * temp / 1000.0
    output["inverse_temperature"] = 1000.0 / temp

    # 전처리에서 만든 열처리 가혹도를 합산한다. 누락된 경우에도 안전하게 계산한다.
    for prefix in ("N", "T", "A"):
        temp_col, time_col = f"{prefix}temp", f"{prefix}time"
        if temp_col in output and time_col in output:
            treatment_temp = pd.to_numeric(output[temp_col], errors="coerce").fillna(0.0)
            treatment_time = np.maximum(
                pd.to_numeric(output[time_col], errors="coerce").fillna(0.0).to_numpy(float),
                1e-6,
            )
            severity = treatment_temp.to_numpy(float) * (20.0 + np.log10(treatment_time))
            output[f"{prefix}_severity"] = np.where(treatment_temp > 0, severity, 0.0)
    severity_cols = [name for name in ("N_severity", "T_severity", "A_severity") if name in output]
    if severity_cols:
        output["total_heat_treatment_severity"] = output[severity_cols].sum(axis=1)
    else:
        heat_cols = [name for name in HEAT_TREATMENT_COLS if name in output]
        output["total_heat_treatment_severity"] = output[heat_cols].sum(axis=1) if heat_cols else 0.0
    return output


def features(frame):
    """타깃을 포함하지 않는 고정 순서 34개 입력 피처를 반환한다."""
    return add_physics_features(frame).drop(
        columns=["LMP", "lifetime", "log_lifetime"], errors="ignore"
    )


def group_holdout_split(X, y, groups, test_size, seed):
    """행이 아닌 조성 그룹 단위로 분할해 동일 합금의 누수를 막는다."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups.to_numpy())
    rng.shuffle(unique_groups)
    target_count = max(1, int(round(len(X) * test_size)))
    selected, selected_count = [], 0
    group_values = groups.to_numpy()
    for group in unique_groups:
        selected.append(group)
        selected_count += int(np.sum(group_values == group))
        if selected_count >= target_count:
            break
    test_mask = np.isin(group_values, np.asarray(selected, dtype=object))
    train_mask = ~test_mask
    return SplitResult(
        X.loc[train_mask].reset_index(drop=True),
        X.loc[test_mask].reset_index(drop=True),
        y.loc[train_mask].reset_index(drop=True),
        y.loc[test_mask].reset_index(drop=True),
        groups.loc[train_mask].reset_index(drop=True),
        groups.loc[test_mask].reset_index(drop=True),
    )


def metrics(y, prediction):
    """log10 수명과 원래 시간 단위의 평가 지표를 함께 계산한다."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = y - prediction
    hours_error = 10.0 ** y - 10.0 ** prediction
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
        "rmse_hours": float(np.sqrt(np.mean(hours_error ** 2))),
        "mae_hours": float(np.mean(np.abs(hours_error))),
    }


def augmented(X, y, groups, seed):
    """공통 LMP 증강 모듈을 호출하고 합성 행에 0.35 가중치를 부여한다."""
    config = AugmentationConfig(
        synthetic_ratio=0.5,
        max_synthetic_per_group=40,
        target_margin_log=0.45,
        random_state=seed,
    )
    result = augment_training_data(X, y, groups_train=groups, config=config)
    assert_synthetic_within_source_bounds(result, result.X)
    weights = np.full(len(result.y), 0.35)
    weights[:len(y)] = 1.0
    return features(result.X).to_numpy(float), np.asarray(result.y), weights, result, config


# ---------------------------------------------------------------------------
# 2. Histogram Gradient Boosting
# ---------------------------------------------------------------------------

class HistogramTree:
    """히스토그램 누적합으로 최적 분할을 찾는 가중 CART 회귀 트리."""

    def __init__(self, depth=5, min_leaf=10, l2=3.0, seed=42):
        self.depth = depth
        self.min_leaf = min_leaf
        self.l2 = l2
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, weights):
        def build(indices, depth):
            local_weights = weights[indices]
            target = y[indices]
            total_weight = local_weights.sum()
            total_gradient = np.dot(local_weights, target)
            value = float(total_gradient / (total_weight + self.l2))
            if depth == self.depth or len(indices) < 2 * self.min_leaf:
                return value
            best_gain, best_split = 1e-9, None
            feature_ids = self.rng.choice(
                X.shape[1], max(1, int(X.shape[1] * 0.85)), replace=False
            )
            for feature_id in feature_ids:
                bins = X[indices, feature_id]
                count = np.bincount(bins, minlength=128).cumsum()[:-1]
                left_weight = np.bincount(
                    bins, weights=local_weights, minlength=128
                ).cumsum()[:-1]
                left_gradient = np.bincount(
                    bins, weights=local_weights * target, minlength=128
                ).cumsum()[:-1]
                gain = (
                    left_gradient ** 2 / np.maximum(left_weight + self.l2, 1e-15)
                    + (total_gradient - left_gradient) ** 2
                    / np.maximum(total_weight - left_weight + self.l2, 1e-15)
                    - total_gradient ** 2 / (total_weight + self.l2)
                )
                gain[(count < self.min_leaf) | (len(indices) - count < self.min_leaf)] = -np.inf
                threshold = int(np.argmax(gain))
                if gain[threshold] > best_gain:
                    best_gain = gain[threshold]
                    best_split = (int(feature_id), threshold)
            if best_split is None:
                return value
            feature_id, threshold = best_split
            left_mask = X[indices, feature_id] <= threshold
            return (
                feature_id,
                threshold,
                build(indices[left_mask], depth + 1),
                build(indices[~left_mask], depth + 1),
            )

        self.root = build(np.arange(len(y)), 0)
        return self

    def predict(self, X):
        output = np.empty(len(X))

        def visit(node, indices):
            if not isinstance(node, tuple):
                output[indices] = node
                return
            feature_id, threshold, left_node, right_node = node
            left_mask = X[indices, feature_id] <= threshold
            visit(left_node, indices[left_mask])
            visit(right_node, indices[~left_mask])

        visit(self.root, np.arange(len(X)))
        return output


class HistogramBoosting:
    """고정 학습률로 잔차를 순차 보정하는 부스팅 모델."""

    def __init__(self, depth, seed, rounds, min_leaf=10, l2=3.0, bins=128, rate=0.045):
        self.depth = depth
        self.seed = seed
        self.rounds = rounds
        self.min_leaf = min_leaf
        self.l2 = l2
        self.bins = bins
        self.rate = rate

    def encode(self, X):
        return np.column_stack([
            np.searchsorted(edge, X[:, index], side="left")
            for index, edge in enumerate(self.edges)
        ]).astype(np.uint16)

    def fit(self, X, y, weights):
        # 경계는 합성 행을 제외한 실제 학습 관측치에서만 만든다.
        real = X[weights == 1]
        self.edges = [
            np.unique(np.quantile(real[:, index], np.linspace(0, 1, self.bins + 1)[1:-1]))
            for index in range(X.shape[1])
        ]
        encoded = self.encode(X)
        self.base = float(np.average(y, weights=weights))
        prediction = np.full(len(y), self.base)
        self.trees = []
        rng = np.random.default_rng(self.seed)
        for iteration in range(self.rounds):
            sample = rng.choice(len(y), int(0.85 * len(y)), replace=False)
            tree = HistogramTree(
                self.depth, self.min_leaf, self.l2, self.seed + iteration
            ).fit(encoded[sample], (y - prediction)[sample], weights[sample])
            self.trees.append(tree)
            prediction += self.rate * tree.predict(encoded)
        return self

    def predict(self, X):
        encoded = self.encode(np.asarray(X))
        prediction = np.full(len(X), self.base)
        for tree in self.trees:
            prediction += self.rate * tree.predict(encoded)
        return prediction


def train_tree_bundle(X, y, weights, names, smoke, final_fit):
    """검증에서 확정한 트리 수·깊이·가중치로 트리 묶음을 학습한다."""
    rounds = (3, 3, 3) if smoke else TREE_ROUNDS
    seed_offsets = (42,) if smoke else ((42, 10042, 20042) if final_fit else (42,))
    models, model_weights = [], []
    for offset in seed_offsets:
        for index, depth in enumerate(TREE_DEPTHS):
            model = HistogramBoosting(depth, offset + 1000 * index, rounds[index]).fit(
                X, y, weights
            )
            models.append(model)
            model_weights.append(TREE_BLEND[index] / len(seed_offsets))
    return {
        "models": models,
        "weights": np.asarray(model_weights),
        "features": list(names),
        "smoke": smoke,
    }


def predict_bundle(bundle, frame):
    """저장된 트리 묶음으로 DataFrame을 예측한다."""
    X = features(frame)[bundle["features"]].to_numpy(float)
    return np.column_stack([model.predict(X) for model in bundle["models"]]) @ bundle["weights"]


# ---------------------------------------------------------------------------
# 3. RBF와 기본 앙상블
# ---------------------------------------------------------------------------

class RadialBasisRegressor:
    """학습 행을 중심으로 사용하는 가중 RBF ridge 회귀."""

    def __init__(self, gamma=0.5, ridge=0.01, seed=42, centers=600):
        self.gamma = gamma
        self.ridge = ridge
        self.seed = seed
        self.n_centers = centers

    def scale(self, X):
        return np.clip((np.asarray(X) - self.mean) / self.std, -8, 8) * self.axis_weights

    def basis(self, X):
        scaled = self.scale(X)
        distance = np.maximum(
            np.sum(scaled * scaled, axis=1)[:, None]
            + np.sum(self.centers * self.centers, axis=1)[None, :]
            - 2 * scaled @ self.centers.T,
            0,
        )
        return np.exp(-self.gamma * distance)

    def fit(self, X, y, weights, names):
        real = np.asarray(X)[weights == 1]
        self.mean = real.mean(axis=0)
        self.std = real.std(axis=0)
        self.std[self.std < 1e-10] = 1
        operating = {
            "temp", "stress", "operating_severity",
            "stress_temperature_interaction", "inverse_temperature",
        }
        self.axis_weights = np.asarray([
            1 / np.sqrt(19) if name in COMPOSITION_COLS
            else 1 / np.sqrt(5) if name in operating
            else 1 / np.sqrt(10)
            for name in names
        ])
        rng = np.random.default_rng(self.seed)
        center_rows = rng.choice(len(real), min(self.n_centers, len(real)), replace=False)
        self.centers = self.scale(real[center_rows])
        self.base = float(np.average(y, weights=weights))
        basis = self.basis(X)
        weighted = basis * np.sqrt(weights[:, None])
        eigenvalues, vectors = np.linalg.eigh(weighted.T @ weighted)
        right = vectors.T @ (basis.T @ (weights * (y - self.base)))
        self.spectrum = (np.maximum(eigenvalues, 0), vectors, right)
        self.set_ridge(self.ridge)
        return self

    def set_ridge(self, ridge):
        self.ridge = ridge
        eigenvalues, vectors, right = self.spectrum
        self.coef = vectors @ (right / (eigenvalues + ridge))

    def predict(self, X):
        return self.base + self.basis(X) @ self.coef


class MixedEnsemble:
    """Histogram Boosting과 두 RBF 모델을 결합한다."""

    def __init__(self, tree_bundle, rbf_models, weights):
        self.tree_bundle = tree_bundle
        self.rbf_models = rbf_models
        self.weights = np.asarray(weights)

    def predict(self, frame):
        X = features(frame)[self.tree_bundle["features"]].to_numpy(float)
        predictions = [predict_bundle(self.tree_bundle, frame)]
        predictions.extend(model.predict(X) for model in self.rbf_models)
        return np.column_stack(predictions) @ self.weights


def train_base_ensemble(X, y, weights, names, smoke, final_fit):
    """현재 학습 데이터로 트리와 RBF 구성원을 학습한다."""
    centers = 50 if smoke else 600
    tree_bundle = train_tree_bundle(X, y, weights, names, smoke, final_fit)
    rbf_models = [
        RadialBasisRegressor(0.5, 0.001, seed=42, centers=centers).fit(X, y, weights, names),
        RadialBasisRegressor(1.5, 1.0, seed=42, centers=centers).fit(X, y, weights, names),
    ]
    for model in rbf_models:
        del model.spectrum
    return MixedEnsemble(tree_bundle, rbf_models, BASE_WEIGHTS)


# ---------------------------------------------------------------------------
# 4. 추가 구성원: LMP-RBF, 이웃 회귀, 무작위 비선형 특징망
# ---------------------------------------------------------------------------

class TransformedRBFRegressor:
    """log10 수명 또는 Larson-Miller 타깃을 학습하는 RBF 모델."""

    def __init__(self, target="lmp", lmp_constant=20.0, gamma=0.5, ridge=0.01,
                 seed=42, centers=600):
        self.target = target
        self.lmp_constant = float(lmp_constant)
        self.model = RadialBasisRegressor(
            gamma=gamma, ridge=ridge, seed=seed, centers=centers
        )

    def fit(self, X, y, weights, names):
        self.temp_index = names.index("temp")
        transformed = np.asarray(y, dtype=float)
        if self.target == "lmp":
            transformed = (
                np.asarray(X, dtype=float)[:, self.temp_index]
                * (self.lmp_constant + transformed)
                / 1000.0
            )
        self.model.fit(X, transformed, weights, names)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        prediction = self.model.predict(X)
        if self.target == "lmp":
            prediction = 1000.0 * prediction / X[:, self.temp_index] - self.lmp_constant
        return prediction

    def compact(self):
        if hasattr(self.model, "spectrum"):
            del self.model.spectrum
        return self


class KernelNeighborRegressor:
    """NumPy 거리 계산으로 구현한 가중 k-최근접 이웃 회귀."""

    def __init__(self, neighbors=80, bandwidth=1.0, composition_weight=1.0,
                 condition_weight=1.0):
        self.neighbors = int(neighbors)
        self.bandwidth = float(bandwidth)
        self.composition_weight = float(composition_weight)
        self.condition_weight = float(condition_weight)

    def fit(self, X, y, weights, names):
        X = np.asarray(X, dtype=float)
        weights = np.asarray(weights, dtype=float)
        real = X[weights == 1]
        self.mean = real.mean(axis=0)
        self.std = real.std(axis=0)
        self.std[self.std < 1e-10] = 1.0
        operating = {
            "temp", "stress", "operating_severity",
            "stress_temperature_interaction", "inverse_temperature",
        }
        axes = []
        for name in names:
            if name in COMPOSITION_COLS:
                axes.append(self.composition_weight / np.sqrt(len(COMPOSITION_COLS)))
            elif name in operating:
                axes.append(self.condition_weight / np.sqrt(len(operating)))
            else:
                axes.append(1.0 / np.sqrt(10.0))
        self.axes = np.asarray(axes)
        self.X = np.clip((X - self.mean) / self.std, -8, 8) * self.axes
        self.y = np.asarray(y, dtype=float)
        self.sample_weights = weights
        return self

    def predict(self, X):
        query = np.clip((np.asarray(X) - self.mean) / self.std, -8, 8) * self.axes
        output = np.empty(len(query))
        k = min(self.neighbors, len(self.X))
        train_norm = np.sum(self.X * self.X, axis=1)
        for start in range(0, len(query), 256):
            q = query[start:start + 256]
            distance = np.maximum(
                np.sum(q * q, axis=1)[:, None] + train_norm[None, :] - 2 * q @ self.X.T,
                0.0,
            )
            indices = np.argpartition(distance, k - 1, axis=1)[:, :k]
            local_distance = np.take_along_axis(distance, indices, axis=1)
            scale = np.maximum(np.median(local_distance, axis=1, keepdims=True), 1e-10)
            kernel = np.exp(-local_distance / (self.bandwidth * scale))
            kernel *= self.sample_weights[indices]
            output[start:start + len(q)] = np.sum(kernel * self.y[indices], axis=1) / np.maximum(
                np.sum(kernel, axis=1), 1e-12
            )
        return output


class RandomFeatureRegressor:
    """무작위 은닉층과 닫힌형 ridge 해를 사용하는 비선형 특징망."""

    def __init__(self, hidden=600, scale=0.7, ridge=1.0, activation="tanh", seed=42):
        self.hidden = int(hidden)
        self.scale = float(scale)
        self.ridge = float(ridge)
        self.activation = activation
        self.seed = int(seed)

    def _hidden(self, X):
        z = np.clip((np.asarray(X) - self.mean) / self.std, -8, 8) * self.axes
        value = z @ self.projection + self.bias
        if self.activation == "relu":
            return np.maximum(value, 0.0)
        return np.tanh(value)

    def fit(self, X, y, weights, names):
        X = np.asarray(X, dtype=float)
        weights = np.asarray(weights, dtype=float)
        real = X[weights == 1]
        self.mean = real.mean(axis=0)
        self.std = real.std(axis=0)
        self.std[self.std < 1e-10] = 1.0
        operating = {
            "temp", "stress", "operating_severity",
            "stress_temperature_interaction", "inverse_temperature",
        }
        self.axes = np.array([
            1 / np.sqrt(19) if name in COMPOSITION_COLS
            else 1 / np.sqrt(5) if name in operating
            else 1 / np.sqrt(10)
            for name in names
        ])
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(0.0, self.scale, (X.shape[1], self.hidden))
        self.bias = rng.uniform(-np.pi, np.pi, self.hidden)
        hidden = self._hidden(X)
        self.hidden_mean = np.average(hidden, axis=0, weights=weights)
        self.hidden_std = np.sqrt(np.average((hidden - self.hidden_mean) ** 2, axis=0, weights=weights))
        self.hidden_std[self.hidden_std < 1e-8] = 1.0
        hidden = (hidden - self.hidden_mean) / self.hidden_std
        self.base = float(np.average(y, weights=weights))
        weighted = hidden * np.sqrt(weights[:, None])
        self.coef = np.linalg.solve(
            weighted.T @ weighted + self.ridge * np.eye(self.hidden),
            hidden.T @ (weights * (np.asarray(y) - self.base)),
        )
        return self

    def predict(self, X):
        hidden = (self._hidden(X) - self.hidden_mean) / self.hidden_std
        return self.base + hidden @ self.coef


class LifetimeEnsemble:
    """기본 앙상블과 검증에서 선택된 추가 모델을 결합한다."""

    def __init__(self, base_model, extra_models, weights, intercept=0.0, slope=1.0):
        self.base_model = base_model
        self.extra_models = extra_models
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)
        self.slope = float(slope)

    def predict(self, frame):
        names = self.base_model.tree_bundle["features"]
        X = features(frame)[names].to_numpy(float)
        predictions = [self.base_model.predict(frame)]
        predictions.extend(model.predict(X) for model in self.extra_models)
        blended = np.column_stack(predictions) @ self.weights
        return self.intercept + self.slope * blended


def forward_blend(matrix, target, rounds=5):
    """내부 검증 오차만 사용해 음수 없는 결합 가중치를 선택한다."""
    single_losses = np.mean((matrix - target[:, None]) ** 2, axis=0)
    weights = np.zeros(matrix.shape[1])
    weights[int(np.argmin(single_losses))] = 1.0
    for _ in range(rounds):
        old = weights.copy()
        best = np.mean((target - matrix @ weights) ** 2)
        for column in range(matrix.shape[1]):
            for alpha in np.linspace(0.0, 0.5, 51):
                trial = old * (1.0 - alpha)
                trial[column] += alpha
                loss = np.mean((target - matrix @ trial) ** 2)
                if loss < best - 1e-12:
                    best, weights = loss, trial
        if np.allclose(old, weights):
            break
    return weights


def predict_base_validation(X, y, sample_weights, names, frame, centers):
    """내부 학습 세트만 사용해 누수 없는 기본 모델 검증 예측을 만든다."""
    smoke = centers < 600
    return train_base_ensemble(
        X, y, sample_weights, names, smoke=smoke, final_fit=False
    ).predict(frame)


def candidate_specs(smoke):
    centers = 50 if smoke else 600
    return [
        {"kind": "lmp_rbf", "constant": 15.0, "gamma": 0.3, "ridge": 0.003, "seed": 142, "centers": centers},
        {"kind": "lmp_rbf", "constant": 20.0, "gamma": 0.3, "ridge": 0.003, "seed": 242, "centers": centers},
        {"kind": "lmp_rbf", "constant": 25.0, "gamma": 0.5, "ridge": 0.01, "seed": 142, "centers": centers},
        {"kind": "lmp_rbf", "constant": 20.0, "gamma": 0.8, "ridge": 0.03, "seed": 242, "centers": centers},
        {"kind": "life_rbf", "gamma": 0.3, "ridge": 0.003, "seed": 142, "centers": centers},
        {"kind": "life_rbf", "gamma": 0.5, "ridge": 0.01, "seed": 242, "centers": centers},
        {"kind": "random_feature", "activation": "tanh", "scale": 0.3, "ridge": 3.0, "seed": 342, "hidden": centers},
        {"kind": "random_feature", "activation": "tanh", "scale": 0.7, "ridge": 10.0, "seed": 442, "hidden": centers},
        {"kind": "random_feature", "activation": "relu", "scale": 0.5, "ridge": 3.0, "seed": 542, "hidden": centers},
        {"kind": "random_feature", "activation": "relu", "scale": 1.0, "ridge": 10.0, "seed": 642, "hidden": centers},
        {"kind": "knn", "neighbors": 40, "bandwidth": 0.5, "composition_weight": 1.5, "condition_weight": 1.0},
        {"kind": "knn", "neighbors": 120, "bandwidth": 0.7, "composition_weight": 1.0, "condition_weight": 1.5},
    ]


def build_model(spec, X, y, sample_weights, names):
    if spec["kind"] in {"lmp_rbf", "life_rbf"}:
        model = TransformedRBFRegressor(
            target="lmp" if spec["kind"] == "lmp_rbf" else "life",
            lmp_constant=spec.get("constant", 20.0),
            gamma=spec["gamma"],
            ridge=spec["ridge"],
            seed=spec["seed"],
            centers=spec["centers"],
        )
    elif spec["kind"] == "knn":
        model = KernelNeighborRegressor(
            neighbors=spec["neighbors"],
            bandwidth=spec["bandwidth"],
            composition_weight=spec["composition_weight"],
            condition_weight=spec["condition_weight"],
        )
    else:
        model = RandomFeatureRegressor(
            hidden=spec["hidden"], scale=spec["scale"], ridge=spec["ridge"],
            activation=spec["activation"], seed=spec["seed"],
        )
    return model.fit(X, y, sample_weights, names)


# ---------------------------------------------------------------------------
# 5. 전체 학습, 내부 검증 선택, 저장, 외부 테스트 평가
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """데이터 분할, 모델 선택, 최종 학습과 저장을 순서대로 수행한다."""
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    artifact_path = Path(args.artifact_path)
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
        artifact_path = artifact_path.with_name(artifact_path.stem + "_smoke.pkl")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset(use_scaler=False)
    outer = group_holdout_split(data.X, data.y, data.groups, args.test_size, args.seed)
    inner = group_holdout_split(
        outer.X_train, outer.y_train, outer.groups_train, args.validation_size, args.seed + 17
    )
    inner_X, inner_y, inner_w, inner_aug, _ = augmented(
        inner.X_train, inner.y_train, inner.groups_train, args.seed + 31
    )
    names = list(features(data.X).columns)
    validation_X = features(inner.X_test)[names].to_numpy(float)
    validation_y = np.asarray(inner.y_test, dtype=float)
    centers = 50 if args.smoke else 600

    base_prediction = predict_base_validation(
        inner_X, inner_y, inner_w, names, inner.X_test, centers
    )
    predictions = [base_prediction]
    records = [{"name": "base_ensemble", "validation": metrics(validation_y, base_prediction)}]
    specs = candidate_specs(args.smoke)
    for index, spec in enumerate(specs):
        model = build_model(spec, inner_X, inner_y, inner_w, names)
        prediction = model.predict(validation_X)
        record = {"name": f"candidate_{index}", **spec, "validation": metrics(validation_y, prediction)}
        records.append(record)
        predictions.append(prediction)
        print("CANDIDATE=" + json.dumps(record), flush=True)

    matrix = np.column_stack(predictions)
    weights = forward_blend(matrix, validation_y)
    raw_validation = matrix @ weights
    slope = float(np.cov(raw_validation, validation_y, ddof=0)[0, 1] / np.var(raw_validation))
    intercept = float(validation_y.mean() - slope * raw_validation.mean())
    calibrated = intercept + slope * raw_validation
    if metrics(validation_y, calibrated)["rmse"] >= metrics(validation_y, raw_validation)["rmse"] - 0.001:
        intercept, slope, calibrated = 0.0, 1.0, raw_validation

    selection = {
        "records": records,
        "weights": weights.tolist(),
        "intercept": intercept,
        "slope": slope,
        "validation": metrics(validation_y, calibrated),
        "inner_original_rows": int(len(inner.X_train)),
        "inner_synthetic_rows": int(inner_aug.synthetic_count),
        "seeds": {"outer_split": 42, "inner_split": 59, "inner_augmentation": 73},
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("SELECTION=" + json.dumps(selection), flush=True)

    outer_X, outer_y, outer_w, outer_aug, _ = augmented(
        outer.X_train, outer.y_train, outer.groups_train, args.seed + 53
    )
    base_ensemble = train_base_ensemble(
        outer_X, outer_y, outer_w, names, smoke=args.smoke, final_fit=True
    )
    chosen_models = []
    chosen_weights = [weights[0]]
    for index, spec in enumerate(specs, start=1):
        if weights[index] <= 1e-12:
            continue
        model = build_model(spec, outer_X, outer_y, outer_w, names)
        if isinstance(model, TransformedRBFRegressor):
            model.compact()
        chosen_models.append(model)
        chosen_weights.append(weights[index])
    ensemble = LifetimeEnsemble(
        base_ensemble, chosen_models, chosen_weights, intercept=intercept, slope=slope
    )
    with artifact_path.open("wb") as stream:
        pickle.dump(ensemble, stream)

    test_y = np.asarray(outer.y_test, dtype=float)
    test_prediction = ensemble.predict(outer.X_test)
    with artifact_path.open("rb") as stream:
        np.testing.assert_allclose(pickle.load(stream).predict(outer.X_test), test_prediction)
    result = {
        "test": metrics(test_y, test_prediction),
        "validation": selection["validation"],
        "weights": chosen_weights,
        "intercept": intercept,
        "slope": slope,
        "outer_original_rows": int(len(outer.X_train)),
        "outer_synthetic_rows": int(outer_aug.synthetic_count),
        "test_rows": int(len(outer.X_test)),
        "seconds": time.perf_counter() - started,
        "smoke": args.smoke,
        "seeds": {
            "outer_split": args.seed,
            "inner_split": args.seed + 17,
            "inner_augmentation": args.seed + 31,
            "outer_augmentation": args.seed + 53,
        },
        "artifact_path": str(artifact_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "group": outer.groups_test,
        "actual_log": test_y,
        "predicted_log": test_prediction,
        "actual_hours": 10.0 ** test_y,
        "predicted_hours": 10.0 ** test_prediction,
    }).to_csv(output_dir / "predictions.csv", index=False)
    print("RESULT=" + json.dumps(result), flush=True)


@dataclass
class PredictionResult:
    log_lifetime: float
    lifetime_hours: float
    lifetime_years: float


class CustomHistogramPredictor:
    """저장된 앙상블을 DataFrame 또는 단일 조건 입력에 적용한다."""

    def __init__(self, model: LifetimeEnsemble, artifact_path: Path) -> None:
        self.model = model
        self.artifact_path = artifact_path
        self.feature_names = list(model.base_model.tree_bundle["features"])

    @property
    def is_smoke_model(self) -> bool:
        return self.artifact_path.stem.endswith("_smoke")

    def _prepare_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        prepared = rows.copy()
        required = ["stress", "temp", *COMPOSITION_COLS, *HEAT_TREATMENT_COLS]
        for column in required:
            if column not in prepared:
                prepared[column] = 0.0
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
        for prefix in ("N", "T", "A"):
            temp_col, time_col = f"{prefix}temp", f"{prefix}time"
            severity_col = f"{prefix}_severity"
            if severity_col not in prepared:
                safe_time = np.maximum(prepared[time_col].to_numpy(float), 1e-6)
                severity = prepared[temp_col].to_numpy(float) * (20.0 + np.log10(safe_time))
                prepared[severity_col] = np.where(prepared[temp_col] > 0, severity, 0.0)
        return prepared

    def predict_dataframe(self, rows: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_features(rows)
        log_lifetime = self.model.predict(prepared)
        lifetime_hours = 10.0 ** log_lifetime
        return pd.DataFrame({
            "log_lifetime": log_lifetime,
            "lifetime_hours": lifetime_hours,
            "lifetime_years": lifetime_hours / 8760.0,
        })

    def predict_one(self, stress, temp, composition, heat_treatment) -> PredictionResult:
        row = {"stress": float(stress), "temp": float(temp)}
        row.update({key: float(value) for key, value in composition.items()})
        row.update({key: float(value) for key, value in heat_treatment.items()})
        prediction = self.predict_dataframe(pd.DataFrame([row])).iloc[0]
        return PredictionResult(
            log_lifetime=float(prediction["log_lifetime"]),
            lifetime_hours=float(prediction["lifetime_hours"]),
            lifetime_years=float(prediction["lifetime_years"]),
        )


def load_custom_histogram_predictor(artifact_path=None) -> CustomHistogramPredictor:
    """학습된 모델을 CPU 추론용 객체로 불러온다."""
    path = Path(artifact_path) if artifact_path else MODEL_DIR / "custom_histogram.pkl"
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    return CustomHistogramPredictor(model, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Histogram boosting + radial basis 앙상블")
    parser.add_argument("--test-size", type=float, default=0.2, help="테스트 비율")
    parser.add_argument("--validation-size", type=float, default=0.15, help="내부 검증 비율")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--smoke", action="store_true", help="빠른 실행 확인용 축소 설정")
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(MODEL_DIR / "custom_histogram.pkl"),
        help="학습된 모델 저장 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODEL_DIR / "custom_histogram_output"),
        help="지표와 테스트 예측 저장 폴더",
    )
    return parser.parse_args()


if __name__ == "__main__":
    module_name = "models.custom_histogram"
    sys.modules[module_name] = sys.modules[__name__]
    for serializable_class in (
        HistogramTree,
        HistogramBoosting,
        RadialBasisRegressor,
        MixedEnsemble,
        TransformedRBFRegressor,
        KernelNeighborRegressor,
        RandomFeatureRegressor,
        LifetimeEnsemble,
    ):
        serializable_class.__module__ = module_name
    run_pipeline(parse_args())
