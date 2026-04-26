"""
Transformer + 트리 앙상블 기반 크립 수명 예측 모델.

- 전처리 파이프라인은 데이터전처리.py를 사용
- Transformer 인코더와 트리 기반 잔차 보정 앙상블 모델
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "데이터전처리.py").exists() else SCRIPT_DIR.parent
MODEL_DIR = ROOT_DIR / "models"

sys.path.insert(0, str(ROOT_DIR))
from 데이터전처리 import ( 
    COMPOSITION_COLS,
    CONDITION_COLS,
    HEAT_TREATMENT_COLS,
    prepare_dataset,
)


PHYSICS_FEATURES = [
    "operating_severity",
    "stress_temperature_interaction",
    "inverse_temperature",
    "total_heat_treatment_severity",
]

FEATURE_KR = {
    "stress": "응력",
    "temp": "온도",
    "operating_severity": "운전 가혹도 지수",
    "stress_temperature_interaction": "응력-온도 상호작용",
    "inverse_temperature": "역온도",
    "total_heat_treatment_severity": "총 열처리 가혹도",
    "N_severity": "노말라이징 가혹도",
    "T_severity": "템퍼링 가혹도",
    "A_severity": "어닐링 가혹도",
}


class TerminalLogger:
    """터미널에 출력한 내용을 같은 형태로 로그 파일에도 저장한다."""

    def __init__(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self._file = output_path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    val_rmse_log: List[float]
    best_epoch: int
    best_val_rmse_log: float


class StandardScalerCustom:
    """sklearn 의존성을 피하기 위해 직접 구현한 표준화 스케일러."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "StandardScalerCustom":
        arr = np.asarray(X, dtype=np.float64)
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("스케일러가 아직 학습되지 않았습니다.")
        arr = np.asarray(X, dtype=np.float64)
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TargetScalerCustom(StandardScalerCustom):
    def fit_transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y.reshape(-1, 1)).transform(y.reshape(-1, 1)).reshape(-1)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return self.transform(y.reshape(-1, 1)).reshape(-1)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("타깃 스케일러가 아직 학습되지 않았습니다.")
        arr = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
        return (arr * self.scale_ + self.mean_).reshape(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """타깃 누수 없이 사용할 수 있는 물리 기반 파생 변수를 추가한다.

    LMP는 log10(수명)을 포함하므로 입력 피처로 사용하지 않는다.
    LMP는 물리 반응 검증 단계에서만 사후 계산한다.
    """
    out = X.copy()
    stress = pd.to_numeric(out["stress"], errors="coerce").fillna(0.0).clip(lower=1e-9)
    temp = pd.to_numeric(out["temp"], errors="coerce").fillna(0.0).clip(lower=1e-9)

    out["operating_severity"] = (temp / 1000.0) * np.log10(stress + 1.0)
    out["stress_temperature_interaction"] = (stress * temp) / 1000.0
    out["inverse_temperature"] = 1000.0 / temp

    severity_cols = [col for col in ["N_severity", "T_severity", "A_severity"] if col in out.columns]
    if severity_cols:
        out["total_heat_treatment_severity"] = out[severity_cols].sum(axis=1)
    else:
        heat_cols = [col for col in HEAT_TREATMENT_COLS if col in out.columns]
        out["total_heat_treatment_severity"] = out[heat_cols].sum(axis=1)

    return out


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
    selected: List[object] = []
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


def infer_feature_group_ids(feature_names: List[str]) -> List[int]:
    condition = set(CONDITION_COLS)
    composition = set(COMPOSITION_COLS)
    physics = set(PHYSICS_FEATURES)

    group_ids: List[int] = []
    for name in feature_names:
        if name in condition:
            group_ids.append(0)
        elif name in composition:
            group_ids.append(1)
        elif name in physics:
            group_ids.append(3)
        else:
            group_ids.append(2)
    return group_ids


def group_summary(group_ids: List[int]) -> Dict[str, int]:
    return {
        "운전조건": int(sum(g == 0 for g in group_ids)),
        "합금조성": int(sum(g == 1 for g in group_ids)),
        "열처리": int(sum(g == 2 for g in group_ids)),
        "물리파생": int(sum(g == 3 for g in group_ids)),
    }


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class FeatureTokenizer(nn.Module):
    """각 수치 피처를 Transformer 입력 토큰으로 변환한다."""

    def __init__(self, n_features: int, d_model: int, group_ids: List[int], dropout: float) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.empty(n_features, d_model))
        self.group_embedding = nn.Embedding(max(group_ids) + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("group_ids", torch.tensor(group_ids, dtype=torch.long))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value_tokens = self.value_projection(x.unsqueeze(-1))
        feature_tokens = self.feature_embedding.unsqueeze(0)
        group_tokens = self.group_embedding(self.group_ids).unsqueeze(0)
        return self.dropout(value_tokens + feature_tokens + group_tokens)


class ManualMultiHeadSelfAttention(nn.Module):
    """nn.TransformerEncoder를 쓰지 않고 직접 구현한 다중 헤드 자기어텐션."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(score, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tokens, self.d_model)
        return self.out_projection(out)


class ManualTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = ManualMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class CustomTransformerRegressor(nn.Module):
    """log10 크리프 수명 예측을 위한 정형 데이터용 Transformer."""

    def __init__(
        self,
        n_features: int,
        group_ids: List[int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model, group_ids, dropout)
        self.blocks = nn.ModuleList(
            [ManualTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.pool_score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.numeric_shortcut = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)

        weight = torch.softmax(self.pool_score(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        embedding = torch.sum(tokens * weight, dim=1)
        pred = self.prediction_head(embedding).squeeze(1)
        pred = pred + self.numeric_shortcut(x).squeeze(1)

        if return_embedding:
            return pred, embedding
        return pred


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + 0.05 * F.smooth_l1_loss(pred, target)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 1e-12:
        return float("nan")
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def evaluate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_hours = np.power(10.0, y_true_log)
    y_pred_hours = np.power(10.0, y_pred_log)
    return {
        "rmse_log": rmse_np(y_true_log, y_pred_log),
        "r2_log": r2_np(y_true_log, y_pred_log),
        "rmse_hours": rmse_np(y_true_hours, y_pred_hours),
        "r2_hours": r2_np(y_true_hours, y_pred_hours),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: TargetScalerCustom,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    pred_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(float(regression_loss(pred, yb).item()))
            pred_chunks.append(pred.detach().cpu().numpy())
            target_chunks.append(yb.detach().cpu().numpy())

    pred_log = target_scaler.inverse_transform_target(np.concatenate(pred_chunks))
    target_log = target_scaler.inverse_transform_target(np.concatenate(target_chunks))
    return float(np.mean(losses)), rmse_np(target_log, pred_log)


def train_transformer(
    model: CustomTransformerRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: TargetScalerCustom,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    logger: TerminalLogger,
    verbose_every: int,
) -> TrainHistory:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_rmse = float("inf")
    stale = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_rmses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = regression_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss, val_rmse = evaluate_loader(model, val_loader, target_scaler, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if verbose_every > 0 and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
            logger.log(
                f"[Transformer] epoch={epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.6f} | val_rmse_log={val_rmse:.6f}"
            )

        if stale >= patience:
            logger.log(f"[조기 종료] best_epoch={best_epoch}, best_val_rmse_log={best_rmse:.6f}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return TrainHistory(train_losses, val_losses, val_rmses, best_epoch, best_rmse)


def train_transformer_fixed_epochs(
    model: CustomTransformerRegressor,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    logger: TerminalLogger,
    verbose_every: int,
) -> CustomTransformerRegressor:
    loader = make_loader(X, y, batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = regression_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        if verbose_every > 0 and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
            logger.log(f"[최종 Transformer] epoch={epoch:03d}/{epochs} | loss={float(np.mean(batch_losses)):.6f}")

    return model


def predict_transformer(
    model: CustomTransformerRegressor,
    X_scaled: np.ndarray,
    target_scaler: TargetScalerCustom,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    pred_chunks: List[np.ndarray] = []
    embedding_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred_scaled, embedding = model(xb, return_embedding=True)
            pred_chunks.append(pred_scaled.detach().cpu().numpy())
            embedding_chunks.append(embedding.detach().cpu().numpy())

    pred_log = target_scaler.inverse_transform_target(np.concatenate(pred_chunks))
    embedding = np.concatenate(embedding_chunks, axis=0)
    return pred_log, embedding


@dataclass
class TreeNode:
    value: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None or self.left is None or self.right is None


class CustomRegressionTree:
    """NumPy로 직접 구현한 CART 방식 회귀트리."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 12,
        feature_subsample: float = 0.75,
        max_bins: int = 24,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_subsample = feature_subsample
        self.max_bins = max_bins
        self.random_state = random_state
        self.root_: Optional[TreeNode] = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomRegressionTree":
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float64)
        self.root_ = self._build(X_np, y_np, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("회귀트리가 아직 학습되지 않았습니다.")
        X_np = np.asarray(X, dtype=np.float32)
        return np.array([self._predict_one(row, self.root_) for row in X_np], dtype=np.float64)

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.value

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node_value = float(np.mean(y))
        node = TreeNode(value=node_value)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_leaf * 2
            or float(np.var(y)) < 1e-12
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float]]:
        n_samples, n_features = X.shape
        n_candidate_features = max(1, int(round(n_features * self.feature_subsample)))
        feature_indices = self.rng_.choice(n_features, size=n_candidate_features, replace=False)

        parent_sse = self._sse(y)
        best_gain = 1e-12
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for feature_index in feature_indices:
            col = X[:, feature_index]
            if float(np.max(col) - np.min(col)) <= 1e-12:
                continue

            quantiles = np.linspace(0.05, 0.95, self.max_bins)
            thresholds = np.unique(np.quantile(col, quantiles))
            for threshold in thresholds:
                left_mask = col <= threshold
                left_count = int(left_mask.sum())
                right_count = n_samples - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]
                right_y = y[~left_mask]
                gain = parent_sse - self._sse(left_y) - self._sse(right_y)
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        centered = y - float(np.mean(y))
        return float(np.sum(centered * centered))


class CustomTreeEnsemble:

    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        min_samples_leaf: int,
        feature_subsample: float,
        max_bins: int,
        random_state: int,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_subsample = feature_subsample
        self.max_bins = max_bins
        self.random_state = random_state
        self.trees_: List[CustomRegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray, logger: Optional[TerminalLogger] = None) -> "CustomTreeEnsemble":
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n_samples = len(y_np)
        self.trees_ = []

        for tree_idx in range(self.n_trees):
            sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = CustomRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_subsample=self.feature_subsample,
                max_bins=self.max_bins,
                random_state=self.random_state + tree_idx,
            )
            tree.fit(X_np[sample_idx], y_np[sample_idx])
            self.trees_.append(tree)

            if logger is not None and (tree_idx == 0 or (tree_idx + 1) % 10 == 0 or tree_idx + 1 == self.n_trees):
                logger.log(f"[트리 앙상블] {tree_idx + 1:03d}/{self.n_trees}개 트리 학습 완료")

        return self

    def predict_members(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("트리 앙상블이 아직 학습되지 않았습니다.")
        return np.vstack([tree.predict(X) for tree in self.trees_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_members(X).mean(axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        members = self.predict_members(X)
        return members.std(axis=0)


def make_correction_features(
    X_scaled: np.ndarray,
    embedding: np.ndarray,
    transformer_pred_log: np.ndarray,
) -> np.ndarray:
    return np.hstack(
        [
            np.asarray(X_scaled, dtype=np.float32),
            np.asarray(embedding, dtype=np.float32),
            np.asarray(transformer_pred_log, dtype=np.float32).reshape(-1, 1),
        ]
    )


def calibrate_correction_weight(
    y_val: np.ndarray,
    transformer_pred: np.ndarray,
    residual_pred: np.ndarray,
) -> Tuple[float, float]:
    best_weight = 1.0
    best_rmse = float("inf")
    for weight in np.linspace(0.0, 1.25, 26):
        pred = transformer_pred + weight * residual_pred
        rmse = rmse_np(y_val, pred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight, best_rmse


def predict_full_ensemble(
    model: CustomTransformerRegressor,
    tree_ensemble: CustomTreeEnsemble,
    scaler: StandardScalerCustom,
    target_scaler: TargetScalerCustom,
    feature_names: List[str],
    correction_weight: float,
    X_raw: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    X_aug = add_physics_features(X_raw)[feature_names]
    X_scaled = scaler.transform(X_aug)
    transformer_pred, embedding = predict_transformer(
        model,
        X_scaled,
        target_scaler,
        device,
        batch_size,
    )
    correction_X = make_correction_features(X_scaled, embedding, transformer_pred)
    residual_pred = tree_ensemble.predict(correction_X)
    residual_std = tree_ensemble.predict_std(correction_X)
    ensemble_pred = transformer_pred + correction_weight * residual_pred
    return {
        "transformer_pred": transformer_pred,
        "residual_pred": residual_pred,
        "residual_std": residual_std,
        "ensemble_pred": ensemble_pred,
    }


def larson_miller_parameter(temp_k: np.ndarray, log_lifetime: np.ndarray, constant: float = 20.0) -> np.ndarray:
    return temp_k * (constant + log_lifetime) / 1000.0


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rank_b = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(rank_a) < 1e-12 or np.std(rank_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def permutation_importance_physics(
    model: CustomTransformerRegressor,
    tree_ensemble: CustomTreeEnsemble,
    scaler: StandardScalerCustom,
    target_scaler: TargetScalerCustom,
    feature_names: List[str],
    correction_weight: float,
    X_raw: pd.DataFrame,
    y_true_log: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> List[Tuple[str, float]]:
    baseline = predict_full_ensemble(
        model,
        tree_ensemble,
        scaler,
        target_scaler,
        feature_names,
        correction_weight,
        X_raw,
        device,
        batch_size,
    )["ensemble_pred"]
    baseline_rmse = rmse_np(y_true_log, baseline)

    X_aug = add_physics_features(X_raw)
    candidates = [
        "temp",
        "stress",
        "operating_severity",
        "stress_temperature_interaction",
        "inverse_temperature",
        "total_heat_treatment_severity",
        "N_severity",
        "T_severity",
        "A_severity",
    ]
    candidates = [name for name in candidates if name in X_aug.columns]
    rng = np.random.default_rng(seed)
    rows: List[Tuple[str, float]] = []

    for feature in candidates:
        permuted_aug = X_aug.copy()
        permuted_aug[feature] = rng.permutation(permuted_aug[feature].to_numpy())
        X_scaled = scaler.transform(permuted_aug[feature_names])
        transformer_pred, embedding = predict_transformer(
            model,
            X_scaled,
            target_scaler,
            device,
            batch_size,
        )
        correction_X = make_correction_features(X_scaled, embedding, transformer_pred)
        residual_pred = tree_ensemble.predict(correction_X)
        pred = transformer_pred + correction_weight * residual_pred
        rows.append((feature, rmse_np(y_true_log, pred) - baseline_rmse))

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def scenario_sweep(
    model: CustomTransformerRegressor,
    tree_ensemble: CustomTreeEnsemble,
    scaler: StandardScalerCustom,
    target_scaler: TargetScalerCustom,
    feature_names: List[str],
    correction_weight: float,
    X_raw: pd.DataFrame,
    sweep_feature: str,
    device: torch.device,
    batch_size: int,
    fixed_updates: Optional[Dict[str, float]] = None,
    n_points: int = 40,
) -> Tuple[float, float, float]:
    fixed_updates = fixed_updates or {}
    base = X_raw.median(numeric_only=True).to_frame().T
    values = np.linspace(
        float(X_raw[sweep_feature].quantile(0.05)),
        float(X_raw[sweep_feature].quantile(0.95)),
        n_points,
    )
    scenario = pd.concat([base] * n_points, ignore_index=True)
    scenario[sweep_feature] = values
    for key, value in fixed_updates.items():
        scenario[key] = value

    pred = predict_full_ensemble(
        model,
        tree_ensemble,
        scaler,
        target_scaler,
        feature_names,
        correction_weight,
        scenario,
        device,
        batch_size,
    )["ensemble_pred"]
    slope = float(np.polyfit(values, pred, deg=1)[0])
    return slope, float(pred[0]), float(pred[-1])


def run_pipeline(args: argparse.Namespace) -> None:
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.n_trees = min(args.n_trees, 8)
        args.d_model = min(args.d_model, 32)
        args.n_layers = min(args.n_layers, 1)
        args.patience = min(args.patience, 3)
        args.verbose_every = 1

    logger = TerminalLogger(Path(args.output_path))
    started_at = time.time()

    try:
        logger.log("=" * 72)
        logger.log("Transformer + 트리 기반 앙상블 크리프 수명 예측")
        logger.log("=" * 72)
        logger.log("Transformer 구현: 다중 헤드 자기어텐션 + 인코더 블록")
        logger.log("트리 앙상블 구현: CART 회귀트리 + 부트스트랩 앙상블")
        logger.log("전처리 모듈: 데이터전처리.py")
        logger.log(f"출력 저장 파일: {Path(args.output_path).resolve()}")
        logger.log("")

        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"실행 장치: {device}")

        dataset = prepare_dataset(rounding_decimals=args.rounding, use_scaler=False)
        raw_outer = group_holdout_split(
            dataset.X,
            dataset.y,
            dataset.groups,
            test_size=args.test_size,
            seed=args.seed,
        )
        raw_inner = group_holdout_split(
            raw_outer.X_train,
            raw_outer.y_train,
            raw_outer.groups_train,
            test_size=args.val_size,
            seed=args.seed + 17,
        )

        X_inner_train_aug = add_physics_features(raw_inner.X_train)
        X_inner_val_aug = add_physics_features(raw_inner.X_test)
        X_full_train_aug = add_physics_features(raw_outer.X_train)
        X_test_aug = add_physics_features(raw_outer.X_test)

        feature_names = list(X_full_train_aug.columns)
        group_ids = infer_feature_group_ids(feature_names)

        logger.log("[데이터 분할]")
        logger.log(f"전체 샘플: {len(dataset.X)}")
        logger.log(f"최종 학습 샘플: {len(raw_outer.X_train)} | 테스트 샘플: {len(raw_outer.X_test)}")
        logger.log(f"내부 학습 샘플: {len(raw_inner.X_train)} | 검증 샘플: {len(raw_inner.X_test)}")
        logger.log(f"피처 수: {len(feature_names)}")
        logger.log(f"피처 그룹: {group_summary(group_ids)}")
        logger.log("")

        inner_scaler = StandardScalerCustom()
        X_inner_train = inner_scaler.fit_transform(X_inner_train_aug[feature_names])
        X_inner_val = inner_scaler.transform(X_inner_val_aug[feature_names])
        inner_target_scaler = TargetScalerCustom()
        y_inner_train = inner_target_scaler.fit_transform_target(raw_inner.y_train.to_numpy(dtype=float))
        y_inner_val = inner_target_scaler.transform_target(raw_inner.y_test.to_numpy(dtype=float))

        selector_model = CustomTransformerRegressor(
            n_features=X_inner_train.shape[1],
            group_ids=group_ids,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        train_loader = make_loader(X_inner_train, y_inner_train, args.batch_size, shuffle=True)
        val_loader = make_loader(X_inner_val, y_inner_val, args.batch_size, shuffle=False)

        logger.log("[1단계] Transformer 변수 상호작용 학습")
        history = train_transformer(
            model=selector_model,
            train_loader=train_loader,
            val_loader=val_loader,
            target_scaler=inner_target_scaler,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            logger=logger,
            verbose_every=args.verbose_every,
        )
        logger.log(f"Transformer 내부 검증 best_epoch: {history.best_epoch}")
        logger.log(f"Transformer 내부 검증 best RMSE(log10): {history.best_val_rmse_log:.6f}")
        logger.log("")

        logger.log("[2단계] 직접 구현 트리 앙상블로 Transformer 잔차 보정")
        inner_train_pred, inner_train_embedding = predict_transformer(
            selector_model,
            X_inner_train,
            inner_target_scaler,
            device,
            args.batch_size,
        )
        inner_val_pred, inner_val_embedding = predict_transformer(
            selector_model,
            X_inner_val,
            inner_target_scaler,
            device,
            args.batch_size,
        )
        inner_train_correction_X = make_correction_features(X_inner_train, inner_train_embedding, inner_train_pred)
        inner_val_correction_X = make_correction_features(X_inner_val, inner_val_embedding, inner_val_pred)
        inner_residual = raw_inner.y_train.to_numpy(dtype=float) - inner_train_pred

        selector_tree = CustomTreeEnsemble(
            n_trees=args.n_trees,
            max_depth=args.tree_depth,
            min_samples_leaf=args.min_samples_leaf,
            feature_subsample=args.feature_subsample,
            max_bins=args.max_bins,
            random_state=args.seed,
        ).fit(inner_train_correction_X, inner_residual, logger=logger)

        val_residual_pred = selector_tree.predict(inner_val_correction_X)
        correction_weight, weighted_val_rmse = calibrate_correction_weight(
            raw_inner.y_test.to_numpy(dtype=float),
            inner_val_pred,
            val_residual_pred,
        )
        logger.log(f"잔차 보정 계수: {correction_weight:.3f}")
        logger.log(f"보정 후 내부 검증 RMSE(log10): {weighted_val_rmse:.6f}")
        logger.log("")

        logger.log("[3단계] 최종 학습 세트 전체로 재학습")
        final_scaler = StandardScalerCustom()
        X_full_train = final_scaler.fit_transform(X_full_train_aug[feature_names])
        X_test = final_scaler.transform(X_test_aug[feature_names])
        final_target_scaler = TargetScalerCustom()
        y_full_train = final_target_scaler.fit_transform_target(raw_outer.y_train.to_numpy(dtype=float))

        final_model = CustomTransformerRegressor(
            n_features=X_full_train.shape[1],
            group_ids=group_ids,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        final_epochs = max(1, history.best_epoch)
        final_model = train_transformer_fixed_epochs(
            model=final_model,
            X=X_full_train,
            y=y_full_train,
            device=device,
            epochs=final_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            logger=logger,
            verbose_every=args.verbose_every,
        )

        train_pred, train_embedding = predict_transformer(
            final_model,
            X_full_train,
            final_target_scaler,
            device,
            args.batch_size,
        )
        train_correction_X = make_correction_features(X_full_train, train_embedding, train_pred)
        train_residual = raw_outer.y_train.to_numpy(dtype=float) - train_pred

        final_tree = CustomTreeEnsemble(
            n_trees=args.n_trees,
            max_depth=args.tree_depth,
            min_samples_leaf=args.min_samples_leaf,
            feature_subsample=args.feature_subsample,
            max_bins=args.max_bins,
            random_state=args.seed + 101,
        ).fit(train_correction_X, train_residual, logger=logger)

        test_pred = predict_full_ensemble(
            final_model,
            final_tree,
            final_scaler,
            final_target_scaler,
            feature_names,
            correction_weight,
            raw_outer.X_test,
            device,
            args.batch_size,
        )
        y_test_log = raw_outer.y_test.to_numpy(dtype=float)
        transformer_metrics = evaluate_metrics(y_test_log, test_pred["transformer_pred"])
        ensemble_metrics = evaluate_metrics(y_test_log, test_pred["ensemble_pred"])

        logger.log("")
        logger.log("[최종 테스트 성능]")
        logger.log(f"최종 RMSE(log10): {ensemble_metrics['rmse_log']:.6f}")
        logger.log(f"최종 R2(log10): {ensemble_metrics['r2_log']:.6f}")
        logger.log(f"최종 RMSE(hours): {ensemble_metrics['rmse_hours']:.3f}")
        logger.log(f"트리 앙상블 평균 잔차 불확실성(log10 std): {float(np.mean(test_pred['residual_std'])):.6f}")
        logger.log("")

        logger.log("[물리 기반 변수 반응 및 해석 가능성 검증]")
        temp = raw_outer.X_test["temp"].to_numpy(dtype=float)
        lmp_actual = larson_miller_parameter(temp, y_test_log)
        lmp_pred = larson_miller_parameter(temp, test_pred["ensemble_pred"])
        logger.log(f"LMP RMSE: {rmse_np(lmp_actual, lmp_pred):.6f}")
        logger.log(f"LMP R2: {r2_np(lmp_actual, lmp_pred):.6f}")

        test_aug = add_physics_features(raw_outer.X_test)
        severity_corr = spearman_corr(
            test_aug["operating_severity"].to_numpy(dtype=float),
            test_pred["ensemble_pred"],
        )
        logger.log(f"운전 가혹도-예측 수명 Spearman 상관: {severity_corr:.6f}")

        high_temp = float(raw_outer.X_test["temp"].quantile(0.75))
        temp_slope, temp_low_pred, temp_high_pred = scenario_sweep(
            final_model,
            final_tree,
            final_scaler,
            final_target_scaler,
            feature_names,
            correction_weight,
            raw_outer.X_test,
            "temp",
            device,
            args.batch_size,
        )
        stress_slope, stress_low_pred, stress_high_pred = scenario_sweep(
            final_model,
            final_tree,
            final_scaler,
            final_target_scaler,
            feature_names,
            correction_weight,
            raw_outer.X_test,
            "stress",
            device,
            args.batch_size,
            fixed_updates={"temp": high_temp},
        )
        logger.log(f"온도 sweep 기울기: {temp_slope:.8f} | 낮은 온도 예측={temp_low_pred:.4f}, 높은 온도 예측={temp_high_pred:.4f}")
        logger.log(f"고온 조건 응력 sweep 기울기: {stress_slope:.8f} | 낮은 응력 예측={stress_low_pred:.4f}, 높은 응력 예측={stress_high_pred:.4f}")

        importance_rows = permutation_importance_physics(
            final_model,
            final_tree,
            final_scaler,
            final_target_scaler,
            feature_names,
            correction_weight,
            raw_outer.X_test,
            y_test_log,
            device,
            args.batch_size,
            args.seed + 777,
        )
        logger.log("물리 변수 순열 중요도 상위:")
        for feature, importance in importance_rows[:5]:
            logger.log(f"  - {FEATURE_KR.get(feature, feature)}: RMSE 증가량 {importance:.6f}")
        logger.log("")
        logger.log(f"총 실행 시간: {time.time() - started_at:.1f}초")
        logger.log("=" * 72)
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="독자 구현 Transformer + 트리 앙상블")
    parser.add_argument("--test-size", type=float, default=0.2, help="테스트 비율")
    parser.add_argument("--val-size", type=float, default=0.15, help="내부 검증 비율")
    parser.add_argument("--rounding", type=int, default=3, help="조성 그룹 반올림 자리수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")

    parser.add_argument("--epochs", type=int, default=18, help="Transformer 최대 epoch")
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    parser.add_argument("--d-model", type=int, default=64, help="토큰 임베딩 차원")
    parser.add_argument("--n-heads", type=int, default=4, help="어텐션 헤드 수")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer 인코더 블록 수")
    parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="가중치 감쇠")
    parser.add_argument("--patience", type=int, default=8, help="조기 종료 patience")
    parser.add_argument("--verbose-every", type=int, default=3, help="로그 출력 주기")

    parser.add_argument("--n-trees", type=int, default=40, help="트리 개수")
    parser.add_argument("--tree-depth", type=int, default=4, help="트리 최대 깊이")
    parser.add_argument("--min-samples-leaf", type=int, default=14, help="리프 최소 샘플 수")
    parser.add_argument("--feature-subsample", type=float, default=0.75, help="노드별 피처 샘플링 비율")
    parser.add_argument("--max-bins", type=int, default=24, help="분할 후보 quantile 개수")

    parser.add_argument("--smoke", action="store_true", help="빠른 실행 확인용 축소 설정")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(MODEL_DIR / "transformer_and_tree_ensemble_output.txt"),
        help="터미널 출력 저장 파일",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
