"""
Tabular ResNet + Optuna Hyperparameter Optimization
선행 조건: python models/select_features.py 실행 후 selected_features.json 생성

Optuna 탐색 범위:
  - hidden_size : 16 ~ 32
  - num_blocks  : 2 ~ 6
  - dropout     : 0.0 ~ 0.4
  - lr          : 1e-4 ~ 1e-2 (log scale)
  - batch_size  : {32, 64, 128}
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
from 데이터전처리 import preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS    = 30
MAX_EPOCHS  = 300
PATIENCE    = 25   # early stopping patience
DB_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_study.db")
STUDY_NAME  = "resnet_creep"


# ── Architecture ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation ResNet block (BN → ReLU → Linear → BN → ReLU → Linear + skip)."""
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabResNet(nn.Module):
    """
    Tabular ResNet:
      stem  : Linear(in_features → hidden_size)
      blocks: N × ResBlock(hidden_size)
      head  : Linear(hidden_size → 1)
    """
    def __init__(self, in_features: int, hidden_size: int, num_blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Linear(in_features, hidden_size)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_size, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x))).squeeze(1)


# ── Data utilities ────────────────────────────────────────────────────────

def to_tensor(arr) -> torch.Tensor:
    if hasattr(arr, "values"):
        arr = arr.values
    return torch.tensor(arr, dtype=torch.float32)


def make_loaders(X_train, y_train, X_test, y_test, batch_size: int):
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=len(test_ds), shuffle=False)
    return train_loader, test_loader


# ── Train / eval helpers ──────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    Xb, yb = next(iter(loader))
    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
    preds   = model(Xb).cpu().numpy()
    targets = yb.cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    r2   = float(r2_score(targets, preds))
    return rmse, r2, preds, targets


# ── Optuna objective ──────────────────────────────────────────────────────

def make_objective(X_train, y_train, X_test, y_test, in_features: int):
    def objective(trial: optuna.Trial) -> float:
        # ── 탐색 공간 (hidden_size: 16~32) ──
        hidden_size = trial.suggest_int("hidden_size", 16, 32)
        num_blocks  = trial.suggest_int("num_blocks",  2,  6)
        dropout     = trial.suggest_float("dropout",   0.0, 0.4)
        lr          = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = TabResNet(in_features, hidden_size, num_blocks, dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size)

        best_val = float("inf")
        patience_cnt = 0

        for epoch in range(MAX_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            val_rmse, _, _, _ = evaluate(model, test_loader)

            if val_rmse < best_val:
                best_val = val_rmse
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val

    return objective


# ── Final training ────────────────────────────────────────────────────────

def train_final(params: dict, X_train, y_train, X_test, y_test, in_features: int):
    model = TabResNet(
        in_features,
        params["hidden_size"],
        params["num_blocks"],
        params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    final_epochs = MAX_EPOCHS * 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs)

    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test, params["batch_size"]
    )

    history = []
    best_rmse  = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(final_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        val_rmse, val_r2, _, _ = evaluate(model, test_loader)
        history.append({"epoch": epoch + 1, "train_loss": loss,
                         "val_rmse": val_rmse, "val_r2": val_r2})

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE * 2:
                break

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d} | loss={loss:.5f} | "
                  f"val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ── Visualization ─────────────────────────────────────────────────────────

def plot_results(preds, targets, history):
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Pred vs Actual
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)
    lo = min(targets.min(), preds.min()) - 0.1
    hi = max(targets.max(), preds.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="완벽 예측선")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"실제값 $\log_{10}$(수명/시간)", fontsize=11)
    ax.set_ylabel(r"예측값 $\log_{10}$(수명/시간)", fontsize=11)
    ax.set_title(f"ResNet — Pred vs Actual\nRMSE={rmse:.4f}  R²={r2:.4f}", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2. Val RMSE curve
    ax2 = axes[1]
    ax2.plot(history["epoch"], history["val_rmse"], color="#E53935")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Val RMSE (log10)", fontsize=11)
    ax2.set_title("Validation RMSE Curve", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Val R² curve
    ax3 = axes[2]
    ax3.plot(history["epoch"], history["val_r2"], color="#1E88E5")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Val R²", fontsize=11)
    ax3.set_title("Validation R² Curve", fontsize=12)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("TabResNet (Optuna 최적 하이퍼파라미터)", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "resnet_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"저장: {path}")
    plt.close(fig)

    return rmse, r2


def plot_optuna_history(study):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Optimization history
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = [min(vals[:i+1]) for i in range(len(vals))]
    axes[0].plot(vals, "o-", alpha=0.5, label="Trial RMSE", markersize=4)
    axes[0].plot(best_so_far, "r-", lw=2, label="Best so far")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val RMSE")
    axes[0].set_title("Optuna Optimization History"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parameter importance (hidden_size, num_blocks, dropout, lr, batch_size)
    importance = optuna.importance.get_param_importances(study)
    axes[1].barh(list(importance.keys()), list(importance.values()),
                 color="#5C6BC0", edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Hyperparameter Importance")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "optuna_history.png")
    fig.savefig(path, dpi=150)
    print(f"저장: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TabResNet + Optuna (hidden_size 16~32)")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    # 선택된 피처 로드
    feat_path = os.path.join(BASE_DIR, "selected_features.json")
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} 없음. 먼저 select_features.py를 실행하세요.")
        sys.exit(1)
    with open(feat_path, encoding="utf-8") as f:
        selected = json.load(f)
    print(f"사용 피처 ({len(selected)}개): {selected}\n")

    # 전처리 후 피처 서브셋
    data = preprocess(save=False)
    X_train = data["X_train"][selected]
    X_test  = data["X_test"][selected]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    in_features = len(selected)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {in_features}")

    # ── Optuna 탐색 ──
    print(f"\nOptuna 탐색 시작 ({N_TRIALS} trials)...\n")
    print(f"Study DB: {DB_PATH}")
    print(f"대시보드 명령어: optuna-dashboard sqlite:///{DB_PATH}\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage(f"sqlite:///{DB_PATH}")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(X_train, y_train, X_test, y_test, in_features),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print("  Best Trial")
    print("=" * 60)
    print(f"  Val RMSE : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<15s}: {v}")

    plot_optuna_history(study)

    # ── 최종 학습 ──
    print("\n최적 파라미터로 최종 학습 중...")
    model, history = train_final(best.params, X_train, y_train, X_test, y_test, in_features)

    # 최종 평가
    test_loader = make_loaders(X_train, y_train, X_test, y_test, best.params["batch_size"])[1]
    _, _, preds, targets = evaluate(model, test_loader)

    rmse_log = np.sqrt(mean_squared_error(targets, preds))
    r2_log   = r2_score(targets, preds)
    rmse_h   = np.sqrt(mean_squared_error(10**targets, 10**preds))
    r2_h     = r2_score(10**targets, 10**preds)

    plot_results(preds, targets, history)

    print("\n" + "=" * 60)
    print("  최종 평가 결과")
    print("=" * 60)
    print(f"  [log10 스케일]  RMSE = {rmse_log:.4f},  R² = {r2_log:.4f}")
    print(f"  [시간 스케일]    RMSE = {rmse_h:.1f} 시간,  R² = {r2_h:.4f}")
    print("=" * 60)

    # 모델 저장
    model_path = os.path.join(BASE_DIR, "resnet_best.pt")
    torch.save({
        "model_state": model.state_dict(),
        "params": best.params,
        "features": selected,
        "in_features": in_features,
    }, model_path)
    print(f"\n모델 저장: {model_path}")

    return model, best.params


if __name__ == "__main__":
    main()
