"""
Feature Selection: XGBoost importance + correlation heatmap → 13 features.
Saves selected feature list to models/selected_features.json.
Run this BEFORE resnet_optuna.py.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
from 데이터전처리 import preprocess

N_SELECT = 13
CORR_THRESHOLD = 0.85  # 두 피처 간 |corr| > 임계값이면 중요도 낮은 쪽 제거


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    return model


def select_features(importance_series, X_train, n=N_SELECT, threshold=CORR_THRESHOLD):
    """
    중요도 내림차순으로 순회하며 이미 선택된 피처와 |corr| < threshold 인 경우만 추가.
    n개가 채워지거나 피처가 소진될 때까지 반복.
    """
    corr = X_train.corr().abs()
    selected = []
    for feat in importance_series.index:
        if len(selected) >= n:
            break
        if not selected:
            selected.append(feat)
            continue
        max_corr = corr.loc[feat, selected].max()
        if max_corr < threshold:
            selected.append(feat)

    # 상관관계 필터로 n개를 못 채운 경우 나머지를 순서대로 채움
    if len(selected) < n:
        for feat in importance_series.index:
            if len(selected) >= n:
                break
            if feat not in selected:
                selected.append(feat)

    return selected


def plot_importance(importance_series, top_n=20, save_path=None):
    top = importance_series.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3"] * top_n
    ax.barh(top.index[::-1], top.values[::-1], color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title(f"XGBoost Top {top_n} Feature Importance (전체 39개 피처)", fontsize=13)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.close(fig)


def plot_full_heatmap(X, selected, save_path=None):
    """전체 피처 상관관계 히트맵 — 선택된 피처를 좌상단에 배치 + 녹색 테두리."""
    rest = [c for c in X.columns if c not in selected]
    order = selected + rest
    corr = X[order].corr()

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.2,
        annot=False,
        cbar_kws={"shrink": 0.7},
    )

    # 선택된 블록에 녹색 테두리
    n = len(selected)
    total = len(order)
    ax.add_patch(plt.Rectangle(
        (0, total - n), n, n,
        fill=False, edgecolor="lime", lw=3,
        transform=ax.transData, clip_on=False,
    ))

    ax.set_title(
        f"전체 피처 상관관계 히트맵 (녹색 박스 = 선택된 {n}개 피처)",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.close(fig)


def plot_selected_heatmap(X, selected, save_path=None):
    """선택된 13개 피처만의 상관관계 히트맵 (값 표시)."""
    corr = X[selected].corr()
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"선택된 {len(selected)}개 피처 — 상관관계 히트맵", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  Feature Selection: XGBoost + Correlation Heatmap")
    print("=" * 60)

    data = preprocess(save=False)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]

    X_all = pd.concat([X_train, X_test])

    # 1. XGBoost 전체 학습
    print("\n[1/4] XGBoost 전체 피처 학습...")
    model = train_xgboost(X_train, y_train, X_test, y_test)

    importance_series = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    # 2. 피처 중요도 그래프
    print("\n[2/4] 피처 중요도 그래프 저장...")
    plot_importance(
        importance_series, top_n=20,
        save_path=os.path.join(PLOT_DIR, "feature_importance_all.png"),
    )

    # 3. 피처 선택
    print(f"\n[3/4] 상위 중요도 + 상관관계 필터(threshold={CORR_THRESHOLD})로 {N_SELECT}개 선택...")
    selected = select_features(importance_series, X_train, n=N_SELECT, threshold=CORR_THRESHOLD)

    print(f"\n선택된 {len(selected)}개 피처:")
    for i, f in enumerate(selected, 1):
        print(f"  {i:2d}. {f:<20s}  importance={importance_series[f]:.5f}")

    # 4. 히트맵
    print("\n[4/4] 히트맵 생성...")
    plot_full_heatmap(
        X_all, selected,
        save_path=os.path.join(PLOT_DIR, "heatmap_all_features.png"),
    )
    plot_selected_heatmap(
        X_all, selected,
        save_path=os.path.join(PLOT_DIR, "heatmap_selected.png"),
    )

    # 저장
    out_path = os.path.join(BASE_DIR, "selected_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"\n선택된 피처 목록 저장: {out_path}")

    print("\n완료. 다음 단계: python models/resnet_optuna.py")
    return selected


if __name__ == "__main__":
    main()
