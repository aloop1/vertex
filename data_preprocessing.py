"""
Data Preprocessing Pipeline for Creep Rupture Life Prediction.
Loads taka.xlsx, applies transformations per CLAUDE.md rules, and saves
the preprocessor object and processed data splits.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "taka.xlsx")

# ── Column definitions ──────────────────────────────────────────────────
TARGET = "lifetime"

COOLING_COLS = ["Cooling1", "Cooling2", "Cooling3"]
COOLING_LABELS = {0: "Furnace", 1: "Air", 2: "Oil", 3: "Water"}
# Furnace: 노냉, Air: 공냉, Oil: 유냉, Water: 수냉

COMPOSITION_COLS = [
    "C", "Si", "Mn", "P", "S", "Cr", "Mo", "W",
    "Ni", "Cu", "V", "Nb", "N", "Al", "B", "Co", "Ta", "O",
]
CONDITION_COLS = ["stress", "temp"]
HEAT_TREATMENT_COLS = [
    "Ntemp", "Ntime", "Ttemp", "Ttime", "Atemp", "Atime",
]
EXTRA_COLS = ["Rh"]

NUMERIC_COLS = CONDITION_COLS + COMPOSITION_COLS + HEAT_TREATMENT_COLS + EXTRA_COLS


# ── Helper functions ─────────────────────────────────────────────────────
def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Excel file."""
    df = pd.read_excel(path)
    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on domain logic.
    - Composition columns: Fill NaN with 0 (assumed as 'not added').
    - Essential columns: Drop rows if Target, Stress, or Temp is missing.
    """
    # 결측치 처리
    df = df.copy()
    df[COMPOSITION_COLS] = df[COMPOSITION_COLS].fillna(0)
    df = df.dropna(subset=[TARGET, "stress", "temp"])
    print("Missing values handled: Composition filled with 0, Essential rows dropped.")
    return df

def check_outliers(df: pd.DataFrame, col: str):
    # 이상치 존재 확인 (미제거, 관찰용)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outliers.empty:
        print(f"  - {col}: {len(outliers)}개의 통계적 이상치 발견")
    else:
        print(f"  - {col}: 이상치 없음")

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically impossible values only.
    - Negative lifetime, stress, or temperature
    - Zero or negative lifetime (log transform needs > 0)
    NOTE: 0 in composition columns is intentional (element not added).
    """
    # 데이터 무결성 검사
    n_before = len(df)
    mask = (
        (df[TARGET] > 0)
        & (df["stress"] > 0)
        & (df["temp"] > 0)
        & (df["Ntemp"] >= 0)
        & (df["Ttemp"] >= 0)
        & (df["Atemp"] >= 0)
    )
    df = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        print(f"Removed {n_removed} rows with physically impossible values")
    else:
        print("No invalid rows found")
    return df


def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log10 to the target variable (lifetime)."""
    # 수명 데이터의 넓은 범위를 축소하여 이분산성 해결
    # 모델이 큰 수치에만 편향되지 않고 고르게 학습하도록 안정화
    df = df.copy()
    df["log_lifetime"] = np.log10(df[TARGET])
    print(
        f"Log10(lifetime) range: [{df['log_lifetime'].min():.3f}, "
        f"{df['log_lifetime'].max():.3f}]"
    )
    return df


def one_hot_encode_cooling(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode cooling rate columns (categorical: 0-3)."""
    df = df.copy()
    for col in COOLING_COLS:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        # Ensure all categories exist
        for val, label in COOLING_LABELS.items():
            expected_col = f"{col}_{val}"
            if expected_col not in dummies.columns:
                dummies[expected_col] = 0
        # Sort columns for consistency
        dummies = dummies[[f"{col}_{v}" for v in sorted(COOLING_LABELS.keys())]]
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=COOLING_COLS)
    print(f"One-hot encoded {COOLING_COLS} -> {len(COOLING_LABELS) * len(COOLING_COLS)} columns")
    return df


def build_preprocessor(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the numeric features of the training set."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"StandardScaler fitted on {X_train.shape[1]} features")
    return scaler


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns after one-hot encoding."""
    exclude = {TARGET, "log_lifetime"}
    return [c for c in df.columns if c not in exclude]


# ── Main preprocessing pipeline ─────────────────────────────────────────
def preprocess(
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Remove physically impossible values
      3. Log-transform target
      4. One-hot encode cooling rate columns
      5. Train/test split
      6. Fit StandardScaler on train set, transform both
      7. Save preprocessor and processed data
    Returns dict with keys: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("=" * 60)
    print("  Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_raw_data()

    # 2. Handle missing values
    df = handle_missing_values(df)

    # 3. Remove invalid rows
    df = remove_invalid_rows(df)

    # 4. Outlier detection
    print("\n[Outlier Detection (Observation Only)]")
    for col in [TARGET, "stress", "temp"]:
        check_outliers(df, col)

    # 5. Log-transform target
    df = log_transform_target(df)

    # 6. One-hot encode cooling
    df = one_hot_encode_cooling(df)

    # 7. Physics-informed feature engineering
    # Hollomon-Jaffe Parameter 기반 가혹도(severity) 파생 변수 생성
    # Formula: T * (C + log10(t)) | T: Kelvin, C: 20, t: hours

    print("\nGenerating severity features (Using Kelvin)...")

    for p in ['N', 'T', 'A']:
        temp_col, time_col = f"{p}temp", f"{p}time"
        # 0인 경우는 해당 공정 생략이므로 계산 제외
        df[f'{p}_severity'] = np.where(
            df[temp_col] > 0,
            df[temp_col] * (20 + np.log10(df[time_col] + 1e-6)),
            0
        )

    # 8. Visualization
    print("\nGenerating correlation heatmap...")
    corr_df = df.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_df, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=15)
    
    heatmap_path = os.path.join(DATA_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close() 
    print(f"Heatmap saved to: {heatmap_path}")
    
    # 9. Feature / target split
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["log_lifetime"]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Samples: {len(X)}")

    # 10. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 11. Scale
    scaler = build_preprocessor(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=feature_cols, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index,
    )

    # 12. Save artifacts
    if save:
        preprocessor_path = os.path.join(DATA_DIR, "preprocessor.pkl")
        joblib.dump(
            {
                "scaler": scaler,
                "feature_names": feature_cols,
                "target": "log_lifetime",
                "cooling_labels": COOLING_LABELS,
            },
            preprocessor_path,
        )
        print(f"\nPreprocessor saved to {preprocessor_path}")

    print("=" * 60)
    print("  Preprocessing complete")
    print("=" * 60)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    result = preprocess()
    print(f"\nX_train shape: {result['X_train'].shape}")
    print(f"X_test  shape: {result['X_test'].shape}")
    print(f"y_train range: [{result['y_train'].min():.3f}, {result['y_train'].max():.3f}]")
    print(f"y_test  range: [{result['y_test'].min():.3f}, {result['y_test'].max():.3f}]")
