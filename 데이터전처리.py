"""통합 크립 데이터 전처리 모듈

지원 데이터:
1) taka.xlsx
2) creep.csv
3) creep_data.csv

- 추가된 데이터와 다른 스키마를 공통 피처 체계로 정규화
- 조성 그룹 기반 분할용 인터페이스 제공
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# 프로젝트 기본 경로
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TAKA_PATH = DATA_DIR / "taka.xlsx"
CREEP_CSV_PATH = DATA_DIR / "creep.csv"
CREEP_DATA_CSV_PATH = DATA_DIR / "creep_data.csv"
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.pkl"

TARGET_COL = "lifetime"
LOG_TARGET_COL = "log_lifetime"

CONDITION_COLS: List[str] = ["stress", "temp"]
COMPOSITION_COLS: List[str] = [
    "C", "Si", "Mn", "P", "S",
    "Cr", "Mo", "W", "Ni", "Cu",
    "V", "Nb", "N", "Al", "B", "Co", "Ta", "O", "Re",
]
HEAT_TREATMENT_COLS: List[str] = [
    "Ntemp", "Ntime", "Ttemp", "Ttime", "Atemp", "Atime",
]
COOLING_COLS: List[str] = ["Cooling1", "Cooling2", "Cooling3"]
EXTRA_COLS: List[str] = ["Re"]

CORE_COLUMNS: List[str] = [
    TARGET_COL,
    *CONDITION_COLS,
    *COMPOSITION_COLS,
    *HEAT_TREATMENT_COLS[:2],
    COOLING_COLS[0],
    *HEAT_TREATMENT_COLS[2:4],
    COOLING_COLS[1],
    *HEAT_TREATMENT_COLS[4:6],
    COOLING_COLS[2],
]

TAKA_COLUMN_NAMES: List[str] = [
    "lifetime",
    "stress",
    "temp",
    "C", "Si", "Mn", "P", "S",
    "Cr", "Mo", "W", "Ni", "Cu",
    "V", "Nb", "N", "Al", "B", "Co", "Ta", "O",
    "Ntemp", "Ntime", "Cooling1",
    "Ttemp", "Ttime", "Cooling2",
    "Atemp", "Atime", "Cooling3",
    "Re",
]


@dataclass
class PreprocessResult:
    """전처리 결과 묶음."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    feature_names: List[str]
    raw_df: pd.DataFrame
    scaler: Optional[StandardScaler] = None


@dataclass
class GroupSplitResult:
    """조성 그룹 기반 홀드아웃 분할 결과."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")


def _detect_header(path: Path) -> bool:
    """엑셀 파일의 헤더 존재 여부를 간단히 판별한다."""
    probe = pd.read_excel(path, nrows=1)
    first_col = probe.columns[0]
    first_col_str = str(first_col)
    is_numeric_like = first_col_str.replace(".", "", 1).lstrip("-").isdigit()
    return not is_numeric_like


def _normalize_text(value: str) -> str:
    return (
        str(value)
        .replace("\xa0", " ")
        .replace("ЎЙ", "C")
        .replace("Ґг", "gamma")
        .replace("'", "")
        .replace('"', "")
        .strip()
        .lower()
    )


def _find_column(columns: Iterable[str], *keywords: str) -> str:
    normalized_map = {_normalize_text(col): col for col in columns}
    for normalized, original in normalized_map.items():
        if all(keyword in normalized for keyword in keywords):
            return original
    raise KeyError(f"컬럼을 찾을 수 없습니다. keywords={keywords}")


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _empty_canonical_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({col: np.zeros(n_rows, dtype=float) for col in CORE_COLUMNS})


def _convert_celsius_to_kelvin(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") + 273.15


def load_taka_data(path: Path | str = TAKA_PATH) -> pd.DataFrame:
    """taka.xlsx를 공통 스키마로 로드한다."""
    path = Path(path)
    _ensure_exists(path)

    probe = pd.read_excel(path, nrows=1)
    if probe.shape[1] != len(TAKA_COLUMN_NAMES):
        raise ValueError(
            f"taka.xlsx 컬럼 수 불일치: expected={len(TAKA_COLUMN_NAMES)}, found={probe.shape[1]}"
        )

    if _detect_header(path):
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(path, header=None)

    df.columns = TAKA_COLUMN_NAMES
    df = _to_numeric(df, TAKA_COLUMN_NAMES)
    return df[CORE_COLUMNS].copy()


def load_creep_csv_data(path: Path | str = CREEP_CSV_PATH) -> pd.DataFrame:
    """creep.csv를 공통 스키마로 정규화한다.

    이 데이터의 온도는 초합금 문헌 관례를 따라 섭씨로 가정하고 Kelvin으로 변환한다.
    """
    path = Path(path)
    _ensure_exists(path)

    df = pd.read_csv(path)
    out = _empty_canonical_frame(len(df))

    out[TARGET_COL] = pd.to_numeric(df["creep_life"], errors="coerce")
    out["stress"] = pd.to_numeric(df["Stress"], errors="coerce")
    out["temp"] = _convert_celsius_to_kelvin(df["Temperature"])

    direct_map = {
        "C": "C",
        "Si": "Si",
        "Cr": "Cr",
        "Mo": "Mo",
        "W": "W",
        "Ni": "Ni",
        "Nb": "Nb",
        "Al": "Al",
        "B": "B",
        "Co": "Co",
        "Ta": "Ta",
        "Re": "Re",
    }
    for target_col, source_col in direct_map.items():
        if source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors="coerce")

    return out


def load_creep_data_csv(path: Path | str = CREEP_DATA_CSV_PATH) -> pd.DataFrame:
    """creep_data.csv를 공통 스키마로 정규화한다."""
    path = Path(path)
    _ensure_exists(path)

    df = pd.read_csv(path)
    out = _empty_canonical_frame(len(df))

    temp_col = _find_column(df.columns, "test temperature")
    stress_col = _find_column(df.columns, "test stress")
    life_col = _find_column(df.columns, "creep rupture life")
    solution_temp_col = _find_column(df.columns, "solution treatment temperature")
    solution_time_col = _find_column(df.columns, "solution treatment time")
    stable_temp_col = _find_column(df.columns, "stable", "aging", "temperature")
    stable_time_col = _find_column(df.columns, "stable", "aging", "time")
    aging_temp_col = _find_column(df.columns, "aging", "temperature")
    aging_time_col = _find_column(df.columns, "aging", "time")

    out[TARGET_COL] = pd.to_numeric(df[life_col], errors="coerce")
    out["stress"] = pd.to_numeric(df[stress_col], errors="coerce")
    out["temp"] = _convert_celsius_to_kelvin(df[temp_col])

    direct_map = {
        "C": "C",
        "Cr": "Cr",
        "Mo": "Mo",
        "W": "W",
        "Ni": "Ni",
        "Nb": "Nb",
        "Al": "Al",
        "B": "B",
        "Co": "Co",
        "Re": "Re",
    }
    for target_col, source_col in direct_map.items():
        if source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors="coerce")

    out["Ntemp"] = _convert_celsius_to_kelvin(df[solution_temp_col])
    out["Ntime"] = pd.to_numeric(df[solution_time_col], errors="coerce")
    out["Ttemp"] = _convert_celsius_to_kelvin(df[stable_temp_col])
    out["Ttime"] = pd.to_numeric(df[stable_time_col], errors="coerce")
    out["Atemp"] = _convert_celsius_to_kelvin(df[aging_temp_col])
    out["Atime"] = pd.to_numeric(df[aging_time_col], errors="coerce")

    return out


def load_raw_data() -> pd.DataFrame:
    """세 개의 원본 데이터를 공통 스키마로 합쳐 반환한다."""
    frames = [
        load_taka_data(),
        load_creep_csv_data(),
        load_creep_data_csv(),
    ]

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged[CORE_COLUMNS].copy()
    return merged


def clean_domain_values(df: pd.DataFrame) -> pd.DataFrame:
    """도메인 규칙 기반 결측/타입 정리."""
    out = df.copy()
    numeric_cols = [col for col in CORE_COLUMNS if col in out.columns]
    out = _to_numeric(out, numeric_cols)
    out[numeric_cols] = out[numeric_cols].fillna(0.0)

    for col in COOLING_COLS:
        out[col] = out[col].round().fillna(0).astype(int)

    return out


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """물리적으로 불가능한 값만 제거한다."""
    out = df.copy()
    n_before = len(out)

    mask = (
        (out[TARGET_COL] > 0)
        & (out["stress"] > 0)
        & (out["temp"] > 0)
        & (out["Ntemp"] >= 0)
        & (out["Ttemp"] >= 0)
        & (out["Atemp"] >= 0)
        & (out["Ntime"] >= 0)
        & (out["Ttime"] >= 0)
        & (out["Atime"] >= 0)
    )

    out = out.loc[mask].reset_index(drop=True)
    removed = n_before - len(out)
    print(f"물리적으로 불가능한 행 제거: {removed}개")
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """열처리 가혹도(severity) 파생 특성을 추가한다."""
    out = df.copy()

    for prefix in ["N", "T", "A"]:
        temp_col = f"{prefix}temp"
        time_col = f"{prefix}time"
        safe_time = np.maximum(out[time_col].astype(float), 1e-6)
        severity = out[temp_col].astype(float) * (20.0 + np.log10(safe_time))
        out[f"{prefix}_severity"] = np.where(out[temp_col] > 0, severity, 0.0)
    """크리프 LMP 계산 추가 / T(켈빈) * (C + log10(t)) / 1000, TARGET_COL = lifetime"""
    C = 20.0 
    out['LMP'] = out['temp'] * (C + np.log10(out[TARGET_COL])) / 1000.0

    return out


def drop_cooling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """냉각 방식 변수는 사용하지 않으므로 피처에서 제외한다."""
    out = df.copy()
    out = out.drop(columns=COOLING_COLS)
    return out


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """수명 타깃을 log10 스케일로 변환한다."""
    out = df.copy()
    safe_lifetime = np.clip(out[TARGET_COL].astype(float), a_min=1e-12, a_max=None)
    out[LOG_TARGET_COL] = np.log10(safe_lifetime)
    return out

def generate_inference_grid(
    base_row: pd.Series, 
    temp_range: Tuple[float, float, float] = (600, 810, 10), 
    stress_range: Tuple[float, float, float] = (100, 310, 10)
) -> pd.DataFrame:
    """
    특정 합금 조성 및 열처리 조건에 대한 온도-응력 가상 격자(Inference Grid) 데이터를 생성한다.
    
    Args:
        base_row (pd.Series): 조성 및 열처리 조건이 포함된 기준 샘플 데이터
        temp_range (tuple): 온도 범위 및 간격 (start, stop, step)
        stress_range (tuple): 응력 범위 및 간격 (start, stop, step)
        
    Returns:
        pd.DataFrame: 시각화 및 추론용 가상 격자 데이터프레임
    """
    # 1. 격자 포인트 생성 및 조합(Meshgrid)
    temps = np.arange(*temp_range)
    stresses = np.arange(*stress_range)
    t_grid, s_grid = np.meshgrid(temps, stresses)
    
    # 2. 데이터프레임 초기화 및 평탄화(Flatten)
    grid_df = pd.DataFrame({
        "temp": t_grid.flatten(),
        "stress": s_grid.flatten()
    })
    
    # 3. 기준 샘플의 고정 피처(조성 및 공정 조건) 복제
    for col, value in base_row.items():
        if col not in ["temp", "stress"]:
            grid_df[col] = value
            
    # NOTE: 모델 추론 전 학습 시 사용된 feature_names 순서로 컬럼 재정렬 권장
    return grid_df

def make_composition_group_id(
    df: pd.DataFrame,
    rounding_decimals: int = 3,
    prefix: str = "grp",
) -> pd.Series:
    """반올림된 조성 벡터 기준 그룹 ID를 생성한다."""
    rounded = df[COMPOSITION_COLS].round(rounding_decimals)
    key = rounded.apply(
        lambda row: "|".join(f"{value:.{rounding_decimals}f}" for value in row.values),
        axis=1,
    )
    return prefix + "_" + key


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """모델 입력 X, 타깃 y를 생성한다."""
    exclude = {TARGET_COL, LOG_TARGET_COL}
    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].copy()
    y = df[LOG_TARGET_COL].copy()
    return X, y, feature_cols


def prepare_dataset(
    data_path: Path | str | None = None,
    rounding_decimals: int = 3,
    use_scaler: bool = False,
) -> PreprocessResult:
    """학습용 데이터셋(X, y, groups)을 구성한다."""
    if data_path is not None:
        path = Path(data_path)
        if path.suffix.lower() == ".xlsx":
            df = load_taka_data(path)
        elif path.name.lower() == "creep.csv":
            df = load_creep_csv_data(path)
        elif path.name.lower() == "creep_data.csv":
            df = load_creep_data_csv(path)
        else:
            raise ValueError(f"지원하지 않는 데이터 파일입니다: {path}")
    else:
        df = load_raw_data()

    df = clean_domain_values(df)
    df = remove_invalid_rows(df)
    df = add_engineered_features(df)
    df = drop_cooling_columns(df)
    df = add_log_target(df)

    groups = make_composition_group_id(df, rounding_decimals=rounding_decimals)
    X, y, feature_names = build_feature_matrix(df)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)

    return PreprocessResult(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        raw_df=df,
        scaler=scaler,
    )


def split_by_composition_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> GroupSplitResult:
    """조성 그룹 기준으로 누수 없는 홀드아웃 분할을 수행한다."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    overlap = set(groups_train.unique()).intersection(set(groups_test.unique()))
    if overlap:
        raise RuntimeError("그룹 누수 발생: train/test 간 조성 그룹이 겹칩니다.")

    return GroupSplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups_train=groups_train,
        groups_test=groups_test,
    )


def summarize_group_split(split: GroupSplitResult) -> Dict[str, float]:
    """분할 결과 기본 통계를 요약한다."""
    return {
        "n_train": float(len(split.X_train)),
        "n_test": float(len(split.X_test)),
        "n_group_train": float(split.groups_train.nunique()),
        "n_group_test": float(split.groups_test.nunique()),
        "y_train_mean": float(split.y_train.mean()),
        "y_test_mean": float(split.y_test.mean()),
    }


def preprocess(
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
    use_scaler: bool = True,
) -> dict:
    """기존 학습 코드 호환용 전처리 파이프라인."""
    print("=" * 60)
    print("  통합 데이터 전처리 파이프라인")
    print("=" * 60)

    dataset = prepare_dataset(rounding_decimals=3, use_scaler=False)
    split = split_by_composition_group(
        dataset.X,
        dataset.y,
        dataset.groups,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    scaler.fit(split.X_train)

    if use_scaler:
        X_train = pd.DataFrame(
            scaler.transform(split.X_train),
            columns=dataset.feature_names,
            index=split.X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(split.X_test),
            columns=dataset.feature_names,
            index=split.X_test.index,
        )
    else:
        X_train = split.X_train.copy()
        X_test = split.X_test.copy()

    if save:
        PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": scaler,
                "feature_names": dataset.feature_names,
                "target": LOG_TARGET_COL,
            },
            PREPROCESSOR_PATH,
        )
        print(f"전처리기 저장: {PREPROCESSOR_PATH}")

    print(f"전체 샘플 수: {len(dataset.X)}")
    print(f"학습 샘플 수: {len(X_train)}")
    print(f"테스트 샘플 수: {len(X_test)}")
    print(f"특성 수: {len(dataset.feature_names)}")
    print("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": split.y_train,
        "y_test": split.y_test,
        "groups_train": split.groups_train,
        "groups_test": split.groups_test,
        "groups": dataset.groups,
        "scaler": scaler,
        "feature_names": dataset.feature_names,
        "raw_df": dataset.raw_df,
    }


if __name__ == "__main__":
    result = preprocess(save=False, use_scaler=True)
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test shape: {result['X_test'].shape}")
    print(
        f"y_train range: [{result['y_train'].min():.3f}, {result['y_train'].max():.3f}]"
    )
    print(
        f"y_test range: [{result['y_test'].min():.3f}, {result['y_test'].max():.3f}]"
    )
