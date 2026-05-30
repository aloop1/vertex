from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_FEATURES = [
    "C", "Si", "Mn", "Cr", "Mo", "W",
    "Ni", "V", "Nb", "N", "B"
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원소 컬럼 이름의 공백/대소문자 문제를 줄이기 위한 정규화.
    예: ' cr ' -> 'Cr'
    """

    rename = {}

    canonical = {
        "c": "C",
        "si": "Si",
        "mn": "Mn",
        "p": "P",
        "s": "S",
        "cr": "Cr",
        "mo": "Mo",
        "w": "W",
        "ni": "Ni",
        "cu": "Cu",
        "v": "V",
        "nb": "Nb",
        "n": "N",
        "al": "Al",
        "b": "B",
        "co": "Co",
        "ta": "Ta",
        "o": "O",
        "re": "Re",
        "rh": "Re",  # 기존 데이터의 Rh/Re 혼용 방어. 실제로는 발표 시 제거/0처리가 안전.
    }

    for col in df.columns:
        key = str(col).strip().lower()
        if key in canonical:
            rename[col] = canonical[key]

    return df.rename(columns=rename)


def filter_ferritic_steel_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ni-base 초합금 데이터가 섞이면 OOD 기준이 망가진다.
    9-12Cr ferritic/martensitic steel 근처만 reference로 사용한다.
    """

    out = df.copy()

    if "Cr" in out.columns:
        out = out[
            pd.to_numeric(out["Cr"], errors="coerce").between(8.0, 12.5)
        ]

    if "Ni" in out.columns:
        out = out[
            pd.to_numeric(out["Ni"], errors="coerce").fillna(0.0) <= 2.0
        ]

    # 합금 총량이 비정상적으로 큰 행 제거
    available = [c for c in DEFAULT_FEATURES if c in out.columns]

    if available:
        total = out[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        out = out[total <= 20.0]

    return out


def build_ood_reference(df: pd.DataFrame, features: list[str], quantile: float, reg_factor: float):
    missing = [f for f in features if f not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing required OOD feature columns: {missing}\n"
            f"현재 df columns: {list(df.columns)}"
        )

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="any")

    if len(X) < max(20, len(features) + 5):
        raise RuntimeError(
            f"OOD reference rows too small: {len(X)} rows for {len(features)} features. "
            f"최소 20행 이상, 가능하면 100행 이상 권장."
        )

    # 분산이 거의 없는 feature 제거
    std = X.std(axis=0)
    active_features = [f for f in features if float(std[f]) > 1e-12]

    if len(active_features) < 2:
        raise RuntimeError("Active OOD features are too few after variance filtering.")

    X = X[active_features].to_numpy(dtype=float)

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    dim = cov.shape[0]
    base_scale = float(np.trace(cov) / dim) if dim > 0 else 1.0
    reg = max(base_scale, 1.0) * float(reg_factor)

    cov_reg = cov + reg * np.eye(dim)
    cov_inv = np.linalg.pinv(cov_reg)

    diff = X - mean
    dist_sq = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    dist_sq = np.maximum(dist_sq, 0.0)
    distances = np.sqrt(dist_sq)

    threshold = float(np.quantile(distances, quantile))

    payload = {
        "ood_reference_features": active_features,
        "ood_reference_mean": mean.tolist(),
        "ood_reference_cov_inv": cov_inv.tolist(),
        "ood_distance_threshold": threshold,
        "ood_quantile": float(quantile),
        "ood_train_distance_mean": float(np.mean(distances)),
        "ood_train_distance_max": float(np.max(distances)),
        "n_reference_rows": int(len(X)),
        "cov_regularization": float(reg),
        "source": None,
    }

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fe계 학습 데이터 파일. 예: data/taka.xlsx")
    parser.add_argument("--output", default="data/ood_reference.pkl")
    parser.add_argument("--json-output", default="data/ood_reference_summary.json")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--reg", type=float, default=1e-6)
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    json_path = (PROJECT_ROOT / args.json_output).resolve()

    print(f"[OOD] input: {input_path}")

    df = read_table(input_path)
    df = normalize_columns(df)

    print(f"[OOD] raw rows: {len(df)}")

    df = filter_ferritic_steel_rows(df)

    print(f"[OOD] ferritic-filtered rows: {len(df)}")

    payload = build_ood_reference(
        df=df,
        features=DEFAULT_FEATURES,
        quantile=args.quantile,
        reg_factor=args.reg,
    )

    payload["source"] = str(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(payload, output_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OOD] saved: {output_path}")
    print(f"[OOD] summary: {json_path}")
    print(f"[OOD] features: {payload['ood_reference_features']}")
    print(f"[OOD] n rows: {payload['n_reference_rows']}")
    print(f"[OOD] threshold: {payload['ood_distance_threshold']:.4f}")


if __name__ == "__main__":
    main()