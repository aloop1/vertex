"""Vertex — 크립 수명 예측 (추론 전용 웹앱)

제품 조성·열처리 데이터(수명 컬럼 불필요)를 업로드하면
온도·응력 sweep을 통해 크립 수명을 예측하고 시각화합니다.
"""

import io
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")  # headless backend before any pyplot import

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from 데이터전처리 import (
    COMPOSITION_COLS,
    HEAT_TREATMENT_COLS,
    EXTRA_COLS,
)
from models.transformer_and_tree_ensemble import load_transformer_tree_predictor

# ── Paths ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Global model state ─────────────────────────────────────────────────────
_PREDICTOR = None

def _ensure_model() -> None:
    global _PREDICTOR
    if _PREDICTOR is not None:
        return
    artifact_override = os.environ.get("VERTEX_MODEL_PATH")
    _PREDICTOR = load_transformer_tree_predictor(
        artifact_path=artifact_override,
        allow_smoke_fallback=True,
    )
    print(
        f"[Vertex] 앙상블 모델 로드 완료: {_PREDICTOR.artifact_path.name} | "
        f"피처 수: {len(_PREDICTOR.feature_names)}"
    )

# ── Feature engineering ────────────────────────────────────────────────────
_ALL_COMP = list(dict.fromkeys([*COMPOSITION_COLS, *EXTRA_COLS]))
_HT_COLS  = HEAT_TREATMENT_COLS             # Ntemp, Ntime, Ttemp, Ttime, Atemp, Atime


def _fixed_features(product: dict) -> dict:
    """Composition + heat-treatment + severity features (no stress/temp yet)."""
    d: dict[str, float] = {}
    for col in _ALL_COMP:
        val = product.get(col)
        if val is None and col == "Re":
            val = product.get("Rh")
        d[col] = float(val or 0.0)
    for col in _HT_COLS:
        d[col] = float(product.get(col) or 0.0)
    for prefix in ("N", "T", "A"):
        t_val    = d[f"{prefix}temp"]
        time_val = d[f"{prefix}time"]
        d[f"{prefix}_severity"] = (
            t_val * (20.0 + np.log10(max(time_val, 1e-6))) if t_val > 0 else 0.0
        )
    return d


def _predict_batch(fixed: dict, stresses: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Vectorised prediction → log10(hours), preserves input shape."""
    _ensure_model()
    shape = stresses.shape
    rows  = []
    for s, t in zip(stresses.ravel(), temps.ravel()):
        row = dict(fixed)
        row["stress"] = float(s)
        row["temp"]   = float(t)
        rows.append(row)
    pred_df = _PREDICTOR.predict_dataframe(pd.DataFrame(rows), batch_size=1024)
    return pred_df["log_lifetime"].to_numpy(dtype=float).reshape(shape)

# ── File parsing ───────────────────────────────────────────────────────────
_LIFETIME_COLS = {"lifetime", "log_lifetime", "rupture_time", "creep_life",
                  "creep_lifetime", "hours", "rupture_hours"}


def _parse_upload(file_storage) -> list[dict]:
    buf  = io.BytesIO(file_storage.read())
    name = (file_storage.filename or "").lower()
    df   = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={"Rh": "Re", "rh": "Re"}, inplace=True)

    # Drop any lifetime-like columns (inference only)
    df.drop(columns=[c for c in df.columns if c.lower() in _LIFETIME_COLS],
            inplace=True, errors="ignore")

    # Ensure a product name column
    nc = next((c for c in df.columns if c.lower() == "name"), None)
    if nc is None:
        df.insert(0, "name", [f"제품 {i+1}" for i in range(len(df))])
    else:
        df.rename(columns={nc: "name"}, inplace=True)

    for col in df.columns: 
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if len(df) == 0:
        raise ValueError("유효한 제품 행이 없습니다.")
    return df.to_dict(orient="records")

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.get("/health")
def health():
    try:
        _ensure_model()
        return jsonify(
            {
                "status": "ok",
                "model_path": str(_PREDICTOR.artifact_path),
                "feature_count": len(_PREDICTOR.feature_names),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.get("/")
def index():
    return render_template("index.html", error=None)


@app.post("/predict")
def predict():
    _ensure_model()

    file = request.files.get("file")
    if not file or file.filename == "":
        return "<h3 style='color:#f76b6b;font-family:sans-serif'>파일을 선택해주세요.</h3>", 400

    try:
        products = _parse_upload(file)
    except Exception as exc:
        return f"<h3 style='color:#f76b6b;font-family:sans-serif'>파일 파싱 오류: {exc}</h3>", 400

    def fv(key, default):
        try:
            return float(request.form.get(key, default))
        except ValueError:
            return float(default)

    temp_min     = fv("temp_min",     700)
    temp_max     = fv("temp_max",    1200)
    stress_min   = fv("stress_min",    10)
    stress_max   = fv("stress_max",   600)
    fixed_stress = fv("fixed_stress", 150)
    fixed_temp   = fv("fixed_temp",   873)

    # Sweep arrays
    temps    = np.linspace(temp_min, temp_max, 80)
    stresses = np.logspace(np.log10(max(stress_min, 1.0)),
                           np.log10(stress_max), 80)

    # Heatmap grid (35×35)
    hm_t = np.linspace(temp_min, temp_max, 35)
    hm_s = np.logspace(np.log10(max(stress_min, 1.0)), np.log10(stress_max), 35)
    hm_T, hm_S = np.meshgrid(hm_t, hm_s)   # both shape (35, 35)

    product_data = []
    for i, p in enumerate(products):
        fixed = _fixed_features(p)

        # ── Sweeps ──
        ts_log = _predict_batch(fixed, np.full(80, fixed_stress), temps)
        ss_log = _predict_batch(fixed, stresses, np.full(80, fixed_temp))
        hm_log = _predict_batch(fixed, hm_S, hm_T)   # shape (35, 35)

        # LMP from stress sweep: LMP = T*(20+log10(t_pred))/1000
        lmp_vals = (fixed_temp * (20.0 + ss_log) / 1000.0).tolist()

        # Only show non-zero composition and HT for the table
        key_comp = {c: round(float(p.get(c) or 0.0), 4)
                    for c in _ALL_COMP if float(p.get(c) or 0.0) > 0}
        key_ht   = {c: round(float(p.get(c) or 0.0), 1)
                    for c in _HT_COLS   if float(p.get(c) or 0.0) > 0}

        # Composition pie (Fe balance = remainder)
        comp_total = sum(key_comp.values())
        fe_bal = round(max(0.0, 100.0 - comp_total), 3)
        comp_pie = {"Fe (bal.)": fe_bal, **key_comp} if fe_bal > 0 else dict(key_comp)

        # Heat treatment stages for thermal cycle chart
        ht_stages = []
        for pfx, lbl in [("N", "노말라이징"), ("T", "템퍼링"), ("A", "시효처리")]:
            t_k = float(p.get(f"{pfx}temp") or 0.0)
            t_h = float(p.get(f"{pfx}time") or 0.0)
            if t_k > 0:
                ht_stages.append({"label": lbl, "prefix": pfx,
                                   "temp_k": t_k, "temp_c": round(t_k - 273.15, 1),
                                   "time_h": round(t_h, 2)})

        product_data.append({
            "name":           str(p.get("name", f"제품 {i+1}")),
            "composition":    key_comp,
            "heat_treatment": key_ht,
            "comp_pie":       comp_pie,
            "ht_stages":      ht_stages,
            "_features":      fixed,          # stored for client-side /resweep calls
            "temp_sweep": {
                "temps_k":     temps.tolist(),
                "temps_c":     (temps - 273.15).tolist(),
                "log10_hours": ts_log.tolist(),
                "hours":       np.power(10, ts_log).tolist(),
            },
            "stress_sweep": {
                "stresses_mpa": stresses.tolist(),
                "log10_hours":  ss_log.tolist(),
                "hours":        np.power(10, ss_log).tolist(),
            },
            "lmp": {
                "stresses_mpa": stresses.tolist(),
                "lmp_vals":     lmp_vals,
            },
            "heatmap": {
                "temps_k":           hm_t.tolist(),
                "stresses_mpa":      hm_s.tolist(),
                "log10_hours_grid":  hm_log.tolist(),
            },
        })

    # ── GA 결과 로드 및 sweep 실행 ───────────────────────────────────────────
    ga_data: dict = {"best": None, "candidates": [], "meta": {}, "sweep": None}

    best_path  = PROJECT_ROOT / "ga" / "best_alloy.json"
    pareto_path = PROJECT_ROOT / "ga" / "pareto_top30.json"

    if best_path.exists():
        try:
            with open(best_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "best" in raw:
                ga_data["best"] = raw["best"]
                ga_data["meta"] = {k: v for k, v in raw.items() if k != "best"}
            else:
                ga_data["best"] = raw
        except Exception:
            pass

    if pareto_path.exists():
        try:
            with open(pareto_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                ga_data["candidates"] = raw[:5]
            elif isinstance(raw, dict):
                ga_data["candidates"] = raw.get("candidates", [])[:5]
                if not ga_data["meta"]:
                    ga_data["meta"] = {k: v for k, v in raw.items() if k != "candidates"}
        except Exception:
            pass

    if ga_data["best"]:
        try:
            b = ga_data["best"]
            ga_prod_feat = {elem: float(b.get(elem, 0.0)) for elem in _ALL_COMP}
            ga_ht_feat   = {col:  float(b.get(col,  0.0)) for col  in _HT_COLS}
            ga_fixed = _fixed_features({**ga_prod_feat, **ga_ht_feat})

            ga_ts = _predict_batch(ga_fixed, np.full(80, fixed_stress), temps)
            ga_ss = _predict_batch(ga_fixed, stresses, np.full(80, fixed_temp))

            ga_comp_nz = {c: round(float(b.get(c, 0.0)), 4)
                          for c in _ALL_COMP if float(b.get(c, 0.0)) > 0}
            ga_fe_bal  = round(max(0.0, 100.0 - sum(ga_comp_nz.values())), 3)
            ga_comp_pie = {"Fe (bal.)": ga_fe_bal, **ga_comp_nz} if ga_fe_bal > 0 else dict(ga_comp_nz)

            ga_data["sweep"] = {
                "temp_sweep":  {
                    "temps_k":     temps.tolist(),
                    "temps_c":     (temps - 273.15).tolist(),
                    "log10_hours": ga_ts.tolist(),
                },
                "stress_sweep": {
                    "stresses_mpa": stresses.tolist(),
                    "log10_hours":  ga_ss.tolist(),
                },
                "comp_pie": ga_comp_pie,
                "composition": ga_comp_nz,
            }
        except Exception as e:
            print(f"[GA sweep] {e}")

    rd = {
        "products":     product_data,
        "fixed_stress": fixed_stress,
        "fixed_temp":   fixed_temp,
        "temp_min":     temp_min,   "temp_max":  temp_max,
        "stress_min":   stress_min, "stress_max": stress_max,
        "ga":           ga_data,
    }
    return render_template("result.html", payload=rd)


@app.post("/suggest_params")
def suggest_params():
    """파일의 stress/temp 컬럼을 읽어 sweep 파라미터 추천값을 반환."""
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({}), 200
    try:
        buf = io.BytesIO(file.read())
        name = (file.filename or "").lower()
        df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
        df.columns = [str(c).strip() for c in df.columns]
        col_lower = {c.lower(): c for c in df.columns}
        result = {}

        sc = next((col_lower[k] for k in ["stress", "rupture_stress", "applied_stress"] if k in col_lower), None)
        if sc:
            s = pd.to_numeric(df[sc], errors="coerce").dropna()
            s = s[s > 0]
            if len(s) >= 2:
                result["stress_min"] = round(float(s.min()), 1)
                result["stress_max"] = round(float(s.max()), 1)
                result["fixed_stress"] = round(float(s.median()), 1)

        tc = next((col_lower[k] for k in ["temp", "temperature", "test_temp"] if k in col_lower), None)
        if tc:
            t = pd.to_numeric(df[tc], errors="coerce").dropna()
            t = t[t > 0]
            if len(t) >= 2:
                result["temp_min"] = int(round(float(t.min())))
                result["temp_max"] = int(round(float(t.max())))
                result["fixed_temp"] = int(round(float(t.median())))

        return jsonify(result)
    except Exception:
        return jsonify({}), 200


@app.post("/resweep")
def resweep():
    """재계산 엔드포인트: 파라미터만 바꿔 sweep 재실행, JSON 반환."""
    _ensure_model()
    data = request.get_json(force=True)

    temp_min     = float(data.get("temp_min",     700))
    temp_max     = float(data.get("temp_max",    1200))
    stress_min   = float(data.get("stress_min",    10))
    stress_max   = float(data.get("stress_max",   600))
    fixed_stress = float(data.get("fixed_stress", 150))
    fixed_temp   = float(data.get("fixed_temp",   873))

    temps    = np.linspace(temp_min, temp_max, 80)
    stresses = np.logspace(np.log10(max(stress_min, 1.0)), np.log10(stress_max), 80)
    hm_t = np.linspace(temp_min, temp_max, 35)
    hm_s = np.logspace(np.log10(max(stress_min, 1.0)), np.log10(stress_max), 35)
    hm_T, hm_S = np.meshgrid(hm_t, hm_s)

    results = []
    for feat in data.get("products_features", []):
        fixed = {k: float(v) for k, v in feat.items()}

        ts_log = _predict_batch(fixed, np.full(80, fixed_stress), temps)
        ss_log = _predict_batch(fixed, stresses, np.full(80, fixed_temp))
        hm_log = _predict_batch(fixed, hm_S, hm_T)
        lmp_vals = (fixed_temp * (20.0 + ss_log) / 1000.0).tolist()

        results.append({
            "temp_sweep": {
                "temps_k":     temps.tolist(),
                "temps_c":     (temps - 273.15).tolist(),
                "log10_hours": ts_log.tolist(),
                "hours":       np.power(10, ts_log).tolist(),
            },
            "stress_sweep": {
                "stresses_mpa": stresses.tolist(),
                "log10_hours":  ss_log.tolist(),
                "hours":        np.power(10, ss_log).tolist(),
            },
            "lmp": {
                "stresses_mpa": stresses.tolist(),
                "lmp_vals":     lmp_vals,
            },
            "heatmap": {
                "temps_k":          hm_t.tolist(),
                "stresses_mpa":     hm_s.tolist(),
                "log10_hours_grid": hm_log.tolist(),
            },
        })

    return jsonify({
        "products":     results,
        "fixed_stress": fixed_stress,
        "fixed_temp":   fixed_temp,
        "temp_min":     temp_min,   "temp_max":  temp_max,
        "stress_min":   stress_min, "stress_max": stress_max,
    })


@app.get("/alloy_design")
def alloy_design():
    best_path  = PROJECT_ROOT / "ga" / "best_alloy.json"
    pareto_path = PROJECT_ROOT / "ga" / "pareto_top30.json"

    best_raw = None
    if best_path.exists():
        try:
            with open(best_path, encoding="utf-8") as f:
                raw = json.load(f)
            best_raw = raw.get("best", raw) if isinstance(raw, dict) and "best" in raw else raw
        except Exception:
            pass

    pareto_candidates = []
    pareto_meta: dict = {}
    if pareto_path.exists():
        try:
            with open(pareto_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                pareto_candidates = raw
            elif isinstance(raw, dict):
                pareto_candidates = raw.get("candidates", [])
                pareto_meta = {k: v for k, v in raw.items() if k != "candidates"}
        except Exception:
            pass

    payload = {
        "best": best_raw,
        "candidates": pareto_candidates,
        "meta": pareto_meta,
    }
    return render_template("alloy_design.html", payload=payload)





# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ensure_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
