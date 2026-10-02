"""Vertex web app — creep-life prediction + AI assistant.

현재 웹의 핵심 기능
1) 제품 조성/열처리 업로드 -> Predictor sweep
2) 결과 대시보드 -> 조건 탐색/재계산
3) Vertex AI -> analysis.pipeline.run_assistant_pipeline 연결

"""
from __future__ import annotations

import io
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 데이터전처리 import COMPOSITION_COLS, HEAT_TREATMENT_COLS, EXTRA_COLS
from models.custom_histogram import load_custom_histogram_predictor
from analysis import pipeline as assistant_pipeline

UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

_PREDICTOR = None
_ALL_COMP = list(dict.fromkeys([*COMPOSITION_COLS, *EXTRA_COLS]))
_HT_COLS = list(HEAT_TREATMENT_COLS)
_LIFETIME_COLS = {
    "lifetime", "log_lifetime", "rupture_time", "creep_life",
    "creep_lifetime", "hours", "rupture_hours",
}


def _ensure_model() -> None:
    """Load one real predictor and share the same instance with analysis.pipeline."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return

    artifact_override = os.environ.get("VERTEX_MODEL_PATH")
    _PREDICTOR = load_custom_histogram_predictor(
        artifact_path=artifact_override,
    )
    if getattr(_PREDICTOR, "is_smoke_model", False):
        raise RuntimeError("실제 수명예측 모델이 아닌 smoke-test 모델이 로드되었습니다.")

    # UI sweep과 assistant Prediction/What-if가 같은 predictor 인스턴스를 재사용한다.
    if hasattr(assistant_pipeline, "_life_predictor"):
        assistant_pipeline._life_predictor = _PREDICTOR

    artifact = getattr(_PREDICTOR, "artifact_path", None)
    print(
        f"[Vertex] predictor ready: {getattr(artifact, 'name', artifact)} | "
        f"features={len(getattr(_PREDICTOR, 'feature_names', []))}"
    )


def _fixed_features(product: dict) -> dict[str, float]:
    d: dict[str, float] = {}
    for col in _ALL_COMP:
        val = product.get(col)
        if val is None and col == "Re":
            val = product.get("Rh")
        d[col] = float(val or 0.0)
    for col in _HT_COLS:
        d[col] = float(product.get(col) or 0.0)
    for prefix in ("N", "T", "A"):
        t_val = d[f"{prefix}temp"]
        time_val = d[f"{prefix}time"]
        d[f"{prefix}_severity"] = (
            t_val * (20.0 + np.log10(max(time_val, 1e-6))) if t_val > 0 else 0.0
        )
    return d


def _predict_batch(fixed: dict, stresses: np.ndarray, temps: np.ndarray) -> np.ndarray:
    _ensure_model()
    shape = stresses.shape
    rows = []
    for stress, temp in zip(stresses.ravel(), temps.ravel()):
        row = dict(fixed)
        row["stress"] = float(stress)
        row["temp"] = float(temp)
        rows.append(row)
    pred_df = _PREDICTOR.predict_dataframe(pd.DataFrame(rows))
    # Histogram predictor already returns log10(hours); preserve the web contract.
    return pred_df["log_lifetime"].to_numpy(dtype=float).reshape(shape)


def _finite_float(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _positive_for_model(value: float, default: float = 1.0) -> float:
    x = abs(_finite_float(value, default))
    return float(default if x < 1e-6 else x)


def _ordered_range(a: float, b: float, min_gap: float = 1.0) -> tuple[float, float]:
    a = _finite_float(a, 0.0)
    b = _finite_float(b, a + min_gap)
    if a == b:
        b = a + min_gap
    if a > b:
        a, b = b, a
    return float(a), float(b)


def _sanitize_sweep_params(temp_min, temp_max, stress_min, stress_max, fixed_stress, fixed_temp):
    temp_min, temp_max = _ordered_range(temp_min, temp_max, 1.0)
    stress_min = _positive_for_model(stress_min, 1.0)
    stress_max = _positive_for_model(stress_max, max(stress_min + 1.0, 2.0))
    stress_min, stress_max = _ordered_range(stress_min, stress_max, 1.0)
    fixed_stress = _positive_for_model(fixed_stress, 1.0)
    fixed_temp = _positive_for_model(fixed_temp, 1.0)
    return temp_min, temp_max, stress_min, stress_max, fixed_stress, fixed_temp


def _parse_upload(file_storage) -> list[dict]:
    buf = io.BytesIO(file_storage.read())
    name = (file_storage.filename or "").lower()
    df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={"Rh": "Re", "rh": "Re"}, inplace=True)
    df.drop(columns=[c for c in df.columns if c.lower() in _LIFETIME_COLS], inplace=True, errors="ignore")

    name_col = next((c for c in df.columns if c.lower() == "name"), None)
    if name_col is None:
        df.insert(0, "name", [f"제품 {i + 1}" for i in range(len(df))])
    else:
        df.rename(columns={name_col: "name"}, inplace=True)

    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if len(df) == 0:
        raise ValueError("유효한 제품 행이 없습니다.")
    return df.to_dict(orient="records")


def _jsonable(value: Any) -> Any:
    """Convert numpy / non-finite values before Flask jsonify."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _assistant_state(raw: Any):
    if not isinstance(raw, dict) or not raw:
        return None, None, None

    comp_raw = raw.get("composition") or {}
    ht_raw = raw.get("heat_treatment") or {}
    cond_raw = raw.get("conditions") or {}

    composition = {col: _finite_float(comp_raw.get(col, 0.0), 0.0) for col in _ALL_COMP}
    heat_treatment = {col: _finite_float(ht_raw.get(col, 0.0), 0.0) for col in _HT_COLS}

    if "stress" not in cond_raw or "temp" not in cond_raw:
        conditions = None
    else:
        conditions = {
            "stress": _positive_for_model(cond_raw.get("stress"), 1.0),
            "temp": _positive_for_model(cond_raw.get("temp"), 1.0),
        }

    return composition, conditions, heat_treatment


@app.get("/")
def index():
    return render_template("index.html", error=None)


@app.get("/health")
def health():
    try:
        _ensure_model()
        return jsonify({
            "status": "ok",
            "feature_count": len(getattr(_PREDICTOR, "feature_names", [])),
            "assistant_entrypoint": hasattr(assistant_pipeline, "run_assistant_pipeline"),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/predict")
def predict():
    _ensure_model()
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", error="파일을 선택해주세요."), 400

    try:
        products = _parse_upload(file)
    except Exception as exc:
        return render_template("index.html", error=f"파일 파싱 오류: {exc}"), 400

    def fv(key, default):
        return _finite_float(request.form.get(key, default), default)

    temp_min, temp_max, stress_min, stress_max, fixed_stress, fixed_temp = _sanitize_sweep_params(
        fv("temp_min", 600), fv("temp_max", 700),
        fv("stress_min", 30), fv("stress_max", 130),
        fv("fixed_stress", 100), fv("fixed_temp", 650),
    )

    temps = np.linspace(temp_min, temp_max, 80)
    stresses = np.logspace(np.log10(max(stress_min, 1.0)), np.log10(stress_max), 80)
    hm_t = np.linspace(temp_min, temp_max, 35)
    hm_s = np.logspace(np.log10(max(stress_min, 1.0)), np.log10(stress_max), 35)
    hm_T, hm_S = np.meshgrid(hm_t, hm_s)

    product_data = []
    for i, product in enumerate(products):
        fixed = _fixed_features(product)
        ts_log = _predict_batch(fixed, np.full(80, fixed_stress), temps)
        ss_log = _predict_batch(fixed, stresses, np.full(80, fixed_temp))
        hm_log = _predict_batch(fixed, hm_S, hm_T)
        lmp_vals = (fixed_temp * (20.0 + ss_log) / 1000.0).tolist()

        key_comp = {
            c: round(float(product.get(c) or 0.0), 4)
            for c in _ALL_COMP if float(product.get(c) or 0.0) != 0.0
        }
        key_ht = {
            c: round(float(product.get(c) or 0.0), 3)
            for c in _HT_COLS if float(product.get(c) or 0.0) != 0.0
        }
        comp_total = sum(key_comp.values())
        fe_balance = round(max(0.0, 100.0 - comp_total), 3)
        comp_pie = {"Fe (bal.)": fe_balance, **key_comp} if fe_balance > 0 else dict(key_comp)

        ht_stages = []
        for prefix, label in (("N", "노말라이징"), ("T", "템퍼링"), ("A", "시효처리")):
            temp_k = float(product.get(f"{prefix}temp") or 0.0)
            time_h = float(product.get(f"{prefix}time") or 0.0)
            if temp_k > 0:
                ht_stages.append({
                    "label": label,
                    "prefix": prefix,
                    "temp_k": temp_k,
                    "temp_c": round(temp_k - 273.15, 1),
                    "time_h": round(time_h, 3),
                })

        product_data.append({
            "name": str(product.get("name", f"제품 {i + 1}")),
            "composition": key_comp,
            "heat_treatment": key_ht,
            "comp_pie": comp_pie,
            "ht_stages": ht_stages,
            "_features": fixed,
            "temp_sweep": {
                "temps_k": temps.tolist(), "temps_c": (temps - 273.15).tolist(),
                "log10_hours": ts_log.tolist(), "hours": np.power(10, ts_log).tolist(),
            },
            "stress_sweep": {
                "stresses_mpa": stresses.tolist(), "log10_hours": ss_log.tolist(),
                "hours": np.power(10, ss_log).tolist(),
            },
            "lmp": {"stresses_mpa": stresses.tolist(), "lmp_vals": lmp_vals},
            "heatmap": {
                "temps_k": hm_t.tolist(), "stresses_mpa": hm_s.tolist(),
                "log10_hours_grid": hm_log.tolist(),
            },
        })

    payload = {
        "products": product_data,
        "fixed_stress": fixed_stress, "fixed_temp": fixed_temp,
        "temp_min": temp_min, "temp_max": temp_max,
        "stress_min": stress_min, "stress_max": stress_max,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return render_template("result.html", payload=payload)


@app.post("/suggest_params")
def suggest_params():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({})
    try:
        buf = io.BytesIO(file.read())
        name = (file.filename or "").lower()
        df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}
        result = {}

        sc = next((lower[k] for k in ("stress", "rupture_stress", "applied_stress") if k in lower), None)
        if sc:
            s = pd.to_numeric(df[sc], errors="coerce").dropna()
            s = s[s > 0]
            if len(s) >= 2:
                result.update(stress_min=round(float(s.min()), 1), stress_max=round(float(s.max()), 1), fixed_stress=round(float(s.median()), 1))

        tc = next((lower[k] for k in ("temp", "temperature", "test_temp") if k in lower), None)
        if tc:
            t = pd.to_numeric(df[tc], errors="coerce").dropna()
            t = t[t > 0]
            if len(t) >= 2:
                result.update(temp_min=int(round(float(t.min()))), temp_max=int(round(float(t.max()))), fixed_temp=int(round(float(t.median()))))
        return jsonify(result)
    except Exception:
        return jsonify({})


@app.post("/resweep")
def resweep():
    _ensure_model()
    data = request.get_json(force=True) or {}
    temp_min, temp_max, stress_min, stress_max, fixed_stress, fixed_temp = _sanitize_sweep_params(
        _finite_float(data.get("temp_min", 600), 600), _finite_float(data.get("temp_max", 700), 700),
        _finite_float(data.get("stress_min", 30), 30), _finite_float(data.get("stress_max", 130), 130),
        _finite_float(data.get("fixed_stress", 100), 100), _finite_float(data.get("fixed_temp", 650), 650),
    )
    temps = np.linspace(temp_min, temp_max, 80)
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
        results.append({
            "temp_sweep": {"temps_k": temps.tolist(), "temps_c": (temps - 273.15).tolist(), "log10_hours": ts_log.tolist(), "hours": np.power(10, ts_log).tolist()},
            "stress_sweep": {"stresses_mpa": stresses.tolist(), "log10_hours": ss_log.tolist(), "hours": np.power(10, ss_log).tolist()},
            "lmp": {"stresses_mpa": stresses.tolist(), "lmp_vals": (fixed_temp * (20.0 + ss_log) / 1000.0).tolist()},
            "heatmap": {"temps_k": hm_t.tolist(), "stresses_mpa": hm_s.tolist(), "log10_hours_grid": hm_log.tolist()},
        })
    return jsonify(_jsonable({
        "products": results,
        "fixed_stress": fixed_stress, "fixed_temp": fixed_temp,
        "temp_min": temp_min, "temp_max": temp_max,
        "stress_min": stress_min, "stress_max": stress_max,
    }))


@app.get("/assistant")
def assistant_page():
    return render_template("alloy_design.html")


@app.get("/alloy_design")
def legacy_alloy_design():
    """Backward-compatible URL from the first-semester UI."""
    return redirect(url_for("assistant_page"), code=302)


@app.post("/api/assistant")
def assistant_api():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "message": "질문을 입력해주세요."}), 400

    composition, conditions, heat_treatment = _assistant_state(data.get("state"))
    try:
        result = assistant_pipeline.run_assistant_pipeline(
            question=question,
            current_composition=composition,
            current_conditions=conditions,
            current_heat_treatment=heat_treatment,
            knowledge_top_k=5,
        )
        payload = _jsonable(result)
        payload["ok"] = True
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Vertex assistant failed")
        return jsonify({
            "ok": False,
            "message": f"AI Assistant 처리 중 오류가 발생했습니다: {type(exc).__name__}: {exc}",
        }), 503


if __name__ == "__main__":
    _ensure_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
