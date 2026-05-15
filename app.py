"""Vertex — 크립 수명 예측 (추론 전용 웹앱)

제품 조성·열처리 데이터(수명 컬럼 불필요)를 업로드하면
온도·응력 sweep을 통해 크립 수명을 예측하고 시각화합니다.
"""

import io
import json
import sys
import matplotlib
matplotlib.use("Agg")  # headless backend before any pyplot import

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing import (
    preprocess,
    COMPOSITION_COLS,
    HEAT_TREATMENT_COLS,
    EXTRA_COLS,
)

# ── Paths ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT_DIR / "uploads"
MODEL_PATH = ROOT_DIR / "models" / "xgb_baseline.json"
PREP_PATH  = ROOT_DIR / "data"   / "preprocessor.pkl"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Global model state ─────────────────────────────────────────────────────
_MODEL: XGBRegressor | None = None
_SCALER        = None
_FEATURE_NAMES: list[str] = []

def _ensure_model() -> None:
    global _MODEL, _SCALER, _FEATURE_NAMES
    if _MODEL is not None:
        return
    if MODEL_PATH.exists() and PREP_PATH.exists():
        m = XGBRegressor()
        m.load_model(str(MODEL_PATH))
        prep = joblib.load(PREP_PATH)
        _MODEL = m
        _SCALER = prep["scaler"]
        _FEATURE_NAMES = list(prep["feature_names"])
        print(f"[Vertex] 모델 로드 완료  피처 수: {len(_FEATURE_NAMES)}")
    else:
        print("[Vertex] 저장된 모델 없음 — XGBoost 학습 시작...")
        data = preprocess(save=True)
        m = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        )
        m.fit(data["X_train"], data["y_train"])
        m.save_model(str(MODEL_PATH))
        _MODEL = m
        _SCALER = data["scaler"]
        _FEATURE_NAMES = list(data["feature_names"])
        print(f"[Vertex] 학습 완료  저장: {MODEL_PATH}")

# ── Feature engineering ────────────────────────────────────────────────────
_ALL_COMP = COMPOSITION_COLS + EXTRA_COLS   # 18 base + Rh = 19
_HT_COLS  = HEAT_TREATMENT_COLS             # Ntemp, Ntime, Ttemp, Ttime, Atemp, Atime


def _fixed_features(product: dict) -> dict:
    """Composition + heat-treatment + severity features (no stress/temp yet)."""
    d: dict[str, float] = {}
    for col in _ALL_COMP:
        val = product.get(col)
        if val is None and col == "Rh":          # accept 'Re' as alias
            val = product.get("Re")
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
    shape = stresses.shape
    rows  = []
    for s, t in zip(stresses.ravel(), temps.ravel()):
        row = dict(fixed)
        row["stress"] = float(s)
        row["temp"]   = float(t)
        rows.append([row.get(col, 0.0) for col in _FEATURE_NAMES])
    X_scaled = _SCALER.transform(np.array(rows, dtype=float))
    return _MODEL.predict(X_scaled).reshape(shape)

# ── File parsing ───────────────────────────────────────────────────────────
_LIFETIME_COLS = {"lifetime", "log_lifetime", "rupture_time", "creep_life",
                  "creep_lifetime", "hours", "rupture_hours"}


def _parse_upload(file_storage) -> list[dict]:
    buf  = io.BytesIO(file_storage.read())
    name = (file_storage.filename or "").lower()
    df   = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop any lifetime-like columns (inference only)
    df.drop(columns=[c for c in df.columns if c.lower() in _LIFETIME_COLS],
            inplace=True, errors="ignore")

    # Ensure a product name column
    nc = next((c for c in df.columns if c.lower() == "name"), None)
    if nc is None:
        df.insert(0, "name", [f"Product {i+1}" for i in range(len(df))])
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


@app.get("/")
def index():
    return _INDEX_HTML


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
        for pfx, lbl in [("N", "Normalizing"), ("T", "Tempering"), ("A", "Aging")]:
            t_k = float(p.get(f"{pfx}temp") or 0.0)
            t_h = float(p.get(f"{pfx}time") or 0.0)
            if t_k > 0:
                ht_stages.append({"label": lbl, "prefix": pfx,
                                   "temp_k": t_k, "temp_c": round(t_k - 273.15, 1),
                                   "time_h": round(t_h, 2)})

        product_data.append({
            "name":           str(p.get("name", f"Product {i+1}")),
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

    rd = {
        "products":     product_data,
        "fixed_stress": fixed_stress,
        "fixed_temp":   fixed_temp,
        "temp_min":     temp_min,   "temp_max":  temp_max,
        "stress_min":   stress_min, "stress_max": stress_max,
    }
    return _RESULT_TMPL.replace("__DATA_JSON__", json.dumps(rd))


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


# ── HTML: Upload form ──────────────────────────────────────────────────────
_INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertex — 크립 수명 예측</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
.card{background:#1a1f30;border:1px solid #2a3048;border-radius:14px;padding:40px 48px;max-width:740px;width:100%;box-shadow:0 8px 40px rgba(0,0,0,.45)}
h1{font-size:1.9rem;color:#7eb8f7;margin-bottom:6px;font-weight:700;letter-spacing:-.01em}
.sub{color:#8898aa;font-size:.88rem;line-height:1.6;margin-bottom:32px}
.sub strong{color:#c0c8d8}
.sec{font-size:.75rem;font-weight:700;color:#7eb8f7;letter-spacing:.09em;text-transform:uppercase;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid #2a3048}
.upload-zone{border:2px dashed #3a4668;border-radius:10px;padding:32px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;margin-bottom:6px}
.upload-zone:hover{border-color:#7eb8f7;background:rgba(126,184,247,.04)}
.upload-zone input[type=file]{display:none}
.uz-icon{font-size:2.4rem;margin-bottom:8px}
.uz-label{font-size:.95rem;color:#c0c8d8;margin-bottom:4px}
.uz-hint{font-size:.75rem;color:#5a6a88}
.fname{font-size:.8rem;color:#7eb8f7;margin-top:5px;min-height:18px;text-align:center}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.field{display:flex;flex-direction:column;gap:4px}
.lbl{font-size:.78rem;color:#8898aa}
input[type=number]{background:#0f1117;border:1px solid #2a3048;border-radius:7px;color:#e0e0e0;padding:9px 11px;font-size:.88rem;width:100%;transition:border-color .2s}
input[type=number]:focus{outline:none;border-color:#7eb8f7}
.hint{font-size:.73rem;color:#5a6a88;margin-top:3px}
.btn{margin-top:28px;width:100%;padding:14px;background:linear-gradient(90deg,#3a6cf4,#7eb8f7);border:none;border-radius:9px;color:#fff;font-size:1rem;font-weight:700;cursor:pointer;letter-spacing:.04em;transition:opacity .2s}
.btn:hover{opacity:.87}
</style>
</head>
<body>
<div class="card">
  <h1>⚙ Vertex</h1>
  <p class="sub">제품 조성·열처리 데이터를 업로드하면 <strong>온도 및 응력 범위별 크립 수명</strong>을 예측합니다.<br>
  수명(lifetime) 컬럼 없이 구성 데이터만으로 분석 가능합니다.</p>

  <form method="POST" action="/predict" enctype="multipart/form-data">
    <div class="sec">데이터 파일</div>
    <div class="upload-zone" onclick="document.getElementById('fup').click()">
      <label for="fup" style="cursor:pointer">
        <div class="uz-icon">📂</div>
        <div class="uz-label">클릭하여 파일 선택</div>
        <div class="uz-hint">xlsx 또는 csv &nbsp;·&nbsp; 조성·열처리 컬럼 포함 &nbsp;·&nbsp; 수명 컬럼 불필요</div>
      </label>
      <input type="file" id="fup" name="file" accept=".xlsx,.csv" required
             onchange="document.getElementById('fname').textContent=this.files[0]?.name||''">
    </div>
    <div class="fname" id="fname"></div>

    <div class="sec">온도 Sweep 설정</div>
    <div class="g3">
      <div class="field">
        <span class="lbl">최소 온도 (K)</span>
        <input type="number" name="temp_min" value="700" step="10" min="300" max="2000">
      </div>
      <div class="field">
        <span class="lbl">최대 온도 (K)</span>
        <input type="number" name="temp_max" value="1200" step="10" min="300" max="2000">
      </div>
      <div class="field">
        <span class="lbl">고정 응력 (MPa)</span>
        <input type="number" name="fixed_stress" value="150" step="10" min="1" max="1000">
        <span class="hint">온도 sweep 시 고정값</span>
      </div>
    </div>

    <div class="sec">응력 Sweep 설정</div>
    <div class="g3">
      <div class="field">
        <span class="lbl">최소 응력 (MPa)</span>
        <input type="number" name="stress_min" value="10" step="5" min="1" max="1000">
      </div>
      <div class="field">
        <span class="lbl">최대 응력 (MPa)</span>
        <input type="number" name="stress_max" value="600" step="10" min="1" max="2000">
      </div>
      <div class="field">
        <span class="lbl">고정 온도 (K)</span>
        <input type="number" name="fixed_temp" value="873" step="10" min="300" max="2000">
        <span class="hint">응력 sweep 시 고정값</span>
      </div>
    </div>

    <button type="submit" class="btn">▶ &nbsp;수명 예측 실행</button>
  </form>
</div>
</body>
</html>"""


# ── HTML: Result page ──────────────────────────────────────────────────────
_RESULT_TMPL = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertex — 예측 결과</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif}
header{background:#1a1f30;border-bottom:1px solid #2a3048;padding:14px 32px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100}
header h1{font-size:1.2rem;color:#7eb8f7;font-weight:700}
.back{background:#2a3048;border:1px solid #3a4668;color:#c0c8d8;padding:7px 16px;border-radius:6px;text-decoration:none;font-size:.82rem;transition:background .2s}
.back:hover{background:#3a4668}
#ctrl{background:#12172a;border-bottom:1px solid #2a3048;padding:12px 28px;display:flex;align-items:flex-end;gap:20px;flex-wrap:wrap;position:sticky;top:49px;z-index:99}
.ctrl-grp{display:flex;flex-direction:column;gap:6px}
.ctrl-grp-title{font-size:.7rem;font-weight:700;color:#7eb8f7;letter-spacing:.07em;text-transform:uppercase}
.ctrl-row{display:flex;gap:10px;align-items:center}
.ctrl-field{display:flex;flex-direction:column;gap:3px}
.ctrl-lbl{font-size:.7rem;color:#8898aa}
.ctrl-inp{background:#0f1117;border:1px solid #2a3048;border-radius:5px;color:#e0e0e0;padding:5px 8px;font-size:.82rem;width:80px;transition:border-color .2s}
.ctrl-inp:focus{outline:none;border-color:#7eb8f7}
.ctrl-sep{width:1px;height:44px;background:#2a3048;margin:0 6px}
#apply-btn{padding:8px 20px;background:linear-gradient(90deg,#3a6cf4,#7eb8f7);border:none;border-radius:7px;color:#fff;font-size:.85rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:opacity .2s;margin-left:auto}
#apply-btn:hover{opacity:.85}
#apply-btn:disabled{opacity:.5;cursor:not-allowed}
.wrap{max-width:1440px;margin:0 auto;padding:24px 28px 48px}
.sec{font-size:.73rem;font-weight:700;color:#7eb8f7;letter-spacing:.09em;text-transform:uppercase;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid #2a3048}
.prod-wrap{overflow-x:auto}
table.pt{width:100%;border-collapse:collapse;font-size:.8rem;white-space:nowrap}
table.pt th{background:#1a1f30;color:#7eb8f7;padding:8px 14px;text-align:left;font-weight:600;border-bottom:1px solid #2a3048}
table.pt td{padding:7px 14px;border-bottom:1px solid #1e2538;color:#c0c8d8;vertical-align:top}
table.pt tr:hover td{background:#1e2538}
.pill{display:inline-block;background:#1e2846;color:#7eb8f7;border-radius:4px;padding:2px 8px;margin:2px 2px 2px 0;font-size:.72rem;white-space:nowrap}
.ch-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:4px}
.ch-box{background:#1a1f30;border:1px solid #2a3048;border-radius:10px;padding:16px 14px}
.ch-sub{font-size:.78rem;color:#8898aa;margin-bottom:10px}
.hm-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.hm-btn{background:#1e2538;border:1px solid #3a4668;color:#aab4cc;padding:5px 13px;border-radius:5px;font-size:.78rem;cursor:pointer;transition:background .15s,color .15s}
.hm-btn.active,.hm-btn:hover{background:#4a7cf7;border-color:#4a7cf7;color:#fff}
.ref{font-size:.72rem;color:#5a6a88;margin-top:8px;line-height:1.5}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin-top:4px}
.stat-card{background:#1a1f30;border:1px solid #2a3048;border-radius:8px;padding:10px 18px;flex:1;min-width:140px}
.stat-lbl{font-size:.72rem;color:#8898aa;margin-bottom:3px}
.stat-val{font-size:1.1rem;color:#7eb8f7;font-weight:700}
.ht-card{background:#12172a;border-radius:10px;padding:14px 16px;margin-bottom:10px;border-left:4px solid transparent}
.ht-card-name{font-size:.85rem;font-weight:700;margin-bottom:10px}
.ht-stage{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #1e2538}
.ht-stage:last-child{border-bottom:none}
.ht-badge{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:800;flex-shrink:0}
.ht-badge.N{background:#1a3a5c;color:#7eb8f7}
.ht-badge.T{background:#3a1a1a;color:#f76b6b}
.ht-badge.A{background:#1a3a1a;color:#6bf7c6}
.ht-right{flex:1}
.ht-stage-title{font-size:.78rem;font-weight:600;color:#c0c8d8;margin-bottom:3px}
.ht-stage-vals{display:flex;gap:12px;font-size:.75rem;color:#8898aa;margin-bottom:5px}
.ht-stage-vals b{color:#e0e0e0;font-weight:600}
.ht-bar-wrap{background:#0f1117;border-radius:3px;height:5px;overflow:hidden}
.ht-bar{height:100%;border-radius:3px}
.ht-bar.N{background:linear-gradient(90deg,#1a3a5c,#7eb8f7)}
.ht-bar.T{background:linear-gradient(90deg,#3a1a1a,#f76b6b)}
.ht-bar.A{background:linear-gradient(90deg,#1a3a1a,#6bf7c6)}
/* Loading overlay */
#lov{position:fixed;inset:0;background:rgba(8,11,20,.93);z-index:2000;display:none;flex-direction:column;align-items:center;justify-content:center}
#lov.show{display:flex}
#lv-canvas{position:absolute;inset:0;width:100%;height:100%}
.lv-content{position:relative;z-index:1;text-align:center;pointer-events:none}
.lv-spin{width:54px;height:54px;border:3px solid #1e2846;border-top-color:#4a7cf7;border-radius:50%;animation:lv-rot 1s linear infinite;margin:0 auto 22px}
@keyframes lv-rot{to{transform:rotate(360deg)}}
.lv-title{font-size:1.1rem;color:#7eb8f7;font-weight:700;letter-spacing:.04em;margin-bottom:10px}
.lv-step{font-size:.82rem;color:#6a7a9a;min-height:20px;transition:opacity .3s}
/* Dashboard */
.dash-box{background:#1a1f30;border:1px solid #2a3048;border-radius:12px;padding:20px;margin-top:4px}
.dash-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.dash-tab{background:#12172a;border:1px solid #2a3048;color:#8898aa;padding:6px 16px;border-radius:6px;font-size:.8rem;cursor:pointer;transition:all .15s;border-left-width:3px}
.dash-tab.active{color:#e0e0e0;background:#1e2846;border-color:#4a7cf7}
.dash-inner{display:flex;gap:20px;align-items:stretch}
.sliders-col{display:flex;gap:28px;align-items:center;justify-content:center;padding:16px 24px;background:#12172a;border-radius:10px;flex-shrink:0;border:1px solid #1e2538}
.sl-wrap{display:flex;flex-direction:column;align-items:center;gap:10px;user-select:none}
.sl-val{font-size:.92rem;font-weight:700;color:#7eb8f7;min-width:90px;text-align:center;background:#0f1117;border:1px solid #2a3048;border-radius:6px;padding:5px 8px;font-family:'Courier New',monospace}
.sl-track{height:200px;display:flex;align-items:center;justify-content:center}
input.vslider{writing-mode:vertical-lr;direction:rtl;-webkit-appearance:slider-vertical;width:44px;height:200px;cursor:pointer;accent-color:#4a7cf7;background:transparent;outline:none}
.sl-lbl{font-size:.68rem;color:#5a6a88;text-align:center;font-weight:700;text-transform:uppercase;letter-spacing:.07em}
.chart-col{flex:1;display:flex;flex-direction:column;gap:12px;min-width:0}
.life-row{display:flex;align-items:center;gap:24px;padding:14px 18px;background:#12172a;border-radius:10px;flex-wrap:wrap;border:1px solid #1e2538}
.life-big{font-size:2.8rem;font-weight:800;font-family:'Courier New',monospace;transition:color .4s;line-height:1}
.life-unit{font-size:.72rem;color:#5a6a88;margin-top:4px}
.life-hours{font-size:1rem;color:#c0c8d8;font-weight:600}
.life-sub{font-size:.72rem;color:#5a6a88;margin-top:2px}
.life-badge{padding:6px 16px;border-radius:20px;font-size:.78rem;font-weight:700;letter-spacing:.04em;white-space:nowrap}
.life-badge.safe{background:#0a2e1a;color:#6bf7c6;border:1px solid #1a5a3a}
.life-badge.caution{background:#2e2a0a;color:#f7e06b;border:1px solid #5a4a0a}
.life-badge.danger{background:#2e190a;color:#f7a06b;border:1px solid #5a2a0a}
.life-badge.critical{background:#2e0a0a;color:#f76b6b;border:1px solid #5a0a0a}
/* Collapsible details */
.det-section{margin-top:28px}
.det-header{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:8px 0;border-bottom:1px solid #2a3048;user-select:none}
.det-header .sec{margin:0;border:none;padding:0;pointer-events:none}
.det-arrow{font-size:1rem;color:#5a6a88;transition:transform .35s;display:flex;align-items:center;gap:6px;font-size:.78rem;color:#5a6a88}
.det-arrow.open{color:#7eb8f7}
.det-arrow.open .arr{transform:rotate(180deg)}
.arr{transition:transform .35s;display:inline-block}
.det-body{overflow:hidden;max-height:0;transition:max-height .55s cubic-bezier(.4,0,.2,1)}
.det-body.open{max-height:9999px}
</style>
</head>
<body>
<header>
  <h1>⚙ Vertex — 크립 수명 예측 결과</h1>
  <a class="back" href="/">← 새 분석</a>
</header>

<div id="ctrl">
  <div class="ctrl-grp">
    <div class="ctrl-grp-title">온도 Sweep</div>
    <div class="ctrl-row">
      <div class="ctrl-field"><span class="ctrl-lbl">최소 (K)</span><input class="ctrl-inp" id="c-tmin" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최대 (K)</span><input class="ctrl-inp" id="c-tmax" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">고정 응력 (MPa)</span><input class="ctrl-inp" id="c-fs" type="number" step="10" style="width:90px"></div>
    </div>
  </div>
  <div class="ctrl-sep"></div>
  <div class="ctrl-grp">
    <div class="ctrl-grp-title">응력 Sweep</div>
    <div class="ctrl-row">
      <div class="ctrl-field"><span class="ctrl-lbl">최소 (MPa)</span><input class="ctrl-inp" id="c-smin" type="number" step="5"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최대 (MPa)</span><input class="ctrl-inp" id="c-smax" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">고정 온도 (K)</span><input class="ctrl-inp" id="c-ft" type="number" step="10" style="width:90px"></div>
    </div>
  </div>
  <button id="apply-btn" onclick="applyParams()">▶ 적용</button>
</div>

<!-- Loading overlay -->
<div id="lov">
  <canvas id="lv-canvas"></canvas>
  <div class="lv-content">
    <div class="lv-spin"></div>
    <div class="lv-title">예측 계산 중...</div>
    <div class="lv-step" id="lv-step">데이터 준비 중…</div>
  </div>
</div>

<div class="wrap">

<div class="sec">분석 요약</div>
<div class="stats" id="stat-bar"></div>

<div class="sec">제품 정보</div>
<div class="prod-wrap"><table class="pt" id="prod-tbl"></table></div>

<div class="sec">조성 분석 (원소 구성, wt%)</div>
<div class="ch-box"><div id="ch-comp" style="height:320px"></div></div>

<div class="sec">열처리 공정</div>
<div class="ch-grid">
  <div class="ch-box">
    <div class="ch-sub">열처리 사이클 (온도-시간 선도)</div>
    <div id="ch-ht" style="height:300px"></div>
    <p class="ref">구간 너비 = 실제 유지 시간(h) 비율 반영 · 승온/냉각은 개략 표시</p>
  </div>
  <div class="ch-box">
    <div class="ch-sub">단계별 조건 요약</div>
    <div id="ht-cards" style="overflow-y:auto;max-height:320px;padding-right:4px"></div>
  </div>
</div>

<!-- Real-time Dashboard -->
<div class="sec">실시간 크립 예측 대시보드</div>
<div class="dash-box">
  <div class="dash-tabs" id="dash-tabs"></div>
  <div class="dash-inner">
    <div class="sliders-col">
      <div class="sl-wrap">
        <div class="sl-val" id="sl-t-val">— K</div>
        <div class="sl-track">
          <input type="range" class="vslider" id="sl-temp" min="0" max="1000" value="500">
        </div>
        <div class="sl-lbl">온도 (K)</div>
      </div>
      <div class="sl-wrap">
        <div class="sl-val" id="sl-s-val">— MPa</div>
        <div class="sl-track">
          <input type="range" class="vslider" id="sl-stress" min="0" max="1000" value="500">
        </div>
        <div class="sl-lbl">응력 (MPa)</div>
      </div>
    </div>
    <div class="chart-col">
      <div id="ch-dash" style="height:300px"></div>
      <div class="life-row">
        <div>
          <div class="life-big" id="life-val">—</div>
          <div class="life-unit">log₁₀(수명 / 시간)</div>
        </div>
        <div>
          <div class="life-hours" id="life-hours">—</div>
          <div class="life-sub">예측 수명</div>
        </div>
        <div class="life-badge safe" id="life-badge">—</div>
      </div>
    </div>
  </div>
</div>

<!-- Collapsible details -->
<div class="det-section">
  <div class="det-header" onclick="toggleDetails()">
    <span class="sec">상세 분석 차트</span>
    <span class="det-arrow" id="det-arrow">펼치기 &nbsp;<span class="arr">▼</span></span>
  </div>
  <div class="det-body" id="det-body">
    <div style="padding-top:16px">

<div class="sec">온도별 크립 수명 &nbsp;<span id="lbl-fs" style="color:#aab4cc;font-weight:400;text-transform:none;font-size:.8rem;letter-spacing:0"></span></div>
<div class="ch-box">
  <div class="ch-sub" id="sub-temp"></div>
  <div id="ch-temp" style="height:400px"></div>
</div>

<div class="sec">응력별 크립 수명 &nbsp;<span id="lbl-ft" style="color:#aab4cc;font-weight:400;text-transform:none;font-size:.8rem;letter-spacing:0"></span></div>
<div class="ch-box">
  <div class="ch-sub" id="sub-stress"></div>
  <div id="ch-stress" style="height:400px"></div>
</div>

<div class="ch-grid">
  <div class="ch-box">
    <div class="ch-sub">라르손-밀러 선도</div>
    <div id="ch-lmp" style="height:360px"></div>
    <p class="ref">LMP = T(K) × (20 + log₁₀(t[h])) / 1000 &nbsp;—&nbsp; 온도·시간의 등가 파라미터<br>
    고정 온도 <span id="lbl-lmp-t"></span> K 에서 응력을 변화시켜 산출</p>
  </div>
  <div class="ch-box">
    <div class="ch-sub">운전 안전 영역 히트맵</div>
    <div class="hm-bar" id="hm-sel"></div>
    <div id="ch-hm" style="height:320px"></div>
    <p class="ref">색상: 예측 log₁₀(수명 / 시간) &nbsp;·&nbsp; 등고선: 100h / 1,000h / 10,000h / 100,000h</p>
  </div>
</div>

    </div>
  </div>
</div>

</div><!-- /wrap -->

<script>
const RD = __DATA_JSON__;

const COLORS=['#4a7cf7','#f76b6b','#6bf7c6','#f7c66b','#c66bf7','#7bf76b','#f76bc6','#6bc6f7','#f7a06b','#a0f76b'];
const BASE={
  paper_bgcolor:'#1a1f30',plot_bgcolor:'#0f1117',
  font:{color:'#c0c8d8',size:12},
  margin:{l:65,r:20,t:24,b:56},
  legend:{bgcolor:'rgba(0,0,0,0)',bordercolor:'#2a3048',borderwidth:1,orientation:'v',yanchor:'top',y:1,xanchor:'left',x:1.01},
  xaxis:{gridcolor:'#1e2538',linecolor:'#2a3048',zerolinecolor:'#2a3048'},
  yaxis:{gridcolor:'#1e2538',linecolor:'#2a3048',zerolinecolor:'#2a3048'},
  hovermode:'x unified',
};
function lyt(e){return Object.assign({},BASE,e);}

const REF=[{v:2,label:'100h',color:'#f76b6b'},{v:3,label:'1,000h',color:'#f7c66b'},{v:4,label:'10,000h',color:'#6bf7c6'},{v:5,label:'100,000h',color:'#7eb8f7'}];
function refShapesY(){return REF.map(r=>({type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:r.v,y1:r.v,line:{color:r.color,width:1,dash:'dot'}}));}
function refAnnotY(){return REF.map(r=>({xref:'paper',x:0.01,yref:'y',y:r.v,text:r.label,showarrow:false,font:{size:10,color:r.color},xanchor:'left',yanchor:'bottom'}));}

(function(){
  document.getElementById('c-tmin').value=RD.temp_min;
  document.getElementById('c-tmax').value=RD.temp_max;
  document.getElementById('c-fs').value=RD.fixed_stress;
  document.getElementById('c-smin').value=RD.stress_min;
  document.getElementById('c-smax').value=RD.stress_max;
  document.getElementById('c-ft').value=RD.fixed_temp;
})();

function updateLabels(){
  document.getElementById('lbl-fs').textContent='(고정 응력 '+RD.fixed_stress+' MPa)';
  document.getElementById('lbl-ft').textContent='(고정 온도 '+RD.fixed_temp+' K)';
  document.getElementById('lbl-lmp-t').textContent=RD.fixed_temp;
  document.getElementById('sub-temp').textContent='응력 '+RD.fixed_stress+' MPa 고정 — 온도 '+RD.temp_min+'–'+RD.temp_max+' K sweep';
  document.getElementById('sub-stress').textContent='온도 '+RD.fixed_temp+' K 고정 — 응력 '+RD.stress_min+'–'+RD.stress_max+' MPa sweep (log)';
}
updateLabels();

(function(){
  const tbl=document.getElementById('prod-tbl');
  const hdr=tbl.insertRow();
  ['제품명','조성 (wt%)','열처리 조건'].forEach(h=>{const th=document.createElement('th');th.textContent=h;hdr.appendChild(th);});
  RD.products.forEach((p,i)=>{
    const tr=tbl.insertRow();
    tr.insertAdjacentHTML('beforeend',`<td><span style="color:${COLORS[i%COLORS.length]};font-weight:700">${p.name}</span></td><td>${Object.entries(p.composition).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('')||'—'}</td><td>${Object.entries(p.heat_treatment).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('')||'—'}</td>`);
  });
})();

(function(){
  const bar=document.getElementById('stat-bar');
  RD.products.forEach((p,i)=>{
    const maxLog=Math.max(...p.temp_sweep.log10_hours),minLog=Math.min(...p.temp_sweep.log10_hours);
    const clr=COLORS[i%COLORS.length];
    bar.insertAdjacentHTML('beforeend',`<div class="stat-card" style="border-top:3px solid ${clr}"><div class="stat-lbl">${p.name}</div><div class="stat-val">${maxLog.toFixed(2)} log₁₀h</div><div style="font-size:.72rem;color:#8898aa;margin-top:2px">온도 범위 내 최대 수명</div><div style="font-size:.78rem;color:#c0c8d8;margin-top:4px">범위: ${minLog.toFixed(2)} – ${maxLog.toFixed(2)}</div></div>`);
  });
})();

function renderTempChart(){
  Plotly.react('ch-temp',RD.products.map((p,i)=>({x:p.temp_sweep.temps_k,y:p.temp_sweep.log10_hours,customdata:p.temp_sweep.hours,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'T=%{x:.0f} K<br>log₁₀(수명)=%{y:.3f}<br>수명=%{customdata:,.0f} h<extra>'+p.name+'</extra>'})),
    lyt({xaxis:{title:'온도 (K)',gridcolor:'#1e2538'},yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},shapes:refShapesY(),annotations:refAnnotY()}),{responsive:true,displayModeBar:false});
}
function renderStressChart(){
  Plotly.react('ch-stress',RD.products.map((p,i)=>({x:p.stress_sweep.stresses_mpa,y:p.stress_sweep.log10_hours,customdata:p.stress_sweep.hours,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'응력=%{x:.1f} MPa<br>log₁₀(수명)=%{y:.3f}<br>수명=%{customdata:,.0f} h<extra>'+p.name+'</extra>'})),
    lyt({xaxis:{title:'응력 (MPa)',type:'log',gridcolor:'#1e2538'},yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},shapes:refShapesY(),annotations:refAnnotY()}),{responsive:true,displayModeBar:false});
}
function renderLMPChart(){
  Plotly.react('ch-lmp',RD.products.map((p,i)=>({x:p.lmp.lmp_vals,y:p.lmp.stresses_mpa,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'LMP=%{x:.3f}<br>응력=%{y:.1f} MPa<extra>'+p.name+'</extra>'})),
    lyt({margin:{l:65,r:20,t:24,b:56},xaxis:{title:'Larson-Miller Parameter (×10³)',gridcolor:'#1e2538'},yaxis:{title:'응력 (MPa)',type:'log',gridcolor:'#1e2538'},hovermode:'closest'}),{responsive:true,displayModeBar:false});
}
let _hmIdx=0;
function renderHeatmap(idx){
  if(idx!==undefined)_hmIdx=idx;
  document.querySelectorAll('.hm-btn').forEach((b,i)=>b.classList.toggle('active',i===_hmIdx));
  const hm=RD.products[_hmIdx].heatmap;
  const cs=[[0,'#0f1117'],[0.1,'#1a1f30'],[0.25,'#2a1f50'],[0.4,'#6b2070'],[0.55,'#c03060'],[0.7,'#e87840'],[0.85,'#f7c66b'],[1,'#ffffff']];
  Plotly.react('ch-hm',[
    {x:hm.temps_k,y:hm.stresses_mpa,z:hm.log10_hours_grid,type:'heatmap',zmin:1,zmax:6,colorscale:cs,colorbar:{title:{text:'log₁₀(h)',side:'right'},tickvals:[1,2,3,4,5,6],ticktext:['10h','100h','1k h','10k h','100k h','1M h'],thickness:14,len:0.9,tickcolor:'#c0c8d8',tickfont:{color:'#c0c8d8'},titlefont:{color:'#c0c8d8'}},hovertemplate:'T=%{x:.0f} K<br>응력=%{y:.0f} MPa<br>수명≈10^%{z:.2f} h<extra></extra>'},
    {x:hm.temps_k,y:hm.stresses_mpa,z:hm.log10_hours_grid,type:'contour',contours:{start:2,end:5,size:1,showlabels:true,labelfont:{size:9,color:'rgba(255,255,255,0.8)'}},colorscale:[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],showscale:false,hoverinfo:'skip',line:{color:'rgba(255,255,255,0.4)',width:1.2}},
  ],lyt({margin:{l:65,r:80,t:10,b:56},xaxis:{title:'온도 (K)',gridcolor:'#1e2538'},yaxis:{title:'응력 (MPa)',type:'log',gridcolor:'#1e2538'},hovermode:'closest'}),{responsive:true,displayModeBar:false});
}

const ELEM_CLR={'Fe (bal.)':'#2a3a50','Cr':'#e74c3c','Mo':'#9b59b6','W':'#3498db','Ni':'#27ae60','Co':'#1abc9c','V':'#f1c40f','Nb':'#e67e22','Ta':'#d35400','Al':'#bdc3c7','C':'#ecf0f1','Si':'#95a5a6','Mn':'#7f8c8d','N':'#5dade2','B':'#a9cce3','Cu':'#f39c12','P':'#f9e79f','S':'#fad7a0','Rh':'#c39bd3','Re':'#a2d9ce','O':'#f8c471'};
function renderCompositionChart(){
  const N=RD.products.length,cols=Math.min(N,4),rows=Math.ceil(N/cols);
  document.getElementById('ch-comp').style.height=Math.max(280,rows*280)+'px';
  Plotly.react('ch-comp',RD.products.map((p,i)=>{
    const r=Math.floor(i/cols),c=i%cols,W=1/cols,H=1/rows;
    const labels=Object.keys(p.comp_pie),values=Object.values(p.comp_pie);
    return {labels,values,type:'pie',hole:0.48,name:p.name,textinfo:'label+percent',textposition:'inside',insidetextorientation:'radial',textfont:{size:10,color:'#fff'},marker:{colors:labels.map(l=>ELEM_CLR[l]||'#5a6a88'),line:{color:'#0f1117',width:1.5}},domain:{x:[c*W+0.01,(c+1)*W-0.01],y:[1-(r+1)*H+0.02,1-r*H-0.02]},title:{text:'<b>'+p.name+'</b>',font:{color:COLORS[i%COLORS.length],size:12},position:'bottom center'},hovertemplate:'%{label}<br>%{value:.3f} wt%<br>%{percent}<extra></extra>'};
  }),{paper_bgcolor:'#1a1f30',plot_bgcolor:'#0f1117',font:{color:'#c0c8d8',size:11},showlegend:false,margin:{l:10,r:10,t:20,b:20}},{responsive:true,displayModeBar:false});
}
function renderThermalCycle(){
  const RAMP=0.4,GAP=0.3,RT=25;
  function buildTrace(p,ci){
    if(!p.ht_stages||!p.ht_stages.length)return null;
    const xs=[0],ys=[RT],txt=['상온 (25°C)'];let t=0;
    p.ht_stages.forEach((s,si)=>{
      t+=RAMP;xs.push(t);ys.push(s.temp_c);txt.push(s.label+': '+s.temp_c+'°C');
      t+=s.time_h;xs.push(t);ys.push(s.temp_c);txt.push(s.label+': '+s.time_h+'h 유지');
      t+=RAMP;xs.push(t);ys.push(RT);txt.push('냉각');
      if(si<p.ht_stages.length-1){t+=GAP;xs.push(t);ys.push(RT);txt.push('');}
    });
    const clr=COLORS[ci%COLORS.length];
    return {x:xs,y:ys,text:txt,name:p.name,type:'scatter',mode:'lines+markers',line:{color:clr,width:2.5},marker:{size:5,color:clr},fill:'tozeroy',fillcolor:'rgba('+parseInt(clr.slice(1,3),16)+','+parseInt(clr.slice(3,5),16)+','+parseInt(clr.slice(5,7),16)+',0.08)',hovertemplate:'%{text}<extra>'+p.name+'</extra>'};
  }
  const annots=[];
  if(RD.products.length===1){let t=0;(RD.products[0].ht_stages||[]).forEach(s=>{t+=RAMP;annots.push({x:t+s.time_h/2,y:s.temp_c,text:'<b>'+s.label+'</b><br>'+s.temp_c+'°C · '+s.time_h+'h',showarrow:false,yanchor:'bottom',yshift:8,font:{size:10,color:'#e0e0e0'},bgcolor:'rgba(26,31,48,0.85)',bordercolor:'#2a3048',borderwidth:1,borderpad:4});t+=s.time_h+RAMP+GAP;});}
  Plotly.react('ch-ht',RD.products.map(buildTrace).filter(Boolean),{paper_bgcolor:'#1a1f30',plot_bgcolor:'#0f1117',font:{color:'#c0c8d8',size:12},margin:{l:65,r:20,t:24,b:56},xaxis:{title:'시간 (h)',gridcolor:'#1e2538'},yaxis:{title:'온도 (°C)',gridcolor:'#1e2538',rangemode:'tozero'},legend:{bgcolor:'rgba(0,0,0,0)',bordercolor:'#2a3048',borderwidth:1},annotations:annots,hovermode:'x unified'},{responsive:true,displayModeBar:false});
}
function buildHTCards(){
  const STAGE_FULL={Normalizing:'Normalizing (소둔)',Tempering:'Tempering (템퍼링)',Aging:'Aging (시효)'};
  let html='';
  RD.products.forEach((p,i)=>{
    if(!p.ht_stages||!p.ht_stages.length)return;
    const clr=COLORS[i%COLORS.length],maxT=Math.max(...p.ht_stages.map(s=>s.temp_c));
    html+=`<div class="ht-card" style="border-left-color:${clr}"><div class="ht-card-name" style="color:${clr}">${p.name}</div>`;
    p.ht_stages.forEach(s=>{const barW=Math.round(s.temp_c/maxT*100);html+=`<div class="ht-stage"><div class="ht-badge ${s.prefix}">${s.prefix}</div><div class="ht-right"><div class="ht-stage-title">${STAGE_FULL[s.label]||s.label}</div><div class="ht-stage-vals"><span>🌡 <b>${s.temp_c}°C</b> (${s.temp_k} K)</span><span>⏱ <b>${s.time_h} h</b></span></div><div class="ht-bar-wrap"><div class="ht-bar ${s.prefix}" style="width:${barW}%"></div></div></div></div>`;});
    html+='</div>';
  });
  document.getElementById('ht-cards').innerHTML=html||'<p style="color:#5a6a88;font-size:.82rem">열처리 데이터 없음</p>';
}

(function(){
  const sel=document.getElementById('hm-sel');
  RD.products.forEach((p,i)=>{const b=document.createElement('button');b.className='hm-btn'+(i===0?' active':'');b.style.cssText='border-left:3px solid '+COLORS[i%COLORS.length];b.textContent=p.name;b.onclick=()=>renderHeatmap(i);sel.appendChild(b);});
  renderCompositionChart();renderThermalCycle();buildHTCards();
})();

/* ── Loading overlay ── */
let _raf=null,_stepTimer=null;
const LOV_STEPS=['데이터 준비 중…','온도 조건 계산 중…','응력 조건 계산 중…','안전 영역 분석 중…','결과 시각화 중…'];
function showLoading(){
  const canvas=document.getElementById('lv-canvas');
  const ctx=canvas.getContext('2d');
  canvas.width=window.innerWidth;canvas.height=window.innerHeight;
  const pts=Array.from({length:70},()=>({x:Math.random()*canvas.width,y:Math.random()*canvas.height,r:Math.random()*2+.8,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,a:Math.random()*.6+.15}));
  function frame(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(let i=0;i<pts.length;i++){
      const p=pts[i];p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=canvas.width;if(p.x>canvas.width)p.x=0;
      if(p.y<0)p.y=canvas.height;if(p.y>canvas.height)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(74,124,247,${p.a})`;ctx.fill();
      for(let j=i+1;j<pts.length;j++){const dx=p.x-pts[j].x,dy=p.y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<110){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle=`rgba(74,124,247,${(1-d/110)*.15})`;ctx.lineWidth=.7;ctx.stroke();}}
    }
    _raf=requestAnimationFrame(frame);
  }
  frame();
  let si=0;const stepEl=document.getElementById('lv-step');stepEl.textContent=LOV_STEPS[0];
  _stepTimer=setInterval(()=>{si=(si+1)%LOV_STEPS.length;stepEl.textContent=LOV_STEPS[si];},650);
  document.getElementById('lov').classList.add('show');
}
function hideLoading(){
  if(_raf){cancelAnimationFrame(_raf);_raf=null;}
  clearInterval(_stepTimer);
  document.getElementById('lov').classList.remove('show');
}

/* ── Dashboard ── */
let _dashIdx=0,_curTemp=RD.fixed_temp,_curStress=RD.fixed_stress;

function bilinearInterp(pidx,temp_k,stress_mpa){
  const hm=RD.products[pidx].heatmap;
  const ts=hm.temps_k,ss=hm.stresses_mpa,grid=hm.log10_hours_grid;
  const t=Math.max(ts[0],Math.min(ts[ts.length-1],temp_k));
  const s=Math.max(ss[0],Math.min(ss[ss.length-1],stress_mpa));
  let ti=ts.length-2;for(let i=0;i<ts.length-1;i++){if(ts[i]<=t&&t<=ts[i+1]){ti=i;break;}}
  let si=ss.length-2;for(let i=0;i<ss.length-1;i++){if(ss[i]<=s&&s<=ss[i+1]){si=i;break;}}
  const ft=(ts[ti+1]-ts[ti])>0?(t-ts[ti])/(ts[ti+1]-ts[ti]):0;
  const fs=(ss[si+1]-ss[si])>0?(s-ss[si])/(ss[si+1]-ss[si]):0;
  return grid[si][ti]*(1-ft)*(1-fs)+grid[si][ti+1]*ft*(1-fs)+grid[si+1][ti]*(1-ft)*fs+grid[si+1][ti+1]*ft*fs;
}

function updateDashboard(){
  const pidx=_dashIdx,hm=RD.products[pidx].heatmap;
  const logL=bilinearInterp(pidx,_curTemp,_curStress);
  const hrs=Math.pow(10,logL);
  const curve=hm.temps_k.map(t=>bilinearInterp(pidx,t,_curStress));
  let badge,cls;
  if(logL>=5){badge='장기 안전 ✅';cls='safe';}
  else if(logL>=4){badge='정상 운전 ✅';cls='safe';}
  else if(logL>=3){badge='주의 ⚠';cls='caution';}
  else if(logL>=2){badge='위험 ⚠';cls='danger';}
  else{badge='심각 ✕';cls='critical';}
  const lifeEl=document.getElementById('life-val');
  lifeEl.textContent=logL.toFixed(3);
  lifeEl.style.color=cls==='safe'?'#6bf7c6':cls==='caution'?'#f7e06b':cls==='danger'?'#f7a06b':'#f76b6b';
  document.getElementById('life-hours').textContent=hrs>=1e6?(hrs/1e6).toFixed(2)+' M시간':hrs>=1000?(hrs/1000).toFixed(1)+' K시간':hrs.toFixed(0)+' 시간';
  const badgeEl=document.getElementById('life-badge');badgeEl.textContent=badge;badgeEl.className='life-badge '+cls;
  document.getElementById('sl-t-val').textContent=Math.round(_curTemp)+' K';
  document.getElementById('sl-s-val').textContent=_curStress.toFixed(0)+' MPa';
  const clr=COLORS[pidx%COLORS.length];
  Plotly.react('ch-dash',[
    {x:hm.temps_k,y:curve,name:RD.products[pidx].name,type:'scatter',mode:'lines',line:{color:clr,width:2.5},hovertemplate:'T=%{x:.0f} K<br>log₁₀(수명)=%{y:.3f}<extra></extra>'},
    {x:[_curTemp],y:[logL],name:'현재 조건',type:'scatter',mode:'markers',marker:{size:14,color:'#fff',symbol:'circle',line:{color:clr,width:3}},hovertemplate:'T=%{x:.0f} K<br>log₁₀(수명)=%{y:.3f}<extra>현재 조건</extra>'},
  ],lyt({margin:{l:60,r:20,t:16,b:50},xaxis:{title:'온도 (K)',gridcolor:'#1e2538'},yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},shapes:[...refShapesY(),{type:'line',xref:'x',x0:_curTemp,x1:_curTemp,yref:'paper',y0:0,y1:1,line:{color:'rgba(255,255,255,.35)',width:1.5,dash:'dash'}}],annotations:refAnnotY(),hovermode:'closest',legend:{bgcolor:'rgba(0,0,0,0)',bordercolor:'#2a3048',borderwidth:1,orientation:'h',y:-0.18}}),{responsive:true,displayModeBar:false});
}

function initDashboard(){
  const tabs=document.getElementById('dash-tabs');tabs.innerHTML='';
  RD.products.forEach((p,i)=>{
    const btn=document.createElement('button');btn.className='dash-tab'+(i===0?' active':'');
    btn.style.borderLeftColor=COLORS[i%COLORS.length];btn.textContent=p.name;
    btn.onclick=()=>{_dashIdx=i;document.querySelectorAll('.dash-tab').forEach((b,j)=>b.classList.toggle('active',j===i));updateDashboard();};
    tabs.appendChild(btn);
  });
  const hm=RD.products[0].heatmap;
  const tMin=hm.temps_k[0],tMax=hm.temps_k[hm.temps_k.length-1];
  const sMin=hm.stresses_mpa[0],sMax=hm.stresses_mpa[hm.stresses_mpa.length-1];
  const sLogMin=Math.log10(sMin),sLogMax=Math.log10(sMax);
  _curTemp=Math.max(tMin,Math.min(tMax,RD.fixed_temp));
  _curStress=Math.max(sMin,Math.min(sMax,RD.fixed_stress));
  const slT=document.getElementById('sl-temp'),slS=document.getElementById('sl-stress');
  slT.min=0;slT.max=1000;slT.value=Math.round((_curTemp-tMin)/(tMax-tMin)*1000);
  slS.min=0;slS.max=1000;slS.value=Math.round((Math.log10(_curStress)-sLogMin)/(sLogMax-sLogMin)*1000);
  slT.oninput=()=>{_curTemp=tMin+(slT.value/1000)*(tMax-tMin);updateDashboard();};
  slS.oninput=()=>{_curStress=Math.pow(10,sLogMin+(slS.value/1000)*(sLogMax-sLogMin));updateDashboard();};
  updateDashboard();
}

/* ── Collapsible ── */
let _detOpen=false,_detRendered=false;
function toggleDetails(){
  _detOpen=!_detOpen;
  document.getElementById('det-body').classList.toggle('open',_detOpen);
  const arr=document.getElementById('det-arrow');arr.classList.toggle('open',_detOpen);
  arr.innerHTML=(_detOpen?'접기':'펼치기')+' &nbsp;<span class="arr">▼</span>';
  if(_detOpen&&!_detRendered){_detRendered=true;setTimeout(()=>{renderTempChart();renderStressChart();renderLMPChart();renderHeatmap(0);},100);}
}

/* ── Recalculate ── */
async function applyParams(){
  document.getElementById('apply-btn').disabled=true;
  showLoading();
  const params={
    temp_min:+document.getElementById('c-tmin').value,
    temp_max:+document.getElementById('c-tmax').value,
    fixed_stress:+document.getElementById('c-fs').value,
    stress_min:+document.getElementById('c-smin').value,
    stress_max:+document.getElementById('c-smax').value,
    fixed_temp:+document.getElementById('c-ft').value,
    products_features:RD.products.map(p=>p._features),
  };
  try{
    const resp=await fetch('/resweep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    if(!resp.ok)throw new Error('서버 오류 '+resp.status);
    const result=await resp.json();
    result.products.forEach((p,i)=>{RD.products[i].temp_sweep=p.temp_sweep;RD.products[i].stress_sweep=p.stress_sweep;RD.products[i].lmp=p.lmp;RD.products[i].heatmap=p.heatmap;});
    Object.assign(RD,{fixed_stress:result.fixed_stress,fixed_temp:result.fixed_temp,temp_min:params.temp_min,temp_max:params.temp_max,stress_min:params.stress_min,stress_max:params.stress_max});
    updateLabels();
    if(_detOpen){renderTempChart();renderStressChart();renderLMPChart();renderHeatmap();}
    _dashIdx=0;initDashboard();
  }catch(e){alert('재계산 실패: '+e.message);}
  finally{hideLoading();document.getElementById('apply-btn').disabled=false;}
}

initDashboard();
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ensure_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
