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
from flask import Flask, request, jsonify

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
    return _RESULT_TMPL.replace("__DATA_JSON__", json.dumps(rd, ensure_ascii=False))


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
    return _ALLOY_DESIGN_TMPL.replace(
        "__DATA_JSON__",
        json.dumps(payload, ensure_ascii=False, default=str)
    )


# ── HTML: Alloy design dashboard ──────────────────────────────────────────
_ALLOY_DESIGN_TMPL = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertex — 합금 설계 결과</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;min-height:100vh;padding:24px 32px}
a{color:#7eb8f7;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:1.75rem;color:#7eb8f7;font-weight:700;margin-bottom:4px}
.page-sub{color:#8898aa;font-size:.85rem;margin-bottom:28px}
.meta-bar{background:#1a1f30;border:1px solid #2a3048;border-radius:10px;padding:12px 18px;display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;font-size:.8rem;color:#8898aa}
.meta-bar span{color:#c0c8d8}.meta-bar strong{color:#7eb8f7}
.sec{font-size:.72rem;font-weight:700;color:#7eb8f7;letter-spacing:.09em;text-transform:uppercase;margin:28px 0 10px;padding-bottom:5px;border-bottom:1px solid #2a3048}
.best-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px}
@media(max-width:700px){.best-grid{grid-template-columns:1fr}}
.card{background:#1a1f30;border:1px solid #2a3048;border-radius:12px;padding:20px 24px}
.card-title{font-size:.72rem;color:#7eb8f7;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px}
.stat-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;font-size:.88rem}
.stat-row .lbl{color:#8898aa}
.stat-row .val{color:#e0e0e0;font-weight:600}
.stat-row .val.ok{color:#6bf7c6}
.stat-row .val.warn{color:#f7e06b}
.stat-row .val.bad{color:#f76b6b}
.life-num{font-size:2.2rem;font-weight:800;color:#6bf7c6;letter-spacing:-.02em;margin-bottom:2px}
.life-sub{font-size:.82rem;color:#8898aa}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pill{background:#12172a;border:1px solid #3a4668;border-radius:6px;padding:4px 10px;font-size:.78rem;color:#c0c8d8}
.pill strong{color:#7eb8f7}
.bar-row{margin-bottom:8px}
.bar-label{display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:3px;color:#8898aa}
.bar-label .val{color:#c0c8d8}
.bar-bg{background:#12172a;border-radius:4px;height:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;background:#4a7cf7;transition:width .4s}
.bar-fill.ok{background:#6bf7c6}
.bar-fill.warn{background:#f7e06b}
.bar-fill.bad{background:#f76b6b}
.candidates-scroll{display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;margin-top:4px}
.candidates-scroll::-webkit-scrollbar{height:5px}
.candidates-scroll::-webkit-scrollbar-track{background:#12172a}
.candidates-scroll::-webkit-scrollbar-thumb{background:#35466f;border-radius:3px}
.cand-card{background:#1a1f30;border:1px solid #2a3048;border-radius:10px;padding:16px;min-width:200px;max-width:220px;flex-shrink:0}
.cand-rank{font-size:.65rem;color:#7eb8f7;font-weight:700;margin-bottom:6px}
.cand-life{font-size:1.2rem;font-weight:700;color:#6bf7c6}
.cand-sub{font-size:.75rem;color:#8898aa;margin-bottom:8px}
.cand-pills{display:flex;flex-wrap:wrap;gap:4px}
.cand-pill{background:#12172a;border:1px solid #2a3048;border-radius:4px;padding:2px 7px;font-size:.72rem;color:#c0c8d8}
.empty{color:#5a6a88;font-size:.88rem;padding:24px 0;text-align:center}
.ood-ok{color:#6bf7c6}.ood-out{color:#f76b6b}.ood-unk{color:#8898aa}
</style>
</head>
<body>
<h1>⚙ Vertex — 합금 설계 결과</h1>
<p class="page-sub"><a href="/">← 크립 수명 예측으로 돌아가기</a></p>
<div id="meta-bar" class="meta-bar"></div>
<div id="best-section"></div>
<div class="sec">Pareto Top 후보</div>
<div id="candidates-section"></div>
<script>
var D=__DATA_JSON__;
function fmt(v,d){if(v==null||v===undefined)return'—';return parseFloat(v).toFixed(d!=null?d:3);}
function fmtHrs(log10){if(log10==null)return'—';var h=Math.pow(10,parseFloat(log10));if(h>87600)return(h/8760).toFixed(0)+'년';if(h>1000)return(h/1000).toFixed(1)+'k h';return h.toFixed(0)+' h';}
function barCls(v,okMax,warnMax){if(v<=okMax)return'ok';if(v<=warnMax)return'warn';return'bad';}

/* Meta bar */
var mb=document.getElementById('meta-bar');
var m=D.meta||{};
var cond=m.conditions||{};
var parts=[];
if(m.generated_at) parts.push('<span>생성: <strong>'+m.generated_at+'</strong></span>');
if(cond.temp_c!=null) parts.push('<span>온도: <strong>'+(cond.temp_c)+' °C</strong></span>');
if(cond.stress_mpa!=null) parts.push('<span>응력: <strong>'+cond.stress_mpa+' MPa</strong></span>');
if(cond.cost_limit!=null) parts.push('<span>비용 한도: <strong>$'+cond.cost_limit+'/kg</strong></span>');
if(m.count!=null) parts.push('<span>후보 수: <strong>'+m.count+'</strong></span>');
if(!parts.length) parts.push('<span>GA 결과 파일을 읽었습니다.</span>');
mb.innerHTML=parts.join('');

/* Best alloy */
var bs=document.getElementById('best-section');
if(!D.best){
  bs.innerHTML='<div class="empty">아직 GA 최적화 결과가 없습니다. <code>python ga/engine.py</code>를 실행해 결과를 생성해 주세요.</div>';
} else {
  var b=D.best;
  var logL=b.predicted_log_life;
  var lifeCls=logL>=5?'ok':logL>=4?'ok':logL>=3?'warn':'bad';
  var costCls=b.material_cost<30?'ok':b.material_cost<60?'warn':'bad';
  var riskCls=b.physics_risk<10?'ok':b.physics_risk<50?'warn':'bad';
  var oodTxt=b.ood_is_out_of_distribution===true?'<span class="ood-out">범위 이탈</span>':b.ood_is_out_of_distribution===false?'<span class="ood-ok">범위 내</span>':'<span class="ood-unk">—</span>';
  var calTxt=b.thermo_success?'<span class="ood-ok">성공</span>':'<span class="ood-out">미수행/실패</span>';

  /* Composition pills */
  var ELEMS=['C','Si','Mn','P','S','Cr','Mo','W','Ni','Cu','V','Nb','N','Al','B','Co','Ta','O','Re'];
  var pillsHtml='';
  for(var i=0;i<ELEMS.length;i++){var e=ELEMS[i];var v=b[e];if(v&&parseFloat(v)>0) pillsHtml+='<span class="pill"><strong>'+e+'</strong> '+fmt(v,3)+'%</span>';}
  if(b.Fe_balance&&parseFloat(b.Fe_balance)>0) pillsHtml='<span class="pill"><strong>Fe</strong> '+fmt(b.Fe_balance,2)+'%</span>'+pillsHtml;

  /* Phase bars */
  var phases=[
    {key:'bcc_fraction',label:'BCC_A2',okMax:.8,warnMax:1.0},
    {key:'m23c6_fraction',label:'M₂₃C₆',okMax:.05,warnMax:.1},
    {key:'laves_fraction',label:'Laves',okMax:.02,warnMax:.05},
    {key:'sigma_fraction',label:'Sigma',okMax:.01,warnMax:.03},
    {key:'fcc_fraction',label:'FCC',okMax:.05,warnMax:.1},
  ];
  var phBarsHtml='';
  for(var pi=0;pi<phases.length;pi++){
    var ph=phases[pi];var v=b[ph.key]!=null?parseFloat(b[ph.key]):null;
    var pct=v!=null?Math.min(v*100,100):0;
    var cls=v!=null?barCls(v,ph.okMax,ph.warnMax):'';
    phBarsHtml+='<div class="bar-row"><div class="bar-label"><span>'+ph.label+'</span><span class="val">'+(v!=null?pct.toFixed(2)+'%':'—')+'</span></div><div class="bar-bg"><div class="bar-fill '+cls+'" style="width:'+pct+'%"></div></div></div>';
  }

  bs.innerHTML='<div class="sec">최적 합금 후보</div>'+
  '<div class="best-grid">'+
  '<div class="card">'+
  '<div class="card-title">예측 수명</div>'+
  '<div class="life-num '+lifeCls+'">'+fmtHrs(logL)+'</div>'+
  '<div class="life-sub">log₁₀(수명) = '+(logL!=null?fmt(logL,3):'—')+'</div>'+
  '<div style="margin-top:18px">'+
  '<div class="stat-row"><span class="lbl">재료 비용</span><span class="val '+costCls+'">$'+fmt(b.material_cost,2)+'/kg</span></div>'+
  '<div class="stat-row"><span class="lbl">물리 위험도</span><span class="val '+riskCls+'">'+fmt(b.physics_risk,2)+'</span></div>'+
  '<div class="stat-row"><span class="lbl">OOD 상태</span><span class="val">'+oodTxt+'</span></div>'+
  '<div class="stat-row"><span class="lbl">CALPHAD</span><span class="val">'+calTxt+'</span></div>'+
  '</div>'+
  '<div style="margin-top:12px"><div class="card-title" style="margin-bottom:8px">야금 지표</div>'+
  '<div class="stat-row"><span class="lbl">KN</span><span class="val">'+fmt(b.KN,3)+'</span></div>'+
  '<div class="stat-row"><span class="lbl">Ms 온도</span><span class="val">'+(b.Ms_temp!=null?fmt(b.Ms_temp,1)+' °C':'—')+'</span></div>'+
  '<div class="stat-row"><span class="lbl">CEQ</span><span class="val">'+fmt(b.CEQ,3)+'</span></div>'+
  '<div class="stat-row"><span class="lbl">총 합금량</span><span class="val">'+fmt(b.total_alloy_wt,2)+'%</span></div>'+
  '</div>'+
  '</div>'+
  '<div class="card">'+
  '<div class="card-title">조성 (wt%)</div>'+
  '<div class="pills">'+pillsHtml+'</div>'+
  '<div style="margin-top:20px"><div class="card-title" style="margin-bottom:10px">상 안정성 (CALPHAD)</div>'+
  phBarsHtml+
  '</div>'+
  '</div>'+
  '</div>';
}

/* Candidates */
var cs=document.getElementById('candidates-section');
if(!D.candidates||!D.candidates.length){
  cs.innerHTML='<div class="empty">후보 목록이 없습니다.</div>';
} else {
  var html='<div class="candidates-scroll">';
  for(var ci=0;ci<D.candidates.length;ci++){
    var c=D.candidates[ci];
    var logL=c.predicted_log_life;
    var cls=logL>=5?'ok':logL>=4?'ok':logL>=3?'warn':'bad';
    var ELEMS2=['C','Si','Mn','Cr','Mo','W','Ni','V','Nb','N','B'];
    var cpHtml='';
    for(var ei=0;ei<ELEMS2.length;ei++){var e=ELEMS2[ei];var v=c[e];if(v&&parseFloat(v)>0.001) cpHtml+='<span class="cand-pill">'+e+' '+fmt(v,2)+'</span>';}
    html+='<div class="cand-card">'+
      '<div class="cand-rank">#'+(ci+1)+'</div>'+
      '<div class="cand-life '+cls+'">'+fmtHrs(logL)+'</div>'+
      '<div class="cand-sub">$'+fmt(c.material_cost,1)+'/kg &nbsp;·&nbsp; risk '+fmt(c.physics_risk,1)+'</div>'+
      '<div class="cand-pills">'+cpHtml+'</div>'+
      '</div>';
  }
  html+='</div>';
  cs.innerHTML=html;
}
</script>
</body>
</html>"""


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
.upload-zone label{display:block;width:100%;cursor:pointer}
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
  수명(lifetime) 컬럼 없이 구성 데이터만으로 분석 가능합니다.<br>
  <a href="/alloy_design" style="color:#7eb8f7;font-size:.83rem">⚗ GA 합금 설계 결과 보기 →</a></p>

  <form method="POST" action="/predict" enctype="multipart/form-data">
    <div class="sec">데이터 파일</div>
    <div class="upload-zone">
      <label for="fup" style="cursor:pointer">
        <div class="uz-icon">📂</div>
        <div class="uz-label">클릭하여 파일 선택</div>
        <div class="uz-hint">xlsx 또는 csv &nbsp;·&nbsp; 조성·열처리 컬럼 포함 &nbsp;·&nbsp; 수명 컬럼 불필요</div>
      </label>
      <input type="file" id="fup" name="file" accept=".xlsx,.csv" required
             onchange="document.getElementById('fname').textContent=this.files[0]?.name||'';suggestParams(this.files[0]);">
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
<script>
function suggestParams(file){
  if(!file) return;
  var fd=new FormData(); fd.append('file',file);
  fetch('/suggest_params',{method:'POST',body:fd})
    .then(function(r){return r.json();})
    .then(function(d){
      function set(name,val){
        if(val===undefined||val===null) return;
        var el=document.querySelector('[name="'+name+'"]');
        if(el) el.value=val;
      }
      set('temp_min',d.temp_min);
      set('temp_max',d.temp_max);
      set('fixed_stress',d.fixed_stress);
      set('stress_min',d.stress_min);
      set('stress_max',d.stress_max);
      set('fixed_temp',d.fixed_temp);
    })
    .catch(function(){});
}
</script>
</body>
</html>"""


# ── HTML: Result page ──────────────────────────────────────────────────────
_RESULT_TMPL = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertex — 예측 결과</title>
<style>
:root{
  --scroll-track:#12172a;
  --scroll-thumb:#35466f;
  --scroll-thumb-hover:#4a7cf7;
}
*{box-sizing:border-box;margin:0;padding:0}
*{
  scrollbar-width:thin;
  scrollbar-color:var(--scroll-thumb) var(--scroll-track);
}
*::-webkit-scrollbar{height:12px;width:12px}
*::-webkit-scrollbar-track{background:var(--scroll-track);border-radius:999px}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--scroll-thumb),#27314c);border:2px solid var(--scroll-track);border-radius:999px}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--scroll-thumb-hover),#5f90ff)}
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
.ctrl-inp{background:#0f1117;border:1px solid #2a3048;border-radius:5px;color:#e0e0e0;padding:7px 9px;font-size:.82rem;width:110px;transition:border-color .2s}
.ctrl-inp:focus{outline:none;border-color:#7eb8f7}
.ctrl-sep{width:1px;height:44px;background:#2a3048;margin:0 6px}
#apply-btn{padding:8px 20px;background:linear-gradient(90deg,#3a6cf4,#7eb8f7);border:none;border-radius:7px;color:#fff;font-size:.85rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:opacity .2s;margin-left:auto}
#apply-btn:hover{opacity:.85}
#apply-btn:disabled{opacity:.5;cursor:not-allowed}
.wrap{max-width:1440px;margin:0 auto;padding:24px 28px 48px}
.sec{font-size:.73rem;font-weight:700;color:#7eb8f7;letter-spacing:.09em;text-transform:uppercase;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid #2a3048}
.scroll-shell{max-width:100%;overflow-x:auto;overflow-y:hidden;padding-bottom:10px}
.prod-grid{display:flex;flex-wrap:nowrap;gap:16px;width:max-content;min-width:100%}
.prod-card{background:#1a1f30;border:1px solid #2a3048;border-radius:12px;padding:16px;flex:0 0 360px;max-width:360px}
.prod-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}
.prod-title{font-size:.98rem;font-weight:700}
.prod-meta{font-size:.72rem;color:#8898aa}
.prod-block{padding-top:10px;margin-top:10px;border-top:1px solid #232a40}
.prod-block:first-of-type{padding-top:0;margin-top:0;border-top:none}
.prod-block-title{font-size:.72rem;color:#7eb8f7;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px}
.pill-wrap{display:flex;flex-wrap:wrap;gap:6px}
.pill-row{display:flex;flex-wrap:nowrap;gap:6px;overflow-x:auto;overflow-y:hidden;padding-bottom:4px}
.pill{display:inline-block;background:#1e2846;color:#7eb8f7;border-radius:4px;padding:2px 8px;margin:2px 2px 2px 0;font-size:.72rem;white-space:nowrap}
.ch-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;margin-top:4px}
.info-layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:22px;align-items:start;margin-top:4px}
@media(max-width:960px){.info-layout{grid-template-columns:1fr}}
.ht-layout{display:grid;grid-template-columns:minmax(0,1fr);gap:18px;margin-top:4px}
.ch-box{min-width:0;background:#1a1f30;border:1px solid #2a3048;border-radius:10px;padding:16px 14px;overflow:hidden}
.ch-sub{font-size:.78rem;color:#8898aa;margin-bottom:10px}
.hm-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.hm-btn{background:#1e2538;border:1px solid #3a4668;color:#aab4cc;padding:5px 13px;border-radius:5px;font-size:.78rem;cursor:pointer;transition:background .15s,color .15s}
.hm-btn.active,.hm-btn:hover{background:#4a7cf7;border-color:#4a7cf7;color:#fff}
.ref{font-size:.72rem;color:#5a6a88;margin-top:8px;line-height:1.5}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin-top:4px}
.stat-card{background:#1a1f30;border:1px solid #2a3048;border-radius:10px;padding:14px 18px;flex:1;min-width:200px}
.stat-lbl{font-size:.72rem;color:#8898aa;margin-bottom:6px}
.stat-val{font-size:1.25rem;color:#dfe8ff;font-weight:800;line-height:1.25}
.stat-sub{font-size:.78rem;color:#7eb8f7;margin-top:6px}
.stat-range{font-size:.78rem;color:#c0c8d8;margin-top:8px;line-height:1.5}
.ht-cards-row{display:flex;flex-wrap:nowrap;gap:14px;width:100%;overflow-x:auto;overflow-y:hidden;padding-bottom:8px}
.ht-card{background:#12172a;border-radius:10px;padding:14px 16px;border-left:4px solid transparent;flex:0 0 320px;max-width:320px}
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
/* GA section */
.ga-layout{display:flex;gap:18px;align-items:stretch;margin-top:4px}
@media(max-width:900px){.ga-layout{flex-direction:column}}
.ga-card-box{flex:0 0 280px;min-width:0;display:flex;flex-direction:column}
.ga-card-title{font-size:.7rem;font-weight:700;color:#7eb8f7;letter-spacing:.07em;text-transform:uppercase;margin-bottom:8px}
.ga-life-num{font-size:2rem;font-weight:800;color:#6bf7c6;letter-spacing:-.02em;margin-bottom:2px}
.ga-life-sub{font-size:.8rem;color:#8898aa;margin-bottom:14px}
.ga-stats{display:flex;flex-direction:column;gap:5px;margin-bottom:4px}
.ga-stat{display:flex;justify-content:space-between;font-size:.82rem}
.ga-stat .lbl{color:#8898aa}.ga-stat .val{font-weight:600;color:#e0e0e0}
.ga-stat .val.ok{color:#6bf7c6}.ga-stat .val.warn{color:#f7e06b}.ga-stat .val.bad{color:#f76b6b}
.ga-bar-row{margin-bottom:6px}
.ga-bar-label{display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:2px;color:#8898aa}
.ga-bar-label span:last-child{color:#c0c8d8}
.ga-bar-bg{background:#12172a;border-radius:3px;height:6px;overflow:hidden}
.ga-bar-fill{height:100%;border-radius:3px}
.ga-bar-fill.ok{background:#6bf7c6}.ga-bar-fill.warn{background:#f7e06b}.ga-bar-fill.bad{background:#f76b6b}
.ga-meta-bar{font-size:.75rem;color:#5a6a88;margin-top:10px;display:flex;flex-wrap:wrap;gap:12px}
.ga-meta-bar span{color:#8898aa}.ga-meta-bar strong{color:#7eb8f7}
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
.dash-inner{display:grid;grid-template-columns:1fr;gap:16px;align-items:stretch}
.sliders-col{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:16px;padding:0;background:transparent;border:none}
.sl-wrap{display:flex;flex-direction:column;gap:10px;user-select:none;background:#12172a;border-radius:12px;border:1px solid #1e2538;padding:14px 16px}
.sl-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.sl-val{font-size:.92rem;font-weight:700;color:#7eb8f7;min-width:110px;text-align:center;background:#0f1117;border:1px solid #2a3048;border-radius:6px;padding:5px 8px;font-family:'Courier New',monospace}
.sl-track{height:auto;display:flex;align-items:center;justify-content:center}
input.vslider{-webkit-appearance:none;appearance:none;width:100%;height:8px;cursor:pointer;accent-color:#4a7cf7;background:transparent;outline:none}
input.vslider::-webkit-slider-runnable-track{height:8px;background:#27314c;border-radius:999px}
input.vslider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;border-radius:50%;background:#7eb8f7;border:2px solid #dfe8ff;margin-top:-5px}
input.vslider::-moz-range-track{height:8px;background:#27314c;border-radius:999px}
input.vslider::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#7eb8f7;border:2px solid #dfe8ff}
.sl-lbl{font-size:.74rem;color:#8898aa;font-weight:700;letter-spacing:.05em}
.sl-note{font-size:.74rem;color:#5a6a88}
.chart-col{flex:1;display:flex;flex-direction:column;gap:12px;min-width:0}
.life-row{display:grid;grid-template-columns:minmax(220px,1.3fr) minmax(200px,1fr) auto;align-items:center;gap:20px;padding:16px 18px;background:#12172a;border-radius:10px;border:1px solid #1e2538}
.life-big{font-size:2rem;font-weight:800;transition:color .4s;line-height:1.2}
.life-unit{font-size:.78rem;color:#5a6a88;margin-top:6px}
.life-hours{font-size:1rem;color:#c0c8d8;font-weight:700}
.life-sub{font-size:.78rem;color:#7eb8f7;margin-top:6px}
.life-meta{font-size:.72rem;color:#8898aa;margin-top:6px;line-height:1.5}
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
@media (max-width: 980px){
  .ch-grid{grid-template-columns:1fr}
  .sliders-col{grid-template-columns:1fr}
  .life-row{grid-template-columns:1fr}
  #ctrl{padding:12px 16px}
  .prod-card,.ht-card{flex-basis:300px;max-width:300px}
}
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
      <div class="ctrl-field"><span class="ctrl-lbl">응력 고정값 (MPa)</span><input class="ctrl-inp" id="c-fs" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최소 온도 (K)</span><input class="ctrl-inp" id="c-tmin" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최대 온도 (K)</span><input class="ctrl-inp" id="c-tmax" type="number" step="10"></div>
    </div>
  </div>
  <div class="ctrl-sep"></div>
  <div class="ctrl-grp">
    <div class="ctrl-grp-title">응력 Sweep</div>
    <div class="ctrl-row">
      <div class="ctrl-field"><span class="ctrl-lbl">온도 고정값 (K)</span><input class="ctrl-inp" id="c-ft" type="number" step="10"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최소 응력 (MPa)</span><input class="ctrl-inp" id="c-smin" type="number" step="5"></div>
      <div class="ctrl-field"><span class="ctrl-lbl">최대 응력 (MPa)</span><input class="ctrl-inp" id="c-smax" type="number" step="10"></div>
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

<div class="info-layout">
  <div>
    <div class="sec">제품 정보</div>
    <div class="scroll-shell">
      <div class="prod-grid" id="prod-grid"></div>
    </div>
  </div>
  <div>
    <div class="sec">조성 분석 (원소 구성, wt%)</div>
    <div class="ch-box" id="comp-box" style="overflow-x:auto;overflow-y:hidden">
      <div id="ch-comp" style="height:320px"></div>
    </div>
  </div>
</div>

<div class="sec">열처리 공정</div>
<div class="ht-layout">
  <div class="ch-box">
    <div class="ch-sub">열처리 사이클 (온도-시간 선도)</div>
    <div id="ch-ht" style="height:300px"></div>
    <p class="ref">구간 너비 = 실제 유지 시간(h) 비율 반영 · 승온/냉각은 개략 표시</p>
  </div>
  <div class="ch-box">
    <div class="ch-sub">단계별 조건 요약</div>
    <div class="ht-cards-row" id="ht-cards"></div>
  </div>
</div>

<!-- Real-time Dashboard -->
<div class="sec">실시간 크립 예측 대시보드</div>
<div class="dash-box">
  <div class="dash-tabs" id="dash-tabs"></div>
  <div class="dash-inner">
    <div class="sliders-col">
      <div class="sl-wrap">
        <div class="sl-head">
          <div class="sl-lbl">온도 조절</div>
          <div class="sl-val" id="sl-t-val">— K</div>
        </div>
        <div class="sl-track">
          <input type="range" class="vslider" id="sl-temp" min="0" max="1000" value="500">
        </div>
        <div class="sl-note">현재 비교 중인 제품의 운전 온도를 조절합니다.</div>
      </div>
      <div class="sl-wrap">
        <div class="sl-head">
          <div class="sl-lbl">응력 조절</div>
          <div class="sl-val" id="sl-s-val">— MPa</div>
        </div>
        <div class="sl-track">
          <input type="range" class="vslider" id="sl-stress" min="0" max="1000" value="500">
        </div>
        <div class="sl-note">현재 비교 중인 제품의 운전 응력을 조절합니다.</div>
      </div>
    </div>
    <div class="chart-col">
      <div id="ch-dash" style="height:300px"></div>
      <div class="life-row">
        <div>
          <div class="life-big" id="life-val">—</div>
          <div class="life-unit">현재 조건에서의 예상 크립 수명</div>
        </div>
        <div>
          <div class="life-hours" id="life-hours">—</div>
          <div class="life-sub" id="life-sub">—</div>
          <div class="life-meta" id="life-meta">—</div>
        </div>
        <div class="life-badge safe" id="life-badge">—</div>
      </div>
    </div>
  </div>
</div>

<!-- GA 합금 설계 추천 -->
<div id="ga-section" style="display:none">
  <div class="sec">⚗ GA 합금 설계 추천</div>
  <div class="ga-layout">
    <div class="ch-box ga-card-box">
      <div class="ga-card-title">최적 합금 후보</div>
      <div id="ga-life-num" class="ga-life-num">—</div>
      <div id="ga-life-sub" class="ga-life-sub">—</div>
      <div class="ga-stats" id="ga-stats"></div>
      <div class="ga-card-title" style="margin-top:16px">조성 (wt%)</div>
      <div class="pills" id="ga-pills"></div>
      <div class="ga-card-title" style="margin-top:16px">상 안정성 (CALPHAD)</div>
      <div id="ga-phases"></div>
    </div>
    <div class="ch-box" style="min-width:0;flex:1">
      <div class="ch-sub">온도별 수명 비교 (고정 응력 기준)</div>
      <div id="ch-ga-comp" style="height:320px"></div>
    </div>
  </div>
  <div class="ga-meta-bar" id="ga-meta-bar"></div>
</div>

<!-- Collapsible details -->
<div class="det-section">
  <div class="det-header" onclick="toggleDetails()">
    <span class="sec">상세 분석 차트</span>
    <span class="det-arrow open" id="det-arrow">접기 &nbsp;<span class="arr">▼</span></span>
  </div>
  <div class="det-body open" id="det-body">
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
let PLOTLY_READY = false;

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

function formatNumberKo(value, digits=0){
  return new Intl.NumberFormat('ko-KR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatHoursCompact(hours, withPrefix=true){
  if(!Number.isFinite(hours) || hours <= 0){
    return '계산 불가';
  }
  const prefix = withPrefix ? '약 ' : '';
  if(hours >= 1e8) return `${prefix}${formatNumberKo(hours / 1e8, 2)}억 시간`;
  if(hours >= 1e6) return `${prefix}${formatNumberKo(hours / 1e6, 2)}백만 시간`;
  if(hours >= 1e4) return `${prefix}${formatNumberKo(hours / 1e4, 1)}만 시간`;
  if(hours >= 1e3) return `${prefix}${formatNumberKo(hours, 0)}시간`;
  if(hours >= 10) return `${prefix}${formatNumberKo(hours, 1)}시간`;
  return `${prefix}${formatNumberKo(hours, 2)}시간`;
}

function formatYearsCompact(hours){
  if(!Number.isFinite(hours) || hours <= 0){
    return '환산 불가';
  }
  const years = hours / 8760;
  if(years >= 100) return `약 ${formatNumberKo(years, 0)}년`;
  if(years >= 10) return `약 ${formatNumberKo(years, 1)}년`;
  return `약 ${formatNumberKo(years, 2)}년`;
}

function setChartFallback(id, message){
  const el=document.getElementById(id);
  if(!el)return;
  el.innerHTML=`<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#8898aa;font-size:.84rem;text-align:center;padding:20px;line-height:1.6">${message}</div>`;
}

function requirePlotly(ids){
  if(window.Plotly){
    return true;
  }
  ids.forEach(id=>setChartFallback(id,'차트 라이브러리를 불러오는 중입니다.<br>잠시 후 다시 시도해주세요.'));
  return false;
}

(function(){
  const grid=document.getElementById('prod-grid');
  RD.products.forEach((p,i)=>{
    const compositionEntries = Object.entries(p.composition);
    const sortedComposition = [...compositionEntries].sort((a,b)=>b[1]-a[1]);
    const topComposition = sortedComposition.slice(0, 8);
    grid.insertAdjacentHTML('beforeend',`
      <article class="prod-card" style="border-top:3px solid ${COLORS[i%COLORS.length]}">
        <div class="prod-head">
          <div>
            <div class="prod-title" style="color:${COLORS[i%COLORS.length]}">${p.name}</div>
            <div class="prod-meta">조성 ${compositionEntries.length}개 원소 · 열처리 ${Object.keys(p.heat_treatment).length}개 조건</div>
          </div>
          <div class="pill">예측 대상</div>
        </div>
        <div class="prod-block">
          <div class="prod-block-title">주요 조성</div>
          <div class="pill-row">${topComposition.map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('') || '—'}</div>
        </div>
        <div class="prod-block">
          <div class="prod-block-title">전체 조성</div>
          <div class="pill-row">${compositionEntries.map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('') || '—'}</div>
        </div>
        <div class="prod-block">
          <div class="prod-block-title">열처리 조건</div>
          <div class="pill-row">${Object.entries(p.heat_treatment).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join('') || '<span class="prod-meta">입력된 열처리 정보 없음</span>'}</div>
        </div>
      </article>
    `);
  });
})();

(function(){
  const bar=document.getElementById('stat-bar');
  RD.products.forEach((p,i)=>{
    const maxLog=Math.max(...p.temp_sweep.log10_hours),minLog=Math.min(...p.temp_sweep.log10_hours);
    const maxHours=Math.pow(10, maxLog), minHours=Math.pow(10, minLog);
    const clr=COLORS[i%COLORS.length];
    bar.insertAdjacentHTML('beforeend',`
      <div class="stat-card" style="border-top:3px solid ${clr}">
        <div class="stat-lbl">${p.name}</div>
        <div class="stat-val">${formatHoursCompact(maxHours)}</div>
        <div class="stat-sub">${formatYearsCompact(maxHours)}</div>
        <div class="stat-range">온도 범위 내 최대 예상 수명<br>범위: ${formatHoursCompact(minHours, false)} - ${formatHoursCompact(maxHours, false)}</div>
      </div>
    `);
  });
})();

function renderTempChart(){
  if(!requirePlotly(['ch-temp']))return;
  Plotly.react('ch-temp',RD.products.map((p,i)=>({x:p.temp_sweep.temps_k,y:p.temp_sweep.log10_hours,customdata:p.temp_sweep.hours,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'T=%{x:.0f} K<br>log₁₀(수명)=%{y:.3f}<br>수명=%{customdata:,.0f} h<extra>'+p.name+'</extra>'})),
    lyt({xaxis:{title:'온도 (K)',gridcolor:'#1e2538'},yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},shapes:refShapesY(),annotations:refAnnotY()}),{responsive:true,displayModeBar:false});
}
function renderStressChart(){
  if(!requirePlotly(['ch-stress']))return;
  Plotly.react('ch-stress',RD.products.map((p,i)=>({x:p.stress_sweep.stresses_mpa,y:p.stress_sweep.log10_hours,customdata:p.stress_sweep.hours,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'응력=%{x:.1f} MPa<br>log₁₀(수명)=%{y:.3f}<br>수명=%{customdata:,.0f} h<extra>'+p.name+'</extra>'})),
    lyt({xaxis:{title:'응력 (MPa)',type:'log',gridcolor:'#1e2538'},yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},shapes:refShapesY(),annotations:refAnnotY()}),{responsive:true,displayModeBar:false});
}
function renderLMPChart(){
  if(!requirePlotly(['ch-lmp']))return;
  Plotly.react('ch-lmp',RD.products.map((p,i)=>({x:p.lmp.lmp_vals,y:p.lmp.stresses_mpa,name:p.name,type:'scatter',mode:'lines',line:{color:COLORS[i%COLORS.length],width:2.5},hovertemplate:'LMP=%{x:.3f}<br>응력=%{y:.1f} MPa<extra>'+p.name+'</extra>'})),
    lyt({margin:{l:65,r:20,t:24,b:56},xaxis:{title:'Larson-Miller Parameter (×10³)',gridcolor:'#1e2538'},yaxis:{title:'응력 (MPa)',type:'log',gridcolor:'#1e2538'},hovermode:'closest'}),{responsive:true,displayModeBar:false});
}
let _hmIdx=0;
function renderHeatmap(idx){
  if(!requirePlotly(['ch-hm']))return;
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
  if(!requirePlotly(['ch-comp']))return;
  requestAnimationFrame(function(){
    const visible = RD.products;
    const N=Math.max(1, visible.length);
    const panelWidth = 260;
    const chartEl = document.getElementById('ch-comp');
    const box = document.getElementById('comp-box');
    const boxInner = box ? Math.max(box.clientWidth - 28, 200) : 400;
    const chartWidth = Math.max(N * panelWidth, boxInner);
    chartEl.style.width = chartWidth + 'px';
    chartEl.style.height = '320px';
    Plotly.react('ch-comp',visible.map((p,i)=>{
      const W=1/N;
      const c=i;
      const labels=Object.keys(p.comp_pie),values=Object.values(p.comp_pie);
      return {labels,values,type:'pie',hole:0.48,name:p.name,textinfo:'label+percent',textposition:'inside',insidetextorientation:'radial',textfont:{size:10,color:'#fff'},marker:{colors:labels.map(l=>ELEM_CLR[l]||'#5a6a88'),line:{color:'#0f1117',width:1.5}},domain:{x:[c*W+0.015,(c+1)*W-0.015],y:[0.08,0.98]},title:{text:'<b>'+p.name+'</b>',font:{color:COLORS[i%COLORS.length],size:12},position:'bottom center'},hovertemplate:'%{label}<br>%{value:.3f} wt%<br>%{percent}<extra></extra>'};
    }),{paper_bgcolor:'#1a1f30',plot_bgcolor:'#0f1117',font:{color:'#c0c8d8',size:11},showlegend:false,margin:{l:10,r:10,t:20,b:20},width:chartWidth,height:320},{responsive:false,displayModeBar:false});
  });
}
function renderThermalCycle(){
  if(!requirePlotly(['ch-ht']))return;
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
  const STAGE_FULL={
    Normalizing:'노말라이징',
    Tempering:'템퍼링',
    Aging:'시효처리',
    '노말라이징':'노말라이징',
    '템퍼링':'템퍼링',
    '시효처리':'시효처리'
  };
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
  buildHTCards();
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
  if(logL>=5){badge='장기 안전';cls='safe';}
  else if(logL>=4){badge='정상 운전';cls='safe';}
  else if(logL>=3){badge='주의!';cls='caution';}
  else if(logL>=2){badge='위험!';cls='danger';}
  else{badge='긴급 점검';cls='critical';}
  const lifeEl=document.getElementById('life-val');
  lifeEl.textContent=formatHoursCompact(hrs);
  lifeEl.style.color=cls==='safe'?'#6bf7c6':cls==='caution'?'#f7e06b':cls==='danger'?'#f7a06b':'#f76b6b';
  document.getElementById('life-hours').textContent=`절대 시간 기준 ${formatNumberKo(hrs, hrs >= 100 ? 0 : 1)}시간`;
  document.getElementById('life-sub').textContent=`수명 환산 ${formatYearsCompact(hrs)}`;
  document.getElementById('life-meta').textContent=`log10 수명 ${logL.toFixed(3)} · 온도 ${Math.round(_curTemp)} K · 응력 ${_curStress.toFixed(0)} MPa`;
  const badgeEl=document.getElementById('life-badge');badgeEl.textContent=badge;badgeEl.className='life-badge '+cls;
  document.getElementById('sl-t-val').textContent=Math.round(_curTemp)+' K';
  document.getElementById('sl-s-val').textContent=_curStress.toFixed(0)+' MPa';
  if(!window.Plotly){
    setChartFallback('ch-dash','차트 라이브러리를 불러오는 중입니다.<br>수명 수치 계산은 완료되었습니다.');
    return;
  }
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

function renderGASection(){
  const ga = RD.ga;
  if(!ga||!ga.best) return;
  document.getElementById('ga-section').style.display='';
  const b=ga.best;
  function fmt(v,d){if(v==null||v===undefined||isNaN(parseFloat(v)))return'—';return parseFloat(v).toFixed(d!=null?d:3);}
  function fmtHrs(log10){if(log10==null)return'—';const h=Math.pow(10,parseFloat(log10));if(h>87600)return(h/8760).toFixed(0)+'년';if(h>1000)return(h/1000).toFixed(1)+'k h';return h.toFixed(0)+' h';}
  function barCls(v,okMax,warnMax){if(v<=okMax)return'ok';if(v<=warnMax)return'warn';return'bad';}
  const logL=b.predicted_log_life;
  const lifeCls=logL>=5||logL>=4?'ok':logL>=3?'warn':'bad';
  document.getElementById('ga-life-num').textContent=fmtHrs(logL);
  document.getElementById('ga-life-num').className='ga-life-num '+lifeCls;
  document.getElementById('ga-life-sub').textContent='log₁₀(수명) = '+fmt(logL,3)+' · GA 최적화 결과';
  // Stats
  const costCls=b.material_cost<30?'ok':b.material_cost<60?'warn':'bad';
  const riskCls=b.physics_risk<10?'ok':b.physics_risk<50?'warn':'bad';
  const oodTxt=b.ood_is_out_of_distribution===true?'<span class="bad">범위 이탈</span>':b.ood_is_out_of_distribution===false?'<span class="ok">범위 내</span>':'—';
  const calTxt=b.thermo_success?'<span class="ok">완료</span>':'<span style="color:#8898aa">미수행</span>';
  document.getElementById('ga-stats').innerHTML=
    '<div class="ga-stat"><span class="lbl">재료 비용</span><span class="val '+costCls+'">$'+fmt(b.material_cost,2)+'/kg</span></div>'+
    '<div class="ga-stat"><span class="lbl">물리 위험도</span><span class="val '+riskCls+'">'+fmt(b.physics_risk,2)+'</span></div>'+
    '<div class="ga-stat"><span class="lbl">OOD 상태</span><span class="val">'+oodTxt+'</span></div>'+
    '<div class="ga-stat"><span class="lbl">CALPHAD</span><span class="val">'+calTxt+'</span></div>'+
    '<div class="ga-stat"><span class="lbl">KN</span><span class="val">'+fmt(b.KN,3)+'</span></div>'+
    '<div class="ga-stat"><span class="lbl">CEQ</span><span class="val">'+fmt(b.CEQ,3)+'</span></div>';
  // Composition pills
  const ELEMS=['C','Si','Mn','Cr','Mo','W','Ni','V','Nb','N','B','P','S','Al','Cu','Co','Ta','O','Re'];
  let pillsHtml='';
  if(b.Fe_balance&&parseFloat(b.Fe_balance)>0) pillsHtml='<span class="pill"><b>Fe</b> '+fmt(b.Fe_balance,2)+'%</span>';
  ELEMS.forEach(e=>{const v=b[e];if(v&&parseFloat(v)>0.001) pillsHtml+='<span class="pill"><b>'+e+'</b> '+fmt(v,3)+'%</span>';});
  document.getElementById('ga-pills').innerHTML=pillsHtml||'—';
  // Phase bars
  const phases=[
    {key:'bcc_fraction',label:'BCC_A2',okMax:.8,warnMax:1.0},
    {key:'m23c6_fraction',label:'M₂₃C₆',okMax:.05,warnMax:.1},
    {key:'laves_fraction',label:'Laves',okMax:.02,warnMax:.05},
    {key:'sigma_fraction',label:'Sigma',okMax:.01,warnMax:.03},
    {key:'fcc_fraction',label:'FCC',okMax:.05,warnMax:.1},
  ];
  let phHtml='';
  phases.forEach(ph=>{
    const v=b[ph.key]!=null?parseFloat(b[ph.key]):null;
    const pct=v!=null?Math.min(v*100,100):0;
    const cls=v!=null?barCls(v,ph.okMax,ph.warnMax):'';
    phHtml+='<div class="ga-bar-row"><div class="ga-bar-label"><span>'+ph.label+'</span><span>'+(v!=null?pct.toFixed(2)+'%':'—')+'</span></div><div class="ga-bar-bg"><div class="ga-bar-fill '+cls+'" style="width:'+pct+'%"></div></div></div>';
  });
  document.getElementById('ga-phases').innerHTML=phHtml;
  // Meta bar
  const meta=ga.meta||{};const cond=meta.conditions||{};
  const mParts=[];
  if(meta.generated_at) mParts.push('<span>생성: <strong>'+meta.generated_at+'</strong></span>');
  if(cond.temp_c!=null) mParts.push('<span>GA 조건 — 온도: <strong>'+cond.temp_c+' °C</strong></span>');
  if(cond.stress_mpa!=null) mParts.push('<span>응력: <strong>'+cond.stress_mpa+' MPa</strong></span>');
  if(cond.cost_limit!=null) mParts.push('<span>비용 한도: <strong>$'+cond.cost_limit+'/kg</strong></span>');
  document.getElementById('ga-meta-bar').innerHTML=mParts.join('');
  // Comparison chart
  if(!window.Plotly||!ga.sweep) return;
  const sw=ga.sweep.temp_sweep;
  const traces=RD.products.map((p,i)=>({
    x:p.temp_sweep.temps_k,y:p.temp_sweep.log10_hours,name:p.name,type:'scatter',mode:'lines',
    line:{color:COLORS[i%COLORS.length],width:1.8,dash:'dot'},
    hovertemplate:'T=%{x:.0f} K<br>log₁₀=%{y:.3f}<extra>'+p.name+'</extra>'
  }));
  traces.push({
    x:sw.temps_k,y:sw.log10_hours,name:'GA 최적 합금',type:'scatter',mode:'lines',
    line:{color:'#f7e06b',width:3},
    hovertemplate:'T=%{x:.0f} K<br>log₁₀=%{y:.3f}<extra>GA 최적 합금</extra>'
  });
  Plotly.react('ch-ga-comp',traces,lyt({
    margin:{l:60,r:20,t:16,b:50},
    xaxis:{title:'온도 (K)',gridcolor:'#1e2538'},
    yaxis:{title:'log₁₀(수명 / 시간)',gridcolor:'#1e2538'},
    legend:{bgcolor:'rgba(0,0,0,0)',bordercolor:'#2a3048',borderwidth:1,orientation:'h',y:-0.2},
    hovermode:'closest'
  }),{responsive:true,displayModeBar:false});
}

function loadPlotlyAndRender(){
  if(window.Plotly){
    PLOTLY_READY = true;
    renderCompositionChart();
    renderThermalCycle();
    initDashboard();
    renderGASection();
    if(_detOpen){
      _detRendered = true;
      renderTempChart();
      renderStressChart();
      renderLMPChart();
      renderHeatmap(0);
    }
    return;
  }
  const script=document.createElement('script');
  script.src='https://cdn.plot.ly/plotly-2.35.2.min.js';
  script.async=true;
  script.onload=()=>{
    PLOTLY_READY = true;
    renderCompositionChart();
    renderThermalCycle();
    initDashboard();
    renderGASection();
    if(_detOpen){
      _detRendered = true;
      renderTempChart();
      renderStressChart();
      renderLMPChart();
      renderHeatmap(0);
    }
  };
  script.onerror=()=>{
    ['ch-comp','ch-ht','ch-dash'].forEach(id=>setChartFallback(id,'차트 라이브러리를 불러오지 못했습니다.<br>인터넷 연결을 확인하거나 잠시 후 다시 시도해주세요.'));
  };
  document.body.appendChild(script);
}

/* ── Collapsible ── */
let _detOpen=true,_detRendered=false;
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
loadPlotlyAndRender();
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ensure_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
