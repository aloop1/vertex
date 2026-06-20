# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vertex** is an AI-powered application for predicting creep rupture lifetime and designing optimal alloy compositions for high-temperature, high-pressure environments. It uses a Transformer + Tree Ensemble model for lifetime prediction and NSGA-II genetic algorithm with Gemini LLM seeding to optimize alloy compositions balancing performance and cost.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Data preprocessing (outputs preprocessor.pkl to data/)
python data_preprocessing.py        # original pipeline (taka.xlsx only)
python 데이터전처리.py              # merged-dataset pipeline (taka + creep CSVs)

# Build OOD reference (required before running GA)
python tools/build_ood_reference.py                         # ferritic-only (default)
python tools/build_ood_reference.py --all-data              # use full dataset
python tools/build_ood_reference.py --ni-max 3 --cr-min 7.5 # relax filter if < 50 rows

# Train prediction model
python models/transformer_and_tree_ensemble.py  # Transformer + Tree Ensemble (production model)
python models/transformer_and_tree_ensemble.py --smoke --pickle-path models/transformer_tree_ensemble_smoke.pkl --pt-path models/transformer_tree_ensemble_smoke.pt  # fast smoke model (3 epochs, 8 trees)
python models/LMP_데이터증강.py                 # LMP-based data augmentation

# Run web application (Flask, inference only)
python web/app.py                    # dev server on http://0.0.0.0:5000
python web/serve.py                  # production server (waitress); PORT/VERTEX_THREADS env vars

# Run GA optimization (interactive CLI)
python ga/engine.py

# Standalone LLM seed test
python ga/llm.py
```

## Architecture

### Four main components

**1. Data Preprocessing** (`data_preprocessing.py`, `데이터전처리.py`)
- Loads from `data/taka.xlsx` (2066×31), `data/creep.csv`, `data/creep_data.csv`
- Fills composition NaNs with 0, drops rows with missing target/stress/temperature
- Adds physics-derived features: operating_severity, stress_temp_interaction, inverse_temp, total_ht_severity
- Uses **group-based shuffled split** (by alloy group ID) to prevent data leakage — the model must generalize to novel alloy groups, not just unseen rows
- Exports `StandardScaler` to `data/preprocessor.pkl`; this same scaler must be loaded for all inference
- Target is `log10(lifetime_hours)` — always work in log scale
- `데이터전처리.py` exports `COMPOSITION_COLS`, `HEAT_TREATMENT_COLS`, `EXTRA_COLS`, and `prepare_dataset()` used by other modules

**2. Predictive Modeling** (`models/transformer_and_tree_ensemble.py`)
- Custom PyTorch `ManualMultiHeadSelfAttention` (not `nn.TransformerEncoder`) plus a residual tree ensemble
- `load_transformer_tree_predictor(artifact_path, allow_smoke_fallback)` — the main entry point; loads `.pkl` or `.pt` artifact
- `predictor.predict_one(stress, temp, composition, heat_treatment)` → returns `.log_lifetime`
- `predictor.predict_dataframe(df, batch_size)` → vectorised batch inference
- **LMP (Larson-Miller Parameter) is excluded from training features** to avoid target leakage; computed only post-hoc for physics validation
- `ALLOY_PREDICTOR_ARTIFACT_PATH` env var overrides the default artifact path

**3. Alloy Design Optimization** (`ga/`)
- `config.py` — element database: `DESIGN_VARIABLES` (11 optimizable elements), `FIXED_ELEMENTS` (8 impurity/fixed elements), `FULL_COMPOSITION_FEATURES` (19 elements total), `FERRITIC_SYSTEM_LIMITS`, `ELEMENT_COST`, `THERMO_CONFIG`, `OOD_CONFIG`, `LLM_CONFIG`, `SEARCH_POLICY`
- `physics.py` — `as_full_composition()` restores 19-element composition from 11 design variables; `evaluate_physics()` computes metallurgy penalty + optional CALPHAD penalty + OOD penalty; `calculate_material_cost()`, `calculate_metallurgical_scores()`
- `engine.py` — NSGA-II (DEAP `eaMuPlusLambda`); tri-objective: maximize predicted lifetime, minimize cost, minimize physics risk; outputs `ga/best_alloy.json`, `ga/best_alloy.csv`, `ga/pareto_top30.json`, `ga/pareto_top30.csv`; JSON files are metadata-wrapped: `{generated_at, conditions, best/candidates}`
- `llm.py` — Gemini API seed generation with cache (`data/seed_cache.json`); controlled by `USE_LLM_API`, `FORCE_LLM_REFRESH`, `STRICT_LLM_SEED_MODE` env flags; falls back to `data/sample_seeds.json` then rule-based seeds when API is disabled
- `ood_analysis.py` — Mahalanobis OOD analysis; loads `data/ood_reference.pkl`; `analyze_ood_candidate()` returns distance, percentile, feature contributions, nearest neighbors

**4. Web Interface** (`web/`)
- Flask app (inference only; no training or GA at runtime)
- `web/app.py` routes only; HTML lives in `web/templates/` (index, result, alloy_design), shared theme assets in `web/static/` (`vertex.css` dual dark/light theme via CSS variables + `data-theme` attr, `theme.js` toggle + Plotly theme helpers `vxBaseLayout`/`vxChartColors`/`VX_SERIES`)
- Theme toggle persists in localStorage; charts re-render on `vx-themechange` event
- `web/serve.py` is the production entry point (waitress)
- Upload CSV/XLSX with composition + heat treatment columns (lifetime column is silently dropped)
- Sweeps predictions over temperature and stress grids (80 points each; 35×35 heatmap)
- Returns interactive HTML with Plotly charts: temperature sweep, stress sweep, LMP diagram, safety heatmap, real-time dashboard with sliders
- `/resweep` — recalculates with updated sweep parameters without re-uploading
- `/suggest_params` — reads uploaded file's stress/temp columns and returns recommended sweep defaults (called on file select)
- `/alloy_design` — GA results dashboard; reads `ga/best_alloy.json` and `ga/pareto_top30.json`; handles both old (flat) and new (metadata-wrapped) JSON formats
- `VERTEX_MODEL_PATH` env var overrides the model artifact path

### Feature set (30 total)
- **Conditions**: stress, temp
- **Composition** (19 elements): C, Si, Mn, P, S, Cr, Mo, W, Ni, Cu, V, Nb, N, Al, B, Co, Ta, O, Re
- **Heat treatment** (6): Ntemp, Ntime, Ttemp, Ttime, Atemp, Atime
- **Physics-derived** (4): operating_severity, stress_temp_interaction, inverse_temp, total_ht_severity

### Required artifacts
- `data/preprocessor.pkl` — StandardScaler, built by preprocessing scripts
- `data/ood_reference.pkl` — Mahalanobis OOD reference, built by `tools/build_ood_reference.py`
- `models/transformer_tree_ensemble.pkl` or `.pt` — production model artifact
- `models/transformer_tree_ensemble_smoke.pkl` — fast smoke model; auto-loaded by `web/app.py` when production artifact is absent (`allow_smoke_fallback=True`)
- `.env` — `GEMINI_API_KEY` for LLM seed generation; `USE_LLM_API=false` disables API calls

### GA element split
- **DESIGN_VARIABLES** (11): C, Si, Mn, Cr, Mo, W, Ni, V, Nb, N, B — the genes evolved by NSGA-II
- **FIXED_ELEMENTS** (8): P, S, O, Co, Re, Al, Cu, Ta — fixed impurity values injected via `as_full_composition()`
- CALPHAD (`pycalphad`) runs only on Top 10 Pareto candidates after GA concludes, not during evolution; controlled by `THERMO_CONFIG["APPLY_THERMO_DURING_GA"]`

## Key Conventions

- **Korean filenames/variables are intentional**: `데이터전처리.py` and Korean variable names exist for domain context; do not rename
- **Dual preprocessing paths**: `data_preprocessing.py` (original) and `데이터전처리.py` (merged datasets) are both maintained; `web/app.py` and `ga/engine.py` import from `데이터전처리`
- **Outlier preservation**: Statistical outliers in Stress (up to 14%) are valid experimental conditions (5–450 MPa range); do not remove them
- **Element costs** in `ga/config.py` are relative indices (not exact $/kg); update together if repricing
- `documents/` contains Korean-language meeting notes (2026-03-06 onwards) tracking project decisions and backlog
