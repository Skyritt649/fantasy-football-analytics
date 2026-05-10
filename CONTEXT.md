# fantasy-football-analytics — Deep Context

> Reference this file when deeper knowledge of the pipeline, model, or architecture is needed.

## Data Pipeline (`RB_Dataset_Builder.ipynb`)

**Data sources:**
1. `nfl_data_py.import_seasonal_data()` — base seasonal stats (2015–2024, 2025 not yet available)
2. `import_pbp_data()` — chunked at 4-year windows; extracts goal-line touches (yardline ≤ 20) and team rushing context
3. `import_snap_counts()` — snap percentages via PFR pfr_player_id → gsis_id crosswalk
4. `import_depth_charts()` — week 1 RB depth positions
5. `import_injuries()` — out/limited games per season
6. `import_ngs_data(stat_type='rushing')` — RB-specific NGS rushing metrics
7. `import_ids()` — gsis_id ↔ sleeper_id crosswalk (77% coverage, used only for snap_counts join)

**Feature engineering:**
- Per-game conversion: divide season totals by games_played
- aDOT: air_yards / targets
- Carry share: player carries / team carries
- YoY deltas: season Y − season Y−1 (returning players only)
- Goal-line: gl_carries_pg, gl_targets_pg, gl_touches_pg (NaN if SKIP_PBP=True)

## Model Architecture

- XGBoost + Optuna (60 trials), objective: minimize −Spearman ρ
- Tuning split: train 2015–2020, validate 2021
- Walk-forward folds: test years 2019–2024 (train on season < test_year−1, test on test_year−1)
- 62 features: prior-season PPG, per-game stats, efficiency, snap %, depth chart, NGS, injury, YoY deltas, player attributes

**Results:**

| Metric | XGBoost | Baseline |
|--------|---------|---------|
| Spearman ρ | 0.732 | 0.721 |
| MAE | 2.91 PPG | 3.15 |
| RMSE | 3.66 PPG | 4.00 |
| Top-12 prec | 55.6% | — |
| Top-24 prec | 72.9% | — |

2025 predictions: 113 returning RBs with RB1/RB2/RB3/Depth tiers. Top: Gibbs (18.8), Bijan (18.6), Barkley (17.7).

## Dependencies

| Package | Version | Note |
|---------|---------|------|
| pandas | 3.0.2 | |
| numpy | 2.4.4 | shap requires ≥2; nfl_data_py conflicts (tolerated) |
| nfl_data_py | 0.3.3 | works with warnings |
| xgboost | 3.2.0 | |
| scikit-learn | 1.8.0 | |
| optuna | 4.8.0 | |
| shap | 0.51.0 | requires numpy≥2 |
| matplotlib | 3.10.9 | |
| seaborn | 0.13.2 | |
| scipy | 1.17.1 | |
| fastparquet | — | only parquet engine installed |

## Scripts Not in Repo

All at `C:\Users\alexg\OneDrive\Documents\iClaude\Python Scripts\`:

- **`build_rb_dataset.py`** — source script for `RB_Dataset_Builder.ipynb`; converted via `generate_notebook.py`
- **`train_rb_ppg_model.py`** — full model pipeline (~2 min); outputs model JSON, predictions CSV, SHAP + validation PNGs
- **`plot_2025_comparison.py`** — fetches 2025 actuals from Sleeper weekly API (17 weeks), plots R² scatter vs baseline and model

## Key Column Definitions

- `ppg` — fantasy_points_ppr / games
- `aDOT` — air_yards / targets
- `carries_share` — player carries / team carries
- `is_rb1_wk1` — depth_team_wk1 == 1 (binary)
- `games_out_seas` — games missed (Out/Doubtful)
- `games_limited_seas` — games limited (Questionable/Limited)
- `ppg_yoy` — current PPG − prior season PPG
- `ngs_ryoe_per_att` — rush yards over expected per attempt

## Extending the Project

**New position (WR/QB):** filter `import_seasonal_data()` by position, adjust feature set (WR: target share; QB: completion %, EPA), retrain with same walk-forward structure.

**Rookie model:** separate pipeline using college stats + draft metrics (round, ADP, 40-time, etc.); target is rookie-year PPG; smaller dataset (~250/year), noisier target.

**REST API:** FastAPI `POST /predict`, load model with `xgb.Booster().load_model('rb_ppg_xgb.json')`, deploy via Docker + Cloud Run.
