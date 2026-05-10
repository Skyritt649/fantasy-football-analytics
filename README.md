# Fantasy Football Analytics

A machine learning-powered fantasy football analytics project featuring **RB PPG prediction models**, historical data pipelines, and advanced feature engineering.

## Features

- **XGBoost RB PPG Model** — Predicts next-season PPR points per game for running backs with walk-forward validation (Spearman ρ: 0.732)
- **Sleeper API Integration** — Real-time player trending, ADP, and roster data
- **61 Advanced Features** — Per-game stats, goal-line touches, snap counts, depth charts, injury history, NGS rushing metrics, and YoY deltas
- **2015–2025 Seasonal Data** — 1,229 RB-season observations from nfl_data_py
- **SHAP Explainability** — Feature importance analysis with beeswarm and bar charts

## Project Structure

```
.
├── RB_Dataset_Builder.ipynb          # Data pipeline: build 61-feature RB dataset
├── Fantasy Football 2026 Claude.ipynb # EDA and model exploration notebook
├── rb_features_2015_2025.csv         # Final dataset (1,229 rows, 71 columns)
├── rb_features_2015_2025.parquet     # Same, parquet format
├── README.md                         # This file
└── CLAUDE.md                         # Codebase guide for Claude
```

## Quick Start

### Requirements
- Python 3.12+
- pandas, numpy, xgboost, scikit-learn, matplotlib, seaborn, optuna, shap
- nfl_data_py 0.3.3 (for data fetching)

### Run the Data Pipeline
```bash
jupyter notebook RB_Dataset_Builder.ipynb
```
Generates `rb_features_2015_2025.csv` with 61 engineered features.

### Train the XGBoost Model
```bash
# From the repo root (requires rb_features_2015_2025.csv)
python train_rb_ppg_model.py
```

Outputs:
- `model_output/rb_ppg_xgb.json` — trained model
- `model_output/rb_predictions_2025.csv` — 2025 predictions with tiers
- `model_output/validation_summary.png` — performance plots
- `model_output/shap_*.png` — feature importance visualizations

## Model Performance

**Walk-Forward Validation (test years 2019–2024):**

| Metric | Score |
|--------|-------|
| Avg Spearman ρ | 0.732 |
| Avg MAE | 2.91 PPG |
| Avg RMSE | 3.66 PPG |
| Top-12 Precision | 55.6% |
| Top-24 Precision | 72.9% |

**vs. Baseline (prior-season PPG):** +0.011 Spearman ρ improvement, -3.3% MAE reduction.

## Data Sources

- **Seasonal & play-by-play:** [nfl_data_py](https://github.com/nflverse/nflverse-py) (`import_seasonal_data`, `import_pbp_data`, `import_snap_counts`, etc.)
- **Sleeper ID mapping:** nfl_data_py's `import_ids()` crosswalk (77% coverage) + `import_seasonal_rosters()` native sleeper_id (99.4% coverage)
- **Trending/ADP:** [Sleeper API](https://docs.sleeper.app/) (no auth required)

## 61 Features

**Base Stats (per game):**
carries_pg, targets_pg, receptions_pg, receiving_yards_pg, rushing_yards_pg, receiving_air_yards_pg, receiving_yards_after_catch_pg, total_yards_pg, rushing_tds_pg, receiving_tds_pg, total_tds_pg

**Advanced Stats (per game):**
rushing_epa_pg, receiving_epa_pg, total_epa_pg, rushing_first_downs_pg, receiving_first_downs_pg, rushing_fumbles_pg, receiving_fumbles_pg, ypc, ypr, catch_rate, scrimmage_yds_per_touch

**Share Stats:**
tgt_sh, ay_sh, yac_sh, ry_sh, rtd_sh, carries_share, dom, w8dom

**Situational:**
gl_carries_pg, gl_targets_pg, gl_touches_pg, avg_snap_pct, off_snaps_pg, depth_team_wk1, is_rb1_wk1

**NGS (Next Gen Stats):**
ngs_efficiency, ngs_pct_gte_8_defenders, ngs_avg_time_to_los, ngs_ryoe_per_att, ngs_rush_pct_over_expected

**Player Attributes:**
age, years_exp, height, weight, draft_number

**Injury History:**
games_out_seas, games_limited_seas, games_out_seas_prev, games_prev

**YoY Deltas:**
ppg_yoy, carries_pg_yoy, targets_pg_yoy, total_yards_pg_yoy, total_epa_pg_yoy, avg_snap_pct_yoy

## Next Steps

- [ ] **Rookie Model** — separate pipeline for predicting rookie RB PPG (pre-draft, college stats + draft metrics)
- [ ] **WR Breakout Model** — extend to wide receivers
- [ ] **QB Model** — quarterback PPG predictions
- [ ] **Portfolio API** — REST endpoint for model predictions (FastAPI)
- [ ] **Full PBP Run** — populate goal-line and team-context features with complete play-by-play data

## License

MIT License — see LICENSE file.

## Author

Built with nfl_data_py, XGBoost, Optuna, and SHAP for portfolio demonstration.
