# fantasy-football-analytics — CLAUDE.md

## Project
XGBoost model predicting RB PPG using nfl_data_py + Sleeper API. Walk-forward validated (2019–2024). Spearman ρ 0.732, MAE 2.91, Top-24 precision 72.9%.

## Key Files
| File | Purpose |
|------|---------|
| `RB_Dataset_Builder.ipynb` | Data pipeline (57 cells) |
| `train_rb_ppg_model.py` | XGBoost training + SHAP |
| `plot_2025_comparison.py` | 2024 vs model vs 2025 actuals |
| `rb_features_2015_2025.csv/.parquet` | Final dataset: 1,229 rows, 71 cols |
| `model_output/` | Model JSON, predictions, PNGs |

Scripts not in repo: `C:\Users\alexg\OneDrive\Documents\iClaude\Python Scripts\`
Venv: same dir as scripts, Python 3.12.10

## Known Issues & Workarounds
- **numpy conflict**: shap needs numpy≥2, nfl_data_py needs numpy<2. Resolution: numpy 2.4.4 + pandas 3.0.2. nfl_data_py works with warnings — reuse cached CSV/parquet, avoid re-fetching.
- **2025 nfl_data_py unavailable**: returns 404. Use Sleeper weekly API for 2025 actuals instead.
- **NGS**: `import_ngs_data(stat_type='receiving')` has no RB data. Use `stat_type='rushing'`.
- **PBP memory**: load in 4-year chunks, extract goal-line + team carries, delete chunk.
- **SKIP_PBP=True** (default): `carries_share`, `gl_*` features are 100% NaN. Full PBP run ~40 min.
- **Parquet**: fastparquet only — no pyarrow installed.
- **Sleeper ID**: use `import_seasonal_rosters()` — has native `sleeper_id` + `birth_date` (99.4% match). No explicit ID join needed.
- **RMSE**: use `np.sqrt(mean_squared_error(...))` — `squared=False` deprecated in sklearn 1.8.
