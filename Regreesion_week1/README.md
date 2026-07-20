# House Prices — Statistics, Regression & Interactive Dashboard

## Project Overview

This project analyzes the **Ames Housing dataset** (`train.csv`, 1,460 homes,
81 features) to understand what drives residential sale prices, and builds
regression models to predict `SalePrice` from house characteristics like
overall quality, living area, garage capacity, basement size, and year built.

The core assignment (Parts 1–8) covers the full statistics and regression
workflow: data cleaning, exploratory data analysis, statistical summaries,
simple and multiple linear regression, a visualization portfolio, and a
written reflection.

**Stretch goals completed:**
- Random Forest model compared against both regression models
- Feature importance analysis (built-in + permutation importance)
- Interactive Plotly visualizations
- Interactive Streamlit dashboard with live price prediction
- Deployed as a live web app *(add your Streamlit Cloud URL here once deployed)*

## How to Run

### Notebook (Parts 1–8 + stretch goals)
1. Open `Week1_Statistics_Regression.ipynb` in Jupyter Notebook, JupyterLab,
   or Google Colab.
2. Make sure `train.csv` is in the same folder as the notebook (or update the
   file path in the first code cell if it's stored elsewhere, e.g. `data/train.csv`).
3. Install the required packages (see below), then run all cells top to
   bottom: **Runtime → Run all** (Colab) or **Kernel → Restart & Run All**
   (Jupyter).
4. All outputs — tables, charts, and markdown explanations — will regenerate
   in order.

### Interactive Dashboard (Streamlit stretch goal)
1. Make sure `app.py`, `house_icon.png`, and `data/train.csv` are all in the
   project folder (see structure below).
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. The dashboard opens automatically in your browser at `http://localhost:8501`.
   Use the sliders to enter house features and get a live price prediction,
   with interactive charts showing where that house lands in the market.

## Libraries Required

Install with:
```bash
pip install -r requirements.txt
```

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — static plotting (notebook)
- `seaborn` — statistical visualizations (heatmaps, styled plots)
- `scikit-learn` — train/test split, Linear Regression, Random Forest,
  evaluation metrics, feature importance
- `streamlit` — interactive dashboard framework
- `plotly` — interactive charts (notebook stretch goal + dashboard)

## Key Findings

1. **`OverallQual` (overall material/finish quality, rated 1–10) is the
   strongest single predictor of sale price** (r = 0.79), outperforming raw
   size measures like living area (r = 0.71).
2. **The 5-feature Multiple Linear Regression model** (`OverallQual`,
   `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`) explains **~79% of
   price variance (R² = 0.794)**, a 22% relative improvement over the
   single-feature Simple LR model (R² = 0.65).
3. **Random Forest outperforms both linear models** (R² = 0.887, MAE ≈
   $19,202), because it captures non-linear relationships and feature
   interactions that linear regression can't — at the cost of being harder
   to interpret directly.
4. **`SalePrice` is right-skewed** (skewness = 1.88) — a small number of
   high-value homes pull the average price above the median, which is typical
   of real-world housing markets.
5. **Two homes with a perfect 10/10 quality rating and over 4,300 sq ft of
   living space sold for only $160,000–$185,000** — well below what
   similarly large, high-quality homes typically sold for. Likely an
   unusual sale circumstance (e.g. foreclosure or family sale) rather than
   a data error.
6. **Neighborhood location causes large price swings** — the most expensive
   neighborhood (NoRidge, ~$335K average) sells for over 3x the cheapest
   (MeadowV, ~$99K average). This feature was not used in the models built
   here, and is flagged as the strongest candidate for future improvement.
7. **The residual plots for both linear models show heteroscedasticity**
   (error variance increases at higher predicted prices), indicating a
   plain linear model is less reliable for predicting expensive homes —
   Random Forest handles this better.
8. **19 of 81 columns had missing values**, but most were not truly missing
   — they encoded "feature not present" (e.g., no pool, no garage), which
   required careful, feature-specific imputation rather than blanket removal.

## Project Structure

```
week1-assignment-python/
├── Week1_Statistics_Regression.ipynb   # Main notebook (code + outputs)
├── Week1_Statistics_Regression.pdf     # PDF/HTML export of the notebook
├── app.py                              # Streamlit interactive dashboard
├── house_icon.png                      # Icon used in the dashboard
├── data/
│   └── train.csv                       # Dataset used
├── screenshots/                        # Key visualizations (3–5 images)
├── README.md                           # This file
└── requirements.txt                    # Python package dependencies
```

## Model Comparison Summary

| Metric | Simple LR | Multiple LR | Random Forest |
|---|---|---|---|
| R² Score | 0.6505 | 0.7939 | 0.8874 |
| MAE ($) | 33,343.24 | 25,414.73 | 19,202.18 |
| RMSE ($) | 51,778.63 | 39,763.30 | 29,386.20 |

## Notes / Limitations

- All models use only 5 numeric features; categorical features like
  `Neighborhood` were cleaned but not included in modeling, despite showing
  a strong relationship with price. Adding them (e.g., via one-hot encoding)
  is a natural next step.
- `SalePrice` was not log-transformed before modeling, despite being
  right-skewed — this is a likely contributor to the heteroscedasticity seen
  in the residual plots.
- The 80/20 train-test split uses a single fixed `random_state=42`; results
  were not validated across multiple splits or via cross-validation.
