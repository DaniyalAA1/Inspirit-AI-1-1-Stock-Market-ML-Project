# Stock Price Prediction with Machine Learning

A comparison of five machine learning regression models for short-term stock price prediction, evaluated both on standard error metrics and through simulated trading strategies. This project was conducted through the Inspirit AI research program and led to a first-authored paper published in the *Curieux Academic Journal* (April 2026).

## Overview

The model predicts a stock's next opening price using its three most recent opening prices as input features. Data is pulled live via the `yfinance` API (5-year historical window).

**Models compared:**
- Linear Regression
- Multi-Layer Perceptron (MLP) Regressor — two hyperparameter configurations
- Decision Tree Regressor — two max-depth configurations
- Random Forest Regressor — two configurations (depth/estimators)
- K-Nearest Neighbors Regressor — k=2 and k=3

## Methodology

1. **Data collection**: Pull 5 years of daily price history for a given ticker via `yfinance`.
2. **Feature engineering**: Use each set of 3 consecutive opening prices to predict the following day's opening price.
3. **Train/test split**: 67% train / 33% test.
4. **Evaluation**:
   - Mean Squared Error (MSE) on held-out test data for each model.
   - An **ensemble average** of all model predictions.
   - An **inverse-MSE-weighted ensemble**, giving more accurate models a larger vote.
5. **Trading simulation**: Using each model's predicted price direction, simulate a simple buy/sell strategy (1 or 2 shares per signal) starting from a fixed cash balance ($1,000 and $10,000 scenarios), to see how prediction accuracy translates into simulated returns.

# Stock Price Prediction with Machine Learning

A comparison of five machine learning regression models for short-term stock price prediction across five sectors, evaluated on both prediction error and simulated trading returns. This project was conducted through the Inspirit AI research program and published as a first-authored paper — *"Investigating Machine Learning Models for Short-Term Stock Price Prediction Across Sectors"* — in the *Curieux Academic Journal* (April 2026), co-authored with Odysseas Drosis (Cornell University).

## Overview

Each model predicts a stock's next opening price using its three most recent opening prices as input, using 5 years of historical data pulled via the `yfinance` API. The same pipeline was run independently across five companies spanning different sectors:

| Company | Sector |
|---|---|
| Apple | Technology |
| Walmart | Retail |
| CVS Health | Healthcare |
| Pfizer | Pharmaceutical |
| Goldman Sachs | Finance |

**Models compared** (two hyperparameter configurations each, except Linear Regression):
- Linear Regression
- Multilayer Perceptron (MLP) Regression
- Decision Tree Regression
- Random Forest Regression
- K-Nearest Neighbors (KNN) Regression

## Methodology

1. **Data collection**: Pull 5 years of daily Open-price history per ticker via `yfinance`. All other fields (Close, Volume, Dividends, Stock Splits) are dropped to isolate the Open-price signal and reduce overfitting.
2. **Feature engineering**: Each 3-day sequence of opening prices is used to predict the following day's opening price.
3. **Train/test split**: 67% train / 33% test.
4. **Evaluation**: Mean Squared Error (MSE) per model per company, plus a uniform-average ensemble and an inverse-MSE-weighted ensemble.
5. **Trading simulation**: Using each model's predicted price direction, simulate daily buy/sell decisions starting from $1,000 and $10,000 baselines, then compare final liquidated portfolio value across models and sectors.

## Key Findings

- **Linear Regression had the lowest MSE on every single stock tested**, consistently outperforming the more complex models (MLP, Decision Tree, Random Forest, KNN) on raw prediction error.
- **Finance (Goldman Sachs) had the highest MSE of any sector** — indicating more overfitting/variance — but paradoxically produced **the largest trading returns of any sector**, with Random Forest (depth=25, 100 estimators) turning $1,000 into $7,172.70 (617% return) and every model in that sector finishing with a positive return.
- **Pharmaceutical (Pfizer) was the weakest-performing sector**, with several models producing negative returns.
- MSE ranking and trading-simulation ranking didn't always agree — a model with better raw prediction accuracy didn't always translate into better simulated trading performance, highlighting a gap between statistical accuracy and downstream decision-usefulness.
- Findings are discussed in the context of prior work (Vijh et al., 2020; Bansal et al., 2022) showing deep learning models often outperform simpler ones for time-series forecasting — our results suggest this doesn't hold uniformly across all sectors when using only short-window Open-price features.

## Tech Stack

- **Language**: Python
- **Libraries**: `yfinance`, `pandas`, `NumPy`, `scikit-learn`
- **Environment**: Google Colab

## Repository Contents

This repo contains 5 notebooks, each running the full pipeline (data collection → model training → MSE evaluation → trading simulation) independently on one company:

- `AAPL_Apple.ipynb` — Apple (Technology)
- `WMT_Walmart.ipynb` — Walmart (Retail)
- `CVS_CVS_Healthcare.ipynb` — CVS Health (Healthcare)
- `PFE_Pfizer.ipynb` — Pfizer (Pharmaceutical)
- `GS_Goldman_Sachs.ipynb` — Goldman Sachs (Finance)

Each notebook is self-contained and can be run independently in Google Colab.

`AI_1_1.ipynb` is the original notebook (Goldman Sachs) submitted alongside the published paper, kept unchanged for reference/provenance. `GS_Goldman_Sachs.ipynb` is a re-run of the same pipeline for consistency with the other 4 company notebooks' naming.

**Note on reproducibility**: `yfinance` pulls a rolling 5-year window of live data, and models like MLP involve randomized initialization, so re-running these notebooks will produce numbers that differ slightly from run to run and from the results reported in the published paper. The paper's tables reflect the results at the time of writing; these notebooks demonstrate the reproducible pipeline.

## Publication

Alvi, D., Drosis, O. "Investigating Machine Learning Models for Short-Term Stock Price Prediction Across Sectors." *Curieux Academic Journal*, April 2026.

## Author

Daniyal Alvi — [daniyalaalvi@gmail.com](mailto:daniyalaalvi@gmail.com)
