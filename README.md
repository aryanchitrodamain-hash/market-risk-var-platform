# Market Risk & VaR Platform

An interactive Streamlit dashboard for exploring portfolio market risk across nine phases: market data and portfolio setup, statistical diagnostics, portfolio P&L, VaR/CVaR, EWMA volatility, GARCH volatility, skewed-t/GJR-GARCH, VaR backtesting, and stress testing.

## Run locally

1. Keep `app_v4.py`, `Main_market_risk (5).ipynb`, and `requirements.txt` in the same folder.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   python -m streamlit run app_v4.py
   ```

## Deploy

Deploy `app_v4.py` from this GitHub repository using [Streamlit Community Cloud](https://share.streamlit.io/). Select `app_v4.py` as the app's main file.

## Methodology

The dashboard presents outputs from the accompanying notebook and provides educational interpretation notes. Results depend on the notebook's data sample, assumptions, units, and model implementation. The project is for educational and portfolio demonstration purposes and is not investment advice.

## Files

- `app_v4.py` — Streamlit application
- `Main_market_risk (5).ipynb` — analysis notebook
- `requirements.txt` — Python dependencies

## Before publishing

- Verify the notebook runs from top to bottom in a fresh environment.
- Remove API keys, credentials, personal data, or any private information.
- Confirm all displayed risk numbers use the intended units, horizons, and sign conventions.
