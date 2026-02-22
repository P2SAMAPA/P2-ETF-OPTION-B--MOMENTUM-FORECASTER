# P2-ETF-ARIMA-FORECASTER-TUST-CHINA

Quantitative ETF trading strategy based on:
> *A Quantitative Trading Strategy Based on A Position Management Model*
> Xu et al., Tianjin University of Science and Technology, 2022

## Strategy Overview

- **ARIMA(p,d,q)** rolling price forecaster per ETF (auto order selection via AIC)
- **Consecutive run analysis** (Apriori-style) to detect statistically overdue reversals
- **Dynamic hold period** selection: 1d / 3d / 5d — chosen by highest expected net return after fees
- **CASH overlay**: 2-day cumulative return ≤ −15% triggers cash protection
- **5 ETFs**: TLT · TBT · VNQ · SLV · GLD
- **Benchmarks**: SPY · AGG
- **Data**: HuggingFace Dataset `P2SAMAPA/fi-etf-macro-signal-master-data` (2008→today)

## Configuration

| Parameter | Options |
|-----------|---------|
| Start year | 2008–2025 |
| Transaction cost | 0–100 bps (steps of 5) |
| Data split | 80% train / 10% val / 10% OOS |
| Auto-lookback | 30 / 45 / 60d (selected by val MAE) |
| Hold periods | 1d / 3d / 5d |

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `HF_TOKEN` as an environment variable or HF Space secret.

## File Structure
```
├── app.py                  # Main Streamlit app
├── loader.py               # HF Dataset loader, ETF availability checks
├── arima_forecaster.py     # Rolling ARIMA, auto order selection, forecasts
├── run_analysis.py         # Consecutive run stats, reversal pressure scoring
├── selector.py             # ETF+hold scoring, CASH overlay, walk-forward backtest
├── components.py           # All Streamlit UI components
├── cache.py                # MD5 cache helpers
├── requirements.txt
└── README.md
```
