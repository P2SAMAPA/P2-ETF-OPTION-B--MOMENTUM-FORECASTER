---
title: P2-ETF-ARIMA-FORECASTER-TUST-CHINA
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# P2-ETF Forecaster

Two-option quantitative ETF rotation strategy using the same dataset, 
same data split, same fee slider and same CASH overlay.

---

## Option A — ARIMA Forecaster
Based on: *A Quantitative Trading Strategy Based on A Position Management Model*
— Xu, Wang, Han et al., Tianjin University of Science & Technology (2022)

- Rolling **ARIMA(p,d,q)** price forecaster per ETF (auto order via AIC)
- **Consecutive run analysis** (Apriori-style) — detects statistically overdue reversals
- **Dynamic hold period**: 1d / 3d / 5d — chosen by highest net return after fees
- Fee charged on every trade

---

## Option B — Cross-Sectional Momentum Rotation

- Ranks all 5 ETFs daily by **composite trailing return**: equal-weight 1m / 3m / 6m
- Always holds **Top 1 ETF**
- **Fee only charged on switches** — not on hold days
- Exit CASH when top-ranked ETF composite score turns positive

---

## Shared Configuration

| Parameter | Detail |
|-----------|--------|
| Start year | Slider: 2008 – 2025 |
| Transaction cost | Slider: 0 – 100 bps (steps of 5) |
| Data split | 80% train · 10% val · 10% OOS |
| ETFs | TLT · TBT · VNQ · SLV · GLD |
| Benchmarks | SPY · AGG |
| CASH overlay | 2-day cumulative return ≤ −15% |
| Data source | HF Dataset `P2SAMAPA/fi-etf-macro-signal-master-data` |

---

## File Structure
```
├── app.py                        # Shared shell + Option A/B toggle
├── loader.py                     # HF Dataset loader, ETF availability checks
├── cache.py                      # MD5 cache helpers
├── components.py                 # All shared + option-specific UI components
├── option_a_arima_forecaster.py  # Option A: rolling ARIMA, auto order, forecasts
├── option_a_run_analysis.py      # Option A: consecutive run stats, reversal pressure
├── option_a_selector.py          # Option A: ETF+hold scoring, backtest
└── option_b_momentum.py          # Option B: momentum ranking, backtest
```

Set `HF_TOKEN` as a secret in HF Space settings.
