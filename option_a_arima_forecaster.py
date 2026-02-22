"""
arima_forecaster.py
Rolling ARIMA forecaster for each ETF.
- Auto-selects ARIMA order via AIC on training window
- Produces 1d, 3d, 5d price direction + magnitude forecasts
- No external data — uses only price series from HF Dataset
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

HOLD_PERIODS  = [1, 3, 5]
ARIMA_ORDERS  = [(1,1,1), (1,1,0), (0,1,1), (2,1,1), (1,1,2)]
MIN_TRAIN_LEN = 30


def _is_stationary(series: np.ndarray) -> bool:
    try:
        return adfuller(series, autolag="AIC")[1] < 0.05
    except Exception:
        return False


def _best_arima_order(series: np.ndarray) -> tuple:
    best_aic   = np.inf
    best_order = (1, 1, 1)
    d = 0 if _is_stationary(series) else 1

    for p, _, q in ARIMA_ORDERS:
        order = (p, d, q)
        try:
            fit = SARIMAX(series, order=order, trend="c",
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=50)
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, order
        except Exception:
            continue

    return best_order


def _fit_arima(price_series: np.ndarray, order: tuple):
    try:
        return SARIMAX(price_series, order=order, trend="c",
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=100)
    except Exception:
        return None


def _forecast_returns(fitted_model, last_price: float,
                      hold_periods: list = HOLD_PERIODS) -> dict:
    results = {}
    try:
        fc = fitted_model.forecast(steps=max(hold_periods))
        for h in hold_periods:
            fp  = float(fc.iloc[h-1]) if hasattr(fc, "iloc") else float(fc[h-1])
            ret = np.clip((fp - last_price) / (last_price + 1e-9), -0.5*h, 0.5*h)
            results[h] = float(ret)
    except Exception:
        for h in hold_periods:
            results[h] = 0.0
    return results


def run_arima_for_etf(price_series: pd.Series, lookback: int,
                      hold_periods: list = HOLD_PERIODS) -> dict:
    clean = price_series.dropna()
    if len(clean) < MIN_TRAIN_LEN:
        return {
            "order": None,
            "forecasts": {h: 0.0 for h in hold_periods},
            "direction": 0,
            "error": f"Insufficient data ({len(clean)} rows)",
        }

    window     = clean.values[-lookback:]
    last_price = float(window[-1])
    order      = _best_arima_order(window)
    model      = _fit_arima(window, order)

    if model is None:
        return {
            "order": order,
            "forecasts": {h: 0.0 for h in hold_periods},
            "direction": 0,
            "error": "ARIMA fit failed",
        }

    forecasts = _forecast_returns(model, last_price, hold_periods)
    return {
        "order":     order,
        "forecasts": forecasts,
        "direction": 1 if forecasts.get(1, 0) > 0 else -1,
        "error":     None,
    }


def run_all_etfs(df: pd.DataFrame, active_etfs: list,
                 lookback: int, hold_periods: list = HOLD_PERIODS) -> dict:
    results = {}
    for etf in active_etfs:
        if etf not in df.columns:
            results[etf] = {
                "order": None,
                "forecasts": {h: 0.0 for h in hold_periods},
                "direction": 0,
                "error": f"{etf} not in dataframe",
            }
            continue
        results[etf] = run_arima_for_etf(df[etf], lookback, hold_periods)
    return results


def select_best_lookback_arima(price_df: pd.DataFrame, active_etfs: list,
                                train_end_idx: int, val_end_idx: int,
                                candidates: list = None) -> int:
    if candidates is None:
        candidates = [30, 45, 60]

    best_lb, best_mae = candidates[0], np.inf

    for lb in candidates:
        maes = []
        for etf in active_etfs:
            if etf not in price_df.columns:
                continue
            series = price_df[etf].dropna().values
            if len(series) < train_end_idx:
                continue

            train_w   = series[train_end_idx - lb: train_end_idx]
            val_prices = series[train_end_idx: val_end_idx]

            if len(train_w) < MIN_TRAIN_LEN or len(val_prices) == 0:
                continue

            try:
                order  = _best_arima_order(train_w)
                errors = []
                window = list(train_w)
                for actual in val_prices[:20]:
                    m = _fit_arima(np.array(window[-lb:]), order)
                    if m is None:
                        break
                    pred = float(m.forecast(steps=1).iloc[0])
                    errors.append(abs(pred - actual))
                    window.append(actual)
                if errors:
                    maes.append(np.mean(errors))
            except Exception:
                continue

        avg = np.mean(maes) if maes else np.inf
        if avg < best_mae:
            best_mae, best_lb = avg, lb

    return best_lb
