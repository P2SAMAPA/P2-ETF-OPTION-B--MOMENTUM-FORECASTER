"""
selector.py
Daily ETF + hold period selection and walk-forward backtest.

Scoring per ETF × hold_period:
  net_score = arima_forecast[h] - fee + momentum_bonus + reversal_pressure_bonus

CASH overlay: 2-day cumulative return <= -15% triggers CASH.
Exit CASH: ARIMA direction turns positive on best ETF.
"""

import numpy as np
import pandas as pd
from datetime import datetime

HOLD_PERIODS          = [1, 3, 5]
CASH_DRAWDOWN_TRIGGER = -0.15
MOMENTUM_WINDOW       = 5
MOMENTUM_WEIGHT       = 0.3
PRESSURE_WEIGHT       = 0.2


def _rolling_momentum(returns: np.ndarray, window: int = MOMENTUM_WINDOW) -> float:
    if len(returns) < window:
        return 0.0
    return float(np.sum(returns[-window:]))


def score_etf_hold(arima_forecast: float, hold_period: int, fee_bps: int,
                   momentum: float, pressure: float, direction: int) -> float:
    fee        = fee_bps / 10000
    net_arima  = arima_forecast - fee
    mom_bonus  = MOMENTUM_WEIGHT * momentum if np.sign(momentum) == direction else 0.0
    press_bonus = PRESSURE_WEIGHT * pressure if direction == 1 else 0.0
    return net_arima + mom_bonus + press_bonus


def select_signal(arima_results: dict, run_scores: dict,
                  df: pd.DataFrame, active_etfs: list,
                  as_of_idx: int, fee_bps: int,
                  hold_periods: list = HOLD_PERIODS) -> dict:
    scores = {}
    for etf in active_etfs:
        ret_col   = f"{etf}_Ret"
        arima     = arima_results.get(etf, {})
        run       = run_scores.get(etf, {})
        recent    = (df[ret_col].iloc[max(0, as_of_idx-MOMENTUM_WINDOW): as_of_idx].values
                     if ret_col in df.columns else np.array([]))
        momentum  = _rolling_momentum(recent)
        pressure  = run.get("pressure", 0.0)
        direction = arima.get("direction", 0)
        forecasts = arima.get("forecasts", {h: 0.0 for h in hold_periods})

        scores[etf] = {}
        for h in hold_periods:
            scores[etf][h] = score_etf_hold(
                forecasts.get(h, 0.0), h, fee_bps, momentum, pressure, direction,
            )

    best_etf, best_h, best_score = active_etfs[0], hold_periods[0], -np.inf
    for etf in active_etfs:
        for h in hold_periods:
            s = scores[etf][h]
            if s > best_score:
                best_score, best_etf, best_h = s, etf, h

    return {
        "etf":             best_etf,
        "hold_period":     best_h,
        "net_score":       best_score,
        "arima_direction": arima_results.get(best_etf, {}).get("direction", 0),
        "scores":          scores,
        "in_cash":         False,
    }


def execute_backtest(df: pd.DataFrame, active_etfs: list,
                     test_slice: slice, run_stats: dict,
                     lookback: int, fee_bps: int,
                     tbill_rate: float,
                     hold_periods: list = HOLD_PERIODS) -> dict:
    from arima_forecaster import run_all_etfs
    from run_analysis import get_reversal_scores

    daily_tbill    = tbill_rate / 252
    fee            = fee_bps / 10000
    today          = datetime.now().date()
    test_indices   = list(range(*test_slice.indices(len(df))))

    if not test_indices:
        return {}

    strat_rets     = []
    audit_trail    = []
    date_index     = []
    in_cash        = False
    recent_rets    = []
    hold_remaining = 0
    current_etf    = active_etfs[0]
    current_h      = hold_periods[0]

    progress = st_progress_placeholder()

    for step, idx in enumerate(test_indices):
        trade_date = df.index[idx]

        # ── Re-evaluate when hold expires ─────────────────────────────────────
        if hold_remaining <= 0:
            arima_res  = run_all_etfs(df.iloc[:idx], active_etfs, lookback, hold_periods)
            rev_scores = get_reversal_scores(df, active_etfs, run_stats, idx)
            signal     = select_signal(arima_res, rev_scores, df,
                                       active_etfs, idx, fee_bps, hold_periods)
            current_etf    = signal["etf"]
            current_h      = signal["hold_period"]
            hold_remaining = current_h

        # ── Realized return ───────────────────────────────────────────────────
        ret_col  = f"{current_etf}_Ret"
        realized = float(np.clip(
            df[ret_col].iloc[idx] if ret_col in df.columns else 0.0,
            -0.5, 0.5,
        ))
        if np.isnan(realized):
            realized = 0.0

        # ── CASH drawdown check ───────────────────────────────────────────────
        recent_rets.append(realized)
        if len(recent_rets) > 2:
            recent_rets.pop(0)
        two_day = ((1 + recent_rets[0]) * (1 + recent_rets[-1]) - 1
                   if len(recent_rets) >= 2 else 0.0)

        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash        = True
            hold_remaining = 0

        # ── Exit CASH when ARIMA turns bullish ────────────────────────────────
        if in_cash and hold_remaining <= 0:
            arima_check = run_all_etfs(df.iloc[:idx], active_etfs, lookback, [1])
            best_dir    = max(active_etfs,
                              key=lambda e: arima_check.get(e, {}).get("forecasts", {}).get(1, -1))
            if arima_check.get(best_dir, {}).get("direction", -1) == 1:
                in_cash = False

        net_ret    = (daily_tbill - fee) if in_cash else (realized - fee)
        signal_etf = "CASH" if in_cash else current_etf

        strat_rets.append(net_ret)
        date_index.append(trade_date)
        hold_remaining = max(0, hold_remaining - 1)

        if trade_date.date() < today:
            audit_trail.append({
                "Date":       trade_date.strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Hold":       f"{current_h}d",
                "Net_Return": net_ret,
                "In_Cash":    in_cash,
            })

    strat_rets = np.array(strat_rets, dtype=np.float64)
    metrics    = _compute_metrics(strat_rets, tbill_rate, date_index)

    return {
        **metrics,
        "strat_rets":  strat_rets,
        "audit_trail": audit_trail,
        "current_etf": current_etf,
        "current_h":   current_h,
    }


def st_progress_placeholder():
    """Dummy — progress shown via app.py spinner."""
    return None


def _compute_metrics(strat_rets, tbill_rate, date_index=None):
    if len(strat_rets) == 0:
        return {}

    cum     = np.cumprod(1 + strat_rets)
    n       = len(strat_rets)
    ann_ret = float(cum[-1] ** (252 / n) - 1)
    excess  = strat_rets - tbill_rate / 252
    sharpe  = float(np.mean(excess) / (np.std(strat_rets) + 1e-9) * np.sqrt(252))
    hit     = float(np.mean(strat_rets[-15:] > 0))
    cum_max = np.maximum.accumulate(cum)
    dd      = (cum - cum_max) / cum_max
    max_dd  = float(np.min(dd))

    worst_idx  = int(np.argmin(strat_rets))
    max_daily  = float(strat_rets[worst_idx])
    worst_date = (date_index[worst_idx].strftime("%Y-%m-%d")
                  if date_index and worst_idx < len(date_index) else "N/A")

    return {
        "cum_returns":    cum,
        "ann_return":     ann_ret,
        "sharpe":         sharpe,
        "hit_ratio":      hit,
        "max_dd":         max_dd,
        "max_daily_dd":   max_daily,
        "max_daily_date": worst_date,
        "cum_max":        cum_max,
    }


def compute_benchmark_metrics(returns, tbill_rate):
    return _compute_metrics(np.array(returns, dtype=np.float64), tbill_rate)
