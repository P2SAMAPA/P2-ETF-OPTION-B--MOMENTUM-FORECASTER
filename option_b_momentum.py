"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation — THREE-FACTOR COMPOSITE.

CASH overlay:
  ENTER: 2-day compound return of held ETF <= -10%
  EXIT:  top-ranked ETF has 1-month trailing return > 0,
         AND at least MIN_CASH_DAYS days have passed (to avoid instant re‑entry),
         OR after MAX_CASH_DAYS days in cash (forced exit on 3rd day).
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER = -0.10
MIN_CASH_DAYS         = 2      # must stay in CASH at least this many days
MAX_CASH_DAYS         = 2      # force exit after this many days in CASH

LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126

W_MOMENTUM = 0.50
W_RS_SPY   = 0.25
W_MA_SLOPE = 0.25


# ── Price helpers (unchanged) ──────────────────────────────────────────────────
def _trailing_return(prices: pd.Series, lookback: int) -> float:
    clean = prices.dropna()
    if len(clean) < lookback + 1:
        return np.nan
    start = float(clean.iloc[-(lookback + 1)])
    end   = float(clean.iloc[-1])
    if start <= 0:
        return np.nan
    return (end - start) / start


def _ma_slope(prices: pd.Series, fast: int = 50, slow: int = 200) -> float:
    clean = prices.dropna()
    if len(clean) < slow:
        return np.nan
    ma_fast = float(clean.iloc[-fast:].mean())
    ma_slow = float(clean.iloc[-slow:].mean())
    if ma_slow <= 0:
        return np.nan
    return (ma_fast / ma_slow) - 1.0


def _relative_strength(etf_prices: pd.Series, spy_prices: pd.Series,
                        lookback: int) -> float:
    r_etf = _trailing_return(etf_prices, lookback)
    r_spy = _trailing_return(spy_prices, lookback)
    if r_etf is None or np.isnan(r_etf):
        return np.nan
    if r_spy is None or np.isnan(r_spy) or abs(r_spy) < 0.001:
        return r_etf - (r_spy if r_spy is not None and not np.isnan(r_spy) else 0.0)
    return r_etf / r_spy


def _rank_dict(values: dict, higher_is_better: bool = True) -> dict:
    sorted_etfs = sorted(
        values.keys(),
        key=lambda e: values[e]
            if (values[e] is not None and not np.isnan(values[e] or np.nan))
            else (-np.inf if higher_is_better else np.inf),
        reverse=higher_is_better,
    )
    return {etf: rank for rank, etf in enumerate(sorted_etfs, start=1)}


# ── Core scoring (unchanged) ──────────────────────────────────────────────────
def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int,
                             lb_short: int = LOOKBACK_1M,
                             lb_mid:   int = LOOKBACK_3M,
                             lb_long:  int = LOOKBACK_6M) -> dict:
    price_slice = df.iloc[:as_of_idx]
    n_etfs      = len(active_etfs)
    lookbacks   = [lb_short, lb_mid, lb_long]

    trail_rets  = {lb: {etf: (_trailing_return(price_slice[etf], lb)
                               if etf in df.columns else np.nan)
                        for etf in active_etfs}
                   for lb in lookbacks}
    trail_ranks = {lb: _rank_dict(trail_rets[lb], higher_is_better=True)
                   for lb in lookbacks}

    spy_prices = price_slice["SPY"] if "SPY" in df.columns else None
    rs_rets    = {lb: {etf: (_relative_strength(price_slice[etf], spy_prices, lb)
                              if spy_prices is not None and etf in df.columns else np.nan)
                       for etf in active_etfs}
                  for lb in lookbacks}
    rs_ranks   = {lb: _rank_dict(rs_rets[lb], higher_is_better=True)
                  for lb in lookbacks}

    ma_slopes = {etf: (_ma_slope(price_slice[etf]) if etf in df.columns else np.nan)
                 for etf in active_etfs}
    ma_rank   = _rank_dict(ma_slopes, higher_is_better=True)

    scores = {}
    for etf in active_etfs:
        momentum_rank = sum(trail_ranks[lb].get(etf, n_etfs) for lb in lookbacks) / 3.0
        rs_rank_avg   = sum(rs_ranks[lb].get(etf, n_etfs)    for lb in lookbacks) / 3.0
        ma_r          = ma_rank.get(etf, n_etfs)
        composite     = (W_MOMENTUM * momentum_rank +
                         W_RS_SPY   * rs_rank_avg   +
                         W_MA_SLOPE * ma_r)
        scores[etf]   = {
            "rank_score":    composite,
            "final_score":   n_etfs + 1 - composite,
            "ret_1m":        trail_rets[lb_short].get(etf, 0.0) or 0.0,
            "ret_3m":        trail_rets[lb_mid].get(etf,   0.0) or 0.0,
            "ret_6m":        trail_rets[lb_long].get(etf,  0.0) or 0.0,
            "rank_1m":       trail_ranks[lb_short].get(etf, n_etfs),
            "rank_3m":       trail_ranks[lb_mid].get(etf,   n_etfs),
            "rank_6m":       trail_ranks[lb_long].get(etf,  n_etfs),
            "rs_spy":        rs_rets[lb_mid].get(etf, 0.0) or 0.0,
            "rs_rank":       round(rs_rank_avg, 2),
            "ma_slope":      ma_slopes.get(etf, 0.0) or 0.0,
            "ma_rank":       ma_r,
            "momentum_rank": round(momentum_rank, 2),
        }
    return scores


def select_top_etf(momentum_scores: dict) -> tuple:
    if not momentum_scores:
        return None, 0.0
    best_etf = min(momentum_scores, key=lambda e: momentum_scores[e]["rank_score"])
    return best_etf, momentum_scores[best_etf]["final_score"]


# ── CASH re‑entry (modified) ──────────────────────────────────────────────────
def should_exit_cash(best_etf: str,
                     momentum_scores: dict,
                     cash_days_held: int) -> bool:
    """
    Exit CASH when:
      1. cash_days_held >= MAX_CASH_DAYS (forced exit)
      OR
      2. cash_days_held >= MIN_CASH_DAYS AND top ETF 1‑month return > 0
         (no requirement on 3‑month return)
    """
    # Forced exit after MAX_CASH_DAYS days
    if cash_days_held >= MAX_CASH_DAYS:
        return True

    # Early exit condition: after minimum days, require only 1m return positive
    if cash_days_held >= MIN_CASH_DAYS:
        info  = momentum_scores.get(best_etf, {})
        ret1m = info.get("ret_1m", -1.0)
        if ret1m is not None and not np.isnan(ret1m) and ret1m > 0:
            return True

    return False


# ── Walk‑forward backtest (unchanged except for the new exit logic) ──────────
def execute_backtest_b(df: pd.DataFrame,
                       active_etfs: list,
                       test_slice: slice,
                       lookback: int,
                       fee_bps: int,
                       tbill_rate: float) -> dict:
    daily_tbill  = tbill_rate / 252
    fee          = fee_bps / 10000
    today        = datetime.now().date()
    test_indices = list(range(*test_slice.indices(len(df))))

    lb_long  = lookback
    lb_mid   = max(lookback // 2, 5)
    lb_short = max(lookback // 3, 3)

    if not test_indices:
        return {}

    strat_rets    = []
    audit_trail   = []
    date_index    = []
    in_cash       = False
    cash_days_held = 0
    ret_history   = [0.0, 0.0]
    current_etf   = None

    for idx in test_indices:
        trade_date = df.index[idx]

        mom_scores           = compute_momentum_scores(
            df, active_etfs, idx, lb_short, lb_mid, lb_long,
        )
        best_etf, best_score = select_top_etf(mom_scores)
        if best_etf is None:
            best_etf = active_etfs[0]

        # ── CASH entry ────────────────────────────────────────────────────────
        two_day = (1 + ret_history[-2]) * (1 + ret_history[-1]) - 1
        if two_day <= CASH_DRAWDOWN_TRIGGER:
            if not in_cash:
                in_cash        = True
                cash_days_held = 0
                current_etf    = None

        # ── CASH exit ─────────────────────────────────────────────────────────
        if in_cash:
            cash_days_held += 1
            if should_exit_cash(best_etf, mom_scores, cash_days_held):
                in_cash        = False
                cash_days_held = 0
                current_etf    = None

        # ── Execute ───────────────────────────────────────────────────────────
        if in_cash:
            signal_etf     = "CASH"
            net_ret        = daily_tbill
            actual_etf_ret = 0.0
        else:
            switched       = (best_etf != current_etf) and (current_etf is not None)
            current_etf    = best_etf
            signal_etf     = best_etf
            ret_col        = f"{current_etf}_Ret"
            raw_ret        = 0.0
            if ret_col in df.columns:
                v = df[ret_col].iloc[idx]
                if not np.isnan(v):
                    raw_ret = float(np.clip(v, -0.5, 0.5))
            net_ret        = raw_ret - (fee if switched else 0.0)
            actual_etf_ret = raw_ret

        ret_history.append(actual_etf_ret)
        ret_history = ret_history[-2:]

        strat_rets.append(net_ret)
        date_index.append(trade_date)

        if trade_date.date() < today:
            audit_trail.append({
                "Date":       trade_date.strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Rank Score": round(mom_scores.get(signal_etf, {}).get("rank_score", 0), 2)
                              if signal_etf != "CASH" else "—",
                "Net_Return": net_ret,
                "In_Cash":    in_cash,
            })

    strat_rets = np.array(strat_rets, dtype=np.float64)
    metrics    = _compute_metrics(strat_rets, tbill_rate, date_index)

    return {
        **metrics,
        "strat_rets":      strat_rets,
        "audit_trail":     audit_trail,
        "current_etf":     current_etf,
        "momentum_scores": mom_scores,
        "ended_in_cash":   in_cash,
        "cash_days_held":  cash_days_held,
    }


def _compute_metrics(strat_rets, tbill_rate, date_index=None):
    if len(strat_rets) == 0:
        return {}
    cum        = np.cumprod(1 + strat_rets)
    n          = len(strat_rets)
    ann_ret    = float(cum[-1] ** (252 / n) - 1)
    excess     = strat_rets - tbill_rate / 252
    sharpe     = float(np.mean(excess) / (np.std(strat_rets) + 1e-9) * np.sqrt(252))
    hit        = float(np.mean(strat_rets[-15:] > 0))
    cum_max    = np.maximum.accumulate(cum)
    dd         = (cum - cum_max) / cum_max
    max_dd     = float(np.min(dd))
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
