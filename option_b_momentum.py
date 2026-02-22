"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation — RANK-BASED.

CASH overlay (corrected):
- Track the ACTUAL daily return of the held ETF each day
- At the START of each day, check if the prior 2 days' returns compound to <= -10%
- If yes: enter CASH for TODAY (not tomorrow)
- Exit CASH: when top-ranked ETF is positive across ALL three lookbacks
- Fee charged only on ETF switches, not on hold days
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER = -0.10  # 2-day compound return threshold

LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126


def _trailing_return(prices: pd.Series, lookback: int) -> float:
    clean = prices.dropna()
    if len(clean) < lookback + 1:
        return np.nan
    start = float(clean.iloc[-(lookback + 1)])
    end   = float(clean.iloc[-1])
    if start <= 0:
        return np.nan
    return (end - start) / start


def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int,
                             lb_short: int = LOOKBACK_1M,
                             lb_mid:   int = LOOKBACK_3M,
                             lb_long:  int = LOOKBACK_6M) -> dict:
    """
    Rank-based composite momentum score.
    For each lookback: rank all ETFs 1=best to N=worst by trailing return.
    Composite rank = average rank across all three lookbacks.
    Lower composite rank = stronger momentum = selected.
    """
    price_slice = df.iloc[:as_of_idx]
    n_etfs      = len(active_etfs)
    lookbacks   = [lb_short, lb_mid, lb_long]

    rets = {lb: {} for lb in lookbacks}
    for etf in active_etfs:
        for lb in lookbacks:
            if etf in df.columns:
                rets[lb][etf] = _trailing_return(price_slice[etf], lb)
            else:
                rets[lb][etf] = np.nan

    ranks = {lb: {} for lb in lookbacks}
    for lb in lookbacks:
        sorted_etfs = sorted(
            active_etfs,
            key=lambda e: rets[lb].get(e) if (
                rets[lb].get(e) is not None and not np.isnan(rets[lb].get(e))
            ) else -np.inf,
            reverse=True,
        )
        for rank, etf in enumerate(sorted_etfs, start=1):
            ranks[lb][etf] = rank

    scores = {}
    for etf in active_etfs:
        r_s = ranks[lb_short].get(etf, n_etfs)
        r_m = ranks[lb_mid].get(etf,   n_etfs)
        r_l = ranks[lb_long].get(etf,  n_etfs)
        composite_rank = (r_s + r_m + r_l) / 3.0
        final_score    = n_etfs + 1 - composite_rank

        scores[etf] = {
            "rank_score":  composite_rank,
            "final_score": final_score,
            "ret_1m":  rets[lb_short].get(etf, 0.0) or 0.0,
            "ret_3m":  rets[lb_mid].get(etf,   0.0) or 0.0,
            "ret_6m":  rets[lb_long].get(etf,  0.0) or 0.0,
            "rank_1m": r_s,
            "rank_3m": r_m,
            "rank_6m": r_l,
        }
    return scores


def select_top_etf(momentum_scores: dict) -> tuple:
    if not momentum_scores:
        return None, 0.0
    best_etf = min(momentum_scores, key=lambda e: momentum_scores[e]["rank_score"])
    return best_etf, momentum_scores[best_etf]["final_score"]


def _all_lookbacks_positive(momentum_scores: dict, etf: str) -> bool:
    """True only if ETF has positive return on ALL three lookbacks — safe to exit CASH."""
    info = momentum_scores.get(etf, {})
    return (info.get("ret_1m", -1) > 0 and
            info.get("ret_3m", -1) > 0 and
            info.get("ret_6m", -1) > 0)


def execute_backtest_b(df: pd.DataFrame,
                       active_etfs: list,
                       test_slice: slice,
                       lookback: int,
                       fee_bps: int,
                       tbill_rate: float) -> dict:
    """
    Walk-forward backtest.

    CASH trigger logic (corrected):
    - We keep a rolling history of ACTUAL daily returns (ret_history)
    - At the START of each day BEFORE deciding what to trade:
        two_day = (1 + ret_history[-2]) * (1 + ret_history[-1]) - 1
        if two_day <= -10%: enter CASH
    - This means Jan 30 loss shows up in ret_history[-1] on Feb 2
      so Feb 2 correctly triggers CASH before any trade is made
    """
    daily_tbill = tbill_rate / 252
    fee         = fee_bps / 10000
    today       = datetime.now().date()
    test_indices = list(range(*test_slice.indices(len(df))))

    lb_long  = lookback
    lb_mid   = max(lookback // 2, 5)
    lb_short = max(lookback // 3, 3)

    if not test_indices:
        return {}

    strat_rets   = []
    audit_trail  = []
    date_index   = []

    in_cash      = False
    ret_history  = [0.0, 0.0]   # rolling last-2 actual daily returns
    current_etf  = None

    for idx in test_indices:
        trade_date = df.index[idx]

        # ── Rank ETFs for today ────────────────────────────────────────────────
        mom_scores           = compute_momentum_scores(
            df, active_etfs, idx, lb_short, lb_mid, lb_long,
        )
        best_etf, best_score = select_top_etf(mom_scores)
        if best_etf is None:
            best_etf = active_etfs[0]

        # ── CASH check at START of day using prior 2 days' returns ────────────
        two_day = (1 + ret_history[-2]) * (1 + ret_history[-1]) - 1
        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash     = True
            current_etf = None

        # ── Exit CASH: top ETF positive on all lookbacks ───────────────────────
        if in_cash and _all_lookbacks_positive(mom_scores, best_etf):
            in_cash = False

        # ── Execute trade ──────────────────────────────────────────────────────
        if in_cash:
            signal_etf = "CASH"
            net_ret    = daily_tbill
        else:
            switched    = (best_etf != current_etf) and (current_etf is not None)
            current_etf = best_etf
            signal_etf  = best_etf
            ret_col     = f"{current_etf}_Ret"
            raw_ret     = 0.0
            if ret_col in df.columns:
                v = df[ret_col].iloc[idx]
                if not np.isnan(v):
                    raw_ret = float(np.clip(v, -0.5, 0.5))
            net_ret = raw_ret - (fee if switched else 0.0)

        # ── Update rolling return history ──────────────────────────────────────
        # Only track ETF returns (not CASH days) for drawdown detection
        actual_ret = net_ret if not in_cash else 0.0
        ret_history.append(actual_ret)
        ret_history = ret_history[-2:]  # keep only last 2

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
    }


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
