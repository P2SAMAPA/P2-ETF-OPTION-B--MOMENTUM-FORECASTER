"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation.

Logic:
- Rank all active ETFs daily by composite trailing return score:
    score = (1m_ret + 3m_ret + 6m_ret) / 3
- Always hold top-ranked ETF
- Transaction fee charged only on switches
- CASH overlay: 2-day cumulative return <= -15%
- Exit CASH: top-ranked ETF composite score > 0
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER = -0.15
LOOKBACK_1M  = 21    # ~1 month trading days
LOOKBACK_3M  = 63    # ~3 months
LOOKBACK_6M  = 126   # ~6 months


# ── Momentum scoring ──────────────────────────────────────────────────────────

def compute_trailing_return(prices: pd.Series, lookback: int) -> float:
    """Compute trailing return over `lookback` days from price series."""
    clean = prices.dropna()
    if len(clean) < lookback + 1:
        return 0.0
    start = float(clean.iloc[-(lookback + 1)])
    end   = float(clean.iloc[-1])
    if start <= 0:
        return 0.0
    return (end - start) / start


def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int) -> dict:
    """
    Compute composite momentum score for each ETF at as_of_idx.
    Score = equal-weight average of 1m, 3m, 6m trailing returns.

    Returns: {etf: {"score": float, "ret_1m": float, "ret_3m": float, "ret_6m": float}}
    """
    scores = {}
    price_slice = df.iloc[:as_of_idx]

    for etf in active_etfs:
        if etf not in df.columns:
            scores[etf] = {"score": 0.0, "ret_1m": 0.0, "ret_3m": 0.0, "ret_6m": 0.0}
            continue

        prices = price_slice[etf]
        r1m    = compute_trailing_return(prices, LOOKBACK_1M)
        r3m    = compute_trailing_return(prices, LOOKBACK_3M)
        r6m    = compute_trailing_return(prices, LOOKBACK_6M)

        # Equal weight composite
        composite = (r1m + r3m + r6m) / 3.0

        scores[etf] = {
            "score":  composite,
            "ret_1m": r1m,
            "ret_3m": r3m,
            "ret_6m": r6m,
        }

    return scores


def rank_etfs(momentum_scores: dict) -> list:
    """
    Rank ETFs by composite momentum score, descending.
    Returns list of (etf, score) tuples.
    """
    ranked = sorted(
        momentum_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )
    return [(etf, info["score"]) for etf, info in ranked]


def select_top_etf(momentum_scores: dict) -> tuple[str, float]:
    """Return (best_etf, score) — top ranked by composite momentum."""
    ranked = rank_etfs(momentum_scores)
    if not ranked:
        return None, 0.0
    return ranked[0]


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def execute_backtest_b(df: pd.DataFrame,
                       active_etfs: list,
                       test_slice: slice,
                       lookback: int,
                       fee_bps: int,
                       tbill_rate: float) -> dict:
    """
    Walk-forward backtest for Option B (cross-sectional momentum).

    - Re-ranks ETFs daily
    - Fee only charged when ETF switches
    - CASH overlay: enter on 2-day <= -15%, exit when top ETF score > 0
    """
    daily_tbill  = tbill_rate / 252
    fee          = fee_bps / 10000
    today        = datetime.now().date()
    test_indices = list(range(*test_slice.indices(len(df))))

    if not test_indices:
        return {}

    strat_rets   = []
    audit_trail  = []
    date_index   = []

    in_cash      = False
    recent_rets  = []
    current_etf  = None

    for idx in test_indices:
        trade_date = df.index[idx]

        # ── Daily momentum ranking ────────────────────────────────────────────
        mom_scores   = compute_momentum_scores(df, active_etfs, idx)
        best_etf, best_score = select_top_etf(mom_scores)

        if best_etf is None:
            best_etf = active_etfs[0]

        # ── Realized return for current ETF ───────────────────────────────────
        ret_col  = f"{current_etf}_Ret" if current_etf else f"{best_etf}_Ret"
        realized = 0.0
        if ret_col in df.columns:
            v = df[ret_col].iloc[idx]
            if not np.isnan(v):
                realized = float(np.clip(v, -0.5, 0.5))

        # ── 2-day CASH drawdown check ─────────────────────────────────────────
        recent_rets.append(realized)
        if len(recent_rets) > 2:
            recent_rets.pop(0)
        two_day = ((1 + recent_rets[0]) * (1 + recent_rets[-1]) - 1
                   if len(recent_rets) >= 2 else 0.0)

        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash     = True
            current_etf = None   # force re-eval on exit

        # ── Exit CASH: top ETF has positive composite momentum ────────────────
        if in_cash and best_score > 0:
            in_cash = False

        # ── Determine signal and net return ───────────────────────────────────
        if in_cash:
            signal_etf = "CASH"
            net_ret    = daily_tbill  # no fee while in cash
        else:
            # Charge fee only on switch
            switched   = (best_etf != current_etf) and (current_etf is not None)
            signal_etf = best_etf
            net_ret    = realized - (fee if switched else 0.0)
            current_etf = best_etf

        strat_rets.append(net_ret)
        date_index.append(trade_date)

        if trade_date.date() < today:
            audit_trail.append({
                "Date":       trade_date.strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Score":      round(best_score, 4),
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
