"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation — RANK-BASED.

Scoring logic (not raw returns):
- For each lookback (1m, 3m, 6m), rank all active ETFs 1..N by trailing return
- Composite rank score = average rank across all 3 lookbacks
- Pick ETF with LOWEST composite rank (rank 1 = best)
- This makes the signal relative — "strongest ETF vs peers" not "biggest absolute return"
- Start year matters because it changes which ETF leads across different regimes

CASH overlay:
- ENTER: 2-day cumulative return <= -15%
- EXIT:  top-ranked ETF has POSITIVE trailing return on ALL three lookbacks
         (i.e. the market is actually trending up, not just relatively less bad)

Fee charged only on ETF switches, not on hold days.
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER = -0.15
LOOKBACK_1M  = 21    # ~1 month trading days
LOOKBACK_3M  = 63    # ~3 months
LOOKBACK_6M  = 126   # ~6 months


# ── Trailing return helper ────────────────────────────────────────────────────

def _trailing_return(prices: pd.Series, lookback: int) -> float:
    clean = prices.dropna()
    if len(clean) < lookback + 1:
        return np.nan
    start = float(clean.iloc[-(lookback + 1)])
    end   = float(clean.iloc[-1])
    if start <= 0:
        return np.nan
    return (end - start) / start


# ── Core rank-based scoring ───────────────────────────────────────────────────

def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int) -> dict:
    """
    Compute rank-based composite momentum score for each ETF.

    For each lookback period:
      - Compute trailing return for all ETFs
      - Rank them 1 (best) to N (worst)

    Composite rank = average rank across 1m / 3m / 6m
    Lower composite rank = stronger momentum = should be selected

    Returns:
      {etf: {
          "rank_score":   float,   # composite rank (lower = better)
          "final_score":  float,   # inverted for display (higher = better)
          "ret_1m":       float,
          "ret_3m":       float,
          "ret_6m":       float,
          "rank_1m":      int,
          "rank_3m":      int,
          "rank_6m":      int,
      }}
    """
    price_slice = df.iloc[:as_of_idx]
    n_etfs      = len(active_etfs)

    # Compute raw trailing returns per lookback
    rets = {lb: {} for lb in [LOOKBACK_1M, LOOKBACK_3M, LOOKBACK_6M]}
    for etf in active_etfs:
        if etf not in df.columns:
            for lb in rets:
                rets[lb][etf] = np.nan
            continue
        for lb in rets:
            rets[lb][etf] = _trailing_return(price_slice[etf], lb)

    # Rank per lookback (rank 1 = highest return = best)
    # NaN gets worst rank
    ranks = {lb: {} for lb in [LOOKBACK_1M, LOOKBACK_3M, LOOKBACK_6M]}
    for lb in [LOOKBACK_1M, LOOKBACK_3M, LOOKBACK_6M]:
        lb_rets   = rets[lb]
        # Sort descending by return, NaN goes last
        sorted_etfs = sorted(
            active_etfs,
            key=lambda e: lb_rets.get(e, np.nan)
                if not np.isnan(lb_rets.get(e, np.nan)) else -np.inf,
            reverse=True,
        )
        for rank, etf in enumerate(sorted_etfs, start=1):
            ranks[lb][etf] = rank

    # Composite rank = average rank across 3 lookbacks (lower = better)
    scores = {}
    for etf in active_etfs:
        r1 = ranks[LOOKBACK_1M].get(etf, n_etfs)
        r3 = ranks[LOOKBACK_3M].get(etf, n_etfs)
        r6 = ranks[LOOKBACK_6M].get(etf, n_etfs)
        composite_rank = (r1 + r3 + r6) / 3.0

        # Invert for display: higher = better (max possible rank = n_etfs)
        final_score = n_etfs + 1 - composite_rank

        scores[etf] = {
            "rank_score":  composite_rank,
            "final_score": final_score,
            "ret_1m":      rets[LOOKBACK_1M].get(etf, 0.0) or 0.0,
            "ret_3m":      rets[LOOKBACK_3M].get(etf, 0.0) or 0.0,
            "ret_6m":      rets[LOOKBACK_6M].get(etf, 0.0) or 0.0,
            "rank_1m":     r1,
            "rank_3m":     r3,
            "rank_6m":     r6,
        }

    return scores


def select_top_etf(momentum_scores: dict) -> tuple:
    """
    Return (best_etf, final_score) — lowest composite rank = winner.
    """
    if not momentum_scores:
        return None, 0.0

    best_etf = min(
        momentum_scores.keys(),
        key=lambda e: momentum_scores[e]["rank_score"],
    )
    return best_etf, momentum_scores[best_etf]["final_score"]


def _all_lookbacks_positive(momentum_scores: dict, etf: str) -> bool:
    """True if ETF has positive return on ALL three lookbacks — safe to exit CASH."""
    info = momentum_scores.get(etf, {})
    return (info.get("ret_1m", -1) > 0 and
            info.get("ret_3m", -1) > 0 and
            info.get("ret_6m", -1) > 0)


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def execute_backtest_b(df: pd.DataFrame,
                       active_etfs: list,
                       test_slice: slice,
                       lookback: int,
                       fee_bps: int,
                       tbill_rate: float) -> dict:
    """
    Walk-forward backtest for Option B (rank-based cross-sectional momentum).

    - Re-ranks ETFs daily using rank-based composite score
    - Fee only charged when ETF switches
    - CASH overlay: enter on 2-day <= -15%
                    exit only when top ETF is positive across ALL lookbacks
    """
    daily_tbill  = tbill_rate / 252
    fee          = fee_bps / 10000
    today        = datetime.now().date()
    test_indices = list(range(*test_slice.indices(len(df))))

    if not test_indices:
        return {}

    strat_rets  = []
    audit_trail = []
    date_index  = []

    in_cash     = False
    recent_rets = []
    current_etf = None

    for idx in test_indices:
        trade_date = df.index[idx]

        # ── Daily rank-based momentum scoring ─────────────────────────────────
        mom_scores           = compute_momentum_scores(df, active_etfs, idx)
        best_etf, best_score = select_top_etf(mom_scores)

        if best_etf is None:
            best_etf = active_etfs[0]

        # ── Realized return for current ETF ───────────────────────────────────
        hold_etf = current_etf if current_etf else best_etf
        ret_col  = f"{hold_etf}_Ret"
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
            current_etf = None

        # ── Exit CASH: top ETF positive across ALL three lookbacks ────────────
        if in_cash and _all_lookbacks_positive(mom_scores, best_etf):
            in_cash = False

        # ── Determine signal and net return ───────────────────────────────────
        if in_cash:
            signal_etf = "CASH"
            net_ret    = daily_tbill
        else:
            switched    = (best_etf != current_etf) and (current_etf is not None)
            signal_etf  = best_etf
            net_ret     = realized - (fee if switched else 0.0)
            current_etf = best_etf

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
