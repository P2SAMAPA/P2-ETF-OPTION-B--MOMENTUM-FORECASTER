"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation — THREE-FACTOR COMPOSITE.

Composite rank score (all rank-based, lower = better):
  50% — Trailing return rank       (price momentum over 3 lookback windows)
  25% — Relative strength vs SPY   (ETF return / SPY return, same windows)
  25% — MA slope rank              (50d MA / 200d MA ratio — trend acceleration)

CASH overlay:
  ENTER: 2-day compound return <= -10% (checked at start of day)
  EXIT:  top-ranked ETF Z-score >= 1.2 sigma vs 63-day rolling distribution
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER  = -0.10
ZSCORE_EXIT_THRESHOLD  = 1.2

LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126

# Factor weights
W_MOMENTUM = 0.50
W_RS_SPY   = 0.25
W_MA_SLOPE = 0.25


# ── Price helpers ─────────────────────────────────────────────────────────────

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
    """
    MA slope = (fast MA / slow MA) - 1
    Positive: short-term trend above long-term = accelerating momentum
    Negative: short-term trend below long-term = decelerating
    """
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
    """
    Relative strength = ETF trailing return / SPY trailing return over lookback.
    > 1.0: ETF outperforming SPY
    < 1.0: ETF underperforming SPY
    If SPY return is near zero or negative, use ETF return minus SPY return instead.
    """
    r_etf = _trailing_return(etf_prices, lookback)
    r_spy = _trailing_return(spy_prices,  lookback)

    if r_etf is None or np.isnan(r_etf):
        return np.nan
    if r_spy is None or np.isnan(r_spy) or abs(r_spy) < 0.001:
        # SPY flat/negative — use excess return instead
        return r_etf - (r_spy if r_spy is not None and not np.isnan(r_spy) else 0.0)
    return r_etf / r_spy


# ── Ranking helper ────────────────────────────────────────────────────────────

def _rank_dict(values: dict, higher_is_better: bool = True) -> dict:
    """
    Rank a dict of {etf: value} from 1 (best) to N (worst).
    NaN values get rank N (worst).
    """
    n = len(values)
    sorted_etfs = sorted(
        values.keys(),
        key=lambda e: values[e]
            if (values[e] is not None and not np.isnan(values[e] or np.nan))
            else (-np.inf if higher_is_better else np.inf),
        reverse=higher_is_better,
    )
    return {etf: rank for rank, etf in enumerate(sorted_etfs, start=1)}


# ── Core scoring ──────────────────────────────────────────────────────────────

def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int,
                             lb_short: int = LOOKBACK_1M,
                             lb_mid:   int = LOOKBACK_3M,
                             lb_long:  int = LOOKBACK_6M) -> dict:
    """
    Three-factor composite rank score for each ETF.

    Factor 1 — Trailing Return Rank (50%)
        Rank ETFs by trailing return over lb_short / lb_mid / lb_long.
        Average the three ranks → momentum_rank.

    Factor 2 — Relative Strength vs SPY Rank (25%)
        Rank ETFs by (ETF_return / SPY_return) over lb_short / lb_mid / lb_long.
        Average the three ranks → rs_rank.

    Factor 3 — MA Slope Rank (25%)
        Rank ETFs by (50d_MA / 200d_MA - 1).
        Single rank per ETF → ma_rank.

    Composite rank = 0.50 * momentum_rank + 0.25 * rs_rank + 0.25 * ma_rank
    Lower composite rank = stronger overall signal = selected.
    """
    price_slice = df.iloc[:as_of_idx]
    n_etfs      = len(active_etfs)
    lookbacks   = [lb_short, lb_mid, lb_long]

    # ── Factor 1: Trailing returns ────────────────────────────────────────────
    trail_rets = {lb: {} for lb in lookbacks}
    for etf in active_etfs:
        for lb in lookbacks:
            trail_rets[lb][etf] = (
                _trailing_return(price_slice[etf], lb)
                if etf in df.columns else np.nan
            )

    trail_ranks = {}
    for lb in lookbacks:
        trail_ranks[lb] = _rank_dict(trail_rets[lb], higher_is_better=True)

    # ── Factor 2: Relative strength vs SPY ───────────────────────────────────
    spy_prices = price_slice["SPY"] if "SPY" in df.columns else None

    rs_rets = {lb: {} for lb in lookbacks}
    for etf in active_etfs:
        for lb in lookbacks:
            if spy_prices is not None and etf in df.columns:
                rs_rets[lb][etf] = _relative_strength(
                    price_slice[etf], spy_prices, lb
                )
            else:
                rs_rets[lb][etf] = np.nan

    rs_ranks = {}
    for lb in lookbacks:
        rs_ranks[lb] = _rank_dict(rs_rets[lb], higher_is_better=True)

    # ── Factor 3: MA slope ────────────────────────────────────────────────────
    ma_slopes = {}
    for etf in active_etfs:
        ma_slopes[etf] = (
            _ma_slope(price_slice[etf])
            if etf in df.columns else np.nan
        )
    ma_rank = _rank_dict(ma_slopes, higher_is_better=True)

    # ── Composite ─────────────────────────────────────────────────────────────
    scores = {}
    for etf in active_etfs:
        # Average momentum ranks across 3 lookbacks
        m_ranks = [trail_ranks[lb].get(etf, n_etfs) for lb in lookbacks]
        momentum_rank = sum(m_ranks) / 3.0

        # Average RS ranks across 3 lookbacks
        r_ranks = [rs_ranks[lb].get(etf, n_etfs) for lb in lookbacks]
        rs_rank_avg = sum(r_ranks) / 3.0

        # MA slope rank (single)
        ma_r = ma_rank.get(etf, n_etfs)

        # Weighted composite rank
        composite_rank = (W_MOMENTUM * momentum_rank +
                          W_RS_SPY   * rs_rank_avg   +
                          W_MA_SLOPE * ma_r)

        # Invert for display (higher = better)
        final_score = n_etfs + 1 - composite_rank

        scores[etf] = {
            "rank_score":     composite_rank,
            "final_score":    final_score,
            # Factor details for UI
            "ret_1m":         trail_rets[lb_short].get(etf, 0.0) or 0.0,
            "ret_3m":         trail_rets[lb_mid].get(etf,   0.0) or 0.0,
            "ret_6m":         trail_rets[lb_long].get(etf,  0.0) or 0.0,
            "rank_1m":        trail_ranks[lb_short].get(etf, n_etfs),
            "rank_3m":        trail_ranks[lb_mid].get(etf,   n_etfs),
            "rank_6m":        trail_ranks[lb_long].get(etf,  n_etfs),
            "rs_spy":         rs_rets[lb_mid].get(etf, 0.0) or 0.0,  # mid window
            "rs_rank":        round(rs_rank_avg, 2),
            "ma_slope":       ma_slopes.get(etf, 0.0) or 0.0,
            "ma_rank":        ma_r,
            "momentum_rank":  round(momentum_rank, 2),
        }

    return scores


def select_top_etf(momentum_scores: dict) -> tuple:
    if not momentum_scores:
        return None, 0.0
    best_etf = min(momentum_scores, key=lambda e: momentum_scores[e]["rank_score"])
    return best_etf, momentum_scores[best_etf]["final_score"]


# ── CASH helpers ──────────────────────────────────────────────────────────────

def _all_lookbacks_positive(momentum_scores: dict, etf: str) -> bool:
    info = momentum_scores.get(etf, {})
    return (info.get("ret_1m", -1) > 0 and
            info.get("ret_3m", -1) > 0 and
            info.get("ret_6m", -1) > 0)


def compute_zscore(df: pd.DataFrame, etf: str, as_of_idx: int,
                   window: int = 63) -> float:
    ret_col = f"{etf}_Ret"
    if ret_col not in df.columns:
        return 0.0
    rets = df[ret_col].iloc[max(0, as_of_idx - window): as_of_idx].dropna().values
    if len(rets) < 10:
        return 0.0
    mean = np.mean(rets)
    std  = np.std(rets)
    if std < 1e-9:
        return 0.0
    today_ret = df[ret_col].iloc[as_of_idx - 1] if as_of_idx > 0 else 0.0
    if np.isnan(today_ret):
        return 0.0
    return float((today_ret - mean) / std)


def should_exit_cash(df: pd.DataFrame, best_etf: str, as_of_idx: int,
                     momentum_scores: dict) -> bool:
    """Exit CASH when top ETF Z-score >= 1.2σ vs 63-day rolling distribution."""
    z = compute_zscore(df, best_etf, as_of_idx)
    return z >= ZSCORE_EXIT_THRESHOLD


# ── Walk-forward backtest ─────────────────────────────────────────────────────

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

    strat_rets  = []
    audit_trail = []
    date_index  = []

    in_cash     = False
    ret_history = [0.0, 0.0]
    current_etf = None

    for idx in test_indices:
        trade_date = df.index[idx]

        mom_scores           = compute_momentum_scores(
            df, active_etfs, idx, lb_short, lb_mid, lb_long,
        )
        best_etf, best_score = select_top_etf(mom_scores)
        if best_etf is None:
            best_etf = active_etfs[0]

        # ── CASH check at START of day ─────────────────────────────────────────
        two_day = (1 + ret_history[-2]) * (1 + ret_history[-1]) - 1
        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash     = True
            current_etf = None

        # ── Exit CASH via Z-score ──────────────────────────────────────────────
        if in_cash and should_exit_cash(df, best_etf, idx, mom_scores):
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
        actual_ret = net_ret if not in_cash else 0.0
        ret_history.append(actual_ret)
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
