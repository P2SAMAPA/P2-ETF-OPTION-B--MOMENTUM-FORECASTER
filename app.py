"""
app.py
P2-ETF Forecaster
Option A (ARIMA) and Option B (Momentum) are fully segregated.
Each option's modules are imported ONLY when that option is selected.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

from loader import (load_dataset, check_data_freshness, prepare_data,
                    dataset_summary, get_next_trading_day, get_est_time)
from cache import (make_cache_key, save_cache, load_cache, make_lb_cache_key)
from components import (show_freshness_status, show_availability_warnings,
                        show_signal_banner, show_metrics_row,
                        show_audit_trail, show_audit_trail_b,
                        show_etf_scores_table, show_hold_period_rationale,
                        show_momentum_scores_table, show_methodology)

st.set_page_config(
    page_title="P2-ETF Forecaster",
    page_icon="📈",
    layout="wide",
)

HF_TOKEN     = os.getenv("HF_TOKEN", "")
HOLD_PERIODS = [1, 3, 5]

LB_MAP = {
    3:  (21,  42,  63),
    6:  (42,  84,  126),
    9:  (63,  126, 189),
    12: (84,  168, 252),
    15: (105, 210, 315),
    18: (126, 252, 378),
}

for key, default in [
    ("output_ready", False), ("result", None), ("arima_results", None),
    ("run_scores", None), ("signal", None), ("momentum_scores", None),
    ("test_slice", None), ("optimal_lookback", None), ("df_ready", None),
    ("tbill_rate", None), ("active_etfs", None), ("availability", None),
    ("spy_ann", None), ("fee_bps", 10), ("option", "Option A — ARIMA Forecaster"),
    ("lb_short", 42), ("lb_mid", 84), ("lb_long", 126),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 EST: {get_est_time().strftime('%H:%M:%S')}")
    st.divider()
    option = st.radio(
        "🧭 Strategy",
        ["Option A — ARIMA Forecaster", "Option B — Momentum Rotation"],
        index=0,
    )
    st.divider()

    if option == "Option A — ARIMA Forecaster":
        start_yr        = st.slider("📅 Start Year", 2008, 2025, 2015)
        lookback_months = None
        st.caption("🔍 **Auto-lookback:** 30 / 45 / 60d via val MAE")
        st.caption("⏱️ **Hold periods:** 1d · 3d · 5d net of fees")
        st.caption("📄 *Based on Xu et al., TUST China (2022)*")
    else:
        start_yr        = None
        lookback_months = st.select_slider(
            "📅 Momentum Lookback",
            options=[3, 6, 9, 12, 15, 18],
            value=6,
            format_func=lambda x: f"{x} months",
        )
        lb_short, lb_mid, lb_long = LB_MAP[lookback_months]
        st.caption(f"📊 Windows: **{lb_short//21}m** · **{lb_mid//21}m** · **{lookback_months}m**")
        st.caption("🏆 **Selection:** Top 1 ETF daily")
        st.caption("💸 **Fee:** Only charged on switches")

    fee_bps = st.select_slider(
        "💰 Transaction Cost (bps)",
        options=list(range(0, 105, 5)),
        value=10,
    )
    st.divider()
    if option == "Option A — ARIMA Forecaster":
        st.caption("📐 **Data Split:** 80% train · 10% val · 10% OOS")
    st.caption("🛡️ **CASH overlay:** 2-day ≤ −10% drawdown")
    st.divider()

    run_button = st.button(
        f"🚀 Run {option.split(' — ')[0]}",
        type="primary",
        use_container_width=True,
    )

st.title("📈 P2-ETF Forecaster")
if option == "Option A — ARIMA Forecaster":
    st.caption(
        "Option A: ARIMA price forecasting · Consecutive run analysis · "
        "Dynamic hold period (1d/3d/5d) · CASH drawdown overlay"
    )
    st.caption("📄 Based on: Xu et al., TUST China (2022)")
else:
    st.caption(
        "Option B: Cross-sectional momentum rotation · "
        "Rank-based composite score · Top 1 ETF · Fee on switches only · CASH overlay"
    )

if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found. Add it in HF Space settings.")
    st.stop()

with st.spinner("📡 Loading dataset from HuggingFace..."):
    df_raw = load_dataset(HF_TOKEN)
    if df_raw.empty:
        st.error("❌ Dataset failed to load.")
        st.stop()
    freshness = check_data_freshness(df_raw)
    show_freshness_status(freshness)

# ── DEBUG ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 DEBUG: Dataset Contents")
etf_cols = [c for c in df_raw.columns
            if any(e in c for e in ["TLT","VCIT","LQD","HYG","VNQ","SLV","GLD"])]
st.write("**ETF Columns Found:**", etf_cols)
for etf in ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD"]:
    if etf in df_raw.columns:
        valid = df_raw[etf].dropna().shape[0]
        total = df_raw.shape[0]
        nans  = df_raw[etf].isna().sum()
        fd    = df_raw[etf].dropna().index[0].strftime("%Y-%m-%d") if valid > 0 else "N/A"
        ld    = df_raw[etf].dropna().index[-1].strftime("%Y-%m-%d") if valid > 0 else "N/A"
        st.write(f"**{etf}:** {valid}/{total} valid rows | {nans} NaN | {fd} → {ld}")
    else:
        st.write(f"**{etf}:** ❌ Column not found in dataset")
st.info("⚠️ If VCIT/LQD/HYG show 6 rows or 'Column not found', the issue is in the dataset or loader.py")
st.divider()
# ── END DEBUG ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    summ = dataset_summary(df_raw)
    if summ:
        st.write(f" Rows: {summ['rows']:,}")
        st.write(f" Range: {summ['start_date']} → {summ['end_date']}")
        st.write(f" ETFs: {', '.join(summ['etfs'])}")
        st.write(f" Benchmarks: {', '.join(summ['benchmarks'])}")
        st.write(f" T-bill: {'✅' if summ['tbill'] else '❌'}")

if run_button:
    st.session_state.output_ready = False
    st.session_state.option       = option

    effective_start = start_yr if option == "Option A — ARIMA Forecaster" else 2008
    with st.spinner("🔧 Preparing data..."):
        df, availability, active_etfs, tbill_rate = prepare_data(df_raw, effective_start)

    show_availability_warnings(availability)
    if not active_etfs:
        st.error("❌ No ETFs available for the selected period.")
        st.stop()

    n             = len(df)
    t1            = int(n * 0.80)
    t2            = int(n * 0.90)
    train_slice   = slice(0, t1)
    test_slice    = slice(t2, n)
    last_date_str = str(freshness.get("last_date", "unknown"))

    if option == "Option A — ARIMA Forecaster":
        st.info(
            f"📅 **{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}**   "
            f"· Train: **{t1}** · Val: **{t2-t1}** · OOS: **{n-t2}** days   "
            f"· ETFs: **{', '.join(active_etfs)}**"
        )
    else:
        st.info(
            f"📅 Data through **{df.index[-1].strftime('%Y-%m-%d')}**   "
            f"· Lookback: up to **18 months** · OOS evaluation: **{n-t2}** days   "
            f"· ETFs: **{', '.join(active_etfs)}**"
        )

    spy_ann = None
    if "SPY_Ret" in df.columns:
        spy_raw = df["SPY_Ret"].iloc[test_slice].values.astype(float)
        spy_raw = np.clip(spy_raw[~np.isnan(spy_raw)], -0.5, 0.5)
        if len(spy_raw) > 5:
            spy_ann = float(np.prod(1 + spy_raw) ** (252 / len(spy_raw)) - 1)

    # ── OPTION A ──────────────────────────────────────────────────────────────
    if option == "Option A — ARIMA Forecaster":

        from option_a_arima_forecaster import run_all_etfs, select_best_lookback_arima
        from option_a_run_analysis     import compute_all_run_stats, get_reversal_scores
        from option_a_selector         import execute_backtest, select_signal

        lb_key    = make_lb_cache_key(last_date_str, start_yr, "80/10/10_A")
        lb_cached = load_cache(lb_key)
        if lb_cached is not None:
            optimal_lookback = lb_cached["lookback"]
            st.success(f"⚡ Cache hit · Lookback: **{optimal_lookback}d**")
        else:
            with st.spinner("🔍 Auto-selecting lookback (30/45/60d)..."):
                optimal_lookback = select_best_lookback_arima(
                    df, active_etfs, t1, t2, candidates=[30, 45, 60],
                )
            save_cache(lb_key, {"lookback": optimal_lookback})
            st.success(f"📐 Optimal lookback: **{optimal_lookback}d**")

        cache_key   = make_cache_key(last_date_str, start_yr, fee_bps,
                                     "80/10/10_A", optimal_lookback)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            result = cached_data["result"]
            st.success("⚡ Results from cache.")
        else:
            with st.spinner("📊 Computing run statistics..."):
                run_stats = compute_all_run_stats(df, active_etfs, train_slice)
            with st.spinner("⏳ Walk-forward ARIMA backtest on OOS..."):
                result = execute_backtest(
                    df=df, active_etfs=active_etfs,
                    test_slice=test_slice, run_stats=run_stats,
                    lookback=optimal_lookback, fee_bps=fee_bps,
                    tbill_rate=tbill_rate, hold_periods=HOLD_PERIODS,
                )
            save_cache(cache_key, {"result": result})

        with st.spinner("🔮 Computing next-day ARIMA signal..."):
            run_stats_live = compute_all_run_stats(df, active_etfs, slice(0, len(df)))
            arima_results  = run_all_etfs(df, active_etfs, optimal_lookback, HOLD_PERIODS)
            rev_scores     = get_reversal_scores(df, active_etfs, run_stats_live, len(df))
            signal         = select_signal(arima_results, rev_scores, df,
                                           active_etfs, len(df), fee_bps, HOLD_PERIODS)

        st.session_state.update({
            "output_ready": True, "result": result,
            "arima_results": arima_results, "run_scores": rev_scores,
            "signal": signal, "momentum_scores": None,
            "test_slice": test_slice, "optimal_lookback": optimal_lookback,
            "df_ready": df, "tbill_rate": tbill_rate,
            "active_etfs": active_etfs, "availability": availability,
            "spy_ann": spy_ann, "fee_bps": fee_bps, "option": option,
            "lb_short": None, "lb_mid": None, "lb_long": None,
        })

    # ── OPTION B ──────────────────────────────────────────────────────────────
    else:

        from option_b_momentum import (execute_backtest_b,
                                        compute_momentum_scores,
                                        select_top_etf,
                                        should_exit_cash)

        lb_short, lb_mid, lb_long = LB_MAP[lookback_months]

        cache_key   = make_cache_key(last_date_str, 2008, fee_bps,
                                     "80/10/10_B", lookback_months)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            result = cached_data["result"]
            st.success("⚡ Results from cache.")
        else:
            with st.spinner("⏳ Running momentum backtest on OOS..."):
                result = execute_backtest_b(
                    df=df, active_etfs=active_etfs,
                    test_slice=test_slice, lookback=lb_long,
                    fee_bps=fee_bps, tbill_rate=tbill_rate,
                )
            save_cache(cache_key, {"result": result})

        with st.spinner("🔮 Computing next-day momentum signal..."):
            mom_scores           = compute_momentum_scores(
                df, active_etfs, len(df), lb_short, lb_mid, lb_long,
            )
            best_etf, best_score = select_top_etf(mom_scores)

            # If backtest ended in CASH, check live re-entry condition.
            # should_exit_cash(best_etf, mom_scores) → True means re-enter.
            if result.get("ended_in_cash", False):
                in_cash_live = not should_exit_cash(best_etf, mom_scores)
            else:
                in_cash_live = False

            signal = {
                "etf":         "CASH" if in_cash_live else best_etf,
                "next_etf":    best_etf,
                "hold_period": 1,
                "net_score":   best_score,
                "in_cash":     in_cash_live,
                "scores":      {etf: {1: mom_scores[etf]["final_score"]}
                                for etf in active_etfs},
            }

        st.session_state.update({
            "output_ready": True, "result": result,
            "arima_results": None, "run_scores": None,
            "signal": signal, "momentum_scores": mom_scores,
            "test_slice": test_slice, "optimal_lookback": None,
            "df_ready": df, "tbill_rate": tbill_rate,
            "active_etfs": active_etfs, "availability": availability,
            "spy_ann": spy_ann, "fee_bps": fee_bps, "option": option,
            "lb_short": lb_short, "lb_mid": lb_mid, "lb_long": lb_long,
        })

# ── Render ────────────────────────────────────────────────────────────────────
if not st.session_state.output_ready:
    st.info("👈 Configure parameters and click 🚀 Run.")
    st.stop()

result         = st.session_state.result
signal         = st.session_state.signal
df             = st.session_state.df_ready
tbill_rate     = st.session_state.tbill_rate
active_etfs    = st.session_state.active_etfs
spy_ann        = st.session_state.spy_ann
fee_bps        = st.session_state.fee_bps
current_option = st.session_state.option
next_date      = get_next_trading_day()

st.divider()
show_signal_banner(
    etf=signal["etf"], hold_period=signal["hold_period"],
    next_date=next_date, net_score=signal["net_score"],
    in_cash=signal["in_cash"], next_etf=signal.get("next_etf", signal["etf"]),
)
st.divider()

if current_option == "Option A — ARIMA Forecaster":
    show_etf_scores_table(
        scores=signal["scores"], arima_results=st.session_state.arima_results,
        run_scores=st.session_state.run_scores, active_etfs=active_etfs,
        hold_periods=HOLD_PERIODS, fee_bps=fee_bps,
    )
    st.divider()
    show_hold_period_rationale(
        best_etf=signal["etf"], best_h=signal["hold_period"],
        scores=signal["scores"], hold_periods=HOLD_PERIODS, fee_bps=fee_bps,
    )
else:
    show_momentum_scores_table(
        momentum_scores=st.session_state.momentum_scores,
        active_etfs=active_etfs, current_etf=signal["etf"],
        lb_short_days=st.session_state.get("lb_short", 42),
        lb_mid_days=st.session_state.get("lb_mid", 84),
        lb_long_days=st.session_state.get("lb_long", 126),
    )

st.divider()
st.subheader("📊 Out-of-Sample Performance Metrics")
show_metrics_row(result, tbill_rate, spy_ann=spy_ann)

st.divider()
st.subheader("📋 Audit Trail — Last 20 Trading Days")
if current_option == "Option A — ARIMA Forecaster":
    show_audit_trail(result.get("audit_trail", []))
else:
    show_audit_trail_b(result.get("audit_trail", []))

if current_option == "Option B — Momentum Rotation":
    show_methodology()
