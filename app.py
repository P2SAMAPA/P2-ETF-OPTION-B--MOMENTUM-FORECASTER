"""
app.py
P2-ETF-ARIMA-FORECASTER-TUST-CHINA
ARIMA-based quantitative ETF trading strategy.
All modules in root directory — flat structure.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from loader           import (load_dataset, check_data_freshness,
                               prepare_data, dataset_summary,
                               get_next_trading_day, get_est_time)
from arima_forecaster import (run_all_etfs, select_best_lookback_arima)
from run_analysis     import (compute_all_run_stats, get_reversal_scores)
from selector         import (execute_backtest, select_signal,
                               compute_benchmark_metrics)
from cache            import (make_cache_key, save_cache, load_cache,
                               make_lb_cache_key)
from components       import (show_freshness_status, show_availability_warnings,
                               show_signal_banner, show_etf_scores_table,
                               show_hold_period_rationale, show_metrics_row,
                               show_audit_trail)

st.set_page_config(
    page_title="P2-ETF-ARIMA Forecaster",
    page_icon="📈",
    layout="wide",
)

HF_TOKEN     = os.getenv("HF_TOKEN", "")
HOLD_PERIODS = [1, 3, 5]

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("output_ready",     False),
    ("result",           None),
    ("arima_results",    None),
    ("run_scores",       None),
    ("signal",           None),
    ("test_slice",       None),
    ("optimal_lookback", None),
    ("df_ready",         None),
    ("tbill_rate",       None),
    ("active_etfs",      None),
    ("availability",     None),
    ("spy_ann",          None),
    ("fee_bps",          10),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    start_yr = st.slider("📅 Start Year", 2008, 2025, 2015)
    fee_bps  = st.select_slider(
        "💰 Transaction Cost (bps)",
        options=list(range(0, 105, 5)),
        value=10,
    )

    st.divider()
    st.caption("📐 **Data Split:** 80% train · 10% val · 10% OOS")
    st.caption("🔍 **Auto-lookback:** 30 / 45 / 60d via val MAE")
    st.caption("⏱️ **Hold periods:** 1d · 3d · 5d net of fees")
    st.caption("🛡️ **CASH overlay:** 2-day ≤ −15% drawdown")
    st.divider()

    run_button = st.button(
        "🚀 Run ARIMA Strategy",
        type="primary",
        use_container_width=True,
    )

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF-ARIMA Forecaster")
st.caption(
    "ARIMA price forecasting · Consecutive run analysis (Apriori) · "
    "Dynamic hold period selection net of fees · CASH drawdown overlay"
)
st.caption(
    "📄 Based on: *A Quantitative Trading Strategy Based on A Position Management Model* "
    "— Xu, Wang, Han et al., Tianjin University of Science & Technology (2022)"
)

if not HF_TOKEN:
    st.error("❌ HF_TOKEN secret not found. Add it in HF Space settings.")
    st.stop()

# ── Load dataset ──────────────────────────────────────────────────────────────
with st.spinner("📡 Loading dataset from HuggingFace..."):
    df_raw = load_dataset(HF_TOKEN)

if df_raw.empty:
    st.error("❌ Dataset failed to load.")
    st.stop()

freshness = check_data_freshness(df_raw)
show_freshness_status(freshness)

# ── Sidebar dataset info ──────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📦 Dataset Info")
    summ = dataset_summary(df_raw)
    if summ:
        st.write(f"**Rows:** {summ['rows']:,}")
        st.write(f"**Range:** {summ['start_date']} → {summ['end_date']}")
        st.write(f"**ETFs:** {', '.join(summ['etfs'])}")
        st.write(f"**Benchmarks:** {', '.join(summ['benchmarks'])}")
        st.write(f"**T-bill:** {'✅' if summ['tbill'] else '❌'}")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_button:
    st.session_state.output_ready = False

    with st.spinner("🔧 Preparing data..."):
        df, availability, active_etfs, tbill_rate = prepare_data(df_raw, start_yr)

    show_availability_warnings(availability)

    if not active_etfs:
        st.error("❌ No ETFs available for the selected start year.")
        st.stop()

    n  = len(df)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)
    train_slice = slice(0, t1)
    val_slice   = slice(t1, t2)
    test_slice  = slice(t2, n)

    st.info(
        f"📅 **{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}**  "
        f"· Train: **{t1}** days · Val: **{t2-t1}** days · OOS: **{n-t2}** days  "
        f"· Active ETFs: **{', '.join(active_etfs)}**"
    )

    last_date_str = str(freshness.get("last_date", "unknown"))

    # ── Auto-select lookback ──────────────────────────────────────────────────
    lb_key    = make_lb_cache_key(last_date_str, start_yr, "80/10/10")
    lb_cached = load_cache(lb_key)

    if lb_cached is not None:
        optimal_lookback = lb_cached["lookback"]
        st.success(f"⚡ Cache hit · Optimal lookback: **{optimal_lookback}d**")
    else:
        with st.spinner("🔍 Auto-selecting lookback (30 / 45 / 60d) via val MAE..."):
            optimal_lookback = select_best_lookback_arima(
                df, active_etfs, t1, t2, candidates=[30, 45, 60],
            )
        save_cache(lb_key, {"lookback": optimal_lookback})
        st.success(f"📐 Optimal lookback: **{optimal_lookback}d** (auto-selected)")

    # ── Check result cache ────────────────────────────────────────────────────
    cache_key   = make_cache_key(last_date_str, start_yr, fee_bps,
                                  "80/10/10", optimal_lookback)
    cached_data = load_cache(cache_key)

    if cached_data is not None:
        result = cached_data["result"]
        st.success("⚡ Results loaded from cache — no retraining needed.")
    else:
        with st.spinner("📊 Computing run statistics on training data..."):
            run_stats = compute_all_run_stats(df, active_etfs, train_slice)

        with st.spinner("⏳ Running walk-forward ARIMA backtest on OOS data..."):
            result = execute_backtest(
                df=df,
                active_etfs=active_etfs,
                test_slice=test_slice,
                run_stats=run_stats,
                lookback=optimal_lookback,
                fee_bps=fee_bps,
                tbill_rate=tbill_rate,
                hold_periods=HOLD_PERIODS,
            )

        save_cache(cache_key, {"result": result})

    # ── Live next-day signal ──────────────────────────────────────────────────
    with st.spinner("🔮 Computing next-day signal..."):
        run_stats_live = compute_all_run_stats(df, active_etfs, slice(0, len(df)))
        arima_results  = run_all_etfs(df, active_etfs, optimal_lookback, HOLD_PERIODS)
        rev_scores     = get_reversal_scores(df, active_etfs, run_stats_live, len(df))
        signal         = select_signal(
            arima_results, rev_scores, df,
            active_etfs, len(df), fee_bps, HOLD_PERIODS,
        )

    # ── SPY return for comparison ─────────────────────────────────────────────
    spy_ann = None
    if "SPY_Ret" in df.columns:
        spy_raw = df["SPY_Ret"].iloc[test_slice].values.astype(float)
        spy_raw = np.clip(spy_raw[~np.isnan(spy_raw)], -0.5, 0.5)
        if len(spy_raw) > 5:
            spy_ann = float(np.prod(1 + spy_raw) ** (252 / len(spy_raw)) - 1)

    # ── Persist to session state ──────────────────────────────────────────────
    st.session_state.update({
        "output_ready":     True,
        "result":           result,
        "arima_results":    arima_results,
        "run_scores":       rev_scores,
        "signal":           signal,
        "test_slice":       test_slice,
        "optimal_lookback": optimal_lookback,
        "df_ready":         df,
        "tbill_rate":       tbill_rate,
        "active_etfs":      active_etfs,
        "availability":     availability,
        "spy_ann":          spy_ann,
        "fee_bps":          fee_bps,
    })

# ── Render (persists across reruns) ──────────────────────────────────────────
if not st.session_state.output_ready:
    st.info("👈 Configure parameters and click **🚀 Run ARIMA Strategy**.")
    st.stop()

result           = st.session_state.result
arima_results    = st.session_state.arima_results
run_scores       = st.session_state.run_scores
signal           = st.session_state.signal
df               = st.session_state.df_ready
tbill_rate       = st.session_state.tbill_rate
active_etfs      = st.session_state.active_etfs
spy_ann          = st.session_state.spy_ann
fee_bps          = st.session_state.fee_bps
optimal_lookback = st.session_state.optimal_lookback

next_date = get_next_trading_day()
st.divider()

# ── Signal banner ─────────────────────────────────────────────────────────────
show_signal_banner(
    etf=signal["etf"],
    hold_period=signal["hold_period"],
    next_date=next_date,
    net_score=signal["net_score"],
    in_cash=signal["in_cash"],
)

st.divider()

# ── ETF breakdown table ───────────────────────────────────────────────────────
show_etf_scores_table(
    scores=signal["scores"],
    arima_results=arima_results,
    run_scores=run_scores,
    active_etfs=active_etfs,
    hold_periods=HOLD_PERIODS,
    fee_bps=fee_bps,
)

st.divider()

# ── Hold period rationale ─────────────────────────────────────────────────────
show_hold_period_rationale(
    best_etf=signal["etf"],
    best_h=signal["hold_period"],
    scores=signal["scores"],
    hold_periods=HOLD_PERIODS,
    fee_bps=fee_bps,
)

st.divider()

# ── OOS Performance ───────────────────────────────────────────────────────────
st.subheader("📊 Out-of-Sample Performance Metrics")
show_metrics_row(result, tbill_rate, spy_ann=spy_ann)

st.divider()

# ── Audit trail ───────────────────────────────────────────────────────────────
st.subheader("📋 Audit Trail — Last 20 Trading Days")
show_audit_trail(result.get("audit_trail", []))
