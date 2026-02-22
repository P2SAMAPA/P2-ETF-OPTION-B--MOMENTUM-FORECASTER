"""
app.py
P2-ETF-ARIMA-FORECASTER-TUST-CHINA
Shared shell with Option A (ARIMA) and Option B (Cross-Sectional Momentum).
Each option is fully self-contained in its own module.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Shared modules ────────────────────────────────────────────────────────────
from loader     import (load_dataset, check_data_freshness, prepare_data,
                         dataset_summary, get_next_trading_day, get_est_time)
from cache      import (make_cache_key, save_cache, load_cache,
                         make_lb_cache_key)
from components import (show_freshness_status, show_availability_warnings,
                         show_signal_banner, show_metrics_row,
                         show_audit_trail, show_audit_trail_b,
                         show_etf_scores_table, show_hold_period_rationale,
                         show_momentum_scores_table)

# ── Option A modules ──────────────────────────────────────────────────────────
from option_a_arima_forecaster import (run_all_etfs,
                                        select_best_lookback_arima)
from option_a_run_analysis     import (compute_all_run_stats,
                                        get_reversal_scores)
from option_a_selector         import (execute_backtest, select_signal,
                                        compute_benchmark_metrics)

# ── Option B modules ──────────────────────────────────────────────────────────
from option_b_momentum import (execute_backtest_b, compute_momentum_scores,
                                select_top_etf, LOOKBACK_6M,
                                _all_lookbacks_positive)

st.set_page_config(
    page_title="P2-ETF Forecaster",
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
    ("momentum_scores",  None),
    ("test_slice",       None),
    ("optimal_lookback", None),
    ("df_ready",         None),
    ("tbill_rate",       None),
    ("active_etfs",      None),
    ("availability",     None),
    ("spy_ann",          None),
    ("fee_bps",          10),
    ("option",           "Option A — ARIMA Forecaster"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"🕒 **EST:** {get_est_time().strftime('%H:%M:%S')}")
    st.divider()

    option = st.radio(
        "🧭 Strategy",
        ["Option A — ARIMA Forecaster", "Option B — Momentum Rotation"],
        index=0,
    )

    st.divider()

    start_yr = None
    if option == "Option A — ARIMA Forecaster":
        start_yr = st.slider("📅 Start Year", 2008, 2025, 2015)

    if option == "Option B — Momentum Rotation":
        lookback_months = st.select_slider(
            "📅 Momentum Lookback",
            options=[1, 3, 6, 9, 12, 15],
            value=6,
            format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
        )
        st.caption("Ranks ETFs by trailing return over selected period")
    else:
        lookback_months = 6
    fee_bps  = st.select_slider(
        "💰 Transaction Cost (bps)",
        options=list(range(0, 105, 5)),
        value=10,
    )

    st.divider()
    st.caption("📐 **Data Split:** 80% train · 10% val · 10% OOS")

    if option == "Option A — ARIMA Forecaster":
        st.caption("🔍 **Auto-lookback:** 30 / 45 / 60d via val MAE")
        st.caption("⏱️ **Hold periods:** 1d · 3d · 5d net of fees")
        st.caption("📄 *Based on Xu et al., TUST China (2022)*")
    else:
        st.caption("📊 **Ranking:** 1m / 3m / 6m composite trailing return")
        st.caption("🏆 **Selection:** Top 1 ETF daily")
        st.caption("💸 **Fee:** Only charged on ETF switches")

    st.caption("🛡️ **CASH overlay:** 2-day ≤ −10% drawdown")
    st.divider()

    run_button = st.button(
        f"🚀 Run {option.split(' — ')[0]}",
        type="primary",
        use_container_width=True,
    )

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📈 P2-ETF Forecaster")

if option == "Option A — ARIMA Forecaster":
    st.caption(
        "**Option A:** ARIMA price forecasting · Consecutive run analysis · "
        "Dynamic hold period (1d/3d/5d) · CASH drawdown overlay"
    )
    st.caption(
        "📄 *Based on: A Quantitative Trading Strategy Based on A Position "
        "Management Model — Xu et al., TUST China (2022)*"
    )
else:
    st.caption(
        "**Option B:** Cross-sectional momentum rotation · "
        "Composite 1m/3m/6m trailing return ranking · "
        "Top 1 ETF · Fee on switches only · CASH drawdown overlay"
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
    st.session_state.option       = option

    with st.spinner("🔧 Preparing data..."):
        effective_start = start_yr if option == "Option A — ARIMA Forecaster" else 2008
        df, availability, active_etfs, tbill_rate = prepare_data(df_raw, effective_start)

    show_availability_warnings(availability)

    if not active_etfs:
        st.error("❌ No ETFs available for the selected start year.")
        st.stop()

    n  = len(df)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)
    train_slice = slice(0, t1)
    test_slice  = slice(t2, n)

    st.info(
        f"📅 **{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}**  "
        f"· Train: **{t1}** · Val: **{t2-t1}** · OOS: **{n-t2}** days  "
        f"· ETFs: **{', '.join(active_etfs)}**"
    )

    last_date_str = str(freshness.get("last_date", "unknown"))

    # ── SPY ann return (shared) ───────────────────────────────────────────────
    spy_ann = None
    if "SPY_Ret" in df.columns:
        spy_raw = df["SPY_Ret"].iloc[test_slice].values.astype(float)
        spy_raw = np.clip(spy_raw[~np.isnan(spy_raw)], -0.5, 0.5)
        if len(spy_raw) > 5:
            spy_ann = float(np.prod(1 + spy_raw) ** (252 / len(spy_raw)) - 1)

    # ═══════════════════════════════════════════════════════════════
    # OPTION A — ARIMA
    # ═══════════════════════════════════════════════════════════════
    if option == "Option A — ARIMA Forecaster":

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
            run_stats_live = compute_all_run_stats(df, active_etfs,
                                                    slice(0, len(df)))
            arima_results  = run_all_etfs(df, active_etfs,
                                           optimal_lookback, HOLD_PERIODS)
            rev_scores     = get_reversal_scores(df, active_etfs,
                                                  run_stats_live, len(df))
            signal         = select_signal(arima_results, rev_scores, df,
                                           active_etfs, len(df),
                                           fee_bps, HOLD_PERIODS)

        st.session_state.update({
            "output_ready":     True,
            "result":           result,
            "arima_results":    arima_results,
            "run_scores":       rev_scores,
            "signal":           signal,
            "momentum_scores":  None,
            "test_slice":       test_slice,
            "optimal_lookback": optimal_lookback,
            "df_ready":         df,
            "tbill_rate":       tbill_rate,
            "active_etfs":      active_etfs,
            "availability":     availability,
            "spy_ann":          spy_ann,
            "fee_bps":          fee_bps,
            "option":           option,
        })

    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # OPTION B — MOMENTUM
    # ═══════════════════════════════════════════════════════════════
    else:
        # Lookback windows scale with user slider
        LB_MAP = {
            1:  (5,   10,  21),   # 1w,  2w,  1m
            3:  (21,  42,  63),   # 1m,  2m,  3m
            6:  (42,  84,  126),  # 2m,  4m,  6m
            9:  (63,  126, 189),  # 3m,  6m,  9m
            12: (126, 189, 252),  # 6m,  9m,  12m
            15: (126, 252, 315),  # 6m,  12m, 15m
        }
        lb_short, lb_mid, lb_long = LB_MAP.get(lookback_months, (42, 84, 126))

        cache_key   = make_cache_key(last_date_str, 2008, fee_bps,
                                      "80/10/10_B", lookback_months)
        cached_data = load_cache(cache_key)

        if cached_data is not None:
            result = cached_data["result"]
            st.success("⚡ Results from cache.")
        else:
            with st.spinner("⏳ Running cross-sectional momentum backtest on OOS..."):
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
            in_cash_live         = not _all_lookbacks_positive(mom_scores, best_etf)
            signal = {
                "etf":         "CASH" if in_cash_live else best_etf,
                "hold_period": 1,
                "net_score":   best_score,
                "in_cash":     in_cash_live,
                "scores":      {etf: {1: mom_scores[etf]["final_score"]}
                                 for etf in active_etfs},
            }

        st.session_state.update({
            "output_ready":     True,
            "result":           result,
            "arima_results":    None,
            "run_scores":       None,
            "signal":           signal,
            "momentum_scores":  mom_scores,
            "test_slice":       test_slice,
            "optimal_lookback": None,
            "df_ready":         df,
            "tbill_rate":       tbill_rate,
            "active_etfs":      active_etfs,
            "availability":     availability,
            "spy_ann":          spy_ann,
            "fee_bps":          fee_bps,
            "option":           option,
            "lb_short":         lb_short,
            "lb_mid":           lb_mid,
            "lb_long":          lb_long,
        })
# ── Render (persists across reruns) ──────────────────────────────────────────
