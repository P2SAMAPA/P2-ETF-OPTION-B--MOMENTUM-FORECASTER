"""
components.py
Streamlit UI components for P2-ETF-ARIMA-FORECASTER.
Shows: signal banner, per-ETF ARIMA scores table,
       hold period rationale, metrics row, audit trail.
"""

import streamlit as st
import pandas as pd
import numpy as np


def show_freshness_status(freshness: dict):
    if freshness.get("fresh"):
        st.success(freshness["message"])
    else:
        st.warning(freshness["message"])


def show_availability_warnings(availability: dict):
    for etf, info in availability.items():
        if not info.get("available"):
            st.warning(info.get("message", f"⚠️ {etf} data unavailable."))


def show_signal_banner(etf: str, hold_period: int, next_date,
                       net_score: float, in_cash: bool):
    is_cash = in_cash or etf == "CASH"
    bg = ("linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%)" if is_cash
          else "linear-gradient(135deg, #00d1b2 0%, #00a896 100%)")
    if is_cash:
        label = "⚠️ DRAWDOWN PROTECTION ACTIVE — CASH"
        sub   = "2-day cumulative return triggered CASH overlay"
    else:
        label = f"🎯 {next_date} → {etf}"
        sub   = f"Hold {hold_period}d · Net Score: {net_score:.4f}"

    st.markdown(f"""
    <div style="background:{bg}; padding:25px; border-radius:15px;
                text-align:center; box-shadow:0 8px 16px rgba(0,0,0,0.3); margin:16px 0;">
      <div style="color:rgba(255,255,255,0.7); font-size:12px;
                  letter-spacing:3px; margin-bottom:6px;">
        ARIMA FORECASTER · NEXT TRADING DAY SIGNAL
      </div>
      <h1 style="color:white; font-size:40px; margin:0 0 8px 0; font-weight:800;">
        {label}
      </h1>
      <div style="color:rgba(255,255,255,0.75); font-size:14px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def show_etf_scores_table(scores: dict, arima_results: dict,
                           run_scores: dict, active_etfs: list,
                           hold_periods: list, fee_bps: int):
    st.subheader("📊 ETF Signal Breakdown")
    st.caption(f"Transaction cost: **{fee_bps} bps** applied to net scores")

    rows = []
    for etf in active_etfs:
        arima         = arima_results.get(etf, {})
        run           = run_scores.get(etf, {})
        err           = arima.get("error")
        direction_str = "↑ Up" if arima.get("direction", 0) == 1 else "↓ Down"
        order_str     = str(arima.get("order", "N/A"))
        pressure      = run.get("pressure", 0.0)
        run_len       = run.get("run_length", 0)
        run_dir       = run.get("direction", "")

        row = {
            "ETF":          etf,
            "ARIMA Order":  order_str,
            "Direction":    direction_str if not err else f"⚠️ {err}",
            "Run Pressure": f"{pressure:.2f} ({run_len}d {run_dir})",
        }
        for h in hold_periods:
            row[f"{h}d Score"] = f"{scores.get(etf, {}).get(h, 0.0):.4f}"
        rows.append(row)

    df_table = pd.DataFrame(rows)

    def _highlight_best(row):
        try:
            vals = [float(r[f"1d Score"]) for r in rows]
            if float(row["1d Score"]) == max(vals):
                return ["background-color: rgba(0,200,150,0.15); font-weight:bold"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    styled = (
        df_table.style
        .apply(_highlight_best, axis=1)
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "13px"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "8px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True)


def show_hold_period_rationale(best_etf: str, best_h: int,
                                scores: dict, hold_periods: list, fee_bps: int):
    etf_scores = scores.get(best_etf, {})
    if not etf_scores:
        return

    st.caption(
        f"**Hold Period Selection for {best_etf}** "
        f"(fee = {fee_bps} bps amortised — higher hold = lower relative cost):"
    )
    cols = st.columns(len(hold_periods))
    for col, h in zip(cols, hold_periods):
        s       = etf_scores.get(h, 0.0)
        is_best = h == best_h
        color   = "#00d1b2" if is_best else "#888"
        badge   = "✅ SELECTED" if is_best else ""
        bg      = "rgba(0,209,178,0.08)" if is_best else "#f8f8f8"

        col.markdown(f"""
        <div style="border:2px solid {color}; border-radius:10px; padding:12px;
                    text-align:center; background:{bg};">
            <div style="font-size:11px; color:{color}; font-weight:700;
                        letter-spacing:1px; margin-bottom:4px;">
                {h}-DAY HOLD {badge}
            </div>
            <div style="font-size:24px; font-weight:800;
                        color:{'#00d1b2' if is_best else '#333'};">
                {s:.4f}
            </div>
            <div style="font-size:11px; color:#999;">net score</div>
        </div>
        """, unsafe_allow_html=True)


def show_metrics_row(result: dict, tbill_rate: float, spy_ann: float = None):
    c1, c2, c3, c4, c5 = st.columns(5)

    if spy_ann is not None:
        diff      = (result["ann_return"] - spy_ann) * 100
        sign      = "+" if diff >= 0 else ""
        delta_str = f"vs SPY: {sign}{diff:.2f}%"
    else:
        delta_str = f"vs T-bill: {(result['ann_return'] - tbill_rate)*100:.2f}%"

    c1.metric("📈 Ann. Return",   f"{result['ann_return']*100:.2f}%", delta=delta_str)
    c2.metric("📊 Sharpe",        f"{result['sharpe']:.2f}",
              delta="Strong" if result["sharpe"] > 1 else "Weak")
    c3.metric("🎯 Hit Ratio 15d", f"{result['hit_ratio']*100:.0f}%",
              delta="Good" if result["hit_ratio"] > 0.55 else "Weak")
    c4.metric("📉 Max Drawdown",  f"{result['max_dd']*100:.2f}%",
              delta="Peak to Trough")

    worst_date = result.get("max_daily_date", "N/A")
    dd_delta   = f"on {worst_date}" if worst_date != "N/A" else "Worst Single Day"
    c5.metric("⚠️ Max Daily DD",  f"{result['max_daily_dd']*100:.2f}%", delta=dd_delta)


def show_audit_trail(audit_trail: list):
    if not audit_trail:
        st.info("No audit trail data available.")
        return

    df = pd.DataFrame(audit_trail).tail(20)

    # Convert In_Cash bool → readable string, drop if never triggered
    if "In_Cash" in df.columns:
        if df["In_Cash"].any():
            df["In_Cash"] = df["In_Cash"].map({True: "🛡️ CASH", False: ""})
        else:
            df = df.drop(columns=["In_Cash"])

    cols = [c for c in ["Date", "Signal", "Hold", "Net_Return", "In_Cash"]
            if c in df.columns]
    df   = df[cols]

    def _color_ret(val):
        return ("color: #00c896; font-weight:bold" if val > 0
                else "color: #ff4b4b; font-weight:bold")

    styled = (
        df.style
        .map(_color_ret, subset=["Net_Return"])
        .format({"Net_Return": "{:.2%}"})
        .set_properties(**{"font-size": "14px", "text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "13px"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "10px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=500)
