"""
Backtest chart viewer — interactive HTML charts.

Generates:
1. Equity curve with drawdown overlay
2. Price chart with trade markers (entry/exit, SL/TP lines)
3. Per-instrument breakdown
4. Trade distribution (win/loss, R:R histogram)

Opens in browser. Saves to backtest_report.html.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from structure import StructureBias, PriceZone, get_recent_structure
from zones import find_order_blocks, find_fair_value_gaps, update_zone_status, ZoneStatus, ZoneType
from liquidity import get_session_levels

logging.basicConfig(level=logging.WARNING)

PAIR_CFG = {
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.0005, "comm": 5.0},
    "US30":   {"ticker": "YM=F",     "spread": 2.0,     "lot_mult": 5,      "min_sl": 20.0,   "comm": 3.0},
    "SP500":  {"ticker": "ES=F",     "spread": 0.5,     "lot_mult": 50,     "min_sl": 5.0,    "comm": 3.0},
}


def fetch_pair(ticker: str) -> dict[str, pd.DataFrame]:
    result = {}
    for tf, interval, period in [("15m", "15m", "60d"), ("1h", "1h", "730d"),
                                  ("daily", "1d", "2y"), ("weekly", "1wk", "5y")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        result[tf] = d

    result["4h"] = result["1h"].resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return result


def in_session(symbol: str, hour: int) -> bool:
    windows = config.SESSION_WINDOWS.get(symbol, [])
    return any(s <= hour < e for s, e in windows) if windows else True


def run_backtest_with_trades(symbol: str, cfg: dict, tf: dict):
    """Run backtest and collect detailed trade data for charting."""
    d15 = tf["15m"]
    warmup = 200
    if len(d15) <= warmup:
        return [], [], d15

    tradeable = d15.index[warmup:]
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = []

    for i, ts in enumerate(tradeable):
        loc = d15.index.get_loc(ts)
        w15 = d15.iloc[max(0, loc - 199):loc + 1]
        w1h = tf["1h"].loc[:ts].iloc[-100:]
        w4h = tf["4h"].loc[:ts].iloc[-100:]
        wd = tf["daily"].loc[:ts].iloc[-60:]

        if len(w15) < 50 or len(w4h) < 10:
            eq_curve.append({"ts": ts, "equity": equity})
            continue

        price = float(w15["close"].iloc[-1])

        # Manage
        if pos is not None:
            bars = i - pos["bar"]
            close_it, reason = False, ""

            if pos["side"] == "long" and price <= pos["sl"]: close_it, reason = True, "SL"
            elif pos["side"] == "short" and price >= pos["sl"]: close_it, reason = True, "SL"
            elif pos["side"] == "long" and price >= pos["tp"]: close_it, reason = True, "TP"
            elif pos["side"] == "short" and price <= pos["tp"]: close_it, reason = True, "TP"
            elif bars >= 60: close_it, reason = True, "Time"

            if close_it:
                raw = ((price - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - price)) * pos["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * pos["lots"]
                pnl = raw - c
                equity += pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": ts,
                    "entry_price": pos["entry"], "exit_price": price,
                    "sl": pos["sl"], "tp": pos["tp"],
                    "side": pos["side"], "pnl": pnl, "reason": reason,
                    "bars": bars,
                })
                pos = None

        # Entry
        if pos is None and in_session(symbol, ts.hour):
            h4_s = get_recent_structure(w4h["high"], w4h["low"], w4h["close"], 3, 50,
                                        w4h["open"] if "open" in w4h else None)
            d_s = get_recent_structure(wd["high"], wd["low"], wd["close"], 3, 30)

            htf_bull = h4_s.bias == StructureBias.BULLISH and d_s.bias != StructureBias.BEARISH
            htf_bear = h4_s.bias == StructureBias.BEARISH and d_s.bias != StructureBias.BULLISH

            if htf_bull or htf_bear:
                if h4_s.dealing_range:
                    z = h4_s.dealing_range.zone_of(price)
                    if htf_bull and z == PriceZone.PREMIUM: htf_bull = False
                    if htf_bear and z == PriceZone.DISCOUNT: htf_bear = False

                if htf_bull or htf_bear:
                    h4_obs = find_order_blocks(w4h["high"], w4h["low"], w4h["open"], w4h["close"], h4_s.breaks, 15)
                    h4_obs = update_zone_status(h4_obs, w4h["high"], w4h["low"], w4h["close"])
                    h1_s = get_recent_structure(w1h["high"], w1h["low"], w1h["close"], 3, 50,
                                                w1h["open"] if "open" in w1h else None)
                    h1_obs = find_order_blocks(w1h["high"], w1h["low"], w1h["open"], w1h["close"], h1_s.breaks, 10)
                    h1_obs = update_zone_status(h1_obs, w1h["high"], w1h["low"], w1h["close"])
                    h1_fvgs = find_fair_value_gaps(w1h["high"], w1h["low"], w1h["close"])
                    h1_fvgs = update_zone_status(h1_fvgs, w1h["high"], w1h["low"], w1h["close"])
                    sessions = get_session_levels(w1h)
                    min_sl = cfg["min_sl"]

                    for zone in h4_obs + h1_obs + h1_fvgs:
                        if zone.status == ZoneStatus.BROKEN: continue
                        zh = zone.top - zone.bottom
                        if zh < min_sl * 0.5: continue
                        if not (zone.bottom * 0.999 <= price <= zone.top * 1.001): continue
                        if htf_bull and zone.zone_type not in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG): continue
                        if htf_bear and zone.zone_type not in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG): continue

                        rejection = False
                        for k in range(-1, -6, -1):
                            if k < -len(w15): break
                            bo, bc = float(w15["open"].iloc[k]), float(w15["close"].iloc[k])
                            bh, bl = float(w15["high"].iloc[k]), float(w15["low"].iloc[k])
                            tot = bh - bl
                            if tot == 0: continue
                            if htf_bull and (min(bo, bc) - bl) / tot > 0.3 and bc > bo: rejection = True; break
                            if htf_bear and (bh - max(bo, bc)) / tot > 0.3 and bc < bo: rejection = True; break
                        if not rejection: continue

                        side = "long" if htf_bull else "short"
                        if zh > min_sl * 10: zh = min_sl * 5
                        if side == "long":
                            sl = min(zone.bottom - zh * 0.3, zone.bottom - min_sl)
                        else:
                            sl = max(zone.top + zh * 0.3, zone.top + min_sl)
                        risk = abs(price - sl)
                        if risk < min_sl: continue
                        if side == "long":
                            tp = sessions.pdh if sessions.pdh > price and abs(sessions.pdh - price) >= risk * 1.5 else price + risk * 3
                        else:
                            tp = sessions.pdl if sessions.pdl < price and abs(price - sessions.pdl) >= risk * 1.5 else price - risk * 3
                        if abs(tp - price) / risk < 1.5: continue

                        spread = cfg["spread"]
                        fill = price + spread if side == "long" else price - spread
                        risk_d = min(equity * 0.01, equity * 0.02)
                        lots = max(0.01, min(0.10, round(risk_d / (risk * cfg["lot_mult"]), 2)))

                        pos = {"side": side, "entry": fill, "sl": sl, "tp": tp,
                               "lots": lots, "bar": i, "entry_time": ts}
                        break

        eq_curve.append({"ts": ts, "equity": equity + (
            ((price - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - price)) * pos["lots"] * cfg["lot_mult"]
            if pos else 0
        )})

    return trades, eq_curve, d15


def build_charts(all_trades: dict, all_eq: dict, all_data: dict):
    """Build interactive plotly charts and save to HTML."""

    # 1. EQUITY CURVE
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=["Portfolio Equity Curve", "Price Chart with Trades", "Trade P&L Distribution"],
        vertical_spacing=0.08,
    )

    # Combined equity
    combined_eq = {}
    for sym, eq in all_eq.items():
        for point in eq:
            ts = point["ts"]
            if ts not in combined_eq:
                combined_eq[ts] = config.ACCOUNT_SIZE
            combined_eq[ts] += point["equity"] - config.ACCOUNT_SIZE

    eq_df = pd.DataFrame(sorted(combined_eq.items(), key=lambda x: x[0]), columns=["time", "equity"])

    # Drawdown
    peak = eq_df["equity"].expanding().max()
    dd = (peak - eq_df["equity"]) / peak * 100

    fig.add_trace(go.Scatter(
        x=eq_df["time"], y=eq_df["equity"],
        name="Equity", line=dict(color="#00d4aa", width=2),
        hovertemplate="$%{y:,.2f}<extra>Equity</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq_df["time"], y=config.ACCOUNT_SIZE + config.ACCOUNT_SIZE * config.PROFIT_TARGET_PHASE1 + eq_df["equity"] * 0,
        name=f"Phase 1 Target (${config.ACCOUNT_SIZE * (1 + config.PROFIT_TARGET_PHASE1):,.0f})",
        line=dict(color="gold", width=1, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq_df["time"], y=eq_df["equity"] * 0 + config.EQUITY_FLOOR,
        name=f"DD Floor (${config.EQUITY_FLOOR:,.0f})",
        line=dict(color="red", width=1, dash="dash"),
    ), row=1, col=1)

    # 2. PRICE CHART WITH TRADES — use the instrument with most trades
    best_sym = max(all_trades.keys(), key=lambda s: len(all_trades[s]))
    d15 = all_data[best_sym]
    trades = all_trades[best_sym]

    fig.add_trace(go.Candlestick(
        x=d15.index[-500:],
        open=d15["open"].iloc[-500:],
        high=d15["high"].iloc[-500:],
        low=d15["low"].iloc[-500:],
        close=d15["close"].iloc[-500:],
        name=best_sym,
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff4757",
    ), row=2, col=1)

    # Trade markers
    for t in trades:
        color = "#00d4aa" if t["pnl"] > 0 else "#ff4757"
        marker = "triangle-up" if t["side"] == "long" else "triangle-down"

        # Entry
        fig.add_trace(go.Scatter(
            x=[t["entry_time"]], y=[t["entry_price"]],
            mode="markers", marker=dict(symbol=marker, size=12, color=color, line=dict(width=1, color="white")),
            name=f"{'WIN' if t['pnl'] > 0 else 'LOSS'} ${t['pnl']:.2f}",
            hovertemplate=f"{t['side'].upper()} Entry: {t['entry_price']:.5g}<br>PnL: ${t['pnl']:.2f}<br>{t['reason']}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)

        # Exit
        fig.add_trace(go.Scatter(
            x=[t["exit_time"]], y=[t["exit_price"]],
            mode="markers", marker=dict(symbol="x", size=8, color=color),
            showlegend=False,
            hovertemplate=f"Exit: {t['exit_price']:.5g}<br>{t['reason']}<extra></extra>",
        ), row=2, col=1)

        # SL/TP lines
        fig.add_trace(go.Scatter(
            x=[t["entry_time"], t["exit_time"]], y=[t["sl"], t["sl"]],
            mode="lines", line=dict(color="red", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[t["entry_time"], t["exit_time"]], y=[t["tp"], t["tp"]],
            mode="lines", line=dict(color="lime", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)

    # 3. PNL DISTRIBUTION
    all_pnls = []
    for sym, trades in all_trades.items():
        for t in trades:
            all_pnls.append({"sym": sym, "pnl": t["pnl"], "reason": t["reason"]})

    if all_pnls:
        pnl_df = pd.DataFrame(all_pnls)
        colors = ["#00d4aa" if p > 0 else "#ff4757" for p in pnl_df["pnl"]]

        fig.add_trace(go.Bar(
            x=list(range(len(pnl_df))),
            y=pnl_df["pnl"],
            marker_color=colors,
            name="Trade P&L",
            hovertemplate="%{text}<br>$%{y:.2f}<extra></extra>",
            text=[f"{r['sym']} {r['reason']}" for _, r in pnl_df.iterrows()],
        ), row=3, col=1)

    # Layout
    fig.update_layout(
        title=dict(
            text="VIPER SMC — Backtest Report",
            font=dict(size=24, color="white"),
        ),
        template="plotly_dark",
        height=1200,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text=f"{best_sym} Price", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_xaxes(title_text="Trade #", row=3, col=1)

    # Stats annotation
    total_trades = sum(len(t) for t in all_trades.values())
    total_wins = sum(1 for sym in all_trades for t in all_trades[sym] if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for sym in all_trades for t in all_trades[sym])
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    stats_text = (
        f"Trades: {total_trades} | WR: {wr:.1f}% | "
        f"PnL: ${total_pnl:,.2f} | DD: {dd.max():.1f}%"
    )
    fig.add_annotation(
        text=stats_text, xref="paper", yref="paper",
        x=0.5, y=1.02, showarrow=False,
        font=dict(size=14, color="white"),
    )

    output = "backtest_report.html"
    fig.write_html(output, auto_open=False)
    print(f"\n  Chart saved to {output}")
    print(f"  Open in browser: file://{output}")
    return output


def main():
    print(f"\n{'='*60}")
    print(f"  VIPER SMC — Backtest Chart Generator")
    print(f"{'='*60}\n")

    all_trades = {}
    all_eq = {}
    all_data = {}

    for sym, cfg in PAIR_CFG.items():
        print(f"  Running {sym}...", end=" ", flush=True)
        tf = fetch_pair(cfg["ticker"])
        trades, eq, d15 = run_backtest_with_trades(sym, cfg, tf)
        all_trades[sym] = trades
        all_eq[sym] = eq
        all_data[sym] = d15
        wins = sum(1 for t in trades if t["pnl"] > 0)
        pnl = sum(t["pnl"] for t in trades)
        print(f"{len(trades)}T {wins}W ${pnl:,.2f}")
        time.sleep(1)

    output = build_charts(all_trades, all_eq, all_data)

    # Summary
    total_t = sum(len(t) for t in all_trades.values())
    total_w = sum(1 for s in all_trades for t in all_trades[s] if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for s in all_trades for t in all_trades[s])
    print(f"\n  Total: {total_t} trades | {total_w}W | ${total_pnl:,.2f}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
