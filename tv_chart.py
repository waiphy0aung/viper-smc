"""
TradingView-style backtest chart viewer.

Uses TradingView's Lightweight Charts — same look, same feel.
Shows candlesticks, trades, zones, equity curve.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import numpy as np
import yfinance as yf
from lightweight_charts import Chart

import config
from structure import StructureBias, PriceZone, get_recent_structure
from zones import (
    find_order_blocks, find_fair_value_gaps, update_zone_status,
    ZoneStatus, ZoneType,
)
from liquidity import get_session_levels

logging.basicConfig(level=logging.WARNING)

PAIR_CFG = {
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.0005, "comm": 5.0, "decimals": 5},
    "US30":   {"ticker": "YM=F",     "spread": 2.0,     "lot_mult": 5,      "min_sl": 20.0,   "comm": 3.0, "decimals": 0},
    "SP500":  {"ticker": "ES=F",     "spread": 0.5,     "lot_mult": 50,     "min_sl": 5.0,    "comm": 3.0, "decimals": 2},
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


def run_backtest(symbol: str, cfg: dict, tf: dict):
    d15 = tf["15m"]
    warmup = 200
    if len(d15) <= warmup:
        return [], []

    tradeable = d15.index[warmup:]
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_data = []

    for i, ts in enumerate(tradeable):
        loc = d15.index.get_loc(ts)
        w15 = d15.iloc[max(0, loc - 199):loc + 1]
        w1h = tf["1h"].loc[:ts].iloc[-100:]
        w4h = tf["4h"].loc[:ts].iloc[-100:]
        wd = tf["daily"].loc[:ts].iloc[-60:]

        if len(w15) < 50 or len(w4h) < 10:
            eq_data.append({"time": ts, "value": equity})
            continue

        price = float(w15["close"].iloc[-1])

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
                pnl = raw - cfg["comm"] * pos["lots"]
                equity += pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": ts,
                    "entry_price": pos["entry"], "exit_price": price,
                    "sl": pos["sl"], "tp": pos["tp"],
                    "side": pos["side"], "pnl": pnl, "reason": reason,
                })
                pos = None

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
                    if side == "long": sl = min(zone.bottom - zh * 0.3, zone.bottom - min_sl)
                    else: sl = max(zone.top + zh * 0.3, zone.top + min_sl)
                    risk = abs(price - sl)
                    if risk < min_sl: continue
                    if side == "long":
                        tp = sessions.pdh if sessions.pdh > price and abs(sessions.pdh - price) >= risk * 1.5 else price + risk * 3
                    else:
                        tp = sessions.pdl if sessions.pdl < price and abs(price - sessions.pdl) >= risk * 1.5 else price - risk * 3
                    if abs(tp - price) / risk < 1.5: continue

                    fill = price + cfg["spread"] if side == "long" else price - cfg["spread"]
                    risk_d = min(equity * 0.01, equity * 0.02)
                    lots = max(0.01, min(0.10, round(risk_d / (risk * cfg["lot_mult"]), 2)))
                    pos = {"side": side, "entry": fill, "sl": sl, "tp": tp,
                           "lots": lots, "bar": i, "entry_time": ts}
                    break

        unrealized = 0
        if pos:
            unrealized = ((price - pos["entry"]) if pos["side"] == "long" else
                          (pos["entry"] - price)) * pos["lots"] * cfg["lot_mult"]
        eq_data.append({"time": ts, "value": equity + unrealized})

    return trades, eq_data


def show_chart(symbol: str):
    """Display TradingView-style chart for a single instrument."""
    cfg = PAIR_CFG[symbol]
    print(f"\n  Fetching {symbol}...", end=" ", flush=True)
    tf = fetch_pair(cfg["ticker"])
    print("done")

    print(f"  Running backtest...", end=" ", flush=True)
    trades, eq_data = run_backtest(symbol, cfg, tf)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)
    print(f"{len(trades)}T {wins}W ${total_pnl:,.2f}")

    # Prepare OHLCV — use last 2000 bars, clean timestamps
    d15 = tf["15m"].tail(2000).copy()
    d15.index = d15.index.tz_localize(None)
    ohlcv = pd.DataFrame({
        "time": d15.index,
        "open": d15["open"].values,
        "high": d15["high"].values,
        "low": d15["low"].values,
        "close": d15["close"].values,
        "volume": d15["volume"].values,
    })

    # Create chart
    chart = Chart(
        width=1400,
        height=800,
        inner_width=1.0,
        inner_height=0.7,
    )
    chart.watermark(f"{symbol} • {len(trades)}T • {wins}W • ${total_pnl:,.2f}")
    chart.set(ohlcv)
    chart.volume_config(up_color="rgba(0, 212, 170, 0.3)", down_color="rgba(255, 71, 87, 0.3)")

    # Trade markers
    for t in trades:
        entry_t = pd.Timestamp(t["entry_time"]).tz_localize(None) if t["entry_time"].tzinfo else pd.Timestamp(t["entry_time"])

        color = "#00d4aa" if t["pnl"] > 0 else "#ff4757"
        shape = "arrow_up" if t["side"] == "long" else "arrow_down"
        pos_str = "below" if t["side"] == "long" else "above"
        text = f"${t['pnl']:.2f} ({t['reason']})"

        try:
            chart.marker(time=entry_t, position=pos_str, shape=shape, color=color, text=text)
        except Exception:
            pass

    # SL/TP lines for the most recent trade
    if trades:
        last = trades[-1]
        try:
            chart.horizontal_line(last["sl"], color="#ff4757", width=1, style="dashed", text="SL")
            chart.horizontal_line(last["tp"], color="#00d4aa", width=1, style="dashed", text="TP")
        except Exception:
            pass

    # Equity subchart
    eq_chart = chart.create_subchart(width=1.0, height=0.3, position="bottom")

    eq_df = pd.DataFrame(eq_data)
    eq_df["time"] = pd.to_datetime(eq_df["time"]).dt.tz_localize(None)
    eq_df = eq_df.rename(columns={"value": "Equity"})
    eq_line = eq_chart.create_line(name="Equity", color="#00d4aa", width=2)
    eq_line.set(eq_df)

    # Target and floor lines
    try:
        target_val = config.ACCOUNT_SIZE * (1 + config.PROFIT_TARGET_PHASE1)
        eq_chart.horizontal_line(target_val, color="gold", width=1, style="dashed", text="Target")
        eq_chart.horizontal_line(config.EQUITY_FLOOR, color="red", width=1, style="dashed", text="Floor")
    except Exception:
        pass  # some versions don't support horizontal_line on subcharts

    print(f"\n  Showing chart... (close window to exit)")
    chart.show(block=True)


def main():
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "GBPUSD"

    if symbol not in PAIR_CFG:
        print(f"  Available: {', '.join(PAIR_CFG.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  VIPER SMC — TradingView Chart")
    print(f"{'='*60}")

    show_chart(symbol)


if __name__ == "__main__":
    main()
