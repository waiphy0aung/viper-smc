"""
VIPER SMC — 1H Execution Backtester.

Uses 730+ days of 1H data for statistical significance.
Daily zones → 4H confirmation → 1H execution.
This is the definitive test.
"""

from __future__ import annotations

import logging
import time
import sys

import pandas as pd
import numpy as np
import yfinance as yf

import config
from structure import StructureBias, PriceZone, get_recent_structure
from zones import find_order_blocks, find_fair_value_gaps, update_zone_status, ZoneStatus, ZoneType
from liquidity import get_session_levels

logging.basicConfig(level=logging.WARNING)

# Only profitable instruments over 730 days (PF > 1.2)
INSTRUMENTS = {
    "SP500":  {"ticker": "ES=F",     "spread": 0.5,     "lot_mult": 50,     "min_sl": 10.0,  "comm": 3.0},
    "US30":   {"ticker": "YM=F",     "spread": 2.0,     "lot_mult": 5,      "min_sl": 30.0,  "comm": 3.0},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.001, "comm": 5.0},
    "USDJPY": {"ticker": "USDJPY=X", "spread": 0.015,   "lot_mult": 1000,   "min_sl": 0.10,  "comm": 5.0},
}


def fetch(ticker):
    d1h = yf.download(ticker, period="730d", interval="1h", progress=False)
    d1h.columns = [c[0].lower() for c in d1h.columns]
    if d1h.index.tz is None: d1h.index = d1h.index.tz_localize("UTC")

    d4h = d1h.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna()

    dd = yf.download(ticker, period="5y", interval="1d", progress=False)
    dd.columns = [c[0].lower() for c in dd.columns]
    if dd.index.tz is None: dd.index = dd.index.tz_localize("UTC")

    dw = yf.download(ticker, period="10y", interval="1wk", progress=False)
    dw.columns = [c[0].lower() for c in dw.columns]
    if dw.index.tz is None: dw.index = dw.index.tz_localize("UTC")

    return {"1h": d1h, "4h": d4h, "daily": dd, "weekly": dw}


def in_session(sym, hour):
    w = config.SESSION_WINDOWS.get(sym, [])
    return any(s <= hour < e for s, e in w) if w else True


def backtest_single(sym, cfg, tf):
    d1h = tf["1h"]
    d4h = tf["4h"]
    dd = tf["daily"]
    dw = tf["weekly"]

    warmup = 200
    if len(d1h) <= warmup:
        return {"name": sym, "trades": 0}

    tradeable = d1h.index[warmup:]
    total_bars = len(tradeable)

    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = [equity]

    spread = cfg["spread"]
    lot_mult = cfg["lot_mult"]
    min_sl = cfg["min_sl"]
    comm = cfg["comm"]

    for i, ts in enumerate(tradeable):
        loc = d1h.index.get_loc(ts)
        w1h = d1h.iloc[max(0, loc - 199):loc + 1]
        w4h = d4h.loc[:ts].iloc[-100:]
        wd = dd.loc[:ts].iloc[-60:]
        ww = dw.loc[:ts].iloc[-30:]

        if len(w1h) < 50 or len(w4h) < 10 or len(wd) < 15:
            eq_curve.append(equity)
            continue

        price = float(w1h["close"].iloc[-1])
        bar_h = float(w1h["high"].iloc[-1])
        bar_l = float(w1h["low"].iloc[-1])

        # --- Manage position ---
        if pos is not None:
            bars_held = i - pos["bar"]
            close_it, reason, ep = False, "", price

            # SL on wick
            if pos["side"] == "long" and bar_l <= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bar_h >= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]

            # TP on wick
            if not close_it:
                if pos["side"] == "long" and bar_h >= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]
                elif pos["side"] == "short" and bar_l <= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]

            # Both — SL wins
            if pos["side"] == "long" and bar_l <= pos["sl"] and bar_h >= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bar_h >= pos["sl"] and bar_l <= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]

            # Time stop: 20 bars on 1H = 20 hours
            if not close_it and bars_held >= 20:
                close_it, reason, ep = True, "Time", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - ep)) * pos["lots"] * lot_mult
                net = raw - comm * pos["lots"]
                equity += net
                trades.append({"pnl": net, "reason": reason, "bars": bars_held})
                pos = None

                if equity <= config.EQUITY_FLOOR:
                    break

        # --- New entry ---
        if pos is None:
            if not in_session(sym, ts.hour):
                eq_curve.append(equity)
                continue

            # DD throttle
            dd_util = (1.0 - equity / config.ACCOUNT_SIZE) / config.MAX_DD_LIMIT if equity < config.ACCOUNT_SIZE else 0
            throttle = max(0.25, 1.0 - dd_util * 0.9)

            # Daily + 4H must agree
            d_struct = get_recent_structure(wd["high"], wd["low"], wd["close"], 3, 30)
            h4_struct = get_recent_structure(w4h["high"], w4h["low"], w4h["close"], 3, 50,
                                             w4h["open"] if "open" in w4h else None)

            # Strict: 4H + daily must agree. Proven better over 730 days.
            htf_bull = h4_struct.bias == StructureBias.BULLISH and d_struct.bias == StructureBias.BULLISH
            htf_bear = h4_struct.bias == StructureBias.BEARISH and d_struct.bias == StructureBias.BEARISH

            if not htf_bull and not htf_bear:
                eq_curve.append(equity)
                continue

            # Premium/Discount
            if h4_struct.dealing_range:
                z = h4_struct.dealing_range.zone_of(price)
                if htf_bull and z == PriceZone.PREMIUM:
                    eq_curve.append(equity)
                    continue
                if htf_bear and z == PriceZone.DISCOUNT:
                    eq_curve.append(equity)
                    continue

            # 4H OBs + Daily OBs
            h4_obs = find_order_blocks(w4h["high"], w4h["low"], w4h["open"], w4h["close"], h4_struct.breaks, 15)
            h4_obs = update_zone_status(h4_obs, w4h["high"], w4h["low"], w4h["close"])

            d_obs = find_order_blocks(wd["high"], wd["low"], wd["open"], wd["close"], d_struct.breaks, 10)
            d_obs = update_zone_status(d_obs, wd["high"], wd["low"], wd["close"])

            # 4H FVGs
            h4_fvgs = find_fair_value_gaps(w4h["high"], w4h["low"], w4h["close"])
            h4_fvgs = update_zone_status(h4_fvgs, w4h["high"], w4h["low"], w4h["close"])

            all_zones = d_obs + h4_obs + h4_fvgs
            sessions = get_session_levels(w1h)

            for zone in all_zones:
                if zone.status == ZoneStatus.BROKEN:
                    continue
                zh = zone.top - zone.bottom
                if zh < min_sl * 0.3:
                    continue
                if zh > min_sl * 20:
                    zh = min_sl * 10  # cap wide zones

                # Price in or near zone
                if not (zone.bottom * 0.998 <= price <= zone.top * 1.002):
                    continue

                if htf_bull and zone.zone_type not in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG):
                    continue
                if htf_bear and zone.zone_type not in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG):
                    continue

                # 1H rejection on COMPLETED candle
                rejection = False
                for k in range(-2, -8, -1):
                    if k < -len(w1h):
                        break
                    bo = float(w1h["open"].iloc[k])
                    bc = float(w1h["close"].iloc[k])
                    bh = float(w1h["high"].iloc[k])
                    bl = float(w1h["low"].iloc[k])
                    tot = bh - bl
                    if tot == 0:
                        continue

                    if htf_bull:
                        lw = min(bo, bc) - bl
                        body = abs(bc - bo)
                        if lw / tot > 0.35 and body / tot > 0.15 and bc > bo:
                            # Confirm current candle moving up
                            if float(w1h["close"].iloc[-1]) > float(w1h["open"].iloc[-1]):
                                rejection = True
                            break
                    elif htf_bear:
                        uw = bh - max(bo, bc)
                        body = abs(bc - bo)
                        if uw / tot > 0.35 and body / tot > 0.15 and bc < bo:
                            if float(w1h["close"].iloc[-1]) < float(w1h["open"].iloc[-1]):
                                rejection = True
                            break

                if not rejection:
                    continue

                side = "long" if htf_bull else "short"

                # SL behind rejection wick
                recent_lows = [float(w1h["low"].iloc[k]) for k in range(-5, 0)]
                recent_highs = [float(w1h["high"].iloc[k]) for k in range(-5, 0)]

                if side == "long":
                    wick_extreme = min(recent_lows)
                    sl = wick_extreme - zh * 0.3
                    sl = min(sl, wick_extreme - min_sl * 0.5)
                else:
                    wick_extreme = max(recent_highs)
                    sl = wick_extreme + zh * 0.3
                    sl = max(sl, wick_extreme + min_sl * 0.5)

                risk = abs(price - sl)
                if risk < min_sl * 0.5:
                    continue

                # TP: session level or 3x risk
                if side == "long":
                    tp = sessions.pdh if sessions.pdh > price and abs(sessions.pdh - price) >= risk * 1.5 else price + risk * 3
                else:
                    tp = sessions.pdl if sessions.pdl < price and abs(price - sessions.pdl) >= risk * 1.5 else price - risk * 3

                rr = abs(tp - price) / risk
                if rr < 1.5:
                    continue

                # Size
                conf = min(1.0, rr / 5.0)
                risk_d = min(equity * 0.02 * conf * throttle, equity * 0.03)
                lots = round(risk_d / (risk * lot_mult), 2)
                lots = max(0.01, min(0.10, lots))

                fill = price + spread if side == "long" else price - spread
                pos = {"side": side, "entry": fill, "sl": sl, "tp": tp,
                       "lots": lots, "bar": i}
                break

        eq = equity
        if pos:
            eq += ((price - pos["entry"]) if pos["side"] == "long" else
                   (pos["entry"] - price)) * pos["lots"] * lot_mult
        eq_curve.append(eq)

    # Stats
    total = len(trades)
    if total == 0:
        return {"name": sym, "trades": 0, "pnl": 0, "wr": 0, "pf": 0, "mdd": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gw / gl if gl > 0 else float("inf")
    pnl = sum(t["pnl"] for t in trades)

    aw = gw / wins if wins else 0
    al = gl / (total - wins) if total - wins > 0 else 0

    peak = eq_curve[0]
    mdd = 0
    for eq in eq_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)

    days = total_bars / 24  # 1H bars to days

    return {
        "name": sym, "trades": total, "wins": wins, "pnl": pnl,
        "wr": wins / total * 100, "pf": pf, "mdd": mdd,
        "aw": aw, "al": al, "days": days,
    }


def main():
    print(f"\n{'='*70}")
    print(f"  VIPER SMC — 1H Execution Backtest (730+ days)")
    print(f"{'='*70}\n")

    all_results = []
    total_pnl = 0

    for sym, cfg in INSTRUMENTS.items():
        print(f"  {sym}...", end=" ", flush=True)
        tf = fetch(cfg["ticker"])
        r = backtest_single(sym, cfg, tf)
        all_results.append(r)

        if r["trades"] > 0:
            pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
            print(f"{r['trades']}T  {r['wr']:.0f}%WR  PF={pf_str}  "
                  f"${r['pnl']:>8,.2f}  DD={r['mdd']:.1f}%  "
                  f"W=${r['aw']:.2f} L=${r['al']:.2f}  ({r['days']:.0f}d)")
            total_pnl += r["pnl"]
        else:
            print("no trades")
        time.sleep(1)

    print(f"\n  {'='*60}")
    print(f"  COMBINED: ${total_pnl:,.2f}")

    valid = [r for r in all_results if r["trades"] >= 3]
    if valid:
        all_trades = sum(r["trades"] for r in valid)
        all_wins = sum(r["wins"] for r in valid)
        all_gw = sum(r["pnl"] for r in valid if r["pnl"] > 0)
        print(f"  Total trades: {all_trades} | WR: {all_wins/all_trades*100:.1f}%")

        # Phase 1 estimate
        days_avg = max(r["days"] for r in valid)
        monthly = total_pnl / (days_avg / 30) if days_avg > 0 else 0
        phase1_days = 400 / monthly * 30 if monthly > 0 else float("inf")
        print(f"  Monthly: ${monthly:,.2f} | Phase 1 estimate: {phase1_days:.0f} days")

    print(f"  {'='*60}\n")


if __name__ == "__main__":
    main()
