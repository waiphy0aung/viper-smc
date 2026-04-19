"""
Multi-pair scanner — backtest SMC engine on all available pairs.
Ranks by profit factor and R:R to find the best instruments.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

import config
from structure import StructureBias, get_recent_structure, find_swing_points, detect_displacement
from zones import find_order_blocks, find_fair_value_gaps, update_zone_status, ZoneStatus, ZoneType
from liquidity import find_liquidity_pools, detect_sweep, get_session_levels, LiquidityType

logging.basicConfig(level=logging.WARNING)

PAIRS = {
    "XAUUSD":  {"ticker": "GC=F",      "spread": 2.5,  "lot_mult": 100, "min_sl": 3.0},
    "XAGUSD":  {"ticker": "SI=F",      "spread": 0.03, "lot_mult": 5000, "min_sl": 0.05},
    "NAS100":  {"ticker": "NQ=F",      "spread": 1.5,  "lot_mult": 20,  "min_sl": 15.0},
    "SP500":   {"ticker": "ES=F",      "spread": 0.5,  "lot_mult": 50,  "min_sl": 5.0},
    "US30":    {"ticker": "YM=F",      "spread": 2.0,  "lot_mult": 5,   "min_sl": 20.0},
    "EURUSD":  {"ticker": "EURUSD=X",  "spread": 0.00012, "lot_mult": 100000, "min_sl": 0.0005},
    "GBPUSD":  {"ticker": "GBPUSD=X",  "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.0008},
    "GBPJPY":  {"ticker": "GBPJPY=X",  "spread": 0.02, "lot_mult": 1000, "min_sl": 0.10},
    "USDJPY":  {"ticker": "USDJPY=X",  "spread": 0.015, "lot_mult": 1000, "min_sl": 0.08},
    "OIL_WTI": {"ticker": "CL=F",      "spread": 0.05, "lot_mult": 1000, "min_sl": 0.20},
}


def fetch_pair(ticker: str) -> dict[str, pd.DataFrame]:
    result = {}
    d = yf.download(ticker, period="60d", interval="15m", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    result["15m"] = d

    d1h = yf.download(ticker, period="730d", interval="1h", progress=False)
    d1h.columns = [c[0].lower() for c in d1h.columns]
    if d1h.index.tz is None:
        d1h.index = d1h.index.tz_localize("UTC")
    result["1h"] = d1h

    result["4h"] = d1h.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    dd = yf.download(ticker, period="2y", interval="1d", progress=False)
    dd.columns = [c[0].lower() for c in dd.columns]
    if dd.index.tz is None:
        dd.index = dd.index.tz_localize("UTC")
    result["daily"] = dd

    dw = yf.download(ticker, period="5y", interval="1wk", progress=False)
    dw.columns = [c[0].lower() for c in dw.columns]
    if dw.index.tz is None:
        dw.index = dw.index.tz_localize("UTC")
    result["weekly"] = dw

    return result


def backtest_pair(name: str, tf: dict, pair_cfg: dict) -> dict:
    """Quick backtest of SMC engine on a pair. Returns stats."""
    df_15m = tf["15m"]
    df_1h = tf["1h"]
    df_4h = tf["4h"]
    df_daily = tf["daily"]
    df_weekly = tf["weekly"]

    spread = pair_cfg["spread"]
    lot_mult = pair_cfg["lot_mult"]
    min_sl = pair_cfg["min_sl"]
    comm_per_lot = 5.0

    warmup = 200
    if len(df_15m) <= warmup:
        return {"trades": 0}

    tradeable = df_15m.index[warmup:]
    equity = 5000.0
    pos = None
    trades = []
    eq_curve = [equity]

    for i, ts in enumerate(tradeable):
        loc = df_15m.index.get_loc(ts)
        w15 = df_15m.iloc[max(0, loc - 199):loc + 1]
        w1h = df_1h.loc[:ts].iloc[-100:]
        w4h = df_4h.loc[:ts].iloc[-100:]
        wd = df_daily.loc[:ts].iloc[-60:]
        ww = df_weekly.loc[:ts].iloc[-30:]

        if len(w15) < 50 or len(w4h) < 10:
            eq_curve.append(equity)
            continue

        price = float(w15["close"].iloc[-1])

        # Manage position
        if pos is not None:
            bars = i - pos["bar"]
            close_it = False
            reason = ""

            if pos["side"] == "long" and price <= pos["sl"]:
                close_it, reason = True, "SL"
            elif pos["side"] == "short" and price >= pos["sl"]:
                close_it, reason = True, "SL"
            elif pos["side"] == "long" and price >= pos["tp"]:
                close_it, reason = True, "TP"
            elif pos["side"] == "short" and price <= pos["tp"]:
                close_it, reason = True, "TP"
            elif bars >= 60:
                close_it, reason = True, "Time"

            if close_it:
                raw = ((price - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - price)) * pos["lots"] * lot_mult
                c = comm_per_lot * pos["lots"]
                equity += raw - c
                trades.append({"pnl": raw - c, "reason": reason})
                pos = None

        # Entry
        if pos is None:
            # HTF bias
            h4_s = get_recent_structure(w4h["high"], w4h["low"], w4h["close"], 3, 50,
                                        w4h["open"] if "open" in w4h else None)
            d_s = get_recent_structure(wd["high"], wd["low"], wd["close"], 3, 30)

            htf_bull = (h4_s.bias == StructureBias.BULLISH and d_s.bias != StructureBias.BEARISH)
            htf_bear = (h4_s.bias == StructureBias.BEARISH and d_s.bias != StructureBias.BULLISH)

            if not htf_bull and not htf_bear:
                eq_curve.append(equity)
                continue

            # Premium/Discount
            if h4_s.dealing_range:
                from structure import PriceZone
                z = h4_s.dealing_range.zone_of(price)
                if htf_bull and z == PriceZone.PREMIUM:
                    eq_curve.append(equity)
                    continue
                if htf_bear and z == PriceZone.DISCOUNT:
                    eq_curve.append(equity)
                    continue

            # Find zones
            h4_obs = find_order_blocks(w4h["high"], w4h["low"], w4h["open"], w4h["close"], h4_s.breaks, 15)
            h4_obs = update_zone_status(h4_obs, w4h["high"], w4h["low"], w4h["close"])

            h1_s = get_recent_structure(w1h["high"], w1h["low"], w1h["close"], 3, 50,
                                        w1h["open"] if "open" in w1h else None)
            h1_obs = find_order_blocks(w1h["high"], w1h["low"], w1h["open"], w1h["close"], h1_s.breaks, 10)
            h1_obs = update_zone_status(h1_obs, w1h["high"], w1h["low"], w1h["close"])
            h1_fvgs = find_fair_value_gaps(w1h["high"], w1h["low"], w1h["close"])
            h1_fvgs = update_zone_status(h1_fvgs, w1h["high"], w1h["low"], w1h["close"])

            all_zones = h4_obs + h1_obs + h1_fvgs
            sessions = get_session_levels(w1h)

            for zone in all_zones:
                if zone.status == ZoneStatus.BROKEN:
                    continue

                zh = zone.top - zone.bottom
                if zh < min_sl * 0.5:
                    continue

                in_zone = zone.bottom * 0.999 <= price <= zone.top * 1.001

                if not in_zone:
                    continue

                # Check direction match
                if htf_bull and zone.zone_type not in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG):
                    continue
                if htf_bear and zone.zone_type not in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG):
                    continue

                # Rejection check
                rejection = False
                for k in range(-1, -6, -1):
                    if k < -len(w15):
                        break
                    bo = float(w15["open"].iloc[k])
                    bc = float(w15["close"].iloc[k])
                    bh = float(w15["high"].iloc[k])
                    bl = float(w15["low"].iloc[k])
                    tot = bh - bl
                    if tot == 0:
                        continue
                    if htf_bull:
                        lw = min(bo, bc) - bl
                        if lw / tot > 0.3 and bc > bo:
                            rejection = True
                            break
                    elif htf_bear:
                        uw = bh - max(bo, bc)
                        if uw / tot > 0.3 and bc < bo:
                            rejection = True
                            break

                if not rejection:
                    continue

                # Entry
                side = "long" if htf_bull else "short"
                if side == "long":
                    sl = zone.bottom - zh * 0.3
                    sl = min(sl, zone.bottom - min_sl)
                else:
                    sl = zone.top + zh * 0.3
                    sl = max(sl, zone.top + min_sl)

                risk = abs(price - sl)
                if risk < min_sl:
                    continue

                # TP
                if side == "long":
                    tp = sessions.pdh if sessions.pdh > price and abs(sessions.pdh - price) >= risk * 1.5 else price + risk * 3
                else:
                    tp = sessions.pdl if sessions.pdl < price and abs(price - sessions.pdl) >= risk * 1.5 else price - risk * 3

                rr = abs(tp - price) / risk
                if rr < 1.5:
                    continue

                risk_d = min(equity * 0.01, equity * 0.02)
                lots = round(risk_d / (risk * lot_mult), 2)
                lots = max(lots, 0.01)
                lots = min(lots, 0.10)

                fill = price + spread if side == "long" else price - spread
                pos = {"side": side, "entry": fill, "sl": sl, "tp": tp,
                       "lots": lots, "bar": i}
                break

        eq = equity
        if pos:
            eq += ((price - pos["entry"]) if pos["side"] == "long" else
                   (pos["entry"] - price)) * pos["lots"] * lot_mult
        eq_curve.append(eq)

    total = len(trades)
    if total == 0:
        return {"name": name, "trades": 0, "pnl": 0, "wr": 0, "pf": 0, "mdd": 0, "avg_rr": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    aw = gw / wins if wins else 0
    al = gl / (total - wins) if (total - wins) > 0 else 0
    pf = gw / gl if gl > 0 else float("inf")

    peak = eq_curve[0]
    mdd = 0
    for eq in eq_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)

    pnl = sum(t["pnl"] for t in trades)
    avg_rr = (aw / al) if al > 0 else 0

    return {
        "name": name, "trades": total, "wins": wins,
        "pnl": pnl, "wr": wins / total * 100,
        "pf": pf, "mdd": mdd, "avg_rr": avg_rr,
        "aw": aw, "al": al,
    }


def main():
    print(f"\n{'='*75}")
    print(f"  VIPER SMC — Multi-Pair Scanner")
    print(f"{'='*75}\n")

    results = []
    for name, cfg in PAIRS.items():
        print(f"  Scanning {name}...", end=" ", flush=True)
        try:
            tf = fetch_pair(cfg["ticker"])
            r = backtest_pair(name, tf, cfg)
            results.append(r)
            if r["trades"] > 0:
                print(f"{r['trades']}T  {r['wr']:.0f}%WR  PF={r['pf']:.2f}  ${r['pnl']:>8,.2f}  DD={r['mdd']:.1f}%  R:R=1:{r['avg_rr']:.1f}")
            else:
                print("no trades")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": name, "trades": 0, "pnl": 0, "wr": 0, "pf": 0, "mdd": 0})
        time.sleep(1)

    # Rank by profit factor (minimum 2 trades)
    valid = [r for r in results if r["trades"] >= 2]
    valid.sort(key=lambda r: r["pf"], reverse=True)

    print(f"\n{'='*75}")
    print(f"  RANKINGS (by Profit Factor, min 2 trades)")
    print(f"{'='*75}")
    print(f"  {'#':>3} {'Pair':10s} {'Trades':>6} {'WR%':>5} {'PF':>6} {'PnL':>10} {'DD%':>6} {'R:R':>6} {'AvgW':>8} {'AvgL':>8}")
    print(f"  {'-'*69}")

    for i, r in enumerate(valid):
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"  {i+1:>3} {r['name']:10s} {r['trades']:>6} {r['wr']:>5.1f} {pf_str:>6} "
              f"${r['pnl']:>9,.2f} {r['mdd']:>5.1f}% {r['avg_rr']:>5.1f}x "
              f"${r.get('aw',0):>7,.2f} ${r.get('al',0):>7,.2f}")

    no_trades = [r for r in results if r["trades"] < 2]
    if no_trades:
        print(f"\n  No trades: {', '.join(r['name'] for r in no_trades)}")

    if valid:
        best = valid[0]
        print(f"\n  BEST PAIR: {best['name']} — PF {best['pf']:.2f}, {best['wr']:.0f}% WR, "
              f"${best['pnl']:,.2f} PnL, {best['mdd']:.1f}% DD")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
