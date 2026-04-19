"""
Portfolio phased backtest — GBPUSD + US30 + SP500.
Funding Pips simulation with all 3 instruments trading simultaneously.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

import config
from structure import StructureBias, PriceZone, get_recent_structure, find_swing_points, detect_displacement
from zones import find_order_blocks, find_fair_value_gaps, update_zone_status, ZoneStatus, ZoneType
from liquidity import get_session_levels, find_liquidity_pools, LiquidityType

logging.basicConfig(level=logging.WARNING)

PAIR_CFG = {
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.0005, "comm": 5.0},
    "US30":   {"ticker": "YM=F",     "spread": 2.0,     "lot_mult": 5,      "min_sl": 20.0,   "comm": 3.0},
    "SP500":  {"ticker": "ES=F",     "spread": 0.5,     "lot_mult": 50,     "min_sl": 5.0,    "comm": 3.0},
}


def fetch_all() -> dict[str, dict[str, pd.DataFrame]]:
    data = {}
    for name, cfg in PAIR_CFG.items():
        print(f"  Fetching {name}...", end=" ", flush=True)
        t = cfg["ticker"]
        d15 = yf.download(t, period="60d", interval="15m", progress=False)
        d15.columns = [c[0].lower() for c in d15.columns]
        if d15.index.tz is None: d15.index = d15.index.tz_localize("UTC")

        d1h = yf.download(t, period="730d", interval="1h", progress=False)
        d1h.columns = [c[0].lower() for c in d1h.columns]
        if d1h.index.tz is None: d1h.index = d1h.index.tz_localize("UTC")

        d4h = d1h.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

        dd = yf.download(t, period="2y", interval="1d", progress=False)
        dd.columns = [c[0].lower() for c in dd.columns]
        if dd.index.tz is None: dd.index = dd.index.tz_localize("UTC")

        dw = yf.download(t, period="5y", interval="1wk", progress=False)
        dw.columns = [c[0].lower() for c in dw.columns]
        if dw.index.tz is None: dw.index = dw.index.tz_localize("UTC")

        data[name] = {"15m": d15, "1h": d1h, "4h": d4h, "daily": dd, "weekly": dw}
        print(f"{len(d15)} bars")
        time.sleep(1)
    return data


def in_session(symbol: str, hour: int) -> bool:
    windows = config.SESSION_WINDOWS.get(symbol, [])
    return any(s <= hour < e for s, e in windows) if windows else True


def find_entry(symbol: str, w15, w1h, w4h, wd, ww, price, cfg) -> dict | None:
    """Find a valid SMC entry for a symbol at current bar."""
    min_sl = cfg["min_sl"]
    spread = cfg["spread"]
    lot_mult = cfg["lot_mult"]

    if len(w15) < 50 or len(w4h) < 10:
        return None

    h4_s = get_recent_structure(w4h["high"], w4h["low"], w4h["close"], 3, 50,
                                w4h["open"] if "open" in w4h else None)
    d_s = get_recent_structure(wd["high"], wd["low"], wd["close"], 3, 30)

    htf_bull = h4_s.bias == StructureBias.BULLISH and d_s.bias != StructureBias.BEARISH
    htf_bear = h4_s.bias == StructureBias.BEARISH and d_s.bias != StructureBias.BULLISH

    if not htf_bull and not htf_bear:
        return None

    if h4_s.dealing_range:
        z = h4_s.dealing_range.zone_of(price)
        if htf_bull and z == PriceZone.PREMIUM: return None
        if htf_bear and z == PriceZone.DISCOUNT: return None

    # Zones
    h4_obs = find_order_blocks(w4h["high"], w4h["low"], w4h["open"], w4h["close"], h4_s.breaks, 15)
    h4_obs = update_zone_status(h4_obs, w4h["high"], w4h["low"], w4h["close"])
    h1_s = get_recent_structure(w1h["high"], w1h["low"], w1h["close"], 3, 50,
                                w1h["open"] if "open" in w1h else None)
    h1_obs = find_order_blocks(w1h["high"], w1h["low"], w1h["open"], w1h["close"], h1_s.breaks, 10)
    h1_obs = update_zone_status(h1_obs, w1h["high"], w1h["low"], w1h["close"])
    h1_fvgs = find_fair_value_gaps(w1h["high"], w1h["low"], w1h["close"])
    h1_fvgs = update_zone_status(h1_fvgs, w1h["high"], w1h["low"], w1h["close"])

    sessions = get_session_levels(w1h)

    for zone in h4_obs + h1_obs + h1_fvgs:
        if zone.status == ZoneStatus.BROKEN: continue
        zh = zone.top - zone.bottom
        if zh < min_sl * 0.5: continue

        if not (zone.bottom * 0.999 <= price <= zone.top * 1.001): continue

        if htf_bull and zone.zone_type not in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG): continue
        if htf_bear and zone.zone_type not in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG): continue

        # Rejection check
        rejection = False
        for k in range(-1, -6, -1):
            if k < -len(w15): break
            bo, bc = float(w15["open"].iloc[k]), float(w15["close"].iloc[k])
            bh, bl = float(w15["high"].iloc[k]), float(w15["low"].iloc[k])
            tot = bh - bl
            if tot == 0: continue
            if htf_bull:
                if (min(bo, bc) - bl) / tot > 0.3 and bc > bo: rejection = True; break
            elif htf_bear:
                if (bh - max(bo, bc)) / tot > 0.3 and bc < bo: rejection = True; break

        if not rejection: continue

        side = "long" if htf_bull else "short"
        if zh > min_sl * 10: zh = min_sl * 5  # cap wide zones

        if side == "long":
            sl = zone.bottom - zh * 0.3
            sl = min(sl, zone.bottom - min_sl)
        else:
            sl = zone.top + zh * 0.3
            sl = max(sl, zone.top + min_sl)

        risk = abs(price - sl)
        if risk < min_sl: continue

        if side == "long":
            tp = sessions.pdh if sessions.pdh > price and abs(sessions.pdh - price) >= risk * 1.5 else price + risk * 3
        else:
            tp = sessions.pdl if sessions.pdl < price and abs(price - sessions.pdl) >= risk * 1.5 else price - risk * 3

        rr = abs(tp - price) / risk
        if rr < 1.5: continue

        fill = price + spread if side == "long" else price - spread
        return {"side": side, "entry": fill, "sl": sl, "tp": tp, "risk": risk, "rr": rr}

    return None


def run():
    data = fetch_all()

    # Common date range
    starts = [d["15m"].index[0] for d in data.values()]
    ends = [d["15m"].index[-1] for d in data.values()]
    start = max(starts)
    end = min(ends)

    master = list(data.keys())[0]
    all_idx = data[master]["15m"].loc[start:end].index
    warmup = 200
    tradeable = all_idx[warmup:]
    total = len(tradeable)

    phases = [
        {"name": "Phase 1", "target_pct": config.PROFIT_TARGET_PHASE1},
        {"name": "Phase 2", "target_pct": config.PROFIT_TARGET_PHASE2},
        {"name": "Funded",  "target_pct": None},
    ]

    print(f"\n{'='*70}")
    print(f"  VIPER SMC — Portfolio Phased Backtest")
    print(f"  Instruments: {', '.join(data.keys())}")
    print(f"  Period: {tradeable[0].date()} to {tradeable[-1].date()} ({total} bars)")
    print(f"{'='*70}")

    bar = 0

    for phase in phases:
        if bar >= total:
            print(f"\n  {phase['name']}: No data.")
            break

        name = phase["name"]
        equity = config.ACCOUNT_SIZE
        target_amt = phase["target_pct"] * equity if phase["target_pct"] else None
        floor = config.EQUITY_FLOOR

        print(f"\n  --- {name} ---")
        print(f"  Account: ${equity:,} | Target: {'${:,.0f}'.format(target_amt) if target_amt else 'None'}")

        pos = {}  # symbol -> position
        trades = []
        eq_curve = [equity]
        daily_start = equity
        cur_date = None
        tot_comm = 0.0
        phase_start = bar
        blown = False

        while bar < total:
            ts = tradeable[bar]
            hour = ts.hour
            day = ts.date()
            if day != cur_date:
                cur_date = day
                daily_start = equity

            # --- Manage positions ---
            for sym in list(pos.keys()):
                cfg = PAIR_CFG[sym]
                d15 = data[sym]["15m"]
                if ts not in d15.index: continue
                p = pos[sym]
                price = float(d15.loc[ts, "close"])
                bh = float(d15.loc[ts, "high"])
                bl = float(d15.loc[ts, "low"])
                bars_held = bar - p["bar"]
                close_it, reason, ep = False, "", price

                # SL on wick
                if p["side"] == "long" and bl <= p["sl"]: close_it, reason, ep = True, "SL", p["sl"]
                elif p["side"] == "short" and bh >= p["sl"]: close_it, reason, ep = True, "SL", p["sl"]
                # TP on wick
                if not close_it:
                    if p["side"] == "long" and bh >= p["tp"]: close_it, reason, ep = True, "TP", p["tp"]
                    elif p["side"] == "short" and bl <= p["tp"]: close_it, reason, ep = True, "TP", p["tp"]
                # Both hit — SL wins
                if p["side"] == "long" and bl <= p["sl"] and bh >= p["tp"]: close_it, reason, ep = True, "SL", p["sl"]
                elif p["side"] == "short" and bh >= p["sl"] and bl <= p["tp"]: close_it, reason, ep = True, "SL", p["sl"]
                # Time
                if not close_it and bars_held >= 60: close_it, reason, ep = True, "Time", price

                if close_it:
                    raw = ((ep - p["entry"]) if p["side"] == "long" else (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                    c = cfg["comm"] * p["lots"]
                    equity += raw - c
                    tot_comm += c
                    trades.append({"sym": sym, "pnl": raw - c, "bars": bars_held, "reason": reason})
                    del pos[sym]

                    if equity <= floor:
                        blown = True; break
                    if target_amt and (equity - config.ACCOUNT_SIZE) >= target_amt:
                        d = (bar - phase_start) * 15 / 60 / 24
                        print(f"  >>> {name} PASSED in {d:.0f} days | ${equity:,.2f} | {len(trades)} trades")
                        bar += 1; break

            if blown or (target_amt and (equity - config.ACCOUNT_SIZE) >= target_amt):
                break

            # --- New entries (max 1 position total) ---
            if len(pos) == 0 and not blown:
                dd = (daily_start - equity) / daily_start if equity < daily_start else 0
                if dd < config.DAILY_DD_LIMIT * 0.8:
                    dd_util = (1.0 - equity / config.ACCOUNT_SIZE) / config.MAX_DD_LIMIT if equity < config.ACCOUNT_SIZE else 0
                    throttle = max(0.25, 1.0 - dd_util * 0.9)

                    for sym in config.SYMBOLS:
                        if not in_session(sym, hour): continue
                        cfg = PAIR_CFG[sym]
                        d = data[sym]
                        d15 = d["15m"]
                        if ts not in d15.index: continue

                        loc = d15.index.get_loc(ts)
                        w15 = d15.iloc[max(0, loc - 199):loc + 1]
                        w1h = d["1h"].loc[:ts].iloc[-100:]
                        w4h = d["4h"].loc[:ts].iloc[-100:]
                        wd = d["daily"].loc[:ts].iloc[-60:]
                        ww = d["weekly"].loc[:ts].iloc[-30:]

                        price = float(w15["close"].iloc[-1])
                        entry = find_entry(sym, w15, w1h, w4h, wd, ww, price, cfg)

                        if entry:
                            risk_d = min(equity * config.MAX_RISK_PER_TRADE * min(1.0, entry["rr"] / 5) * throttle,
                                         equity * 0.02)
                            lots = round(risk_d / (entry["risk"] * cfg["lot_mult"]), 2)
                            lots = max(lots, 0.01)
                            lots = min(lots, 0.10)

                            pos[sym] = {
                                "side": entry["side"], "entry": entry["entry"],
                                "sl": entry["sl"], "tp": entry["tp"],
                                "lots": lots, "bar": bar,
                            }
                            break

            eq = equity
            for sym, p in pos.items():
                d15 = data[sym]["15m"]
                if ts in d15.index:
                    px = float(d15.loc[ts, "close"])
                    eq += ((px - p["entry"]) if p["side"] == "long" else (p["entry"] - px)) * p["lots"] * PAIR_CFG[sym]["lot_mult"]
            eq_curve.append(eq)
            bar += 1

        # Summary
        pnl = equity - config.ACCOUNT_SIZE
        days = (bar - phase_start) * 15 / 60 / 24
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades) * 100 if trades else 0

        peak = eq_curve[0]
        mdd = 0
        for eq in eq_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            mdd = max(mdd, dd)

        status = "BLOWN" if blown else ("PASSED" if target_amt and pnl >= target_amt else "RUNNING" if not target_amt else "NOT YET")

        print(f"\n  {name}: {status}")
        print(f"  PnL: ${pnl:,.2f} ({pnl/config.ACCOUNT_SIZE*100:+.2f}%)")
        print(f"  Trades: {len(trades)} ({wins}W/{len(trades)-wins}L) WR: {wr:.1f}%")
        print(f"  Max DD: {mdd:.2f}% | Comm: ${tot_comm:,.2f} | Days: {days:.0f}")

        if trades:
            wp = [t["pnl"] for t in trades if t["pnl"] > 0]
            lp = [t["pnl"] for t in trades if t["pnl"] < 0]
            aw = sum(wp) / len(wp) if wp else 0
            al = sum(lp) / len(lp) if lp else 0
            gw, gl = sum(wp), abs(sum(lp))
            pf = gw / gl if gl > 0 else float("inf")
            print(f"  Avg Win: ${aw:,.2f} | Avg Loss: ${al:,.2f} | PF: {pf:.2f}")

            # Per instrument
            for sym in sorted(set(t["sym"] for t in trades)):
                st = [t for t in trades if t["sym"] == sym]
                sw = sum(1 for t in st if t["pnl"] > 0)
                sp = sum(t["pnl"] for t in st)
                print(f"    {sym:8s} {len(st):3d}T  {sw/len(st)*100 if st else 0:5.1f}%WR  ${sp:>8,.2f}")

        if blown:
            print(f"\n  FAILED.")
            break

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run()
