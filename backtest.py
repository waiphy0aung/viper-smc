"""
VIPER SMC Backtester v2 — uses the ICT entry model.

Sweep → Displacement → FVG entry → SL behind sweep → TP at liquidity.
HTF bias from 4H structure + daily/weekly confirmation.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import ccxt
import yfinance as yf

import config
from engine import scan_for_setup, SetupQuality

logging.basicConfig(level=logging.WARNING)

COMMISSION = {"XAUUSD": 7.0, "NAS100": 3.0}
LOT_MULT = config.LOT_DOLLAR_PER_POINT
SPREAD = config.SPREAD_POINTS


def fetch_data(symbol: str) -> dict[str, pd.DataFrame]:
    src = config.SYMBOL_DATA[symbol]
    result = {}
    print(f"  Fetching {symbol}...")

    if src["ccxt_exchange"]:
        ex = ccxt.okx({"enableRateLimit": True}) if symbol == "XAUUSD" else getattr(ccxt, src["ccxt_exchange"])({"enableRateLimit": True})
        ccxt_sym = "XAU/USDT:USDT" if symbol == "XAUUSD" else src["ccxt_symbol"]

        for tf, limit in [("15m", 17000), ("1h", 5000), ("4h", 2000)]:
            candles = []
            since = ex.parse8601((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=180)).isoformat())
            while len(candles) < limit:
                batch = ex.fetch_ohlcv(ccxt_sym, tf, since=since, limit=300)
                if not batch:
                    break
                candles.extend(batch)
                since = batch[-1][0] + 1
                time.sleep(ex.rateLimit / 1000 + 0.1)
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            result[tf] = df
            print(f"    {tf}: {len(df)} bars")

    if src["yf_ticker"]:
        t = src["yf_ticker"]
        for tf, interval, period in [("daily", "1d", "2y"), ("weekly", "1wk", "5y")]:
            d = yf.download(t, period=period, interval=interval, progress=False)
            d.columns = [c[0].lower() for c in d.columns]
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            result[tf] = d
            print(f"    {tf}: {len(d)} bars")

        if "15m" not in result:
            d = yf.download(t, period="60d", interval="15m", progress=False)
            d.columns = [c[0].lower() for c in d.columns]
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            result["15m"] = d
            print(f"    15m: {len(d)} bars")

        if "1h" not in result:
            d = yf.download(t, period="730d", interval="1h", progress=False)
            d.columns = [c[0].lower() for c in d.columns]
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            result["1h"] = d

        if "4h" not in result and "1h" in result:
            result["4h"] = result["1h"].resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

    return result


def in_session(symbol: str, hour: int) -> bool:
    windows = config.SESSION_WINDOWS.get(symbol, [])
    return any(s <= hour < e for s, e in windows) if windows else True


def run_phased(symbol: str):
    tf = fetch_data(symbol)
    df_15m = tf["15m"]
    df_4h = tf["4h"]
    df_daily = tf["daily"]
    df_weekly = tf["weekly"]

    warmup = 200
    tradeable = df_15m.index[warmup:]
    total_bars = len(tradeable)

    lot_mult = LOT_MULT.get(symbol, 100)
    spread = SPREAD.get(symbol, 2.5)
    comm = COMMISSION.get(symbol, 5.0)
    strict = config.STRICT_ALIGNMENT.get(symbol, False)

    phases = [
        {"name": "Phase 1", "target_pct": config.PROFIT_TARGET_PHASE1},
        {"name": "Phase 2", "target_pct": config.PROFIT_TARGET_PHASE2},
        {"name": "Funded",  "target_pct": None},
    ]

    print(f"\n{'='*65}")
    print(f"  VIPER SMC v2 — {symbol}")
    print(f"{'='*65}")
    print(f"  Period: {tradeable[0].date()} to {tradeable[-1].date()} ({total_bars} bars)")

    current_bar = 0

    for phase in phases:
        if current_bar >= total_bars:
            print(f"\n  {phase['name']}: No data remaining.")
            break

        name = phase["name"]
        equity = config.ACCOUNT_SIZE
        target = phase["target_pct"] * equity if phase["target_pct"] else None
        floor = config.EQUITY_FLOOR

        print(f"\n  --- {name} ---")
        print(f"  Account: ${equity:,} | Target: {'${:,.0f}'.format(target) if target else 'None'}")

        pos = None
        trades = []
        eq_curve = [equity]
        daily_start = equity
        current_date = None
        total_comm = 0.0
        phase_start = current_bar
        blown = False

        while current_bar < total_bars:
            ts = tradeable[current_bar]
            hour = ts.hour
            loc = df_15m.index.get_loc(ts)

            w15 = df_15m.iloc[max(0, loc - 199):loc + 1]
            w1h = tf["1h"].loc[:ts].iloc[-100:] if "1h" in tf else pd.DataFrame()
            w4h = df_4h.loc[:ts].iloc[-100:]
            w_daily = df_daily.loc[:ts].iloc[-60:]
            w_weekly = df_weekly.loc[:ts].iloc[-30:]

            if len(w15) < 50:
                eq_curve.append(equity)
                current_bar += 1
                continue

            price = float(w15["close"].iloc[-1])

            day = ts.date()
            if day != current_date:
                current_date = day
                daily_start = equity

            # --- Manage position ---
            if pos is not None:
                bars_held = current_bar - pos["bar"]
                close_it = False
                reason = ""
                exit_price = price  # default to close

                # Use HIGH and LOW for SL/TP checks — wicks count in real trading
                bar_high = float(w15["high"].iloc[-1])
                bar_low = float(w15["low"].iloc[-1])

                # SL — check against wick, exit at SL price (not close)
                if pos["side"] == "long" and bar_low <= pos["sl"]:
                    close_it, reason, exit_price = True, "SL", pos["sl"]
                elif pos["side"] == "short" and bar_high >= pos["sl"]:
                    close_it, reason, exit_price = True, "SL", pos["sl"]

                # TP — check against wick, exit at TP price
                if not close_it:
                    if pos["side"] == "long" and bar_high >= pos["tp"]:
                        close_it, reason, exit_price = True, "TP", pos["tp"]
                    elif pos["side"] == "short" and bar_low <= pos["tp"]:
                        close_it, reason, exit_price = True, "TP", pos["tp"]

                # Check if BOTH SL and TP hit on same bar — SL takes priority (worst case)
                # This happens on big wicks. Assume SL hit first.
                if pos["side"] == "long" and bar_low <= pos["sl"] and bar_high >= pos["tp"]:
                    close_it, reason, exit_price = True, "SL", pos["sl"]
                elif pos["side"] == "short" and bar_high >= pos["sl"] and bar_low <= pos["tp"]:
                    close_it, reason, exit_price = True, "SL", pos["sl"]

                # Time stop — exits at close price
                if not close_it and bars_held >= 60:
                    close_it, reason, exit_price = True, "Time", price

                if close_it:
                    raw = ((exit_price - pos["entry"]) if pos["side"] == "long" else
                           (pos["entry"] - exit_price)) * pos["lots"] * lot_mult
                    c = comm * pos["lots"]
                    net = raw - c
                    equity += net
                    total_comm += c
                    trades.append({"pnl": net, "bars": bars_held, "reason": reason,
                                   "rr": abs(raw) / (pos["risk_d"] + 0.01) if raw > 0 else 0})
                    pos = None

                    if equity <= floor:
                        blown = True
                        break
                    if target and (equity - config.ACCOUNT_SIZE) >= target:
                        d = (current_bar - phase_start) * 15 / 60 / 24
                        print(f"  >>> {name} PASSED in {d:.0f} days | ${equity:,.2f} | {len(trades)} trades")
                        current_bar += 1
                        break

            # --- New entry ---
            if pos is None and not blown:
                if not in_session(symbol, hour):
                    eq_curve.append(equity)
                    current_bar += 1
                    continue

                dd = (daily_start - equity) / daily_start if equity < daily_start else 0
                if dd >= config.DAILY_DD_LIMIT * 0.8:
                    eq_curve.append(equity)
                    current_bar += 1
                    continue

                # DD throttle
                dd_util = (1.0 - equity / config.ACCOUNT_SIZE) / config.MAX_DD_LIMIT if equity < config.ACCOUNT_SIZE else 0
                throttle = max(0.25, 1.0 - dd_util * 0.9)

                # Staged SMC engine: POI → price at zone → confirmation
                setup = scan_for_setup(w_weekly, w_daily, w4h, w1h, w15, price, symbol)

                if setup is None:
                    eq_curve.append(equity)
                    current_bar += 1
                    continue

                # Position sizing from zone-based SL (wider = safer)
                risk_dist = setup.risk
                if risk_dist < 3:
                    eq_curve.append(equity)
                    current_bar += 1
                    continue

                # Confidence from setup quality
                conf_map = {SetupQuality.A_PLUS: 1.0, SetupQuality.A: 0.75, SetupQuality.B: 0.5}
                conf = conf_map.get(setup.quality, 0.5)
                risk_d = equity * config.MAX_RISK_PER_TRADE * conf * throttle
                risk_d = min(risk_d, equity * 0.02)  # hard cap 2%

                lots = round(risk_d / (risk_dist * lot_mult), 2)
                lots = max(lots, 0.01)
                lots = min(lots, {"XAUUSD": 0.10, "NAS100": 0.50}.get(symbol, 0.10))

                fill = price + spread if setup.direction == "long" else price - spread

                pos = {
                    "side": setup.direction, "entry": fill,
                    "sl": setup.sl, "tp": setup.tp,
                    "lots": lots, "bar": current_bar, "risk_d": risk_d,
                }

            eq = equity
            if pos:
                eq += ((price - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - price)) * pos["lots"] * lot_mult
            eq_curve.append(eq)
            current_bar += 1

        # Summary
        pnl = equity - config.ACCOUNT_SIZE
        days = (current_bar - phase_start) * 15 / 60 / 24
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades) * 100 if trades else 0

        peak = eq_curve[0]
        mdd = 0
        for eq in eq_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            mdd = max(mdd, dd)

        status = "BLOWN" if blown else ("PASSED" if target and pnl >= target else "RUNNING" if not target else "NOT YET")

        print(f"\n  {name}: {status}")
        print(f"  PnL: ${pnl:,.2f} ({pnl/config.ACCOUNT_SIZE*100:+.2f}%)")
        print(f"  Trades: {len(trades)} ({wins}W/{len(trades)-wins}L) WR: {wr:.1f}%")
        print(f"  Max DD: {mdd:.2f}% | Comm: ${total_comm:,.2f} | Days: {days:.0f}")

        if trades:
            wp = [t["pnl"] for t in trades if t["pnl"] > 0]
            lp = [t["pnl"] for t in trades if t["pnl"] < 0]
            aw = sum(wp) / len(wp) if wp else 0
            al = sum(lp) / len(lp) if lp else 0
            gw = sum(wp)
            gl = abs(sum(lp))
            pf = gw / gl if gl > 0 else float("inf")
            avg_rr = sum(t.get("rr", 0) for t in trades if t["pnl"] > 0) / max(len(wp), 1)
            print(f"  Avg Win: ${aw:,.2f} | Avg Loss: ${al:,.2f} | PF: {pf:.2f} | Avg R:R: 1:{avg_rr:.1f}")

            sorted_t = sorted(trades, key=lambda t: t["pnl"])
            print(f"  Worst: ${sorted_t[0]['pnl']:,.2f} ({sorted_t[0]['reason']})")
            print(f"  Best:  ${sorted_t[-1]['pnl']:,.2f} ({sorted_t[-1]['reason']})")

        if blown:
            print(f"\n  FAILED.")
            break

    print(f"\n{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()
    run_phased(args.symbol)


if __name__ == "__main__":
    main()
