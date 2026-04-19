"""
Backtest replay generator — creates an HTML file with TradingView
Lightweight Charts that replays the backtest bar by bar.

Play/pause, speed control, trade markers, equity curve.
Opens in any browser.
"""

from __future__ import annotations

import json
import logging
import sys
import time

import pandas as pd
import yfinance as yf

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


def fetch(ticker):
    result = {}
    for tf, interval, period in [("15m","15m","60d"),("1h","1h","730d"),("daily","1d","2y"),("weekly","1wk","5y")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None: d.index = d.index.tz_localize("UTC")
        result[tf] = d
    result["4h"] = result["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return result


def in_session(sym, hour):
    w = config.SESSION_WINDOWS.get(sym, [])
    return any(s <= hour < e for s, e in w) if w else True


def run_bt(sym, cfg, tf):
    d15 = tf["15m"]
    warmup = 200
    if len(d15) <= warmup: return [], [], d15

    tradeable = d15.index[warmup:]
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_data = []

    for i, ts in enumerate(tradeable):
        loc = d15.index.get_loc(ts)
        w15 = d15.iloc[max(0,loc-199):loc+1]
        w1h = tf["1h"].loc[:ts].iloc[-100:]
        w4h = tf["4h"].loc[:ts].iloc[-100:]
        wd = tf["daily"].loc[:ts].iloc[-60:]
        if len(w15)<50 or len(w4h)<10:
            eq_data.append({"ts":ts,"eq":equity}); continue
        price = float(w15["close"].iloc[-1])

        if pos:
            bars = i - pos["bar"]
            hit = False; reason = ""
            if pos["side"]=="long" and price<=pos["sl"]: hit,reason=True,"SL"
            elif pos["side"]=="short" and price>=pos["sl"]: hit,reason=True,"SL"
            elif pos["side"]=="long" and price>=pos["tp"]: hit,reason=True,"TP"
            elif pos["side"]=="short" and price<=pos["tp"]: hit,reason=True,"TP"
            elif bars>=60: hit,reason=True,"Time"
            if hit:
                raw=((price-pos["entry"]) if pos["side"]=="long" else (pos["entry"]-price))*pos["lots"]*cfg["lot_mult"]
                pnl=raw-cfg["comm"]*pos["lots"]
                equity+=pnl
                trades.append({"bar_idx":i,"entry_time":str(pos["entry_time"])[:19],"exit_time":str(ts)[:19],
                    "entry_price":pos["entry"],"exit_price":price,"sl":pos["sl"],"tp":pos["tp"],
                    "side":pos["side"],"pnl":round(pnl,2),"reason":reason})
                pos=None

        if pos is None and in_session(sym, ts.hour):
            h4s = get_recent_structure(w4h["high"],w4h["low"],w4h["close"],3,50,w4h["open"] if "open" in w4h else None)
            ds = get_recent_structure(wd["high"],wd["low"],wd["close"],3,30)
            hb = h4s.bias==StructureBias.BULLISH and ds.bias!=StructureBias.BEARISH
            sb = h4s.bias==StructureBias.BEARISH and ds.bias!=StructureBias.BULLISH
            if h4s.dealing_range:
                z=h4s.dealing_range.zone_of(price)
                if hb and z==PriceZone.PREMIUM: hb=False
                if sb and z==PriceZone.DISCOUNT: sb=False
            if hb or sb:
                h4o=find_order_blocks(w4h["high"],w4h["low"],w4h["open"],w4h["close"],h4s.breaks,15)
                h4o=update_zone_status(h4o,w4h["high"],w4h["low"],w4h["close"])
                h1s=get_recent_structure(w1h["high"],w1h["low"],w1h["close"],3,50,w1h["open"] if "open" in w1h else None)
                h1o=find_order_blocks(w1h["high"],w1h["low"],w1h["open"],w1h["close"],h1s.breaks,10)
                h1o=update_zone_status(h1o,w1h["high"],w1h["low"],w1h["close"])
                h1f=find_fair_value_gaps(w1h["high"],w1h["low"],w1h["close"])
                h1f=update_zone_status(h1f,w1h["high"],w1h["low"],w1h["close"])
                sess=get_session_levels(w1h); ms=cfg["min_sl"]
                for zone in h4o+h1o+h1f:
                    if zone.status==ZoneStatus.BROKEN: continue
                    zh=zone.top-zone.bottom
                    if zh<ms*0.5: continue
                    if not(zone.bottom*0.999<=price<=zone.top*1.001): continue
                    if hb and zone.zone_type not in(ZoneType.BULLISH_OB,ZoneType.BULLISH_FVG): continue
                    if sb and zone.zone_type not in(ZoneType.BEARISH_OB,ZoneType.BEARISH_FVG): continue
                    rej=False
                    for k in range(-1,-6,-1):
                        if k<-len(w15): break
                        bo,bc=float(w15["open"].iloc[k]),float(w15["close"].iloc[k])
                        bh,bl=float(w15["high"].iloc[k]),float(w15["low"].iloc[k])
                        tot=bh-bl
                        if tot==0: continue
                        if hb and (min(bo,bc)-bl)/tot>0.3 and bc>bo: rej=True; break
                        if sb and (bh-max(bo,bc))/tot>0.3 and bc<bo: rej=True; break
                    if not rej: continue
                    side="long" if hb else "short"
                    if zh>ms*10: zh=ms*5
                    if side=="long": sl=min(zone.bottom-zh*0.3,zone.bottom-ms)
                    else: sl=max(zone.top+zh*0.3,zone.top+ms)
                    risk=abs(price-sl)
                    if risk<ms: continue
                    if side=="long": tp=sess.pdh if sess.pdh>price and abs(sess.pdh-price)>=risk*1.5 else price+risk*3
                    else: tp=sess.pdl if sess.pdl<price and abs(price-sess.pdl)>=risk*1.5 else price-risk*3
                    if abs(tp-price)/risk<1.5: continue
                    fill=price+cfg["spread"] if side=="long" else price-cfg["spread"]
                    rd=min(equity*0.01,equity*0.02)
                    lots=max(0.01,min(0.10,round(rd/(risk*cfg["lot_mult"]),2)))
                    pos={"side":side,"entry":fill,"sl":sl,"tp":tp,"lots":lots,"bar":i,"entry_time":ts}
                    break

        ur=0
        if pos: ur=((price-pos["entry"]) if pos["side"]=="long" else (pos["entry"]-price))*pos["lots"]*cfg["lot_mult"]
        eq_data.append({"ts":ts,"eq":round(equity+ur,2)})

    return trades, eq_data, d15


def generate_html(sym, ohlcv_data, trades, eq_data):
    """Generate standalone HTML with TradingView Lightweight Charts + replay."""

    candles_json = json.dumps(ohlcv_data)
    trades_json = json.dumps(trades)
    equity_json = json.dumps(eq_data)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VIPER SMC — {sym} Backtest Replay</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; color:#ddd; font-family:'Segoe UI',system-ui,sans-serif; }}
#header {{ padding:12px 20px; display:flex; justify-content:space-between; align-items:center; background:#111; border-bottom:1px solid #222; }}
#header h1 {{ font-size:18px; color:#00d4aa; }}
#stats {{ font-size:14px; color:#888; }}
#stats .win {{ color:#00d4aa; }} #stats .loss {{ color:#ff4757; }}
#controls {{ padding:8px 20px; background:#111; display:flex; gap:12px; align-items:center; border-bottom:1px solid #222; }}
button {{ background:#222; color:#ddd; border:1px solid #333; padding:6px 16px; cursor:pointer; border-radius:4px; font-size:13px; }}
button:hover {{ background:#333; }} button.active {{ background:#00d4aa; color:#000; }}
#speed {{ color:#888; font-size:13px; }}
#chart-container {{ width:100%; height:calc(100vh - 160px); }}
#equity-container {{ width:100%; height:120px; border-top:1px solid #222; }}
#info {{ position:fixed; bottom:130px; left:20px; background:rgba(0,0,0,0.85); padding:10px 16px; border-radius:6px; border:1px solid #333; font-size:13px; display:none; z-index:10; }}
#info .long {{ color:#00d4aa; }} #info .short {{ color:#ff4757; }}
</style>
</head>
<body>

<div id="header">
    <h1>VIPER SMC — {sym} Backtest Replay</h1>
    <div id="stats">
        <span>{len(trades)} trades</span> |
        <span class="win">{wins}W</span> /
        <span class="loss">{len(trades)-wins}L</span> |
        <span class="{'win' if total_pnl > 0 else 'loss'}">${total_pnl:,.2f}</span> |
        <span>WR: {wins/len(trades)*100 if trades else 0:.0f}%</span>
    </div>
</div>

<div id="controls">
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button onclick="setSpeed(1)">1x</button>
    <button onclick="setSpeed(5)" class="active">5x</button>
    <button onclick="setSpeed(20)">20x</button>
    <button onclick="setSpeed(50)">50x</button>
    <button onclick="skipToTrade(-1)">⏮ Prev Trade</button>
    <button onclick="skipToTrade(1)">⏭ Next Trade</button>
    <button onclick="showAll()">Show All</button>
    <span id="speed">Speed: 5x</span>
    <span id="progress" style="margin-left:auto;color:#666"></span>
</div>

<div id="chart-container"></div>
<div id="equity-container"></div>
<div id="info"></div>

<script>
const allCandles = {candles_json};
const allTrades = {trades_json};
const allEquity = {equity_json};

// Main chart
const chartEl = document.getElementById('chart-container');
const chart = LightweightCharts.createChart(chartEl, {{
    width: chartEl.clientWidth,
    height: chartEl.clientHeight,
    layout: {{ background: {{ color: '#0a0a0a' }}, textColor: '#999' }},
    grid: {{ vertLines: {{ color: '#1a1a1a' }}, horzLines: {{ color: '#1a1a1a' }} }},
    crosshair: {{ mode: 0 }},
    rightPriceScale: {{ borderColor: '#222' }},
    timeScale: {{ borderColor: '#222', timeVisible: true, secondsVisible: false }},
}});

const candleSeries = chart.addCandlestickSeries({{
    upColor: '#00d4aa', downColor: '#ff4757',
    borderUpColor: '#00d4aa', borderDownColor: '#ff4757',
    wickUpColor: '#00d4aa', wickDownColor: '#ff4757',
}});

const volumeSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'volume',
}});
chart.priceScale('volume').applyOptions({{ scaleMargins: {{ top: 0.85, bottom: 0 }} }});

// Equity chart
const eqEl = document.getElementById('equity-container');
const eqChart = LightweightCharts.createChart(eqEl, {{
    width: eqEl.clientWidth,
    height: eqEl.clientHeight,
    layout: {{ background: {{ color: '#0a0a0a' }}, textColor: '#999' }},
    grid: {{ vertLines: {{ color: '#1a1a1a' }}, horzLines: {{ color: '#1a1a1a' }} }},
    rightPriceScale: {{ borderColor: '#222' }},
    timeScale: {{ borderColor: '#222', timeVisible: true }},
}});
const eqLine = eqChart.addLineSeries({{ color: '#00d4aa', lineWidth: 2 }});

// State
let currentIdx = 0;
let playing = false;
let speed = 5;
let timer = null;
let currentTradeIdx = -1;

// Build trade lookup by bar timestamp
const tradeEntryMap = {{}};
const tradeExitMap = {{}};
allTrades.forEach((t, i) => {{
    tradeEntryMap[t.entry_time.replace(' ','T')] = i;
    tradeExitMap[t.exit_time.replace(' ','T')] = i;
}});

function addBar(idx) {{
    if (idx >= allCandles.length) return;
    const c = allCandles[idx];
    const t = c.time;

    candleSeries.update({{ time: t, open: c.open, high: c.high, low: c.low, close: c.close }});
    volumeSeries.update({{ time: t, value: c.volume, color: c.close >= c.open ? 'rgba(0,212,170,0.2)' : 'rgba(255,71,87,0.2)' }});

    if (idx < allEquity.length) {{
        eqLine.update({{ time: t, value: allEquity[idx].eq }});
    }}

    // Check for trade markers
    const markers = [];
    const ts = new Date(t * 1000).toISOString().slice(0,19);

    allTrades.forEach(tr => {{
        const entryTs = Math.floor(new Date(tr.entry_time + 'Z').getTime() / 1000);
        const exitTs = Math.floor(new Date(tr.exit_time + 'Z').getTime() / 1000);

        if (entryTs === t) {{
            markers.push({{
                time: t,
                position: tr.side === 'long' ? 'belowBar' : 'aboveBar',
                color: tr.pnl > 0 ? '#00d4aa' : '#ff4757',
                shape: tr.side === 'long' ? 'arrowUp' : 'arrowDown',
                text: (tr.side === 'long' ? '▲' : '▼') + ' $' + tr.pnl.toFixed(2),
            }});
            showTradeInfo(tr, 'ENTRY');
        }}
        if (exitTs === t) {{
            markers.push({{
                time: t,
                position: 'inBar',
                color: tr.pnl > 0 ? '#00d4aa' : '#ff4757',
                shape: 'circle',
                text: '✕ ' + tr.reason,
            }});
        }}
    }});

    if (markers.length > 0) {{
        const existing = candleSeries.markers() || [];
        candleSeries.setMarkers([...existing, ...markers]);
    }}

    document.getElementById('progress').textContent =
        (idx+1) + '/' + allCandles.length + ' bars';
}}

function showTradeInfo(t, label) {{
    const el = document.getElementById('info');
    const cls = t.pnl > 0 ? 'long' : 'short';
    el.innerHTML = `<b>${{label}}</b> <span class="${{t.side}}">${{t.side.toUpperCase()}}</span><br>
        Entry: ${{t.entry_price.toFixed(5)}} → Exit: ${{t.exit_price.toFixed(5)}}<br>
        SL: ${{t.sl.toFixed(5)}} | TP: ${{t.tp.toFixed(5)}}<br>
        <span class="${{cls}}">PnL: $${{t.pnl.toFixed(2)}}</span> | ${{t.reason}}`;
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 5000);
}}

function togglePlay() {{
    playing = !playing;
    document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
    if (playing) startReplay();
    else clearInterval(timer);
}}

function startReplay() {{
    clearInterval(timer);
    timer = setInterval(() => {{
        if (currentIdx >= allCandles.length) {{ togglePlay(); return; }}
        addBar(currentIdx++);
    }}, Math.max(10, 200 / speed));
}}

function setSpeed(s) {{
    speed = s;
    document.getElementById('speed').textContent = 'Speed: ' + s + 'x';
    document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    if (playing) startReplay();
}}

function skipToTrade(dir) {{
    currentTradeIdx += dir;
    if (currentTradeIdx < 0) currentTradeIdx = 0;
    if (currentTradeIdx >= allTrades.length) currentTradeIdx = allTrades.length - 1;

    const t = allTrades[currentTradeIdx];
    const entryTs = Math.floor(new Date(t.entry_time + 'Z').getTime() / 1000);

    // Find bar index for this timestamp
    for (let i = 0; i < allCandles.length; i++) {{
        if (allCandles[i].time >= entryTs) {{
            // Show all bars up to this point
            const barsToShow = allCandles.slice(0, i + 20);
            candleSeries.setData(barsToShow.map(c => ({{ time:c.time, open:c.open, high:c.high, low:c.low, close:c.close }})));
            currentIdx = i + 20;

            // Set markers for all trades up to this point
            const markers = [];
            allTrades.forEach(tr => {{
                const ets = Math.floor(new Date(tr.entry_time + 'Z').getTime() / 1000);
                if (ets <= allCandles[currentIdx-1].time) {{
                    markers.push({{
                        time: ets,
                        position: tr.side === 'long' ? 'belowBar' : 'aboveBar',
                        color: tr.pnl > 0 ? '#00d4aa' : '#ff4757',
                        shape: tr.side === 'long' ? 'arrowUp' : 'arrowDown',
                        text: (tr.side==='long'?'▲':'▼') + ' $' + tr.pnl.toFixed(2),
                    }});
                }}
            }});
            candleSeries.setMarkers(markers);
            showTradeInfo(t, 'TRADE #' + (currentTradeIdx+1));
            chart.timeScale().scrollToPosition(-5, false);
            break;
        }}
    }}
}}

function showAll() {{
    playing = false;
    document.getElementById('playBtn').textContent = '▶ Play';
    clearInterval(timer);

    candleSeries.setData(allCandles.map(c => ({{ time:c.time, open:c.open, high:c.high, low:c.low, close:c.close }})));
    volumeSeries.setData(allCandles.map(c => ({{ time:c.time, value:c.volume, color: c.close>=c.open ? 'rgba(0,212,170,0.2)' : 'rgba(255,71,87,0.2)' }})));
    eqLine.setData(allEquity.map(e => ({{ time: e.time, value: e.eq }})));
    currentIdx = allCandles.length;

    // All trade markers
    const markers = [];
    allTrades.forEach(t => {{
        const ets = Math.floor(new Date(t.entry_time + 'Z').getTime() / 1000);
        markers.push({{
            time: ets,
            position: t.side === 'long' ? 'belowBar' : 'aboveBar',
            color: t.pnl > 0 ? '#00d4aa' : '#ff4757',
            shape: t.side === 'long' ? 'arrowUp' : 'arrowDown',
            text: (t.side==='long'?'▲':'▼') + ' $' + t.pnl.toFixed(2) + ' (' + t.reason + ')',
        }});
    }});
    candleSeries.setMarkers(markers);
    chart.timeScale().fitContent();

    document.getElementById('progress').textContent = allCandles.length + '/' + allCandles.length + ' bars';
}}

// Resize
window.addEventListener('resize', () => {{
    chart.resize(chartEl.clientWidth, chartEl.clientHeight);
    eqChart.resize(eqEl.clientWidth, eqEl.clientHeight);
}});

// Start with all data shown
showAll();
</script>
</body>
</html>"""

    return html


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "GBPUSD"
    if sym not in PAIR_CFG:
        print(f"Available: {', '.join(PAIR_CFG.keys())}"); return

    print(f"\n  VIPER SMC — Generating {sym} replay...\n")

    cfg = PAIR_CFG[sym]
    print(f"  Fetching data...", end=" ", flush=True)
    tf = fetch(cfg["ticker"])
    print("done")

    print(f"  Running backtest...", end=" ", flush=True)
    trades, eq_data, d15 = run_bt(sym, cfg, tf)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    pnl = sum(t["pnl"] for t in trades)
    print(f"{len(trades)}T {wins}W ${pnl:,.2f}")

    # Prepare candle data as JSON-safe list with unix timestamps
    d15_clean = d15.tail(3000).copy()
    d15_clean.index = d15_clean.index.tz_localize(None)

    candles = []
    for ts, row in d15_clean.iterrows():
        candles.append({
            "time": int(ts.timestamp()),
            "open": round(float(row["open"]), 6),
            "high": round(float(row["high"]), 6),
            "low": round(float(row["low"]), 6),
            "close": round(float(row["close"]), 6),
            "volume": round(float(row["volume"]), 2),
        })

    # Equity with matching timestamps
    eq_json = []
    for e in eq_data:
        t = e["ts"]
        if t.tzinfo: t = t.tz_localize(None)
        eq_json.append({"time": int(t.timestamp()), "eq": e["eq"]})

    html = generate_html(sym, candles, trades, eq_json)
    fname = f"replay_{sym.lower()}.html"
    with open(fname, "w") as f:
        f.write(html)

    print(f"\n  Saved: {fname}")
    print(f"  Open in browser: file://{fname}")
    print(f"\n  Controls:")
    print(f"    ▶ Play/Pause — watch candles form in real-time")
    print(f"    1x/5x/20x/50x — replay speed")
    print(f"    ⏮⏭ — jump between trades")
    print(f"    Show All — view complete backtest at once")


if __name__ == "__main__":
    main()
