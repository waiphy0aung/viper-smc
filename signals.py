"""
Signal generator — converts MTF analysis into actionable Telegram signals.

Dynamic R:R: TP targets are real liquidity levels, not arbitrary multiples.
Dynamic lots: size based on setup quality (confluence count) and SL distance.
Partial TP: first target at PDH/PDL, remainder trails to weekly draw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import config
from mtf import MTFAnalysis, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    direction: str          # "BUY" or "SELL"
    symbol: str
    entry: float
    sl: float
    tp1: float              # partial TP (PDH/PDL)
    tp2: float | None       # full TP (weekly draw)
    lot_size: float
    risk_dollars: float
    rr1: float              # R:R to TP1
    rr2: float | None       # R:R to TP2
    confidence: float
    analysis_summary: str   # human-readable reasoning
    session_levels: str     # PDH/PDL/PWH/PWL for reference

    def __repr__(self):
        return f"Signal({self.direction} {self.symbol} @ {self.entry:.2f} SL={self.sl:.2f} TP1={self.tp1:.2f})"


def _in_session(symbol: str) -> bool:
    hour = datetime.now(timezone.utc).hour
    windows = config.SESSION_WINDOWS.get(symbol, [])
    if not windows:
        return True
    return any(start <= hour < end for start, end in windows)


def _calc_lot_size(entry: float, sl: float, confidence: float,
                   symbol: str, equity: float) -> tuple[float, float]:
    """
    Dynamic lot sizing based on:
    - Account equity and base risk %
    - Confidence (higher confluence = bigger size)
    - Actual SL distance (tighter SL from OB = bigger lots naturally)
    """
    risk_pct = config.MAX_RISK_PER_TRADE * confidence
    risk_dollars = equity * risk_pct

    point_risk = abs(entry - sl)
    if point_risk == 0:
        return 0.0, 0.0

    lot_mult = config.LOT_DOLLAR_PER_POINT.get(symbol, 100)
    lots = risk_dollars / (point_risk * lot_mult)
    lots = round(lots, 2)
    lots = max(lots, 0.01)

    return lots, risk_dollars


def generate_signal(analysis: MTFAnalysis, symbol: str,
                    current_price: float, equity: float) -> TradeSignal | None:
    """
    Convert MTF analysis into a trade signal.

    Requirements for a signal:
    1. Direction is LONG or SHORT (3+ TFs aligned)
    2. Entry zone exists (4H OB/FVG or 1H FVG)
    3. 15m trigger fired (CHoCH or sweep)
    4. Within trading session
    5. R:R >= minimum
    """
    # No direction = no trade
    if analysis.direction == SignalDirection.NONE:
        return None

    # Session filter
    if not _in_session(symbol):
        return None

    # Must have entry zone and SL
    if analysis.entry_zone_price is None or analysis.sl_price is None:
        return None

    # Must have at least TP1
    if analysis.tp1_price is None and analysis.tp2_price is None:
        return None

    # 15m confirmation required — CHoCH or sweep
    if not analysis.m15_choch and not analysis.m15_sweep:
        return None

    entry = analysis.entry_zone_price
    sl = analysis.sl_price
    risk = abs(entry - sl)

    if risk == 0:
        return None

    # TP1: nearest target (PDH/PDL)
    # TP2: weekly draw on liquidity
    direction = "BUY" if analysis.direction == SignalDirection.LONG else "SELL"

    if analysis.direction == SignalDirection.LONG:
        tp1 = analysis.tp1_price or (entry + risk * 2)
        tp2 = analysis.tp2_price
        if tp1 <= entry:
            tp1 = entry + risk * 2

    else:
        tp1 = analysis.tp1_price or (entry - risk * 2)
        tp2 = analysis.tp2_price
        if tp1 >= entry:
            tp1 = entry - risk * 2

    # R:R calculation
    rr1 = abs(tp1 - entry) / risk
    rr2 = abs(tp2 - entry) / risk if tp2 else None

    if rr1 < config.MIN_RISK_REWARD:
        return None

    # Dynamic lot sizing
    lots, risk_dollars = _calc_lot_size(entry, sl, analysis.confidence, symbol, equity)
    if lots < 0.01:
        return None

    # Build analysis summary
    summary_parts = []
    summary_parts.append(f"W={analysis.weekly_bias.value}")
    summary_parts.append(f"D={analysis.daily_bias.value}")
    summary_parts.append(f"4H={analysis.h4_bias.value}")

    if analysis.h4_ob:
        summary_parts.append(f"4H OB: {analysis.h4_ob.bottom:.2f}-{analysis.h4_ob.top:.2f}")
    if analysis.h1_entry_zone:
        summary_parts.append(f"1H zone: {analysis.h1_entry_zone.bottom:.2f}-{analysis.h1_entry_zone.top:.2f}")
    if analysis.m15_choch:
        summary_parts.append(f"15m CHoCH {analysis.m15_choch.direction}")
    if analysis.m15_sweep:
        summary_parts.append("15m sweep confirmed")

    session_str = (f"PDH={analysis.pdh:.2f} PDL={analysis.pdl:.2f} "
                   f"PWH={analysis.weekly_draw_up or 0:.2f} PWL={analysis.weekly_draw_down or 0:.2f}")

    return TradeSignal(
        direction=direction,
        symbol=symbol,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        lot_size=lots,
        risk_dollars=risk_dollars,
        rr1=rr1,
        rr2=rr2,
        confidence=analysis.confidence,
        analysis_summary=" | ".join(summary_parts),
        session_levels=session_str,
    )
