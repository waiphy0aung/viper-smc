"""
Multi-Timeframe Analyzer — top-down institutional analysis.

Weekly  → Draw on liquidity (PWH/PWL), weekly structure
Daily   → Daily bias, PDH/PDL, daily OBs
4H      → Current structure, key OBs and FVGs
1H      → Refined entry zones, confirms 4H
15m     → Execution trigger: sweep + CHoCH = signal

Each timeframe narrows the analysis. Only trade when all align.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from structure import (
    StructureBias, StructureState, StructureBreak, BreakType,
    SwingPoint, detect_structure, get_recent_structure,
)
from liquidity import (
    LiquidityPool, LiquidityType, SweepStatus, SessionLevels,
    find_liquidity_pools, detect_sweep, get_session_levels,
    find_draw_on_liquidity,
)
from zones import (
    Zone, ZoneType, ZoneStatus,
    find_order_blocks, find_fair_value_gaps, update_zone_status,
    find_entry_zone,
)

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class MTFAnalysis:
    """Complete top-down analysis result."""
    # Weekly
    weekly_bias: StructureBias
    weekly_draw_up: float | None      # PWH or weekly BSL
    weekly_draw_down: float | None    # PWL or weekly SSL

    # Daily
    daily_bias: StructureBias
    pdh: float
    pdl: float
    daily_ob: Zone | None

    # 4H
    h4_bias: StructureBias
    h4_structure: StructureState
    h4_ob: Zone | None
    h4_fvg: Zone | None

    # 1H
    h1_entry_zone: Zone | None
    h1_fvgs: list[Zone]

    # 15m execution
    m15_choch: StructureBreak | None  # recent CHoCH on 15m
    m15_sweep: bool                    # liquidity swept on 15m
    m15_structure: StructureState

    # Derived
    direction: SignalDirection
    confidence: float                  # 0-1 based on alignment
    entry_zone_price: float | None
    sl_price: float | None
    tp1_price: float | None           # partial TP (PDH/PDL)
    tp2_price: float | None           # full TP (draw on liquidity)

    def __repr__(self):
        return (
            f"MTF({self.direction.value} | "
            f"W={self.weekly_bias.value} D={self.daily_bias.value} "
            f"4H={self.h4_bias.value} | conf={self.confidence:.1f} | "
            f"entry={self.entry_zone_price} SL={self.sl_price} "
            f"TP1={self.tp1_price} TP2={self.tp2_price})"
        )


def analyze(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    current_price: float,
) -> MTFAnalysis:
    """
    Run full top-down multi-timeframe analysis.

    Returns MTFAnalysis with direction, entry zone, SL, TP levels,
    and confidence score based on how many timeframes align.
    """

    # =========================================================================
    # WEEKLY — where's the draw?
    # =========================================================================
    weekly_structure = get_recent_structure(
        df_weekly["high"], df_weekly["low"], df_weekly["close"],
        strength=2, lookback=20,
    )
    weekly_swh, weekly_swl = weekly_structure.swing_highs, weekly_structure.swing_lows
    weekly_pools = find_liquidity_pools(
        df_weekly["high"], df_weekly["low"], weekly_swh, weekly_swl,
    )
    sessions = get_session_levels(df_1h)  # use 1H for session calc
    weekly_draw_up, weekly_draw_down = find_draw_on_liquidity(
        current_price, sessions, weekly_pools,
    )

    # =========================================================================
    # DAILY — bias for the day
    # =========================================================================
    daily_structure = get_recent_structure(
        df_daily["high"], df_daily["low"], df_daily["close"],
        strength=3, lookback=30,
    )
    daily_obs = find_order_blocks(
        df_daily["high"], df_daily["low"], df_daily["open"], df_daily["close"],
        daily_structure.breaks,
    )
    # Most recent valid daily OB
    daily_ob = None
    for ob in reversed(daily_obs):
        if ob.status != ZoneStatus.BROKEN:
            daily_ob = ob
            break

    # =========================================================================
    # 4H — current swing structure
    # =========================================================================
    h4_structure = get_recent_structure(
        df_4h["high"], df_4h["low"], df_4h["close"],
        strength=3, lookback=50,
    )
    h4_obs = find_order_blocks(
        df_4h["high"], df_4h["low"], df_4h["open"], df_4h["close"],
        h4_structure.breaks,
    )
    h4_fvgs = find_fair_value_gaps(df_4h["high"], df_4h["low"], df_4h["close"])

    # Update zone statuses
    h4_obs = update_zone_status(h4_obs, df_4h["high"], df_4h["low"], df_4h["close"])
    h4_fvgs = update_zone_status(h4_fvgs, df_4h["high"], df_4h["low"], df_4h["close"])

    # Best 4H OB and FVG near price
    h4_ob = None
    h4_fvg = None
    if h4_structure.bias == StructureBias.BULLISH:
        h4_ob = find_entry_zone(h4_obs, current_price, "long")
        h4_fvg = find_entry_zone(h4_fvgs, current_price, "long")
    elif h4_structure.bias == StructureBias.BEARISH:
        h4_ob = find_entry_zone(h4_obs, current_price, "short")
        h4_fvg = find_entry_zone(h4_fvgs, current_price, "short")

    # =========================================================================
    # 1H — refined entry zone
    # =========================================================================
    h1_structure = get_recent_structure(
        df_1h["high"], df_1h["low"], df_1h["close"],
        strength=3, lookback=50,
    )
    h1_fvgs = find_fair_value_gaps(df_1h["high"], df_1h["low"], df_1h["close"])
    h1_fvgs = update_zone_status(h1_fvgs, df_1h["high"], df_1h["low"], df_1h["close"])

    # Find 1H FVG inside the 4H OB — the sniper zone
    h1_entry_zone = None
    ref_zone = h4_ob or h4_fvg
    if ref_zone:
        for fvg in h1_fvgs:
            if fvg.status == ZoneStatus.BROKEN:
                continue
            # 1H FVG overlaps with 4H zone
            if fvg.top >= ref_zone.bottom and fvg.bottom <= ref_zone.top:
                h1_entry_zone = fvg
                break

    if h1_entry_zone is None:
        # Fallback: use the 4H zone directly
        h1_entry_zone = ref_zone

    # =========================================================================
    # 15m — execution trigger
    # =========================================================================
    m15_structure = get_recent_structure(
        df_15m["high"], df_15m["low"], df_15m["close"],
        strength=3, lookback=30,
    )

    # Check for recent CHoCH on 15m (last 10 bars)
    m15_choch = None
    for brk in reversed(m15_structure.breaks):
        if brk.break_type == BreakType.CHOCH:
            if brk.index >= len(df_15m) - 15:  # within recent bars
                m15_choch = brk
            break

    # Check for liquidity sweep on 15m
    m15_swh, m15_swl = m15_structure.swing_highs, m15_structure.swing_lows
    m15_pools = find_liquidity_pools(
        df_15m["high"], df_15m["low"], m15_swh, m15_swl,
    )
    m15_sweep = False
    for pool in m15_pools:
        if detect_sweep(pool, df_15m["high"], df_15m["low"], df_15m["close"], lookback=5):
            m15_sweep = True
            pool.status = SweepStatus.SWEPT
            break

    # =========================================================================
    # DERIVE: direction, confidence, levels
    # =========================================================================
    direction = SignalDirection.NONE
    confidence = 0.0
    entry_zone_price = None
    sl_price = None
    tp1_price = None
    tp2_price = None

    # Count alignment
    bullish_score = 0
    bearish_score = 0

    if weekly_structure.bias == StructureBias.BULLISH: bullish_score += 1
    elif weekly_structure.bias == StructureBias.BEARISH: bearish_score += 1

    if daily_structure.bias == StructureBias.BULLISH: bullish_score += 1
    elif daily_structure.bias == StructureBias.BEARISH: bearish_score += 1

    if h4_structure.bias == StructureBias.BULLISH: bullish_score += 1
    elif h4_structure.bias == StructureBias.BEARISH: bearish_score += 1

    if m15_choch:
        if m15_choch.direction == "bullish": bullish_score += 1
        elif m15_choch.direction == "bearish": bearish_score += 1

    if m15_sweep:
        bullish_score += 0.5
        bearish_score += 0.5  # sweep is directionally ambiguous until CHoCH confirms

    # Determine direction — need at least 3 timeframes aligned
    if bullish_score >= 3 and h4_structure.bias == StructureBias.BULLISH:
        direction = SignalDirection.LONG
        confidence = min(1.0, bullish_score / 4.5)

        if h1_entry_zone:
            entry_zone_price = h1_entry_zone.midpoint
            sl_price = h1_entry_zone.bottom - (h1_entry_zone.top - h1_entry_zone.bottom) * 0.5

        tp1_price = sessions.pdh if sessions.pdh > current_price else None
        tp2_price = weekly_draw_up

    elif bearish_score >= 3 and h4_structure.bias == StructureBias.BEARISH:
        direction = SignalDirection.SHORT
        confidence = min(1.0, bearish_score / 4.5)

        if h1_entry_zone:
            entry_zone_price = h1_entry_zone.midpoint
            sl_price = h1_entry_zone.top + (h1_entry_zone.top - h1_entry_zone.bottom) * 0.5

        tp1_price = sessions.pdl if sessions.pdl < current_price else None
        tp2_price = weekly_draw_down

    return MTFAnalysis(
        weekly_bias=weekly_structure.bias,
        weekly_draw_up=weekly_draw_up,
        weekly_draw_down=weekly_draw_down,
        daily_bias=daily_structure.bias,
        pdh=sessions.pdh,
        pdl=sessions.pdl,
        daily_ob=daily_ob,
        h4_bias=h4_structure.bias,
        h4_structure=h4_structure,
        h4_ob=h4_ob,
        h4_fvg=h4_fvg,
        h1_entry_zone=h1_entry_zone,
        h1_fvgs=[f for f in h1_fvgs if f.status != ZoneStatus.BROKEN],
        m15_choch=m15_choch,
        m15_sweep=m15_sweep,
        m15_structure=m15_structure,
        direction=direction,
        confidence=confidence,
        entry_zone_price=entry_zone_price,
        sl_price=sl_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
    )
