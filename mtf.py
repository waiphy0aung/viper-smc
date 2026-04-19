"""
Multi-Timeframe Analyzer v2 — deep SMC with all filters.

New in v2:
- Premium/Discount zones — only long in discount, short in premium
- Displacement required — must see institutional momentum
- Tighter killzones
- Smarter TP: skip partial if PDH/PDL too close, use next liquidity instead
- Entry at FVG inside OB (sniper entry) for tightest SL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

import config
from structure import (
    StructureBias, StructureState, BreakType, PriceZone,
    detect_structure, get_recent_structure, detect_displacement,
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
    # Weekly
    weekly_bias: StructureBias
    weekly_draw_up: float | None
    weekly_draw_down: float | None

    # Daily
    daily_bias: StructureBias
    pdh: float
    pdl: float

    # 4H
    h4_bias: StructureBias
    h4_ob: Zone | None
    h4_fvg: Zone | None
    h4_dealing_range_mid: float | None
    h4_price_zone: PriceZone | None

    # 1H
    h1_entry_zone: Zone | None

    # 15m
    m15_choch: bool
    m15_choch_dir: str | None
    m15_sweep: bool
    m15_displacement: bool
    m15_disp_dir: str | None

    # Derived
    direction: SignalDirection
    confidence: float
    entry_price: float | None
    sl_price: float | None
    tp1_price: float | None
    tp2_price: float | None
    tp1_rr: float
    tp2_rr: float | None
    use_partial: bool           # whether to use partial TP at tp1
    reason: str

    def __repr__(self):
        return (
            f"MTF({self.direction.value} conf={self.confidence:.0%} "
            f"zone={self.h4_price_zone.value if self.h4_price_zone else '?'} "
            f"disp={'Y' if self.m15_displacement else 'N'} "
            f"sweep={'Y' if self.m15_sweep else 'N'})"
        )


def analyze(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    current_price: float,
    symbol: str = "XAUUSD",
) -> MTFAnalysis:

    # Default no-signal result
    no_signal = lambda reason: MTFAnalysis(
        weekly_bias=StructureBias.RANGING, weekly_draw_up=None, weekly_draw_down=None,
        daily_bias=StructureBias.RANGING, pdh=0, pdl=0,
        h4_bias=StructureBias.RANGING, h4_ob=None, h4_fvg=None,
        h4_dealing_range_mid=None, h4_price_zone=None,
        h1_entry_zone=None,
        m15_choch=False, m15_choch_dir=None, m15_sweep=False,
        m15_displacement=False, m15_disp_dir=None,
        direction=SignalDirection.NONE, confidence=0, entry_price=None,
        sl_price=None, tp1_price=None, tp2_price=None,
        tp1_rr=0, tp2_rr=None, use_partial=False, reason=reason,
    )

    # =====================================================================
    # WEEKLY
    # =====================================================================
    if len(df_weekly) < 10:
        return no_signal("Insufficient weekly data")

    w_struct = get_recent_structure(
        df_weekly["high"], df_weekly["low"], df_weekly["close"],
        strength=2, lookback=20,
    )
    sessions = get_session_levels(df_1h)
    w_pools = find_liquidity_pools(
        df_weekly["high"], df_weekly["low"],
        w_struct.swing_highs, w_struct.swing_lows,
    )
    w_draw_up, w_draw_down = find_draw_on_liquidity(current_price, sessions, w_pools)

    # =====================================================================
    # DAILY
    # =====================================================================
    if len(df_daily) < 15:
        return no_signal("Insufficient daily data")

    d_struct = get_recent_structure(
        df_daily["high"], df_daily["low"], df_daily["close"],
        strength=3, lookback=30,
        open_=df_daily["open"] if "open" in df_daily else None,
    )

    # =====================================================================
    # 4H — structure + OBs + premium/discount
    # =====================================================================
    if len(df_4h) < 20:
        return no_signal("Insufficient 4H data")

    h4_struct = get_recent_structure(
        df_4h["high"], df_4h["low"], df_4h["close"],
        strength=3, lookback=50,
        open_=df_4h["open"] if "open" in df_4h else None,
    )
    h4_obs = find_order_blocks(
        df_4h["high"], df_4h["low"], df_4h["open"], df_4h["close"],
        h4_struct.breaks,
    )
    h4_fvgs = find_fair_value_gaps(df_4h["high"], df_4h["low"], df_4h["close"])
    h4_obs = update_zone_status(h4_obs, df_4h["high"], df_4h["low"], df_4h["close"])
    h4_fvgs = update_zone_status(h4_fvgs, df_4h["high"], df_4h["low"], df_4h["close"])

    # Premium / Discount
    h4_dr = h4_struct.dealing_range
    h4_price_zone = None
    h4_dr_mid = None
    if h4_dr:
        h4_price_zone = h4_dr.zone_of(current_price)
        h4_dr_mid = h4_dr.midpoint

    # =====================================================================
    # 1H — FVGs inside 4H OB for sniper entry
    # =====================================================================
    if len(df_1h) < 20:
        return no_signal("Insufficient 1H data")

    h1_fvgs = find_fair_value_gaps(df_1h["high"], df_1h["low"], df_1h["close"])
    h1_fvgs = update_zone_status(h1_fvgs, df_1h["high"], df_1h["low"], df_1h["close"])

    # =====================================================================
    # 15m — execution: CHoCH + sweep + displacement
    # =====================================================================
    if len(df_15m) < 30:
        return no_signal("Insufficient 15m data")

    m15_struct = get_recent_structure(
        df_15m["high"], df_15m["low"], df_15m["close"],
        strength=3, lookback=30,
        open_=df_15m["open"] if "open" in df_15m else None,
    )

    # CHoCH in last 10 bars
    m15_choch = False
    m15_choch_dir = None
    for brk in reversed(m15_struct.breaks):
        if brk.break_type == BreakType.CHOCH and brk.index >= len(df_15m) - 15:
            m15_choch = True
            m15_choch_dir = brk.direction
            break

    # Sweep
    m15_pools = find_liquidity_pools(
        df_15m["high"], df_15m["low"],
        m15_struct.swing_highs, m15_struct.swing_lows,
    )
    m15_sweep = any(detect_sweep(p, df_15m["high"], df_15m["low"], df_15m["close"], 5)
                    for p in m15_pools)

    # Displacement
    m15_disp = detect_displacement(
        df_15m["open"], df_15m["high"], df_15m["low"], df_15m["close"],
        lookback=5,
        min_ratio=config.DISPLACEMENT_BODY_RATIO,
    )
    m15_displacement = m15_disp is not None
    m15_disp_dir = m15_disp.direction if m15_disp else None

    # =====================================================================
    # ALIGNMENT + SIGNAL
    # =====================================================================

    # Count bullish/bearish alignment
    bull = 0
    bear = 0
    reasons = []

    if w_struct.bias == StructureBias.BULLISH:
        bull += 1; reasons.append("W=BULL")
    elif w_struct.bias == StructureBias.BEARISH:
        bear += 1; reasons.append("W=BEAR")

    if d_struct.bias == StructureBias.BULLISH:
        bull += 1; reasons.append("D=BULL")
    elif d_struct.bias == StructureBias.BEARISH:
        bear += 1; reasons.append("D=BEAR")

    if h4_struct.bias == StructureBias.BULLISH:
        bull += 1; reasons.append("4H=BULL")
    elif h4_struct.bias == StructureBias.BEARISH:
        bear += 1; reasons.append("4H=BEAR")

    if m15_choch:
        if m15_choch_dir == "bullish":
            bull += 1; reasons.append("15m CHoCH BULL")
        elif m15_choch_dir == "bearish":
            bear += 1; reasons.append("15m CHoCH BEAR")

    if m15_displacement:
        if m15_disp_dir == "bullish":
            bull += 0.5; reasons.append("15m disp BULL")
        elif m15_disp_dir == "bearish":
            bear += 0.5; reasons.append("15m disp BEAR")

    if m15_sweep:
        reasons.append("15m sweep")

    # Requirements:
    # 1. 4H must have clear bias (mandatory)
    # 2. At least ONE higher TF (weekly or daily) agrees
    # 3. 15m trigger: CHoCH or sweep
    # This is more realistic than requiring 3+ TFs — gold's weekly/daily often disagree
    has_trigger = m15_choch or m15_sweep
    if config.REQUIRE_DISPLACEMENT and not m15_displacement:
        has_trigger = False

    # Higher TF confirmation: daily OR weekly agrees with 4H
    htf_bull = d_struct.bias == StructureBias.BULLISH or w_struct.bias == StructureBias.BULLISH
    htf_bear = d_struct.bias == StructureBias.BEARISH or w_struct.bias == StructureBias.BEARISH

    direction = SignalDirection.NONE
    entry_zone = None
    sl = None
    tp1 = None
    tp2 = None
    tp1_rr = 0.0
    tp2_rr = None
    use_partial = False

    # Confidence: more alignment = bigger position
    # 4H aligned + 1 HTF = base. + 15m CHoCH = bonus. + displacement = bonus.
    # Per-instrument alignment check
    strict = config.STRICT_ALIGNMENT.get(symbol, False) if hasattr(config, 'STRICT_ALIGNMENT') else False

    if strict:
        bull_aligned = (bull >= 3 and h4_struct.bias == StructureBias.BULLISH)
        bear_aligned = (bear >= 3 and h4_struct.bias == StructureBias.BEARISH)
    else:
        bull_aligned = (h4_struct.bias == StructureBias.BULLISH and htf_bull)
        bear_aligned = (h4_struct.bias == StructureBias.BEARISH and htf_bear)

    if bull_aligned and has_trigger:
        # Premium/Discount check — only long in discount
        if config.PREMIUM_DISCOUNT_FILTER and h4_price_zone == PriceZone.PREMIUM:
            return no_signal("LONG blocked — price in premium zone")

        direction = SignalDirection.LONG

        # Find entry zone: 1H FVG inside 4H OB (sniper) or 4H OB alone
        h4_ob = find_entry_zone(h4_obs, current_price, "long")
        h4_fvg_zone = find_entry_zone(h4_fvgs, current_price, "long")

        # Look for 1H FVG inside the 4H OB
        entry_zone = None
        ref = h4_ob or h4_fvg_zone
        if ref:
            for fvg in h1_fvgs:
                if fvg.status == ZoneStatus.BROKEN:
                    continue
                if fvg.zone_type in (ZoneType.BULLISH_FVG,) and \
                   fvg.top >= ref.bottom and fvg.bottom <= ref.top:
                    entry_zone = fvg
                    reasons.append(f"1H FVG in 4H OB")
                    break

        if entry_zone is None:
            entry_zone = ref

        if entry_zone:
            sl = entry_zone.bottom - (entry_zone.top - entry_zone.bottom) * 0.3
            risk = abs(current_price - sl)

            # TP1: PDH if far enough, else next BSL
            if sessions.pdh > current_price and abs(sessions.pdh - current_price) / risk >= config.PARTIAL_MIN_RR:
                tp1 = sessions.pdh
                use_partial = True
                reasons.append(f"TP1=PDH {tp1:.2f}")
            else:
                # Skip partial, go straight to weekly draw
                tp1 = w_draw_up or current_price + risk * 3
                use_partial = False
                reasons.append(f"TP=draw {tp1:.2f}")

            tp2 = w_draw_up if w_draw_up and w_draw_up > tp1 else None
            tp1_rr = abs(tp1 - current_price) / risk if risk > 0 else 0
            tp2_rr = abs(tp2 - current_price) / risk if tp2 and risk > 0 else None

    elif bear_aligned and has_trigger:
        if config.PREMIUM_DISCOUNT_FILTER and h4_price_zone == PriceZone.DISCOUNT:
            return no_signal("SHORT blocked — price in discount zone")

        direction = SignalDirection.SHORT

        h4_ob = find_entry_zone(h4_obs, current_price, "short")
        h4_fvg_zone = find_entry_zone(h4_fvgs, current_price, "short")

        entry_zone = None
        ref = h4_ob or h4_fvg_zone
        if ref:
            for fvg in h1_fvgs:
                if fvg.status == ZoneStatus.BROKEN:
                    continue
                if fvg.zone_type in (ZoneType.BEARISH_FVG,) and \
                   fvg.bottom <= ref.top and fvg.top >= ref.bottom:
                    entry_zone = fvg
                    reasons.append(f"1H FVG in 4H OB")
                    break

        if entry_zone is None:
            entry_zone = ref

        if entry_zone:
            sl = entry_zone.top + (entry_zone.top - entry_zone.bottom) * 0.3
            risk = abs(current_price - sl)

            if sessions.pdl < current_price and abs(current_price - sessions.pdl) / risk >= config.PARTIAL_MIN_RR:
                tp1 = sessions.pdl
                use_partial = True
                reasons.append(f"TP1=PDL {tp1:.2f}")
            else:
                tp1 = w_draw_down or current_price - risk * 3
                use_partial = False
                reasons.append(f"TP=draw {tp1:.2f}")

            tp2 = w_draw_down if w_draw_down and w_draw_down < tp1 else None
            tp1_rr = abs(current_price - tp1) / risk if risk > 0 else 0
            tp2_rr = abs(current_price - tp2) / risk if tp2 and risk > 0 else None

    # Min R:R check
    if direction != SignalDirection.NONE and tp1_rr < config.MIN_RISK_REWARD:
        return no_signal(f"R:R too low ({tp1_rr:.1f} < {config.MIN_RISK_REWARD})")

    # No entry zone found
    if direction != SignalDirection.NONE and entry_zone is None:
        return no_signal("No valid entry zone")

    # Confidence based on confluence count
    if direction != SignalDirection.NONE:
        conf_score = max(bull, bear)
        # Bonus for displacement
        if m15_displacement:
            conf_score += 0.5
        # Bonus for both weekly AND daily agreeing
        if (direction == SignalDirection.LONG and
            w_struct.bias == StructureBias.BULLISH and d_struct.bias == StructureBias.BULLISH):
            conf_score += 0.5
        elif (direction == SignalDirection.SHORT and
              w_struct.bias == StructureBias.BEARISH and d_struct.bias == StructureBias.BEARISH):
            conf_score += 0.5
        confidence = min(1.0, conf_score / 4.0)
    else:
        confidence = 0

    return MTFAnalysis(
        weekly_bias=w_struct.bias, weekly_draw_up=w_draw_up, weekly_draw_down=w_draw_down,
        daily_bias=d_struct.bias, pdh=sessions.pdh, pdl=sessions.pdl,
        h4_bias=h4_struct.bias, h4_ob=h4_ob if direction != SignalDirection.NONE else None,
        h4_fvg=h4_fvg_zone if direction != SignalDirection.NONE else None,
        h4_dealing_range_mid=h4_dr_mid, h4_price_zone=h4_price_zone,
        h1_entry_zone=entry_zone,
        m15_choch=m15_choch, m15_choch_dir=m15_choch_dir,
        m15_sweep=m15_sweep, m15_displacement=m15_displacement, m15_disp_dir=m15_disp_dir,
        direction=direction, confidence=confidence,
        entry_price=entry_zone.midpoint if entry_zone else None,
        sl_price=sl, tp1_price=tp1, tp2_price=tp2,
        tp1_rr=tp1_rr, tp2_rr=tp2_rr, use_partial=use_partial,
        reason=" | ".join(reasons),
    )
