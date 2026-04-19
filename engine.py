"""
SMC Engine v3 — staged approach.

Stage 1: Mark POIs (Points of Interest) from higher TFs
         4H OBs, 1H FVGs, session levels → these are the ONLY zones we trade

Stage 2: Wait for price to enter a POI zone

Stage 3: AT the zone, look for 15m confirmation:
         - Sweep of local 15m liquidity (stop hunt within the zone)
         - Displacement candle (institutional rejection)

Stage 4: If confirmed → generate signal with:
         - Entry at the displacement FVG or zone midpoint
         - SL behind the zone (not the tiny 15m FVG — the whole 4H/1H zone)
         - TP at the next liquidity target

The key insight: SL is behind the ZONE, not behind the 15m FVG.
This gives wider SL = smaller lots = controlled risk.
But we only enter at confirmed zones = high WR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from structure import (
    StructureBias, StructureState, SwingPoint,
    get_recent_structure, find_swing_points, detect_displacement,
)
from zones import (
    Zone, ZoneType, ZoneStatus,
    find_order_blocks, find_fair_value_gaps, update_zone_status,
)
from liquidity import (
    SessionLevels, get_session_levels,
    find_liquidity_pools, detect_sweep, find_draw_on_liquidity,
    LiquidityType,
)


class SetupQuality(Enum):
    A_PLUS = "A+"    # all conditions met + displacement
    A = "A"          # zone + sweep confirmed
    B = "B"          # zone reached, partial confirmation


@dataclass
class POI:
    """Point of Interest — a zone where we expect a reaction."""
    zone: Zone
    direction: str          # "long" (expect bounce up) or "short" (expect rejection down)
    timeframe: str          # "4h" or "1h"
    sl_price: float         # behind the full zone
    target: float           # next liquidity target

    def __repr__(self):
        return f"POI({self.direction} {self.timeframe} {self.zone.bottom:.2f}-{self.zone.top:.2f})"


@dataclass
class TradeSetup:
    """A confirmed trade setup at a POI."""
    direction: str
    entry: float
    sl: float
    tp: float
    quality: SetupQuality
    risk: float
    reward: float
    rr: float
    poi: POI
    sweep_confirmed: bool
    displacement_confirmed: bool
    reason: str

    def __repr__(self):
        return (f"Setup({self.quality.value} {self.direction} "
                f"entry={self.entry:.2f} SL={self.sl:.2f} TP={self.tp:.2f} R:R=1:{self.rr:.1f})")


def identify_pois(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_daily: pd.DataFrame,
    current_price: float,
    h4_bias: StructureBias,
) -> list[POI]:
    """
    Stage 1: Identify Points of Interest from higher timeframes.

    For bullish bias: find bullish OBs and FVGs BELOW current price (pullback zones)
    For bearish bias: find bearish OBs and FVGs ABOVE current price (rally zones)
    """
    pois = []

    # 4H structure and zones
    h4_struct = get_recent_structure(
        df_4h["high"], df_4h["low"], df_4h["close"], 3, 50,
        df_4h["open"] if "open" in df_4h else None,
    )
    h4_obs = find_order_blocks(
        df_4h["high"], df_4h["low"], df_4h["open"], df_4h["close"],
        h4_struct.breaks, lookback_bars=15,
    )
    h4_obs = update_zone_status(h4_obs, df_4h["high"], df_4h["low"], df_4h["close"])

    # 1H FVGs
    h1_fvgs = find_fair_value_gaps(df_1h["high"], df_1h["low"], df_1h["close"])
    h1_fvgs = update_zone_status(h1_fvgs, df_1h["high"], df_1h["low"], df_1h["close"])

    # Session levels for targets
    sessions = get_session_levels(df_1h)

    # Liquidity targets
    h4_swh, h4_swl = h4_struct.swing_highs, h4_struct.swing_lows
    pools = find_liquidity_pools(df_4h["high"], df_4h["low"], h4_swh, h4_swl)

    # Minimum zone height (filter out micro-FVGs that are noise)
    min_zone_height = 3.0  # points — filter out zones < 3 pts
    max_zone_height = 50.0  # filter out massive OBs — use 50% of zone instead

    if h4_bias == StructureBias.BULLISH:
        for zone in h4_obs + h1_fvgs:
            if zone.status == ZoneStatus.BROKEN:
                continue
            if zone.zone_type not in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG):
                continue
            if zone.top > current_price * 1.001:
                continue

            zone_height = zone.top - zone.bottom
            if zone_height < min_zone_height:
                continue

            # For wide OBs, use the bottom half (optimal trade entry)
            if zone_height > max_zone_height:
                zone_height = max_zone_height / 2
                effective_bottom = zone.top - zone_height
            else:
                effective_bottom = zone.bottom

            sl = effective_bottom - zone_height * 0.3
            sl = min(sl, effective_bottom - 5)  # minimum 5 pts below zone

            # TP: use nearest realistic target
            risk = abs(zone.midpoint - sl)
            tp_candidates = []

            # PDH if it gives 2:1+
            if sessions.pdh > current_price and abs(sessions.pdh - current_price) >= risk * 2:
                tp_candidates.append(sessions.pdh)

            # Next BSL
            bsl = [p.level for p in pools if p.liq_type == LiquidityType.BSL and p.level > current_price]
            if bsl:
                for b in sorted(bsl):
                    if abs(b - current_price) >= risk * 1.5:
                        tp_candidates.append(b)
                        break

            # Fallback: 3x risk from entry zone midpoint
            if not tp_candidates:
                tp_candidates.append(zone.midpoint + risk * 3)

            tp = min(tp_candidates)  # take the nearest valid target

            tf_label = "4h" if zone in h4_obs else "1h"
            pois.append(POI(zone=zone, direction="long", timeframe=tf_label, sl_price=sl, target=tp))

    elif h4_bias == StructureBias.BEARISH:
        for zone in h4_obs + h1_fvgs:
            if zone.status == ZoneStatus.BROKEN:
                continue
            if zone.zone_type not in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG):
                continue
            if zone.bottom < current_price * 0.999:
                continue

            zone_height = zone.top - zone.bottom
            if zone_height < min_zone_height:
                continue

            if zone_height > max_zone_height:
                zone_height = max_zone_height / 2
                effective_top = zone.bottom + zone_height
            else:
                effective_top = zone.top

            sl = effective_top + zone_height * 0.3
            sl = max(sl, effective_top + 5)

            risk = abs(sl - zone.midpoint)
            tp_candidates = []

            if sessions.pdl < current_price and abs(current_price - sessions.pdl) >= risk * 2:
                tp_candidates.append(sessions.pdl)

            ssl = [p.level for p in pools if p.liq_type == LiquidityType.SSL and p.level < current_price]
            if ssl:
                for s_level in sorted(ssl, reverse=True):
                    if abs(current_price - s_level) >= risk * 1.5:
                        tp_candidates.append(s_level)
                        break

            if not tp_candidates:
                tp_candidates.append(zone.midpoint - risk * 3)

            tp = max(tp_candidates)

            tf_label = "4h" if zone in h4_obs else "1h"
            pois.append(POI(zone=zone, direction="short", timeframe=tf_label, sl_price=sl, target=tp))

    return pois


def check_confirmation(
    df_15m: pd.DataFrame,
    poi: POI,
    current_price: float,
) -> TradeSetup | None:
    """
    Stage 2+3: Check if price is at the POI and look for confirmation.

    Confirmation = 15m sweep of local liquidity + displacement candle.
    """
    zone = poi.zone

    # Is price in the zone?
    if not (zone.bottom * 0.999 <= current_price <= zone.top * 1.001):
        return None

    # Find 15m swing points local to the zone
    recent_15m = df_15m.iloc[-20:]
    if len(recent_15m) < 10:
        return None

    h = recent_15m["high"]
    l = recent_15m["low"]
    o = recent_15m["open"]
    c = recent_15m["close"]

    # Check for 15m confirmation (optional — adds quality but doesn't block)
    sh, sl_pts = find_swing_points(h, l, 2)
    pools = find_liquidity_pools(h, l, sh, sl_pts)
    sweep_confirmed = any(detect_sweep(p, h, l, c, 5) for p in pools)

    disp = detect_displacement(o, h, l, c, lookback=5, min_ratio=0.45)
    displacement_confirmed = disp is not None
    if poi.direction == "long" and disp and disp.direction != "bullish":
        displacement_confirmed = False
    elif poi.direction == "short" and disp and disp.direction != "bearish":
        displacement_confirmed = False

    # Quality grading — enter at the zone regardless, quality affects lot size
    if sweep_confirmed and displacement_confirmed:
        quality = SetupQuality.A_PLUS
    elif sweep_confirmed or displacement_confirmed:
        quality = SetupQuality.A
    else:
        quality = SetupQuality.B  # zone touch only — still valid, smaller size

    # Entry, SL, TP
    entry = current_price
    sl = poi.sl_price
    tp = poi.target

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0

    if risk < 3:  # minimum 3 points SL for gold
        return None

    if rr < 1.5:
        return None

    reasons = []
    reasons.append(f"{poi.timeframe} {'OB' if 'OB' in zone.zone_type.value else 'FVG'}")
    reasons.append(f"zone {zone.bottom:.2f}-{zone.top:.2f}")
    if sweep_confirmed:
        reasons.append("15m sweep")
    if displacement_confirmed:
        reasons.append(f"15m disp {disp.direction}")

    return TradeSetup(
        direction=poi.direction,
        entry=entry,
        sl=sl,
        tp=tp,
        quality=quality,
        risk=risk,
        reward=reward,
        rr=rr,
        poi=poi,
        sweep_confirmed=sweep_confirmed,
        displacement_confirmed=displacement_confirmed,
        reason=" | ".join(reasons),
    )


def scan_for_setup(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    current_price: float,
    symbol: str = "XAUUSD",
) -> TradeSetup | None:
    """
    Full scan: identify POIs → check if price is at any → confirm entry.

    Returns the best available setup, or None.
    """
    import config

    # HTF bias
    h4_struct = get_recent_structure(
        df_4h["high"], df_4h["low"], df_4h["close"], 3, 50,
        df_4h["open"] if "open" in df_4h else None,
    )
    d_struct = get_recent_structure(df_daily["high"], df_daily["low"], df_daily["close"], 3, 30)
    w_struct = get_recent_structure(df_weekly["high"], df_weekly["low"], df_weekly["close"], 2, 20)

    # Alignment check
    strict = config.STRICT_ALIGNMENT.get(symbol, False)
    if strict:
        htf_bull = (h4_struct.bias == d_struct.bias == w_struct.bias == StructureBias.BULLISH)
        htf_bear = (h4_struct.bias == d_struct.bias == w_struct.bias == StructureBias.BEARISH)
    else:
        htf_bull = (h4_struct.bias == StructureBias.BULLISH and
                    (d_struct.bias == StructureBias.BULLISH or w_struct.bias == StructureBias.BULLISH))
        htf_bear = (h4_struct.bias == StructureBias.BEARISH and
                    (d_struct.bias == StructureBias.BEARISH or w_struct.bias == StructureBias.BEARISH))

    if not htf_bull and not htf_bear:
        return None

    bias = h4_struct.bias

    # Premium/Discount
    if config.PREMIUM_DISCOUNT_FILTER and h4_struct.dealing_range:
        from structure import PriceZone
        zone = h4_struct.dealing_range.zone_of(current_price)
        if htf_bull and zone == PriceZone.PREMIUM:
            return None
        if htf_bear and zone == PriceZone.DISCOUNT:
            return None

    # Stage 1: Identify POIs
    pois = identify_pois(df_4h, df_1h, df_daily, current_price, bias)

    if not pois:
        return None

    # Stage 2+3: Check each POI for confirmation
    best_setup = None
    for poi in pois:
        setup = check_confirmation(df_15m, poi, current_price)
        if setup:
            if best_setup is None or setup.quality.value < best_setup.quality.value:
                best_setup = setup
            elif setup.quality == best_setup.quality and setup.rr > best_setup.rr:
                best_setup = setup

    return best_setup
