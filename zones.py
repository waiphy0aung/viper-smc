"""
Order Blocks and Fair Value Gaps — institutional entry zones.

Order Block (OB):
  The last opposing candle before a break of structure.
  This is where institutions entered — price tends to return here.
  Bullish OB = last bearish candle before a bullish BOS
  Bearish OB = last bullish candle before a bearish BOS

Fair Value Gap (FVG):
  An imbalance between three candles where wicks don't overlap.
  Price tends to fill these gaps. They act as support/resistance.
  Bullish FVG = gap up (candle 1 high < candle 3 low)
  Bearish FVG = gap down (candle 1 low > candle 3 high)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from structure import StructureBreak, BreakType


class ZoneType(Enum):
    BULLISH_OB = "BULL_OB"
    BEARISH_OB = "BEAR_OB"
    BULLISH_FVG = "BULL_FVG"
    BEARISH_FVG = "BEAR_FVG"


class ZoneStatus(Enum):
    FRESH = "FRESH"        # untouched — highest probability
    TESTED = "TESTED"      # price returned once — still valid but weaker
    BROKEN = "BROKEN"      # price closed through it — invalidated


@dataclass
class Zone:
    zone_type: ZoneType
    top: float              # upper boundary
    bottom: float           # lower boundary
    midpoint: float         # (top + bottom) / 2 — ideal entry
    timestamp: pd.Timestamp
    index: int
    status: ZoneStatus = ZoneStatus.FRESH

    def __repr__(self):
        return f"Zone({self.zone_type.value} {self.bottom:.2f}-{self.top:.2f} {self.status.value})"

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_order_blocks(high: pd.Series, low: pd.Series, open_: pd.Series,
                      close: pd.Series, breaks: list[StructureBreak],
                      lookback_bars: int = 10) -> list[Zone]:
    """
    Find order blocks — the last opposing candle before each BOS/CHoCH.

    For a bullish BOS: find the last red (bearish) candle before the break.
    That candle's range is the bullish order block.

    For a bearish BOS: find the last green (bullish) candle before the break.
    """
    obs = []

    for brk in breaks:
        idx = brk.index
        if idx < lookback_bars:
            continue

        if brk.direction == "bullish":
            # Find last bearish candle before this bullish break
            for j in range(idx - 1, max(idx - lookback_bars, 0), -1):
                if close.iloc[j] < open_.iloc[j]:  # bearish candle
                    obs.append(Zone(
                        zone_type=ZoneType.BULLISH_OB,
                        top=float(high.iloc[j]),
                        bottom=float(low.iloc[j]),
                        midpoint=float((high.iloc[j] + low.iloc[j]) / 2),
                        timestamp=high.index[j],
                        index=j,
                    ))
                    break

        elif brk.direction == "bearish":
            # Find last bullish candle before this bearish break
            for j in range(idx - 1, max(idx - lookback_bars, 0), -1):
                if close.iloc[j] > open_.iloc[j]:  # bullish candle
                    obs.append(Zone(
                        zone_type=ZoneType.BEARISH_OB,
                        top=float(high.iloc[j]),
                        bottom=float(low.iloc[j]),
                        midpoint=float((high.iloc[j] + low.iloc[j]) / 2),
                        timestamp=high.index[j],
                        index=j,
                    ))
                    break

    return obs


def find_fair_value_gaps(high: pd.Series, low: pd.Series,
                         close: pd.Series) -> list[Zone]:
    """
    Find Fair Value Gaps (imbalances).

    Bullish FVG: candle[i-1].high < candle[i+1].low → gap up
    Bearish FVG: candle[i-1].low > candle[i+1].high → gap down

    The FVG zone is the space between the non-overlapping wicks.
    """
    fvgs = []

    for i in range(1, len(high) - 1):
        # Bullish FVG: gap between candle i-1 high and candle i+1 low
        if high.iloc[i - 1] < low.iloc[i + 1]:
            fvgs.append(Zone(
                zone_type=ZoneType.BULLISH_FVG,
                top=float(low.iloc[i + 1]),
                bottom=float(high.iloc[i - 1]),
                midpoint=float((low.iloc[i + 1] + high.iloc[i - 1]) / 2),
                timestamp=high.index[i],
                index=i,
            ))

        # Bearish FVG: gap between candle i-1 low and candle i+1 high
        if low.iloc[i - 1] > high.iloc[i + 1]:
            fvgs.append(Zone(
                zone_type=ZoneType.BEARISH_FVG,
                top=float(low.iloc[i - 1]),
                bottom=float(high.iloc[i + 1]),
                midpoint=float((low.iloc[i - 1] + high.iloc[i + 1]) / 2),
                timestamp=high.index[i],
                index=i,
            ))

    return fvgs


def update_zone_status(zones: list[Zone], high: pd.Series, low: pd.Series,
                       close: pd.Series, lookback: int = 5) -> list[Zone]:
    """
    Update zone statuses based on recent price action.
    - FRESH → TESTED if price wicked into the zone but didn't close through
    - FRESH/TESTED → BROKEN if price closed through the zone
    """
    recent_h = high.iloc[-lookback:]
    recent_l = low.iloc[-lookback:]
    recent_c = close.iloc[-lookback:]

    for zone in zones:
        if zone.status == ZoneStatus.BROKEN:
            continue

        for i in range(len(recent_c)):
            h = float(recent_h.iloc[i])
            l = float(recent_l.iloc[i])
            c = float(recent_c.iloc[i])

            if zone.zone_type in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG):
                # Bullish zone — price coming down to it
                if l <= zone.top:  # price reached the zone
                    if c < zone.bottom:
                        zone.status = ZoneStatus.BROKEN
                    elif zone.status == ZoneStatus.FRESH:
                        zone.status = ZoneStatus.TESTED

            elif zone.zone_type in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG):
                # Bearish zone — price coming up to it
                if h >= zone.bottom:  # price reached the zone
                    if c > zone.top:
                        zone.status = ZoneStatus.BROKEN
                    elif zone.status == ZoneStatus.FRESH:
                        zone.status = ZoneStatus.TESTED

    return zones


def find_entry_zone(zones: list[Zone], price: float,
                    direction: str) -> Zone | None:
    """
    Find the best entry zone near current price.

    For longs: look for fresh bullish OB or FVG below price (price pulling back to it)
    For shorts: look for fresh bearish OB or FVG above price
    """
    candidates = []

    for zone in zones:
        if zone.status == ZoneStatus.BROKEN:
            continue

        if direction == "long":
            if zone.zone_type in (ZoneType.BULLISH_OB, ZoneType.BULLISH_FVG):
                # Zone should be at or below current price (pullback zone)
                if zone.top >= price * 0.995:  # within 0.5% — close enough to act on
                    candidates.append(zone)

        elif direction == "short":
            if zone.zone_type in (ZoneType.BEARISH_OB, ZoneType.BEARISH_FVG):
                if zone.bottom <= price * 1.005:
                    candidates.append(zone)

    if not candidates:
        return None

    # Prefer FRESH over TESTED, OB over FVG
    def score(z: Zone) -> tuple:
        freshness = 0 if z.status == ZoneStatus.FRESH else 1
        ob_pref = 0 if "OB" in z.zone_type.value else 1
        return (freshness, ob_pref)

    candidates.sort(key=score)
    return candidates[0]
