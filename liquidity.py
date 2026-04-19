"""
Liquidity detection — where the stops are, where price is drawn to.

Detects:
- Buy-side liquidity (BSL): clusters above swing highs (stop losses from shorts)
- Sell-side liquidity (SSL): clusters below swing lows (stop losses from longs)
- Equal highs/lows: double/triple tops/bottoms = obvious liquidity
- Liquidity sweeps: price raids a level then reverses
- Previous session highs/lows: PDH, PDL, PWH, PWL

Price is drawn toward liquidity. It sweeps it, then moves the other way.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from structure import SwingPoint, SwingType, find_swing_points


class LiquidityType(Enum):
    BSL = "BSL"  # buy-side (above highs)
    SSL = "SSL"  # sell-side (below lows)


class SweepStatus(Enum):
    UNTOUCHED = "UNTOUCHED"
    SWEPT = "SWEPT"


@dataclass
class LiquidityPool:
    level: float
    liq_type: LiquidityType
    strength: int          # number of touches / how obvious it is
    timestamp: pd.Timestamp
    status: SweepStatus = SweepStatus.UNTOUCHED

    def __repr__(self):
        return f"Liq({self.liq_type.value} {self.level:.2f} x{self.strength} {self.status.value})"


@dataclass
class SessionLevels:
    """Previous session highs and lows — primary draw on liquidity."""
    pdh: float          # previous day high
    pdl: float          # previous day low
    pwh: float          # previous week high
    pwl: float          # previous week low
    current_day_high: float
    current_day_low: float

    def __repr__(self):
        return (f"Sessions(PDH={self.pdh:.2f} PDL={self.pdl:.2f} "
                f"PWH={self.pwh:.2f} PWL={self.pwl:.2f})")


def find_liquidity_pools(high: pd.Series, low: pd.Series,
                         swing_highs: list[SwingPoint],
                         swing_lows: list[SwingPoint],
                         tolerance_pct: float = 0.001) -> list[LiquidityPool]:
    """
    Find clusters of liquidity above swing highs (BSL) and below swing lows (SSL).

    Equal highs = strong BSL (shorts have stops above).
    Equal lows = strong SSL (longs have stops below).
    More touches = more obvious = more liquidity = stronger draw.
    """
    pools = []

    # Group nearby swing highs (equal highs detection)
    if swing_highs:
        sorted_sh = sorted(swing_highs, key=lambda s: s.price)
        clusters = [[sorted_sh[0]]]

        for sh in sorted_sh[1:]:
            if abs(sh.price - clusters[-1][-1].price) / clusters[-1][-1].price < tolerance_pct:
                clusters[-1].append(sh)
            else:
                clusters.append([sh])

        for cluster in clusters:
            pools.append(LiquidityPool(
                level=max(s.price for s in cluster),
                liq_type=LiquidityType.BSL,
                strength=len(cluster),
                timestamp=cluster[-1].timestamp,
            ))

    # Group nearby swing lows (equal lows detection)
    if swing_lows:
        sorted_sl = sorted(swing_lows, key=lambda s: s.price)
        clusters = [[sorted_sl[0]]]

        for sl in sorted_sl[1:]:
            if abs(sl.price - clusters[-1][-1].price) / clusters[-1][-1].price < tolerance_pct:
                clusters[-1].append(sl)
            else:
                clusters.append([sl])

        for cluster in clusters:
            pools.append(LiquidityPool(
                level=min(s.price for s in cluster),
                liq_type=LiquidityType.SSL,
                strength=len(cluster),
                timestamp=cluster[-1].timestamp,
            ))

    return pools


def detect_sweep(pool: LiquidityPool, high: pd.Series, low: pd.Series,
                 close: pd.Series, lookback: int = 5) -> bool:
    """
    Check if a liquidity pool was swept in the last `lookback` bars.

    Sweep = price pierced the level (wick beyond) but closed back on the other side.
    This is the key signal — liquidity taken, reversal likely.
    """
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]
    recent_close = close.iloc[-lookback:]

    if pool.liq_type == LiquidityType.BSL:
        # Price wicked above the BSL level but closed below it
        for i in range(len(recent_high)):
            if recent_high.iloc[i] > pool.level and recent_close.iloc[i] < pool.level:
                return True

    elif pool.liq_type == LiquidityType.SSL:
        # Price wicked below the SSL level but closed above it
        for i in range(len(recent_low)):
            if recent_low.iloc[i] < pool.level and recent_close.iloc[i] > pool.level:
                return True

    return False


def get_session_levels(df: pd.DataFrame) -> SessionLevels:
    """
    Get previous day high/low and previous week high/low.
    These are the primary draw-on-liquidity levels.
    """
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    now = df.index[-1]
    today = now.normalize()
    yesterday = today - pd.Timedelta(days=1)

    # Find previous day
    prev_day = df[df.index.normalize() == yesterday]
    if prev_day.empty:
        # Try day before that
        prev_day = df[(df.index.normalize() >= yesterday - pd.Timedelta(days=3)) &
                      (df.index.normalize() < today)]

    # Current day
    curr_day = df[df.index.normalize() == today]
    if curr_day.empty:
        curr_day = df.iloc[-1:]

    # Previous week
    this_monday = today - pd.Timedelta(days=today.weekday())
    last_monday = this_monday - pd.Timedelta(days=7)
    prev_week = df[(df.index >= last_monday) & (df.index < this_monday)]

    pdh = float(prev_day["high"].max()) if not prev_day.empty else float(df["high"].iloc[-2])
    pdl = float(prev_day["low"].min()) if not prev_day.empty else float(df["low"].iloc[-2])
    pwh = float(prev_week["high"].max()) if not prev_week.empty else float(df["high"].max())
    pwl = float(prev_week["low"].min()) if not prev_week.empty else float(df["low"].min())
    cdh = float(curr_day["high"].max())
    cdl = float(curr_day["low"].min())

    return SessionLevels(pdh=pdh, pdl=pdl, pwh=pwh, pwl=pwl,
                         current_day_high=cdh, current_day_low=cdl)


def find_draw_on_liquidity(price: float, session: SessionLevels,
                           pools: list[LiquidityPool]) -> tuple[float | None, float | None]:
    """
    Determine where price is most likely drawn to.

    Returns (upside_target, downside_target) — the nearest untouched
    liquidity levels above and below current price.
    """
    # Combine session levels and pool levels
    upside_levels = []
    downside_levels = []

    # Session levels
    for level in [session.pdh, session.pwh]:
        if level > price * 1.0005:
            upside_levels.append(level)
    for level in [session.pdl, session.pwl]:
        if level < price * 0.9995:
            downside_levels.append(level)

    # Untouched liquidity pools
    for pool in pools:
        if pool.status == SweepStatus.UNTOUCHED:
            if pool.liq_type == LiquidityType.BSL and pool.level > price * 1.0005:
                upside_levels.append(pool.level)
            elif pool.liq_type == LiquidityType.SSL and pool.level < price * 0.9995:
                downside_levels.append(pool.level)

    upside = min(upside_levels) if upside_levels else None
    downside = max(downside_levels) if downside_levels else None

    return upside, downside
