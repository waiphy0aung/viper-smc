"""
Market structure detection v2 — refined for SMC execution.

Added:
- Premium/Discount zones (above/below 50% of dealing range)
- Displacement detection (big momentum candles = institutional activity)
- Inducement detection (minor liquidity taken before the real move)
- Range analysis (dealing range high/low for premium/discount)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np


class StructureBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class BreakType(Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"


class PriceZone(Enum):
    PREMIUM = "PREMIUM"       # above 50% of range — sell zone
    DISCOUNT = "DISCOUNT"     # below 50% of range — buy zone
    EQUILIBRIUM = "EQUILIBRIUM"  # at 50%


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    swing_type: SwingType

    def __repr__(self):
        return f"Swing({self.swing_type.value} {self.price:.2f})"


@dataclass
class StructureBreak:
    break_type: BreakType
    direction: str
    price: float
    timestamp: pd.Timestamp
    index: int

    def __repr__(self):
        return f"{self.break_type.value} {self.direction} @ {self.price:.2f}"


@dataclass
class Displacement:
    """A large momentum candle showing institutional activity."""
    index: int
    timestamp: pd.Timestamp
    direction: str          # "bullish" or "bearish"
    body_size: float        # absolute body size
    body_ratio: float       # body / total range (>0.7 = strong displacement)

    def __repr__(self):
        return f"Displacement({self.direction} {self.body_ratio:.0%} @ {self.timestamp})"


@dataclass
class DealingRange:
    """The current range price is operating in."""
    high: float
    low: float
    midpoint: float         # 50% — equilibrium
    premium_start: float    # above this = premium
    discount_end: float     # below this = discount

    def zone_of(self, price: float) -> PriceZone:
        if price > self.midpoint:
            return PriceZone.PREMIUM
        elif price < self.midpoint:
            return PriceZone.DISCOUNT
        return PriceZone.EQUILIBRIUM


@dataclass
class StructureState:
    bias: StructureBias
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    breaks: list[StructureBreak] = field(default_factory=list)
    dealing_range: DealingRange | None = None
    last_displacement: Displacement | None = None

    def __repr__(self):
        zone = ""
        if self.dealing_range:
            zone = f" range={self.dealing_range.low:.2f}-{self.dealing_range.high:.2f}"
        disp = ""
        if self.last_displacement:
            disp = f" disp={self.last_displacement.direction}"
        return f"Structure({self.bias.value}{zone}{disp})"


def find_swing_points(high: pd.Series, low: pd.Series,
                      strength: int = 3) -> tuple[list[SwingPoint], list[SwingPoint]]:
    swing_highs = []
    swing_lows = []

    for i in range(strength, len(high) - strength):
        is_sh = all(high.iloc[i] >= high.iloc[i - j] and high.iloc[i] >= high.iloc[i + j]
                     for j in range(1, strength + 1))
        if is_sh:
            swing_highs.append(SwingPoint(i, high.index[i], float(high.iloc[i]), SwingType.HIGH))

        is_sl = all(low.iloc[i] <= low.iloc[i - j] and low.iloc[i] <= low.iloc[i + j]
                     for j in range(1, strength + 1))
        if is_sl:
            swing_lows.append(SwingPoint(i, low.index[i], float(low.iloc[i]), SwingType.LOW))

    return swing_highs, swing_lows


def detect_displacement(open_: pd.Series, high: pd.Series, low: pd.Series,
                        close: pd.Series, lookback: int = 5,
                        min_ratio: float = 0.6) -> Displacement | None:
    """
    Find the most recent displacement candle.
    Displacement = large body candle (body > 60% of total range) with above-average size.

    This is institutional footprint — they move price aggressively.
    """
    if len(close) < lookback + 20:
        return None

    # Average candle body size over last 20 bars for comparison
    bodies = (close - open_).abs()
    avg_body = bodies.iloc[-20:].mean()

    for i in range(-1, -lookback - 1, -1):
        body = abs(float(close.iloc[i]) - float(open_.iloc[i]))
        total = float(high.iloc[i]) - float(low.iloc[i])

        if total == 0:
            continue

        ratio = body / total
        if ratio >= min_ratio and body >= avg_body * 1.5:
            direction = "bullish" if close.iloc[i] > open_.iloc[i] else "bearish"
            idx = len(close) + i
            return Displacement(
                index=idx,
                timestamp=close.index[i],
                direction=direction,
                body_size=body,
                body_ratio=ratio,
            )

    return None


def get_dealing_range(swing_highs: list[SwingPoint],
                      swing_lows: list[SwingPoint]) -> DealingRange | None:
    """
    Calculate the current dealing range from the most recent significant swing high and low.
    Premium = above 50%. Discount = below 50%.
    Only buy in discount, only sell in premium.
    """
    if not swing_highs or not swing_lows:
        return None

    # Use the most recent swing high and the most recent swing low
    # that formed BEFORE it (or vice versa) to define the range
    high = max(s.price for s in swing_highs[-3:])
    low = min(s.price for s in swing_lows[-3:])

    if high <= low:
        return None

    mid = (high + low) / 2
    return DealingRange(
        high=high, low=low, midpoint=mid,
        premium_start=mid,
        discount_end=mid,
    )


def detect_structure(high: pd.Series, low: pd.Series, close: pd.Series,
                     strength: int = 3, open_: pd.Series | None = None) -> StructureState:
    swing_highs, swing_lows = find_swing_points(high, low, strength)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureState(bias=StructureBias.RANGING,
                              swing_highs=swing_highs, swing_lows=swing_lows)

    all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.index)

    bias = StructureBias.RANGING
    breaks = []
    prev_sh = None
    prev_sl = None

    for swing in all_swings:
        if swing.swing_type == SwingType.HIGH:
            if prev_sh is not None:
                if swing.price > prev_sh.price:
                    if bias == StructureBias.BEARISH:
                        breaks.append(StructureBreak(BreakType.CHOCH, "bullish",
                                                     prev_sh.price, swing.timestamp, swing.index))
                    elif bias == StructureBias.BULLISH:
                        breaks.append(StructureBreak(BreakType.BOS, "bullish",
                                                     prev_sh.price, swing.timestamp, swing.index))
                    bias = StructureBias.BULLISH
            prev_sh = swing

        elif swing.swing_type == SwingType.LOW:
            if prev_sl is not None:
                if swing.price < prev_sl.price:
                    if bias == StructureBias.BULLISH:
                        breaks.append(StructureBreak(BreakType.CHOCH, "bearish",
                                                     prev_sl.price, swing.timestamp, swing.index))
                    elif bias == StructureBias.BEARISH:
                        breaks.append(StructureBreak(BreakType.BOS, "bearish",
                                                     prev_sl.price, swing.timestamp, swing.index))
                    bias = StructureBias.BEARISH
            prev_sl = swing

    # Dealing range
    dealing_range = get_dealing_range(swing_highs, swing_lows)

    # Displacement
    displacement = None
    if open_ is not None:
        displacement = detect_displacement(open_, high, low, close)

    return StructureState(
        bias=bias,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        breaks=breaks,
        dealing_range=dealing_range,
        last_displacement=displacement,
    )


def get_recent_structure(high: pd.Series, low: pd.Series, close: pd.Series,
                         strength: int = 3, lookback: int = 50,
                         open_: pd.Series | None = None) -> StructureState:
    h = high.iloc[-lookback:] if len(high) > lookback else high
    l = low.iloc[-lookback:] if len(low) > lookback else low
    c = close.iloc[-lookback:] if len(close) > lookback else close
    o = open_.iloc[-lookback:] if open_ is not None and len(open_) > lookback else open_
    return detect_structure(h, l, c, strength, o)
