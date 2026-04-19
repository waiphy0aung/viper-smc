"""
Market structure detection — the foundation of SMC.

Detects:
- Swing highs and swing lows
- Break of Structure (BOS) — trend continuation
- Change of Character (CHoCH) — first sign of reversal
- Current market structure (bullish/bearish/ranging)

Works on any timeframe. The same logic applies from weekly to 1m.
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
    BOS = "BOS"       # Break of Structure — continuation
    CHOCH = "CHOCH"   # Change of Character — reversal


@dataclass
class SwingPoint:
    index: int           # bar index in the dataframe
    timestamp: pd.Timestamp
    price: float
    swing_type: SwingType

    def __repr__(self):
        return f"Swing({self.swing_type.value} {self.price:.2f} @ {self.timestamp})"


@dataclass
class StructureBreak:
    break_type: BreakType
    direction: str       # "bullish" or "bearish"
    price: float         # the level that was broken
    timestamp: pd.Timestamp
    index: int

    def __repr__(self):
        return f"{self.break_type.value} {self.direction} @ {self.price:.2f}"


@dataclass
class StructureState:
    """Current market structure on a given timeframe."""
    bias: StructureBias
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    breaks: list[StructureBreak] = field(default_factory=list)
    last_hh: SwingPoint | None = None   # last higher high
    last_hl: SwingPoint | None = None   # last higher low
    last_lh: SwingPoint | None = None   # last lower high
    last_ll: SwingPoint | None = None   # last lower low

    def __repr__(self):
        return (
            f"Structure({self.bias.value} | "
            f"{len(self.swing_highs)}SH {len(self.swing_lows)}SL | "
            f"last break: {self.breaks[-1] if self.breaks else 'none'})"
        )


def find_swing_points(high: pd.Series, low: pd.Series,
                      strength: int = 3) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Find swing highs and lows using a lookback/lookahead of `strength` bars.

    A swing high: bar's high is the highest within `strength` bars on each side.
    A swing low: bar's low is the lowest within `strength` bars on each side.

    strength=3 works for 15m-1H. Use 5 for 4H-daily.
    """
    swing_highs = []
    swing_lows = []

    for i in range(strength, len(high) - strength):
        # Swing high
        is_sh = True
        for j in range(1, strength + 1):
            if high.iloc[i] < high.iloc[i - j] or high.iloc[i] < high.iloc[i + j]:
                is_sh = False
                break
        if is_sh:
            swing_highs.append(SwingPoint(
                index=i,
                timestamp=high.index[i],
                price=float(high.iloc[i]),
                swing_type=SwingType.HIGH,
            ))

        # Swing low
        is_sl = True
        for j in range(1, strength + 1):
            if low.iloc[i] > low.iloc[i - j] or low.iloc[i] > low.iloc[i + j]:
                is_sl = False
                break
        if is_sl:
            swing_lows.append(SwingPoint(
                index=i,
                timestamp=low.index[i],
                price=float(low.iloc[i]),
                swing_type=SwingType.LOW,
            ))

    return swing_highs, swing_lows


def detect_structure(high: pd.Series, low: pd.Series, close: pd.Series,
                     strength: int = 3) -> StructureState:
    """
    Analyze market structure: detect BOS, CHoCH, and determine bias.

    BOS (Break of Structure):
    - Bullish BOS: price breaks above a previous swing high in an uptrend
    - Bearish BOS: price breaks below a previous swing low in a downtrend

    CHoCH (Change of Character):
    - Bullish CHoCH: price breaks above a swing high during a downtrend (first reversal sign)
    - Bearish CHoCH: price breaks below a swing low during an uptrend (first reversal sign)
    """
    swing_highs, swing_lows = find_swing_points(high, low, strength)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureState(bias=StructureBias.RANGING,
                              swing_highs=swing_highs, swing_lows=swing_lows)

    # Merge and sort all swing points by index
    all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.index)

    # Track structure
    bias = StructureBias.RANGING
    breaks = []
    last_hh = None
    last_hl = None
    last_lh = None
    last_ll = None

    prev_sh: SwingPoint | None = None
    prev_sl: SwingPoint | None = None

    for swing in all_swings:
        if swing.swing_type == SwingType.HIGH:
            if prev_sh is not None:
                if swing.price > prev_sh.price:
                    # Higher high
                    if bias == StructureBias.BEARISH:
                        # Was bearish, now making higher high → CHoCH bullish
                        breaks.append(StructureBreak(
                            break_type=BreakType.CHOCH, direction="bullish",
                            price=prev_sh.price, timestamp=swing.timestamp,
                            index=swing.index,
                        ))
                    elif bias == StructureBias.BULLISH:
                        # Continuation — BOS bullish
                        breaks.append(StructureBreak(
                            break_type=BreakType.BOS, direction="bullish",
                            price=prev_sh.price, timestamp=swing.timestamp,
                            index=swing.index,
                        ))
                    bias = StructureBias.BULLISH
                    last_hh = swing
                else:
                    # Lower high
                    last_lh = swing
            prev_sh = swing

        elif swing.swing_type == SwingType.LOW:
            if prev_sl is not None:
                if swing.price < prev_sl.price:
                    # Lower low
                    if bias == StructureBias.BULLISH:
                        # Was bullish, now making lower low → CHoCH bearish
                        breaks.append(StructureBreak(
                            break_type=BreakType.CHOCH, direction="bearish",
                            price=prev_sl.price, timestamp=swing.timestamp,
                            index=swing.index,
                        ))
                    elif bias == StructureBias.BEARISH:
                        # Continuation — BOS bearish
                        breaks.append(StructureBreak(
                            break_type=BreakType.BOS, direction="bearish",
                            price=prev_sl.price, timestamp=swing.timestamp,
                            index=swing.index,
                        ))
                    bias = StructureBias.BEARISH
                    last_ll = swing
                else:
                    # Higher low
                    last_hl = swing
            prev_sl = swing

    return StructureState(
        bias=bias,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        breaks=breaks,
        last_hh=last_hh,
        last_hl=last_hl,
        last_lh=last_lh,
        last_ll=last_ll,
    )


def get_recent_structure(high: pd.Series, low: pd.Series, close: pd.Series,
                         strength: int = 3, lookback: int = 50) -> StructureState:
    """Detect structure on the most recent `lookback` bars."""
    h = high.iloc[-lookback:] if len(high) > lookback else high
    l = low.iloc[-lookback:] if len(low) > lookback else low
    c = close.iloc[-lookback:] if len(close) > lookback else close
    return detect_structure(h, l, c, strength)
