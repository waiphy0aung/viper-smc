"""
ICT Entry Model — the actual institutional trade setup.

The model:
1. Liquidity is swept (stop hunt)
2. Displacement candle shows institutional entry
3. The displacement creates a fresh FVG (imbalance)
4. Price returns to this FVG → ENTRY
5. SL behind the sweep wick
6. TP at next liquidity pool

This is not "enter at old OB" — this is enter at the imbalance
created by the smart money move that just happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class SweepSetup:
    """A completed liquidity sweep with displacement."""
    sweep_bar: int              # bar where sweep happened
    sweep_price: float          # the wick that swept (SL anchor)
    sweep_direction: str        # "bullish_sweep" (swept lows, expect up) or "bearish_sweep"
    displacement_bar: int       # the big candle after sweep
    displacement_direction: str # "bullish" or "bearish"
    fvg_top: float             # fresh FVG created by displacement
    fvg_bottom: float
    fvg_mid: float             # ideal entry
    target_price: float        # next liquidity target (TP)
    sl_price: float            # behind the sweep wick

    @property
    def risk(self) -> float:
        return abs(self.fvg_mid - self.sl_price)

    @property
    def reward(self) -> float:
        return abs(self.target_price - self.fvg_mid)

    @property
    def rr(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0

    def __repr__(self):
        return (f"Setup({self.sweep_direction} | entry={self.fvg_mid:.2f} "
                f"SL={self.sl_price:.2f} TP={self.target_price:.2f} R:R=1:{self.rr:.1f})")


def find_sweep_and_displacement(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series,
    lookback: int = 20,
    swing_strength: int = 3,
    disp_body_ratio: float = 0.45,
    disp_size_mult: float = 1.0,
) -> list[SweepSetup]:
    """
    Scan recent bars for the ICT entry model:
    sweep → displacement → FVG → entry opportunity.

    Returns list of valid setups found in the lookback window.
    """
    n = len(high)
    if n < lookback + swing_strength * 2 + 5:
        return []

    setups = []

    # Find swing highs and lows for liquidity levels
    swing_highs = []
    swing_lows = []
    for i in range(swing_strength, n - swing_strength):
        if all(high.iloc[i] >= high.iloc[i-j] and high.iloc[i] >= high.iloc[i+j]
               for j in range(1, swing_strength + 1)):
            swing_highs.append((i, float(high.iloc[i])))
        if all(low.iloc[i] <= low.iloc[i-j] and low.iloc[i] <= low.iloc[i+j]
               for j in range(1, swing_strength + 1)):
            swing_lows.append((i, float(low.iloc[i])))

    if not swing_highs or not swing_lows:
        return []

    # Average body size for displacement detection
    bodies = (close - open_).abs()
    avg_body = float(bodies.iloc[-30:].mean()) if n > 30 else float(bodies.mean())

    # Scan the lookback window for sweep + displacement patterns
    start = max(0, n - lookback)

    for i in range(start + 1, n - 2):
        bar_low = float(low.iloc[i])
        bar_high = float(high.iloc[i])
        bar_close = float(close.iloc[i])
        bar_open = float(open_.iloc[i])

        # =============================================
        # BULLISH SWEEP: price wicks below a swing low, closes above it
        # Then displacement candle UP + bullish FVG
        # =============================================
        recent_lows = [p for idx, p in swing_lows if idx < i and idx > i - 50]
        for sl_level in recent_lows:
            if bar_low < sl_level and bar_close > sl_level:
                # Sweep confirmed — wick below, close above
                # Now look for displacement in the NEXT few bars
                for j in range(i + 1, min(i + 4, n)):
                    d_open = float(open_.iloc[j])
                    d_close = float(close.iloc[j])
                    d_high = float(high.iloc[j])
                    d_low = float(low.iloc[j])
                    d_body = abs(d_close - d_open)
                    d_range = d_high - d_low

                    if d_range == 0:
                        continue

                    # Bullish displacement: big green candle
                    if (d_close > d_open and
                        d_body / d_range >= disp_body_ratio and
                        d_body >= avg_body * disp_size_mult):

                        # Check for FVG: gap between bar before displacement and bar after
                        if j + 1 < n:
                            pre_high = float(high.iloc[j - 1])
                            post_low = float(low.iloc[j + 1])

                            # Allow near-FVG: wicks within 0.1% of each other
                            # True FVG on 15m gold is rare — near-gaps still work
                            gap = post_low - pre_high
                            tolerance = pre_high * 0.001

                            if gap > -tolerance:  # gap exists or nearly exists
                                fvg_top = max(post_low, pre_high + tolerance)
                                fvg_bottom = min(pre_high, post_low - tolerance)
                                if fvg_top <= fvg_bottom:
                                    fvg_top = pre_high + abs(gap) + tolerance
                                    fvg_bottom = pre_high
                                fvg_mid = (fvg_top + fvg_bottom) / 2

                                # SL behind the sweep wick with buffer
                                sl_price = bar_low - d_body * 0.3

                                # TP: next swing high above (BSL target)
                                targets = [p for _, p in swing_highs if p > d_high]
                                if not targets:
                                    targets = [d_high + (d_high - sl_price)]  # fallback 1:1
                                tp = min(targets)

                                setup = SweepSetup(
                                    sweep_bar=i, sweep_price=bar_low,
                                    sweep_direction="bullish_sweep",
                                    displacement_bar=j,
                                    displacement_direction="bullish",
                                    fvg_top=fvg_top, fvg_bottom=fvg_bottom,
                                    fvg_mid=fvg_mid,
                                    target_price=tp,
                                    sl_price=sl_price,
                                )
                                if setup.rr >= 1.5:
                                    setups.append(setup)
                        break  # only need one displacement per sweep

        # =============================================
        # BEARISH SWEEP: price wicks above a swing high, closes below it
        # Then displacement candle DOWN + bearish FVG
        # =============================================
        recent_highs = [p for idx, p in swing_highs if idx < i and idx > i - 50]
        for sh_level in recent_highs:
            if bar_high > sh_level and bar_close < sh_level:
                for j in range(i + 1, min(i + 4, n)):
                    d_open = float(open_.iloc[j])
                    d_close = float(close.iloc[j])
                    d_high = float(high.iloc[j])
                    d_low = float(low.iloc[j])
                    d_body = abs(d_close - d_open)
                    d_range = d_high - d_low

                    if d_range == 0:
                        continue

                    if (d_close < d_open and
                        d_body / d_range >= disp_body_ratio and
                        d_body >= avg_body * disp_size_mult):

                        if j + 1 < n:
                            pre_low = float(low.iloc[j - 1])
                            post_high = float(high.iloc[j + 1])

                            gap = pre_low - post_high
                            tolerance = pre_low * 0.001

                            if gap > -tolerance:
                                fvg_top = max(pre_low, post_high + tolerance)
                                fvg_bottom = min(post_high, pre_low - tolerance)
                                if fvg_top <= fvg_bottom:
                                    fvg_top = pre_low
                                    fvg_bottom = pre_low - abs(gap) - tolerance
                                fvg_mid = (fvg_top + fvg_bottom) / 2

                                sl_price = bar_high + d_body * 0.3

                                targets = [p for _, p in swing_lows if p < d_low]
                                if not targets:
                                    targets = [d_low - (sl_price - d_low)]
                                tp = max(targets)

                                setup = SweepSetup(
                                    sweep_bar=i, sweep_price=bar_high,
                                    sweep_direction="bearish_sweep",
                                    displacement_bar=j,
                                    displacement_direction="bearish",
                                    fvg_top=fvg_top, fvg_bottom=fvg_bottom,
                                    fvg_mid=fvg_mid,
                                    target_price=tp,
                                    sl_price=sl_price,
                                )
                                if setup.rr >= 1.5:
                                    setups.append(setup)
                        break

    return setups


def find_active_setup(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series,
    current_price: float,
    lookback: int = 30,
) -> SweepSetup | None:
    """
    Find the most recent valid setup where price is currently in the FVG zone
    (pullback to the entry level) or close to it.

    Returns the best active setup, or None.
    """
    setups = find_sweep_and_displacement(high, low, open_, close, lookback)

    if not setups:
        return None

    # Filter for setups where price is near the FVG (within the zone or approaching)
    active = []
    for s in setups:
        if s.sweep_direction == "bullish_sweep":
            # Price should be pulling back into the FVG zone
            if s.fvg_bottom <= current_price <= s.fvg_top * 1.002:
                active.append(s)
            # Or just above the FVG (just entered)
            elif current_price <= s.fvg_top * 1.005 and current_price >= s.fvg_bottom:
                active.append(s)

        elif s.sweep_direction == "bearish_sweep":
            if s.fvg_bottom * 0.998 <= current_price <= s.fvg_top:
                active.append(s)
            elif current_price >= s.fvg_bottom * 0.995 and current_price <= s.fvg_top:
                active.append(s)

    if not active:
        return None

    # Return the one with best R:R
    return max(active, key=lambda s: s.rr)
