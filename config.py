"""
VIPER SMC — Smart Money Concepts Signal Engine
Configuration
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PROP FIRM RULES — Funding Pips $5k
# =============================================================================
ACCOUNT_SIZE = 5000
DAILY_DD_LIMIT = 0.05
MAX_DD_LIMIT = 0.10
PROFIT_TARGET_PHASE1 = 0.08
PROFIT_TARGET_PHASE2 = 0.05
EQUITY_FLOOR = ACCOUNT_SIZE * (1 - MAX_DD_LIMIT)

# =============================================================================
# INSTRUMENTS
# =============================================================================
SYMBOLS = ["US30", "SP500"]
SYMBOL_DISPLAY = {"US30": "US30", "SP500": "SP500"}

# =============================================================================
# KILLZONES — tight institutional windows, not wide sessions
# =============================================================================
SESSION_FILTER_ENABLED = True
SESSION_WINDOWS = {
    "GBPUSD": [(7, 11), (13, 17)],   # London + NY — GBP moves both sessions
    "US30":   [(13, 17)],              # NY session only
    "SP500":  [(13, 17)],              # NY session only
}

# =============================================================================
# PREMIUM / DISCOUNT — core SMC filter
# =============================================================================
# Only long in discount (below 50% of dealing range)
# Only short in premium (above 50% of dealing range)
PREMIUM_DISCOUNT_FILTER = True

# Alignment requirement per instrument
# Gold: 4H + 1 HTF (more flexible — gold trends intraday against daily)
# NAS100: 4H + daily + weekly must agree (NAS respects higher TF more)
STRICT_ALIGNMENT = {
    "GBPUSD": False,   # 4H + daily not opposing
    "US30":   False,
    "SP500":  False,
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE = 0.01      # 1% base, scaled by confidence (0.5-1.0)
MAX_CONCURRENT_SIGNALS = 1
MIN_RISK_REWARD = 1.5

# Displacement requirement — must see institutional momentum candle
# Displacement adds confidence but shouldn't block entries entirely
REQUIRE_DISPLACEMENT = False   # was True — too restrictive, only 4 trades in 180 days
DISPLACEMENT_BODY_RATIO = 0.6  # candle body must be 60%+ of total range
DISPLACEMENT_SIZE_MULT = 1.5   # body must be 1.5x average body size

# TP Rules — don't partial at PDH/PDL if too close
PARTIAL_MIN_RR = 1.5           # minimum R:R to TP1 for partial close
# If PDH/PDL is < 1.5:1 away, skip partial and go full TP at next liquidity target

# Per-instrument lot multiplier
LOT_DOLLAR_PER_POINT = {
    "GBPUSD": 100000,  # $10 per pip per 0.1 lot
    "US30":   5,        # $5 per point per 1 lot
    "SP500":  50,       # $50 per point per 1 lot
}

# Spread
SPREAD_POINTS = {"GBPUSD": 0.00015, "US30": 2.0, "SP500": 0.5}

# =============================================================================
# STRUCTURE DETECTION
# =============================================================================
# Swing point strength (bars on each side)
SWING_STRENGTH = {
    "weekly": 2,
    "daily": 3,
    "4h": 3,
    "1h": 3,
    "15m": 3,
}

# Structure lookback (bars)
STRUCTURE_LOOKBACK = {
    "weekly": 20,
    "daily": 30,
    "4h": 50,
    "1h": 50,
    "15m": 30,
}

# =============================================================================
# DATA SOURCE
# =============================================================================
SYMBOL_DATA = {
    "GBPUSD": {"ccxt_exchange": None, "ccxt_symbol": None, "yf_ticker": "GBPUSD=X"},
    "US30":   {"ccxt_exchange": None, "ccxt_symbol": None, "yf_ticker": "YM=F"},
    "SP500":  {"ccxt_exchange": None, "ccxt_symbol": None, "yf_ticker": "ES=F"},
}

# =============================================================================
# TELEGRAM
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =============================================================================
# LOOP
# =============================================================================
LOOP_INTERVAL_SECONDS = 60
MAX_RETRIES_ON_ERROR = 5
RETRY_DELAY_SECONDS = 30

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "viper_smc.log"
