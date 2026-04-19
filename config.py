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
SYMBOLS = ["XAUUSD", "NAS100"]
SYMBOL_DISPLAY = {"XAUUSD": "XAUUSD", "NAS100": "NAS100"}

# =============================================================================
# SESSION FILTER
# =============================================================================
SESSION_FILTER_ENABLED = True
SESSION_WINDOWS = {
    "XAUUSD": [(7, 11), (13, 17)],
    "NAS100": [(13, 20)],
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE = 0.01      # 1% base, scaled by confidence (0.5-1.0)
MAX_CONCURRENT_SIGNALS = 1
MIN_RISK_REWARD = 1.5

# Per-instrument lot multiplier
LOT_DOLLAR_PER_POINT = {
    "XAUUSD": 100,
    "NAS100": 20,
}

# Spread
SPREAD_POINTS = {"XAUUSD": 2.5, "NAS100": 1.5}

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
    "XAUUSD": {"ccxt_exchange": "bybit", "ccxt_symbol": "XAU/USDT:USDT", "yf_ticker": "GC=F"},
    "NAS100": {"ccxt_exchange": None, "ccxt_symbol": None, "yf_ticker": "NQ=F"},
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
