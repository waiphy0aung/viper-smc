"""
Multi-timeframe data layer.

Fetches Weekly, Daily, 4H, 1H, 15m data for each instrument.
XAUUSD: Bybit for intraday, yfinance for daily+
NAS100: yfinance for everything
"""

from __future__ import annotations

import logging

import ccxt
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


class MultiTFData:
    """Fetches all timeframes needed for SMC analysis."""

    def __init__(self):
        self.ccxt_clients = {}
        for sym, src in config.SYMBOL_DATA.items():
            if src["ccxt_exchange"]:
                ex_id = src["ccxt_exchange"]
                if ex_id not in self.ccxt_clients:
                    self.ccxt_clients[ex_id] = getattr(ccxt, ex_id)({"enableRateLimit": True})
        logger.info("Multi-TF data layer initialized")

    def fetch_all(self, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Fetch all timeframes for a symbol.
        Returns dict with keys: 'weekly', 'daily', '4h', '1h', '15m'
        """
        src = config.SYMBOL_DATA.get(symbol)
        if not src:
            raise ValueError(f"Unknown symbol: {symbol}")

        result = {}

        if src["ccxt_exchange"] and src["ccxt_symbol"]:
            client = self.ccxt_clients[src["ccxt_exchange"]]
            ccxt_sym = src["ccxt_symbol"]

            # Intraday from CCXT (faster, more data for 15m)
            result["15m"] = self._fetch_ccxt(client, ccxt_sym, "15m", 200)
            result["1h"] = self._fetch_ccxt(client, ccxt_sym, "1h", 200)
            result["4h"] = self._fetch_ccxt(client, ccxt_sym, "4h", 200)

        if src["yf_ticker"]:
            ticker = src["yf_ticker"]

            # Daily and weekly from yfinance (longer history)
            result["daily"] = self._fetch_yf(ticker, "1d", "2y")
            result["weekly"] = self._fetch_yf(ticker, "1wk", "5y")

            # Fill missing intraday from yfinance if CCXT not available
            if "15m" not in result:
                result["15m"] = self._fetch_yf(ticker, "15m", "60d")
            if "1h" not in result:
                result["1h"] = self._fetch_yf(ticker, "1h", "730d")
            if "4h" not in result:
                # yfinance doesn't have 4h — resample from 1h
                if "1h" in result:
                    result["4h"] = result["1h"].resample("4h").agg({
                        "open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum",
                    }).dropna()

        for tf, df in result.items():
            logger.debug(f"{symbol} {tf}: {len(df)} bars")

        return result

    def get_price(self, symbol: str) -> float:
        src = config.SYMBOL_DATA.get(symbol)
        if src and src["ccxt_exchange"]:
            client = self.ccxt_clients[src["ccxt_exchange"]]
            ticker = client.fetch_ticker(src["ccxt_symbol"])
            return float(ticker["last"])
        elif src and src["yf_ticker"]:
            df = self._fetch_yf(src["yf_ticker"], "15m", "5d")
            return float(df["close"].iloc[-1])
        raise ValueError(f"Unknown symbol: {symbol}")

    def _fetch_ccxt(self, client, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        raw = client.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _fetch_yf(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        data.columns = [c[0].lower() for c in data.columns]
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        return data
