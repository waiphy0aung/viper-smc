"""
VIPER SMC — Smart Money Concepts Signal Engine
Main loop: top-down MTF analysis → Telegram signals
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timezone

import config
from data import MultiTFData
from mtf import analyze, SignalDirection
from signal import generate_signal
from notifier import send_signal, send_startup, send_warning


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ]
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=fmt,
        handlers=handlers,
    )


logger = logging.getLogger("main")


def run_cycle(data: MultiTFData, last_signal_time: dict, equity: float):
    """One scan cycle across all instruments."""
    for symbol in config.SYMBOLS:
        try:
            # Fetch all timeframes
            tf_data = data.fetch_all(symbol)
            price = data.get_price(symbol)

            # Run top-down analysis
            analysis = analyze(
                df_weekly=tf_data["weekly"],
                df_daily=tf_data["daily"],
                df_4h=tf_data["4h"],
                df_1h=tf_data["1h"],
                df_15m=tf_data["15m"],
                current_price=price,
            )

            logger.info(f"{symbol} | {analysis}")

            if analysis.direction == SignalDirection.NONE:
                continue

            # Cooldown — 30 min between signals per symbol
            now = time.time()
            last = last_signal_time.get(symbol, 0)
            if now - last < 1800:
                continue

            # Generate signal
            sig = generate_signal(analysis, symbol, price, equity)
            if sig is None:
                continue

            # Send
            send_signal(sig)
            last_signal_time[symbol] = now

            logger.info(f"SIGNAL: {sig}")

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")


def main():
    setup_logging()

    logger.info("=" * 50)
    logger.info("VIPER SMC — Smart Money Concepts Signal Engine")
    logger.info("=" * 50)
    logger.info(f"Instruments: {config.SYMBOLS}")
    logger.info(f"TF Stack: Weekly → Daily → 4H → 1H → 15m")
    logger.info(f"Risk: {config.MAX_RISK_PER_TRADE*100}% × confidence")
    logger.info("=" * 50)

    data_layer = MultiTFData()
    last_signal_time: dict[str, float] = {}
    equity = config.ACCOUNT_SIZE

    send_startup()

    consecutive_errors = 0

    while True:
        try:
            run_cycle(data_layer, last_signal_time, equity)
            consecutive_errors = 0
            time.sleep(config.LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Cycle error #{consecutive_errors}: {e}")
            if consecutive_errors >= config.MAX_RETRIES_ON_ERROR:
                send_warning(f"Bot stopped after {config.MAX_RETRIES_ON_ERROR} errors: {e}")
                break
            time.sleep(config.RETRY_DELAY_SECONDS * consecutive_errors)

    logger.info("VIPER SMC stopped.")


if __name__ == "__main__":
    main()
