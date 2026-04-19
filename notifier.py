"""
Telegram signal sender — SMC-style signals with full context.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

import config
from signal import TradeSignal

logger = logging.getLogger(__name__)


def _send(text: str):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Telegram error: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


def send_signal(sig: TradeSignal):
    emoji = "\U0001f7e2" if sig.direction == "BUY" else "\U0001f534"
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    rr2_line = f"\U0001f3af TP2:    <code>{sig.tp2:.2f}</code>  (R:R 1:{sig.rr2:.1f})\n" if sig.tp2 and sig.rr2 else ""

    text = (
        f"{emoji} <b>{sig.direction} {sig.symbol}</b>\n"
        f"\n"
        f"\U0001f4cd Entry:  <code>{sig.entry:.2f}</code>\n"
        f"\U0001f6d1 SL:     <code>{sig.sl:.2f}</code>\n"
        f"\U0001f3af TP1:    <code>{sig.tp1:.2f}</code>  (R:R 1:{sig.rr1:.1f}) — close 50%\n"
        f"{rr2_line}"
        f"\n"
        f"\U0001f4e6 Lot:    <code>{sig.lot_size:.2f}</code>\n"
        f"\U0001f4b0 Risk:   <code>${sig.risk_dollars:.0f}</code>\n"
        f"\U0001f4aa Conf:   <code>{sig.confidence:.0%}</code>\n"
        f"\n"
        f"\U0001f9e0 <b>Analysis:</b>\n"
        f"<code>{sig.analysis_summary}</code>\n"
        f"\n"
        f"\U0001f4ca <b>Key Levels:</b>\n"
        f"<code>{sig.session_levels}</code>\n"
        f"\n"
        f"\u23f0 {now}"
    )
    _send(text)
    logger.info(f"Signal sent: {sig}")


def send_startup():
    text = (
        f"\U0001f40d <b>VIPER SMC Started</b>\n"
        f"\n"
        f"Strategy: Smart Money Concepts\n"
        f"TF Stack: W → D → 4H → 1H → 15m\n"
        f"Instruments: {', '.join(config.SYMBOLS)}\n"
        f"Risk: {config.MAX_RISK_PER_TRADE*100:.0f}% × confidence\n"
        f"Min R:R: 1:{config.MIN_RISK_REWARD}\n"
        f"\n"
        f"Scanning for liquidity sweeps..."
    )
    _send(text)


def send_warning(msg: str):
    _send(f"\u26a0\ufe0f <b>WARNING</b>\n\n{msg}")
