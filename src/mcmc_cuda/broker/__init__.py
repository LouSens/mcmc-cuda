"""MT5 paper-trading bridge.

Public API:
    MT5Bridge          — context manager that routes orders to MT5 demo
    SignalExporter     — atomic JSON snapshot for an MT5 EA to consume
    SignalSnapshot     — typed payload the EA reads
    OrderTicket        — typed market-order request
    NotDemoAccountError — raised when an order would hit a non-demo account
"""
from __future__ import annotations

from mcmc_cuda.broker.mt5_bridge import (
    LIVE_SIGNAL_FILENAME,
    MT5Bridge,
    NotDemoAccountError,
    OrderTicket,
    SignalExporter,
    SignalSnapshot,
)

__all__ = [
    "LIVE_SIGNAL_FILENAME",
    "MT5Bridge",
    "NotDemoAccountError",
    "OrderTicket",
    "SignalExporter",
    "SignalSnapshot",
]
