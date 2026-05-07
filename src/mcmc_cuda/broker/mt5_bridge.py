"""MT5 paper-trading bridge.

Hard contract: this module **refuses to send any order** unless the connected
MT5 terminal is logged into a demo account. The check is performed on every
single order call, not just at startup, so re-attaching to a live account
mid-session cannot accidentally route real money.

Two surfaces are exposed:

1. `MT5Bridge` — Python-side order routing. Use when running a Python loop
   that wakes on each new bar, asks the strategy for a side, and submits a
   market order with SL/TP attached. Suitable for laptop research; not
   suitable for unattended VPS deployment.

2. `SignalExporter` — writes the latest signal snapshot (side + SL/TP +
   risk fraction + timestamp) to `<artifacts>/live_signal.json` atomically.
   The companion MT5 EA (`mcmc_signal_follower.mq5`, see docs) polls this
   file on each tick and submits / closes positions inside MT5 itself. This
   is the path you want for unattended demo deployment because the EA can
   keep running even if the Python process dies.

Both surfaces share the same demo guard and risk/lot translation logic.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcmc_cuda.config import ARTIFACTS_DIR


LIVE_SIGNAL_FILENAME = "live_signal.json"


class NotDemoAccountError(RuntimeError):
    """Raised when an order would be routed against a non-demo account."""


@dataclass
class OrderTicket:
    side: int                      # +1 long, -1 short
    volume_lots: float             # MT5 "volume" (lots, not oz)
    sl_price: float
    tp_price: float
    deviation_points: int = 20     # max allowed slippage in points
    magic: int = 26_05_2026        # arbitrary EA id; useful for filtering
    comment: str = "mcmc_cuda"

    def as_payload(self, mt5, symbol_info, side_constant_long, side_constant_short, price: float) -> dict:
        return dict(
            action=mt5.TRADE_ACTION_DEAL,
            symbol=symbol_info.name,
            volume=float(self.volume_lots),
            type=side_constant_long if self.side > 0 else side_constant_short,
            price=float(price),
            sl=float(self.sl_price),
            tp=float(self.tp_price),
            deviation=int(self.deviation_points),
            magic=int(self.magic),
            comment=self.comment,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=getattr(mt5, "ORDER_FILLING_IOC", 1),
        )


@dataclass
class SignalSnapshot:
    """The unit the EA reads on each tick.

    Fields are deliberately simple and stable so the .mq5 side can decode
    with `FileReadString` + a tiny JSON parser without needing a library.
    """
    symbol: str
    timeframe: str
    bar_time: str                  # ISO-8601 UTC of the bar this signal refers to
    side: int                      # -1 / 0 / +1
    sl_price: float
    tp_price: float
    risk_fraction: float           # fraction of equity to risk per trade
    confidence: float              # MC confidence score, [0, 1]
    horizon_bars: int
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Signal exporter: file-based handoff to the MT5 EA. Pure Python, no MT5
# dependency — safe to import on Linux / CI.
# --------------------------------------------------------------------------
class SignalExporter:
    """Writes the latest `SignalSnapshot` to disk atomically.

    Atomic = write-to-temp + os.replace, so the EA never reads a half-written
    file. The EA should treat any signal older than `stale_after_seconds` as
    "no opinion" and refuse to act on it.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else (ARTIFACTS_DIR / LIVE_SIGNAL_FILENAME)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, snapshot: SignalSnapshot) -> Path:
        payload = json.dumps(snapshot.to_dict(), indent=2, default=str)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=str(self.path.parent), suffix=".tmp",
            encoding="utf-8",
        )
        try:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.path)
        except Exception:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return self.path

    def read(self) -> Optional[SignalSnapshot]:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return SignalSnapshot(**data)


# --------------------------------------------------------------------------
# MT5 bridge: Python-side order routing. Imports MetaTrader5 lazily so this
# module stays importable on machines without it.
# --------------------------------------------------------------------------
class MT5Bridge:
    """Thin order-routing wrapper with a hard demo-account guard.

    Lifecycle::

        with MT5Bridge() as bridge:
            bridge.assert_demo()
            bridge.submit_market(symbol="XAUUSD", ticket=OrderTicket(...))

    All order entry points (`submit_market`, `close_position`,
    `flatten_all`) re-call `assert_demo()` before talking to the terminal.
    """

    def __init__(self) -> None:
        self._mt5 = None  # populated by __enter__

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "MT5Bridge":
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise RuntimeError(
                f"mt5.initialize() failed: {mt5.last_error()}. "
                "Is the MT5 terminal running and logged in?"
            )
        self._mt5 = mt5
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    @property
    def mt5(self):
        if self._mt5 is None:
            raise RuntimeError(
                "MT5Bridge used outside its context manager — wrap calls in "
                "`with MT5Bridge() as bridge:`."
            )
        return self._mt5

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def assert_demo(self) -> None:
        """Raise NotDemoAccountError unless the connected account is demo."""
        mt5 = self.mt5
        acct = mt5.account_info()
        if acct is None:
            raise NotDemoAccountError(
                "Cannot read account_info(); refusing to trade."
            )
        if getattr(acct, "trade_mode", None) != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise NotDemoAccountError(
                f"Refusing to send orders: account {acct.login} on "
                f"{acct.server!r} is NOT in demo mode "
                f"(trade_mode={acct.trade_mode}). This guard is intentional — "
                "this codebase is paper-trade only."
            )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def submit_market(self, symbol: str, ticket: OrderTicket) -> dict:
        """Submit a market order with SL/TP attached.

        Returns a dict mirroring the MT5 OrderSendResult.
        """
        self.assert_demo()
        mt5 = self.mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol!r}) returned None")
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol!r}) returned None")
        price = tick.ask if ticket.side > 0 else tick.bid

        request = ticket.as_payload(
            mt5,
            symbol_info=info,
            side_constant_long=mt5.ORDER_TYPE_BUY,
            side_constant_short=mt5.ORDER_TYPE_SELL,
            price=price,
        )
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
        return _result_to_dict(result)

    def close_position(self, position_ticket: int) -> dict:
        """Close a single open position by its ticket id."""
        self.assert_demo()
        mt5 = self.mt5
        positions = mt5.positions_get(ticket=position_ticket)
        if not positions:
            raise RuntimeError(f"position {position_ticket} not found")
        pos = positions[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        side_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if side_close == mt5.ORDER_TYPE_SELL else tick.ask
        request = dict(
            action=mt5.TRADE_ACTION_DEAL,
            position=pos.ticket,
            symbol=pos.symbol,
            volume=float(pos.volume),
            type=side_close,
            price=float(price),
            deviation=20,
            magic=int(pos.magic),
            comment=f"close#{pos.ticket}",
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=getattr(mt5, "ORDER_FILLING_IOC", 1),
        )
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send(close) returned None: {mt5.last_error()}")
        return _result_to_dict(result)

    def flatten_all(self, symbol: Optional[str] = None) -> list[dict]:
        """Close every open position (optionally filtered by symbol)."""
        self.assert_demo()
        mt5 = self.mt5
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        results: list[dict] = []
        for pos in positions or ():
            results.append(self.close_position(int(pos.ticket)))
        return results

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def lots_from_oz(oz: float, contract_size: float = 100.0) -> float:
        """Convert an oz size produced by the backtester into MT5 lots."""
        return float(oz) / float(contract_size)


def _result_to_dict(result) -> dict:
    """Best-effort conversion of MT5's OrderSendResult to a plain dict."""
    keys = (
        "retcode", "deal", "order", "volume", "price", "bid", "ask",
        "comment", "request_id", "retcode_external",
    )
    out: dict = {}
    for k in keys:
        try:
            out[k] = getattr(result, k)
        except AttributeError:
            continue
    return out
