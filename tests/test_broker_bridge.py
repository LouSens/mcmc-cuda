"""Broker-bridge surfaces that don't require the MT5 terminal.

The MT5Bridge order-routing path is not exercised here — it needs a live
terminal. We do verify (a) the demo guard refuses to route on a missing /
non-demo account via a tiny fake mt5 module, and (b) SignalExporter writes
atomically and round-trips through `read()`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcmc_cuda.broker import (
    MT5Bridge,
    NotDemoAccountError,
    SignalExporter,
    SignalSnapshot,
)


# ----------------------------------------------------------------------
# SignalExporter
# ----------------------------------------------------------------------
def test_signal_exporter_publishes_atomically(tmp_path: Path) -> None:
    target = tmp_path / "live_signal.json"
    snap = SignalSnapshot(
        symbol="XAUUSD",
        timeframe="M15",
        bar_time="2026-05-06T12:00:00+00:00",
        side=1,
        sl_price=2300.0,
        tp_price=2380.0,
        risk_fraction=0.005,
        confidence=0.41,
        horizon_bars=4,
    )
    exporter = SignalExporter(path=target)
    written = exporter.publish(snap)
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["side"] == 1
    assert payload["symbol"] == "XAUUSD"
    assert "generated_at" in payload  # auto-stamped


def test_signal_exporter_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "live_signal.json"
    snap = SignalSnapshot(
        symbol="XAUUSD", timeframe="M15", bar_time="2026-05-06T12:00:00+00:00",
        side=-1, sl_price=2350.0, tp_price=2280.0,
        risk_fraction=0.005, confidence=0.33, horizon_bars=3,
    )
    exporter = SignalExporter(path=target)
    exporter.publish(snap)
    loaded = exporter.read()
    assert loaded is not None
    assert loaded.side == -1
    assert loaded.tp_price == pytest.approx(2280.0)


def test_signal_exporter_read_missing_returns_none(tmp_path: Path) -> None:
    exporter = SignalExporter(path=tmp_path / "no_such_signal.json")
    assert exporter.read() is None


# ----------------------------------------------------------------------
# Demo-account guard (no real MT5 import)
# ----------------------------------------------------------------------
def test_demo_guard_rejects_when_account_info_missing() -> None:
    bridge = MT5Bridge()
    bridge._mt5 = SimpleNamespace(
        account_info=lambda: None,
        ACCOUNT_TRADE_MODE_DEMO=0,
    )
    with pytest.raises(NotDemoAccountError):
        bridge.assert_demo()


def test_demo_guard_rejects_non_demo_trade_mode() -> None:
    # Simulate a real account: trade_mode=2 (REAL), DEMO constant=0.
    bridge = MT5Bridge()
    bridge._mt5 = SimpleNamespace(
        account_info=lambda: SimpleNamespace(
            login=12345, server="Live-Server", trade_mode=2,
        ),
        ACCOUNT_TRADE_MODE_DEMO=0,
    )
    with pytest.raises(NotDemoAccountError):
        bridge.assert_demo()


def test_demo_guard_passes_on_demo_account() -> None:
    bridge = MT5Bridge()
    bridge._mt5 = SimpleNamespace(
        account_info=lambda: SimpleNamespace(
            login=99999, server="Demo-Server", trade_mode=0,
        ),
        ACCOUNT_TRADE_MODE_DEMO=0,
    )
    bridge.assert_demo()  # must not raise
