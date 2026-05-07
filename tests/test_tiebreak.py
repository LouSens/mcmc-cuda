"""Same-bar TP/SL tiebreaker behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    _resolve_same_bar,
    run_backtest_ohlc,
    trade_log_ohlc,
)


def test_resolve_same_bar_unambiguous_cases():
    assert _resolve_same_bar(1, True, False, 0.0, "by_close") == "tp"
    assert _resolve_same_bar(1, False, True, 0.0, "by_close") == "sl"
    assert _resolve_same_bar(1, False, False, 0.0, "by_close") == "none"


def test_resolve_same_bar_by_close_uses_direction():
    # Long, both touched, bar went up -> TP first
    assert _resolve_same_bar(1, True, True, +1.0, "by_close") == "tp"
    # Long, both touched, bar went down -> SL first
    assert _resolve_same_bar(1, True, True, -1.0, "by_close") == "sl"
    # Short flips
    assert _resolve_same_bar(-1, True, True, -1.0, "by_close") == "tp"
    assert _resolve_same_bar(-1, True, True, +1.0, "by_close") == "sl"


def test_resolve_same_bar_legacy_modes():
    assert _resolve_same_bar(1, True, True, +1.0, "sl_first") == "sl"
    assert _resolve_same_bar(1, True, True, -1.0, "tp_first") == "tp"


def _build_bar_with_both_hit(side: int, bar_dir: int) -> pd.DataFrame:
    """Construct a single trade where the entry bar touches both TP and SL.

    Build a long signal that fires at bar 0; entry happens at bar 1's open.
    On bar 1 we engineer high/low so both TP=1.0*ATR and SL=0.5*ATR are
    touched. `bar_dir` controls whether the bar closes up (+1) or down (-1).
    """
    n = 30
    idx = pd.date_range("2024-06-03 08:00", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 2000.0)
    high = np.full(n, 2000.5)
    low = np.full(n, 1999.5)
    # Need enough bars to seed ATR (~14). Make a stable ATR ~ 1.0 USD.
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.0005, size=n)
    base = 2000.0 * np.exp(np.cumsum(rets))
    close[:] = base
    high[:] = base + 1.0
    low[:]  = base - 1.0
    open_ = np.concatenate([[base[0]], base[:-1]])

    # Engineer the entry bar (index 16) to touch both TP and SL.
    i = 16
    open_[i] = close[i - 1]
    entry = float(open_[i])
    # ATR is ~2.0 USD at this point; use TP=1.0 ATR, SL=0.5 ATR -> expect
    # tp_distance ~2.0, sl_distance ~1.0.
    high[i] = entry + 5.0   # well past TP
    low[i]  = entry - 5.0   # well past SL
    close[i] = entry + (3.0 if bar_dir > 0 else -3.0)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def test_by_close_lets_favorable_bars_take_tp():
    bars = _build_bar_with_both_hit(side=1, bar_dir=+1)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[15] = 1
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.01, atr_mult_tp=1.0, atr_mult_sl=0.5,
        same_bar_tiebreak="by_close", vol_guard_mult=0.0,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    trades = trade_log_ohlc(bt)
    assert not trades.empty
    assert trades["exit_reason"].iat[0] == "tp"


def test_sl_first_always_picks_stop():
    bars = _build_bar_with_both_hit(side=1, bar_dir=+1)
    sig = pd.Series(0, index=bars.index)
    sig.iloc[15] = 1
    cfg = OHLCBacktestConfig(
        risk_per_trade=0.01, atr_mult_tp=1.0, atr_mult_sl=0.5,
        same_bar_tiebreak="sl_first", vol_guard_mult=0.0,
    )
    bt = run_backtest_ohlc(bars, sig, cfg)
    trades = trade_log_ohlc(bt)
    assert not trades.empty
    assert trades["exit_reason"].iat[0] == "sl"
