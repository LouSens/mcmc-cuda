"""Risk module: sizing, daily loss, consecutive-loss cooldown."""
from __future__ import annotations

from datetime import date, timedelta

import math

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.backtest.risk import RiskConfig, RiskState


def test_size_for_entry_respects_per_trade_risk():
    rs = RiskState(cfg=RiskConfig(risk_per_trade=0.01), starting_equity=10_000.0)
    rs.on_new_bar(date(2024, 6, 3), 10_000.0)
    # SL 1.0 USD away, 1% of 10k = $100, => 100 oz exposure.
    sz = rs.size_for_entry(side=1, entry_price=2000.0, stop_price=1999.0)
    assert math.isclose(sz, 100.0, rel_tol=1e-9)


def test_size_for_entry_capped_by_idea_budget():
    rs = RiskState(
        cfg=RiskConfig(risk_per_trade=0.01, max_total_risk_per_idea=0.015),
        starting_equity=10_000.0,
    )
    rs.on_new_bar(date(2024, 6, 3), 10_000.0)
    sz1 = rs.size_for_entry(side=1, entry_price=2000.0, stop_price=1999.0)
    rs.on_layer_added(rs.risk_dollars(sz1, 2000.0, 1999.0))
    # First leg used $100 of $150 idea cap; only $50 budget left -> 50 oz.
    sz2 = rs.size_for_entry(side=1, entry_price=2000.0, stop_price=1999.0)
    assert math.isclose(sz2, 50.0, rel_tol=1e-9)


def test_can_enter_blocks_after_daily_loss_cap():
    # Loose total-DD cap so the daily-loss rule is the predicate under test;
    # otherwise the total-drawdown breaker can fire first and mask it.
    rs = RiskState(
        cfg=RiskConfig(max_daily_loss=0.03, max_total_drawdown=0.50),
        starting_equity=10_000.0,
    )
    rs.on_new_bar(date(2024, 6, 3), 9_500.0)  # -5% intraday
    ok, reason = rs.can_enter()
    assert not ok and reason == "daily_loss_cap"


def test_total_drawdown_breach_is_sticky():
    # Once total DD is breached, can_enter must keep blocking even if
    # equity recovers above the breach threshold (prop-firm rule).
    rs = RiskState(cfg=RiskConfig(max_total_drawdown=0.05), starting_equity=10_000.0)
    rs.on_new_bar(date(2024, 6, 3), 10_000.0)
    rs.on_new_bar(date(2024, 6, 3), 9_400.0)  # -6% from peak -> breach
    ok, reason = rs.can_enter()
    assert not ok and reason == "total_drawdown_breached"
    rs.on_new_bar(date(2024, 6, 4), 10_500.0)  # full recovery
    ok, reason = rs.can_enter()
    assert not ok and reason == "total_drawdown_breached"


def test_consecutive_losses_trigger_cooldown():
    cfg = RiskConfig(max_consecutive_losses=3, cooldown_bars=5)
    rs = RiskState(cfg=cfg, starting_equity=10_000.0)
    rs.on_new_bar(date(2024, 6, 3), 10_000.0)
    for _ in range(3):
        rs.on_trade_closed(net_pnl=-50.0, risk_at_entry=0.0)
    ok, reason = rs.can_enter()
    assert not ok and reason == "cooldown"
    # Five subsequent bars should clear it.
    for _ in range(5):
        rs.on_new_bar(date(2024, 6, 3), 9_900.0)
    ok, _ = rs.can_enter()
    assert ok


def test_stop_distance_ok_rejects_tight_stops():
    rs = RiskState(
        cfg=RiskConfig(min_stop_to_cost_ratio=2.0, min_atr_to_cost_ratio=3.0),
        starting_equity=10_000.0,
    )
    cm = CostModel()
    rt = cm.round_trip_cost_price("london")
    # Stop just barely above friction => fail.
    ok, _ = rs.stop_distance_ok(stop_distance_price=rt * 1.5,
                                atr_value=rt * 5.0, cost=cm, session="london")
    assert not ok
    # Healthy stop & ATR => pass.
    ok, _ = rs.stop_distance_ok(stop_distance_price=rt * 5.0,
                                atr_value=rt * 5.0, cost=cm, session="london")
    assert ok


def test_fixed_lot_path_when_risk_zero():
    rs = RiskState(cfg=RiskConfig(risk_per_trade=0.0, contract_size=100.0), starting_equity=10_000.0)
    rs.on_new_bar(date(2024, 6, 3), 10_000.0)
    sz = rs.size_for_entry(side=-1, entry_price=2000.0, stop_price=2001.0)
    assert sz == -100.0
