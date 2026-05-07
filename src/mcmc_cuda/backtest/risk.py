"""Risk-management primitives independent of the engine internals.

The engine consults a `RiskState` once per bar / per candidate entry to
decide:
- am I over the daily loss cap?
- am I in a consecutive-loss cooldown?
- does this trade idea push total open risk above the per-idea cap?
- is the stop too tight relative to friction?

By keeping these checks out of the inner loop's PnL accounting we keep the
engine simple and the rules unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from mcmc_cuda.backtest.costs import CostModel


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005           # 0.5% of equity per trade (prop-firm safe)
    max_total_risk_per_idea: float = 0.01   # cap on summed open risk for layered idea
    max_daily_loss: float = 0.02            # halt new entries when day P&L <= -2%
    max_total_drawdown: float = 0.05        # hard circuit breaker: stop trading if
                                            # equity drops 5% from peak (prop firm rule)
    max_consecutive_losses: int = 5         # 38% WR → 5-loss streaks are routine
    cooldown_bars: int = 16                 # 4 hours on M15; enough to reset, not kill the day
    min_atr_to_cost_ratio: float = 3.0      # ATR must be >= 3x round-trip cost
    min_stop_to_cost_ratio: float = 2.0     # stop distance >= 2x round-trip cost
    max_lot_oz: float = 1e9
    contract_size: float = 100.0            # used when risk_per_trade <= 0


@dataclass
class RiskState:
    """Mutable state the engine threads through bar by bar."""
    cfg: RiskConfig
    starting_equity: float
    current_equity: float = 0.0
    day_start_equity: float = 0.0
    peak_equity: float = 0.0                # tracks highest equity for DD calc
    current_day: date | None = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    open_idea_risk: float = 0.0             # absolute USD currently at risk on the open idea
    total_dd_breached: bool = False         # once True, stays True (prop firm = game over)

    def __post_init__(self) -> None:
        self.current_equity = self.current_equity or self.starting_equity
        self.day_start_equity = self.day_start_equity or self.starting_equity
        self.peak_equity = self.peak_equity or self.starting_equity

    # ------------------------------------------------------------------
    # Per-bar bookkeeping
    # ------------------------------------------------------------------
    def on_new_bar(self, bar_date: date, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        # Total drawdown circuit breaker — once tripped, stays tripped.
        if not self.total_dd_breached and self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd >= self.cfg.max_total_drawdown:
                self.total_dd_breached = True
        if self.current_day is None:
            # First bar of the run: anchor the day to the starting equity.
            self.current_day = bar_date
        elif self.current_day != bar_date:
            self.current_day = bar_date
            self.day_start_equity = equity
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def on_trade_closed(self, net_pnl: float, risk_at_entry: float) -> None:
        """Update streak and idea-risk counters when a leg of a trade closes."""
        self.open_idea_risk = max(0.0, self.open_idea_risk - max(0.0, risk_at_entry))
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.cooldown_remaining = self.cfg.cooldown_bars
                self.consecutive_losses = 0
        elif net_pnl > 0:
            self.consecutive_losses = 0

    def on_layer_added(self, risk_added: float) -> None:
        self.open_idea_risk += max(0.0, risk_added)

    def on_idea_closed(self) -> None:
        self.open_idea_risk = 0.0

    # ------------------------------------------------------------------
    # Gating predicates
    # ------------------------------------------------------------------
    def can_enter(self) -> tuple[bool, str]:
        """Top-level gate: independent of trade specifics."""
        if self.total_dd_breached:
            return False, "total_drawdown_breached"
        if self.cooldown_remaining > 0:
            return False, "cooldown"
        if self._daily_loss_breached():
            return False, "daily_loss_cap"
        return True, "ok"

    def _daily_loss_breached(self) -> bool:
        if self.day_start_equity <= 0:
            return True
        loss = (self.current_equity - self.day_start_equity) / self.day_start_equity
        return loss <= -abs(self.cfg.max_daily_loss)

    def stop_distance_ok(
        self,
        stop_distance_price: float,
        atr_value: float,
        cost: CostModel,
        session: str | None,
    ) -> tuple[bool, str]:
        """Filter trades whose stop is too tight to survive friction."""
        if not np.isfinite(stop_distance_price) or stop_distance_price <= 0:
            return False, "bad_stop"
        rt_cost = cost.round_trip_cost_price(session)
        if rt_cost > 0 and stop_distance_price < self.cfg.min_stop_to_cost_ratio * rt_cost:
            return False, "stop_below_friction"
        if (
            np.isfinite(atr_value)
            and rt_cost > 0
            and atr_value < self.cfg.min_atr_to_cost_ratio * rt_cost
        ):
            return False, "atr_below_friction"
        return True, "ok"

    def size_for_entry(
        self,
        side: int,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Signed oz position size, capped by per-trade and per-idea risk.

        Falls back to fixed `contract_size` lot when `risk_per_trade <= 0`,
        preserving the original engine's "fixed lot" behavior.
        """
        if side == 0:
            return 0.0
        if self.cfg.risk_per_trade <= 0:
            return float(side) * self.cfg.contract_size

        sl_dist = abs(entry_price - stop_price)
        if sl_dist <= 0 or not np.isfinite(sl_dist):
            return 0.0

        # Per-trade risk budget.
        trade_budget = self.cfg.risk_per_trade * max(self.current_equity, 0.0)

        # Per-idea cap: cannot push open_idea_risk above max_total_risk_per_idea.
        idea_budget_left = (
            self.cfg.max_total_risk_per_idea * max(self.current_equity, 0.0)
            - self.open_idea_risk
        )
        budget = max(0.0, min(trade_budget, idea_budget_left))
        if budget <= 0:
            return 0.0

        size = budget / sl_dist
        size = min(size, self.cfg.max_lot_oz)
        return float(side) * size

    def risk_dollars(self, size_oz: float, entry_price: float, stop_price: float) -> float:
        return abs(size_oz) * abs(entry_price - stop_price)
