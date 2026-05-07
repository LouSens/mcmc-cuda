"""XAUUSD-aware cost model.

XAUUSD specifics that matter for honest backtesting:
- Quoted with 2 decimal digits (point = 0.01); a "pip" colloquially is 0.10.
- Typical retail demo spread: 15-50 points (0.15 - 0.50 USD per oz).
- Contract size: 100 oz per standard lot.
- Swap (overnight rollover) is non-trivial for gold and asymmetric long vs
  short. We treat it per-bar so a multi-bar holding incurs proportional cost.
- Slippage modeled as a fixed-points adverse fill on entry+exit.
- Spread is *strongly session-dependent*: Asia and the London open can
  display 2-3x the overlap spread; we model that with `spread_by_session`.

Costs are quoted in *price units* (USD per oz), not in account currency,
so the engine stays currency-agnostic until lot-sizing.

Public methods are designed for cost-aware trade gating:
- entry_cost_price / exit_cost_price : per-side at fill
- round_trip_cost_price              : total transaction friction
- total_expected_trade_cost_price    : transaction + expected swap
- min_edge_required_price            : what the move must beat to be worth taking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Default session spread profile. Sourced from typical retail XAUUSD broker
# behavior — not authoritative, but realistic and tunable.
DEFAULT_SPREAD_BY_SESSION: dict[str, float] = {
    "overlap": 18.0,    # tightest
    "london":  22.0,
    "ny":      25.0,
    "asia":    45.0,    # wide, illiquid
    "dead":    50.0,    # 21:00 UTC pre-roll
}

# Low-friction profile: tight ECN/raw-spread broker, low commission. Used by
# default in scripts/run_backtest.py so M15 strategies aren't strangled by
# defaults calibrated for retail standard accounts.
LOW_FRICTION_SPREAD_BY_SESSION: dict[str, float] = {
    "overlap":  6.0,
    "london":   8.0,
    "ny":      10.0,
    "asia":    18.0,
    "dead":    22.0,
}


@dataclass
class CostModel:
    spread_points: float = 25.0       # fallback when session unknown / disabled
    slippage_points: float = 5.0      # adverse fill, per side, both directions
    slippage_long_extra: float = 0.0  # asymmetric slippage if your fills suggest it
    slippage_short_extra: float = 0.0
    point: float = 0.01               # XAUUSD point size in price units
    swap_long_per_day: float = -7.0   # USD per lot per day, broker-dependent
    swap_short_per_day: float = 2.5
    bars_per_day: int = 96            # default M15 (96 bars/day)
    spread_by_session: Mapping[str, float] | None = field(
        default_factory=lambda: dict(DEFAULT_SPREAD_BY_SESSION)
    )
    use_session_spread: bool = True
    # Edge threshold: a trade is only allowed if expected gross move >=
    # `min_edge_cost_multiple` * round-trip cost. 1.5 means we want the
    # expected favorable move to be at least 1.5x the friction.
    min_edge_cost_multiple: float = 1.5

    # ------------------------------------------------------------------
    # Spread / slippage helpers (price units)
    # ------------------------------------------------------------------
    def spread_points_for(self, session: str | None) -> float:
        if (
            self.use_session_spread
            and session is not None
            and self.spread_by_session is not None
            and session in self.spread_by_session
        ):
            return float(self.spread_by_session[session])
        return float(self.spread_points)

    def half_spread_price(self, session: str | None = None) -> float:
        return self.spread_points_for(session) * 0.5 * self.point

    def slippage_price(self, side: int) -> float:
        extra = self.slippage_long_extra if side > 0 else self.slippage_short_extra
        return (self.slippage_points + extra) * self.point

    # ------------------------------------------------------------------
    # Per-side costs at a fill (price units, per oz)
    # ------------------------------------------------------------------
    def entry_cost_price(self, side: int, session: str | None = None) -> float:
        """Cost (price units, per oz) booked at entry for a given side."""
        if side == 0:
            return 0.0
        return self.half_spread_price(session) + self.slippage_price(side)

    def exit_cost_price(self, side: int, session: str | None = None) -> float:
        """Cost (price units, per oz) booked at exit."""
        if side == 0:
            return 0.0
        return self.half_spread_price(session) + self.slippage_price(side)

    # ------------------------------------------------------------------
    # Aggregate views
    # ------------------------------------------------------------------
    def round_trip_cost_price(self, session: str | None = None) -> float:
        """Total entry+exit transaction cost in price units (per oz).

        If `session` is None, falls back to the static spread + 2*slippage.
        """
        return self.entry_cost_price(1, session) + self.exit_cost_price(1, session)

    def per_bar_swap_price(self, side: int) -> float:
        """Per-bar swap charge in price units (per oz). side=+1 long, -1 short.

        Daily swap is distributed evenly across bars in the day so equity
        evolves smoothly instead of stepping at rollover.
        """
        if side == 0:
            return 0.0
        daily = self.swap_long_per_day if side == 1 else self.swap_short_per_day
        return daily / self.bars_per_day / 100.0  # per-oz: lot is 100 oz

    def total_expected_trade_cost_price(
        self,
        side: int,
        holding_bars: int,
        session: str | None = None,
    ) -> float:
        """Total expected trade cost (price units, per oz) for a planned
        entry of `side` held for `holding_bars` bars.

        Includes round-trip transaction cost and expected swap accrual.
        Swap sign convention: positive number = cost to us (we negate the
        broker's positive credit so callers see a single "total cost" they
        can subtract from a planned move).
        """
        rt = self.round_trip_cost_price(session)
        swap_per_bar = -self.per_bar_swap_price(side)  # cost-positive convention
        return rt + max(0.0, swap_per_bar) * max(0, int(holding_bars))

    def min_edge_required_price(
        self,
        side: int,
        holding_bars: int,
        session: str | None = None,
    ) -> float:
        """The minimum expected price move (price units) for a trade to be
        worth taking under the cost-multiple threshold. Compare this against
        the strategy's expected favorable excursion (e.g. atr_mult_tp * ATR
        adjusted by p_win) before taking the trade.
        """
        return self.min_edge_cost_multiple * self.total_expected_trade_cost_price(
            side, holding_bars, session
        )

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    @classmethod
    def low_friction(cls, bars_per_day: int = 96) -> "CostModel":
        """Tight-spread, low-slippage, low-swap broker profile.

        Calibrated for an ECN / raw-spread XAUUSD account suitable for
        algorithmic M15 trading. This is the default the production
        backtest script uses; the dataclass defaults remain a more
        conservative retail profile so unit tests stay deterministic.
        """
        return cls(
            spread_points=10.0,
            slippage_points=1.5,
            swap_long_per_day=-2.5,
            swap_short_per_day=0.8,
            bars_per_day=bars_per_day,
            spread_by_session=dict(LOW_FRICTION_SPREAD_BY_SESSION),
            min_edge_cost_multiple=1.2,
        )
