"""OHLC-aware backtester for XAUUSD scalping.

Key properties (vs. the previous version):
- O(n) running equity (no more `bar_pnl[:i+1].sum()` per bar).
- Costs split per bar into spread / slippage / swap.
- Session-aware spread (via the upgraded CostModel).
- Cost-aware trade gating: a trade only opens if expected favorable move
  exceeds friction by `cost.min_edge_cost_multiple`.
- Risk module integration: daily-loss / consecutive-loss cooldowns,
  per-idea risk caps, ATR-vs-friction and stop-vs-friction sanity checks.
- Optional time-stop (`time_stop_bars`) — critical for scalps.
- Optional layered entries (`max_layers`, `add_at_atr_profit`) — pyramid
  one or more extra legs when the first is in profit and risk budget left.
- Per-trade MAE/MFE captured from intra-bar high/low.
- Session label preserved on each bar and per trade.

Sizing modes (unchanged contract):
- Fixed lot:   risk_per_trade <= 0  -> hold contract_size oz when in.
- Risk-based: risk_per_trade > 0    -> size so SL hit loses risk_per_trade
                                       of current equity. Bounded by
                                       max_total_risk_per_idea across legs.

PnL is denominated directly in USD. Equity = initial + cumulative net PnL.

Entry/exit semantics:
- Signal at bar t -> entry at bar t+1's open.
- TP/SL distances are ATR-based (atr_mult_tp / atr_mult_sl).
- Within a bar, SL is checked before TP (pessimistic).
- Both entry and exit pay one half of the round-trip cost.
- Swap accrues per bar while in position.
- Time-stop: if `time_stop_bars > 0`, the trade is force-closed at the
  bar's close after holding `time_stop_bars` bars without TP/SL.

Public API (unchanged signatures + extended config):
    run_backtest_ohlc(bars, signal, cfg=None) -> DataFrame
    iter_backtest_ohlc(bars, signal, cfg=None) -> Iterator[dict]
    trade_log_ohlc(bt) -> DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.backtest.risk import RiskConfig, RiskState
from mcmc_cuda.features.strength import atr
from mcmc_cuda.strategy.sessions import label_sessions


@dataclass
class OHLCBacktestConfig:
    initial_equity: float = 10_000.0
    contract_size: float = 100.0
    risk_per_trade: float = 0.0
    max_lot_oz: float = 1e9
    atr_length: int = 14
    atr_mult_tp: float = 3.0
    atr_mult_sl: float = 1.5

    # --- scalping additions ---
    time_stop_bars: int = 0
    allowed_sessions: tuple[str, ...] = ()    # () = no session filter
    cost_gating: bool = False
    max_layers: int = 1                       # 1 = no pyramiding
    add_at_atr_profit: float = 0.5

    # Same-bar TP/SL tiebreaker:
    #   "by_close" : if both touched, the bar's direction (close vs open / vs
    #                prev_close on later bars) decides which was hit first.
    #                This is the fairest convention without tick data.
    #   "sl_first" : always assume SL was touched first (legacy / pessimistic).
    #   "tp_first" : optimistic; not recommended.
    same_bar_tiebreak: str = "by_close"

    # Break-even mover. When MFE crosses `breakeven_at_atr * ATR` in the
    # trade's favor, the SL is moved to entry_price + side * breakeven_buffer
    # (a small buffer so a re-test of entry doesn't immediately stop us out
    # at exact break-even after costs). Set breakeven_at_atr <= 0 to disable.
    breakeven_at_atr: float = 0.0
    breakeven_buffer_atr: float = 0.05

    # Trailing stop. Once armed (after MFE >= trail_arm_atr * ATR), the SL
    # follows price at distance `trail_distance_atr * ATR_at_entry`. <=0
    # disables. Trailing is mutually compatible with break-even (BE first,
    # then trail).
    trail_arm_atr: float = 0.0
    trail_distance_atr: float = 1.0

    # Volatility guard: skip entry when the entry bar's range (H-L) already
    # exceeds `vol_guard_mult * sl_distance`. Prevents dead-on-arrival trades
    # during news spikes. Set to 0.0 to disable.
    vol_guard_mult: float = 1.5

    cost: CostModel = field(default_factory=CostModel)
    risk: Optional[RiskConfig] = None

    def resolved_risk_cfg(self) -> RiskConfig:
        if self.risk is not None:
            return self.risk
        return RiskConfig(
            risk_per_trade=self.risk_per_trade,
            max_lot_oz=self.max_lot_oz,
            contract_size=self.contract_size,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _prepare(bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")
    df = bars.join(signal.rename("signal"), how="inner").dropna(
        subset=["close", "signal"]
    )
    df["signal"] = df["signal"].astype(int)
    df["atr"] = atr(df["high"], df["low"], df["close"], length=cfg.atr_length)
    df["session"] = label_sessions(df.index).values
    return df


@dataclass
class _Leg:
    entry_price: float
    size_oz: float          # signed
    risk_dollars: float


@dataclass
class _OpenIdea:
    side: int
    entry_idx: int
    entry_session: str
    sl: float
    tp: float
    entry_atr: float = float("nan")
    legs: list[_Leg] = field(default_factory=list)
    bars_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    layers_added: int = 0
    next_add_threshold: float = float("inf")
    breakeven_armed: bool = False
    trail_armed: bool = False

    @property
    def total_size(self) -> float:
        return sum(l.size_oz for l in self.legs)

    @property
    def avg_entry(self) -> float:
        s = self.total_size
        if s == 0:
            return float("nan")
        return sum(l.entry_price * l.size_oz for l in self.legs) / s


def _expected_edge_price(side: int, atr_value: float, cfg: OHLCBacktestConfig) -> float:
    if not np.isfinite(atr_value) or atr_value <= 0:
        return 0.0
    return cfg.atr_mult_tp * atr_value


def _resolve_same_bar(
    side: int,
    tp_hit: bool,
    sl_hit: bool,
    bar_direction: float,
    mode: str,
) -> str:
    """Decide which of TP/SL was hit first when both could have triggered.

    Returns "tp", "sl", or "none". `bar_direction` is `close - open` on the
    entry bar or `close - prev_close` on subsequent bars.
    """
    if not (tp_hit or sl_hit):
        return "none"
    if tp_hit and not sl_hit:
        return "tp"
    if sl_hit and not tp_hit:
        return "sl"
    if mode == "sl_first":
        return "sl"
    if mode == "tp_first":
        return "tp"
    # "by_close": favor whichever direction the bar moved.
    if (side > 0 and bar_direction > 0) or (side < 0 and bar_direction < 0):
        return "tp"
    return "sl"


# ----------------------------------------------------------------------
# Main engine
# ----------------------------------------------------------------------
def run_backtest_ohlc(
    bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig | None = None
) -> pd.DataFrame:
    cfg = cfg or OHLCBacktestConfig()
    df = _prepare(bars, signal, cfg)
    n = len(df)
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    sig = df["signal"].values
    a = df["atr"].values
    sess = df["session"].values
    times = df.index

    risk = RiskState(cfg=cfg.resolved_risk_cfg(), starting_equity=cfg.initial_equity)

    position    = np.zeros(n, dtype=np.int8)
    size_oz     = np.zeros(n)
    entry_price = np.full(n, np.nan)
    exit_price  = np.full(n, np.nan)
    exit_reason = np.array([""] * n, dtype=object)
    bar_pnl     = np.zeros(n)
    spread_cost = np.zeros(n)
    slip_cost   = np.zeros(n)
    swap_cost   = np.zeros(n)
    layer_count = np.zeros(n, dtype=np.int16)
    idea_id     = np.full(n, -1, dtype=np.int64)
    equity_arr  = np.full(n, cfg.initial_equity, dtype=float)
    drawdown    = np.zeros(n)

    cum_pnl = 0.0
    cum_cost = 0.0
    peak_equity = float(cfg.initial_equity)
    cost = cfg.cost
    allowed_sessions = set(cfg.allowed_sessions) if cfg.allowed_sessions else None

    open_idea: Optional[_OpenIdea] = None
    next_idea_id = 0

    # Diagnostic counters for funnel analysis.
    _diag_signal_bars = 0
    _diag_already_in = 0
    _diag_session_block = 0
    _diag_risk_block = 0
    _diag_stop_block = 0
    _diag_edge_block = 0
    _diag_size_zero = 0
    _diag_entries = 0

    def book_exit_costs(i_close: int, side: int, abs_size: float, sess_i: str) -> None:
        nonlocal cum_cost
        sp = cost.half_spread_price(sess_i)
        sl_p = cost.slippage_price(side)
        spread_cost[i_close] += abs_size * sp
        slip_cost[i_close]   += abs_size * sl_p
        cum_cost += abs_size * (sp + sl_p)

    def book_entry_costs(i_open: int, side: int, abs_size: float, sess_i: str) -> None:
        nonlocal cum_cost
        sp = cost.half_spread_price(sess_i)
        sl_p = cost.slippage_price(side)
        spread_cost[i_open] += abs_size * sp
        slip_cost[i_open]   += abs_size * sl_p
        cum_cost += abs_size * (sp + sl_p)

    def close_idea(i_close: int, fill: float, reason: str) -> None:
        """Close out the open idea: book exit costs, mark exit, update streak."""
        nonlocal open_idea
        assert open_idea is not None
        idea = open_idea
        sess_i = sess[i_close]
        book_exit_costs(i_close, idea.side, abs(idea.total_size), sess_i)
        exit_price[i_close]  = fill
        exit_reason[i_close] = reason
        net_for_idea = (
            bar_pnl[idea.entry_idx:i_close + 1].sum()
            - spread_cost[idea.entry_idx:i_close + 1].sum()
            - slip_cost[idea.entry_idx:i_close + 1].sum()
            - swap_cost[idea.entry_idx:i_close + 1].sum()
        )
        risk.on_idea_closed()
        risk.on_trade_closed(net_pnl=float(net_for_idea), risk_at_entry=0.0)
        open_idea = None

    for i in range(n):
        prev_close = c[i - 1] if i > 0 else c[0]
        bar_date = times[i].date() if hasattr(times[i], "date") else None
        risk.on_new_bar(bar_date, cfg.initial_equity + cum_pnl - cum_cost)

        # --------------------------------------------------------------
        # 1) If a trade is open, mark-to-market & check intrabar exits.
        # --------------------------------------------------------------
        if open_idea is not None:
            idea = open_idea
            idea.bars_held += 1
            sess_i = sess[i]
            size_total = idea.total_size  # signed

            # Per-bar swap (cost-positive convention).
            swap_per_bar = -cost.per_bar_swap_price(idea.side)
            swap_cost[i] += abs(size_total) * swap_per_bar
            cum_cost += abs(size_total) * swap_per_bar

            # MAE/MFE update.
            if idea.side > 0:
                idea.mfe_price = max(idea.mfe_price, h[i] - idea.avg_entry)
                idea.mae_price = min(idea.mae_price, l[i] - idea.avg_entry)
            else:
                idea.mfe_price = max(idea.mfe_price, idea.avg_entry - l[i])
                idea.mae_price = min(idea.mae_price, idea.avg_entry - h[i])

            # ---- Stop management: break-even, then trail ----
            if (
                cfg.breakeven_at_atr > 0
                and not idea.breakeven_armed
                and np.isfinite(idea.entry_atr)
                and idea.mfe_price >= cfg.breakeven_at_atr * idea.entry_atr
            ):
                buf = cfg.breakeven_buffer_atr * idea.entry_atr
                new_sl = idea.avg_entry + idea.side * buf
                # Only tighten the stop, never loosen.
                if (idea.side == 1 and new_sl > idea.sl) or (
                    idea.side == -1 and new_sl < idea.sl
                ):
                    idea.sl = new_sl
                idea.breakeven_armed = True

            if (
                cfg.trail_arm_atr > 0
                and np.isfinite(idea.entry_atr)
                and idea.mfe_price >= cfg.trail_arm_atr * idea.entry_atr
            ):
                trail_dist = cfg.trail_distance_atr * idea.entry_atr
                # Reference: side * (current_extreme - trail_dist).
                # Use current bar's extreme on the favorable side.
                ref = h[i] if idea.side == 1 else l[i]
                new_sl = ref - idea.side * trail_dist
                if (idea.side == 1 and new_sl > idea.sl) or (
                    idea.side == -1 and new_sl < idea.sl
                ):
                    idea.sl = new_sl
                idea.trail_armed = True

            tp = idea.tp
            sl = idea.sl
            tp_hit = (idea.side == 1 and h[i] >= tp) or (idea.side == -1 and l[i] <= tp)
            sl_hit = (idea.side == 1 and l[i] <= sl) or (idea.side == -1 and h[i] >= sl)

            position[i]    = idea.side
            size_oz[i]     = size_total
            layer_count[i] = len(idea.legs)
            idea_id[i]     = idea.entry_idx

            decision = _resolve_same_bar(
                idea.side, tp_hit, sl_hit,
                bar_direction=float(c[i] - prev_close),
                mode=cfg.same_bar_tiebreak,
            )

            if decision == "sl":
                bar_pnl[i] += size_total * (sl - prev_close)
                close_idea(i, sl, "sl")
            elif decision == "tp":
                bar_pnl[i] += size_total * (tp - prev_close)
                close_idea(i, tp, "tp")
            else:
                # MTM to close.
                bar_pnl[i] += size_total * (c[i] - prev_close)

                if cfg.time_stop_bars > 0 and idea.bars_held >= cfg.time_stop_bars:
                    # Already MTM'd to close; close at this close.
                    close_idea(i, float(c[i]), "time")
                elif (
                    open_idea is not None
                    and idea.layers_added + 1 < cfg.max_layers
                    and ((idea.side == 1 and c[i] >= idea.next_add_threshold)
                         or (idea.side == -1 and c[i] <= idea.next_add_threshold))
                ):
                    add_size = risk.size_for_entry(idea.side, float(c[i]), idea.sl)
                    if add_size != 0.0:
                        leg_risk = risk.risk_dollars(add_size, float(c[i]), idea.sl)
                        risk.on_layer_added(leg_risk)
                        idea.legs.append(_Leg(
                            entry_price=float(c[i]),
                            size_oz=add_size,
                            risk_dollars=leg_risk,
                        ))
                        idea.layers_added += 1
                        book_entry_costs(i, idea.side, abs(add_size), sess_i)
                        layer_count[i] = len(idea.legs)
                        size_oz[i] = idea.total_size
                        # Mark this bar's add: fill entry_price[i] with the
                        # leg's price so the trade log can detect the add.
                        entry_price[i] = float(c[i])
                        step = cfg.add_at_atr_profit * (a[i] if np.isfinite(a[i]) else 0.0)
                        idea.next_add_threshold = c[i] + idea.side * max(step, 0.0)

        # --------------------------------------------------------------
        # 2) Flat? Check entry from prior bar's signal.
        # --------------------------------------------------------------
        if i > 0 and sig[i - 1] != 0 and np.isfinite(a[i - 1]):
            _diag_signal_bars += 1
            if open_idea is not None:
                _diag_already_in += 1
        if open_idea is None and i > 0 and sig[i - 1] != 0 and np.isfinite(a[i - 1]):
            side = int(sig[i - 1])
            sess_i = sess[i]

            session_ok = allowed_sessions is None or sess_i in allowed_sessions
            risk_ok, risk_reason = risk.can_enter()

            if not session_ok:
                _diag_session_block += 1
            elif not risk_ok:
                _diag_risk_block += 1
            if session_ok and risk_ok:
                entry = float(o[i])
                tp_dist = cfg.atr_mult_tp * a[i - 1]
                sl_dist = cfg.atr_mult_sl * a[i - 1]
                tp = entry + side * tp_dist
                sl = entry - side * sl_dist

                stop_ok, _ = risk.stop_distance_ok(
                    stop_distance_price=sl_dist,
                    atr_value=float(a[i - 1]),
                    cost=cost,
                    session=sess_i,
                )
                edge_ok = True
                if cfg.cost_gating:
                    edge = _expected_edge_price(side, float(a[i - 1]), cfg)
                    required = cost.min_edge_required_price(
                        side=side,
                        holding_bars=max(cfg.time_stop_bars, cfg.atr_length),
                        session=sess_i,
                    )
                    edge_ok = edge >= required

                if not stop_ok:
                    _diag_stop_block += 1
                elif not edge_ok:
                    _diag_edge_block += 1

                # Volatility guard: if the entry bar's range already
                # exceeds the SL distance, the trade is dead on arrival
                # — intrabar noise will hit the SL before the directional
                # edge has any chance to play out.
                if cfg.vol_guard_mult > 0:
                    bar_range = h[i] - l[i]
                    vol_ok = bar_range < sl_dist * cfg.vol_guard_mult
                else:
                    vol_ok = True

                if stop_ok and edge_ok and vol_ok:
                    sized = risk.size_for_entry(side, entry, sl)
                    if sized == 0.0:
                        _diag_size_zero += 1
                    if sized != 0.0:
                        book_entry_costs(i, side, abs(sized), sess_i)
                        entry_price[i] = entry
                        position[i]    = side
                        size_oz[i]     = sized
                        layer_count[i] = 1
                        idea_id[i]     = next_idea_id

                        leg_risk = risk.risk_dollars(sized, entry, sl)
                        risk.on_layer_added(leg_risk)

                        open_idea = _OpenIdea(
                            side=side,
                            entry_idx=i,
                            entry_session=sess_i,
                            sl=sl,
                            tp=tp,
                            entry_atr=float(a[i - 1]),
                            legs=[_Leg(entry, sized, leg_risk)],
                            next_add_threshold=(
                                entry + side * cfg.add_at_atr_profit * float(a[i - 1])
                            ),
                        )
                        next_idea_id += 1
                        _diag_entries += 1

                        # Same-bar TP/SL from open -> close.
                        tp_hit_b = (side == 1 and h[i] >= tp) or (side == -1 and l[i] <= tp)
                        sl_hit_b = (side == 1 and l[i] <= sl) or (side == -1 and h[i] >= sl)

                        if side > 0:
                            open_idea.mfe_price = max(0.0, h[i] - entry)
                            open_idea.mae_price = min(0.0, l[i] - entry)
                        else:
                            open_idea.mfe_price = max(0.0, entry - l[i])
                            open_idea.mae_price = min(0.0, entry - h[i])

                        decision_b = _resolve_same_bar(
                            side, tp_hit_b, sl_hit_b,
                            bar_direction=float(c[i] - entry),
                            mode=cfg.same_bar_tiebreak,
                        )
                        if decision_b == "sl":
                            bar_pnl[i] += sized * (sl - entry)
                            close_idea(i, sl, "sl")
                        elif decision_b == "tp":
                            bar_pnl[i] += sized * (tp - entry)
                            close_idea(i, tp, "tp")
                        else:
                            bar_pnl[i] += sized * (c[i] - entry)

        # --------------------------------------------------------------
        # 3) Equity / drawdown bookkeeping (O(1) per bar).
        # --------------------------------------------------------------
        cum_pnl += bar_pnl[i]
        equity = cfg.initial_equity + cum_pnl - cum_cost
        equity_arr[i] = equity
        if equity > peak_equity:
            peak_equity = equity
        drawdown[i] = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

    df_out = df.copy()
    df_out["position"]    = position
    df_out["size_oz"]     = size_oz
    df_out["entry_price"] = entry_price
    df_out["exit_price"]  = exit_price
    df_out["exit_reason"] = exit_reason
    df_out["gross_pnl"]   = bar_pnl
    df_out["spread_cost"] = spread_cost
    df_out["slip_cost"]   = slip_cost
    df_out["swap_cost"]   = swap_cost
    df_out["costs"]       = spread_cost + slip_cost + swap_cost
    df_out["net_pnl"]     = bar_pnl - df_out["costs"]
    df_out["equity"]      = equity_arr
    df_out["drawdown"]    = drawdown
    df_out["layer_count"] = layer_count
    df_out["idea_id"]     = idea_id

    df_out.attrs["_diag"] = {
        "signal_bars": _diag_signal_bars,
        "already_in_trade": _diag_already_in,
        "session_blocked": _diag_session_block,
        "risk_blocked": _diag_risk_block,
        "stop_blocked": _diag_stop_block,
        "edge_blocked": _diag_edge_block,
        "size_zero": _diag_size_zero,
        "entries": _diag_entries,
    }
    return df_out


def iter_backtest_ohlc(
    bars: pd.DataFrame, signal: pd.Series, cfg: OHLCBacktestConfig | None = None
) -> Iterator[dict]:
    """Stream-friendly view of a completed run; the live UI consumes this."""
    full = run_backtest_ohlc(bars, signal, cfg)
    for ts, row in full.iterrows():
        yield {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "position": int(row["position"]),
            "size_oz": float(row["size_oz"]),
            "entry_price": float(row["entry_price"]) if not np.isnan(row["entry_price"]) else np.nan,
            "exit_price": float(row["exit_price"]) if not np.isnan(row["exit_price"]) else np.nan,
            "exit_reason": str(row["exit_reason"]),
            "equity": float(row["equity"]),
            "drawdown": float(row["drawdown"]),
        }


def trade_log_ohlc(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per *trade idea* (legs aggregated).

    Columns: entry_time, exit_time, side, session, entry_price (avg),
    exit_price, exit_reason, bars, size_oz (sum |abs|), gross_pnl,
    spread_cost, slip_cost, swap_cost, costs, net_pnl, r_multiple,
    mae_price, mfe_price, layers.
    """
    if bt.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    in_trade = False
    entry_idx: Optional[int] = None
    side = 0
    legs: list[tuple[float, float]] = []      # (entry_price, signed_size_at_leg)
    entry_session = ""

    for i in range(len(bt)):
        ep = bt["entry_price"].iat[i]
        xp = bt["exit_price"].iat[i]
        pos = int(bt["position"].iat[i])
        sz = float(bt["size_oz"].iat[i]) if "size_oz" in bt.columns else float(pos)

        if not np.isnan(ep):
            if not in_trade:
                in_trade = True
                entry_idx = i
                side = pos
                entry_session = str(bt["session"].iat[i]) if "session" in bt.columns else ""
                legs = [(float(ep), sz)]
            else:
                # Layered add: incremental size = current total - prior total.
                prev_total = sum(s for _, s in legs)
                add_sz = sz - prev_total
                if add_sz != 0.0:
                    legs.append((float(ep), add_sz))

        if in_trade and not np.isnan(xp):
            assert entry_idx is not None
            seg = bt.iloc[entry_idx:i + 1]
            total_signed = sum(s for _, s in legs)
            avg_entry = (
                sum(p * s for p, s in legs) / total_signed
                if total_signed != 0 else float("nan")
            )
            total_size = sum(abs(s) for _, s in legs)
            exit_p = float(xp)
            gross = float(seg["gross_pnl"].sum())
            sp_c = float(seg["spread_cost"].sum()) if "spread_cost" in seg else 0.0
            sl_c = float(seg["slip_cost"].sum()) if "slip_cost" in seg else 0.0
            sw_c = float(seg["swap_cost"].sum()) if "swap_cost" in seg else 0.0
            tot_c = sp_c + sl_c + sw_c
            net = gross - tot_c

            if side > 0:
                mfe = float((seg["high"] - avg_entry).max())
                mae = float((seg["low"]  - avg_entry).min())
            else:
                mfe = float((avg_entry - seg["low"]).max())
                mae = float((avg_entry - seg["high"]).min())

            r_mult = float("nan")
            try:
                first_p, first_sz = legs[0]
                if seg["exit_reason"].iat[-1] == "sl":
                    risk_unit = abs(first_sz) * abs(first_p - exit_p)
                    if risk_unit > 0:
                        r_mult = net / risk_unit
            except Exception:
                pass

            rows.append(dict(
                entry_time=bt.index[entry_idx],
                exit_time=bt.index[i],
                side=side,
                session=entry_session,
                size_oz=total_size,
                entry_price=float(avg_entry),
                exit_price=exit_p,
                exit_reason=str(bt["exit_reason"].iat[i]),
                bars=int(i - entry_idx),
                gross_pnl=gross,
                spread_cost=sp_c,
                slip_cost=sl_c,
                swap_cost=sw_c,
                costs=tot_c,
                net_pnl=net,
                r_multiple=r_mult,
                mae_price=mae,
                mfe_price=mfe,
                layers=len(legs),
            ))
            in_trade = False
            entry_idx = None
            side = 0
            legs = []

    return pd.DataFrame(rows)
