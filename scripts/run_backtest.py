"""End-to-end backtest: bars -> signal -> backtest -> metrics + chart.

Prop-firm-focused design:
  - MC + Markov ensemble is the sole signal source (no filter noise).
  - Confidence-gated entries with adaptive risk.
  - Smart artifact persistence: only saves when the run beats the leaderboard.

Examples:

    # Default prop-firm setup
    python scripts/run_backtest.py --years 2

    # Tighter risk for funded account
    python scripts/run_backtest.py --years 2 --risk-per-trade 0.003

    # London-only, cost-gated
    python scripts/run_backtest.py --years 2 --session-filter london,overlap --cost-gating

    # Force-save artifacts even if not a new best
    python scripts/run_backtest.py --years 2 --force-save

Outputs (timestamped, only if better than leaderboard) under artifacts/:
- equity_<ts>.png        static equity + drawdown
- trades_<ts>.csv        per-trade log (incl. session, R-multiple, MAE/MFE)
- metrics_<ts>.json      headline + per-session + prop-firm score
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcmc_cuda.broker import SignalExporter, SignalSnapshot
from mcmc_cuda.backtest.costs import CostModel
from mcmc_cuda.backtest.engine_ohlc import (
    OHLCBacktestConfig,
    run_backtest_ohlc,
    trade_log_ohlc,
)
from mcmc_cuda.backtest.metrics import compute as compute_metrics
from mcmc_cuda.backtest.metrics import compute_extended
from mcmc_cuda.backtest.risk import RiskConfig
from mcmc_cuda.config import ARTIFACTS_DIR, LEADERBOARD_PATH, TOP_K_ARTIFACTS
from mcmc_cuda.data.loader import load_bars
from mcmc_cuda.strategy.ensemble import EnsembleConfig, generate_signals

app = typer.Typer(add_completion=False)


# ------------------------------------------------------------------
# Smart artifact management
# ------------------------------------------------------------------
def _load_leaderboard() -> dict:
    """Load existing leaderboard or return empty template."""
    if LEADERBOARD_PATH.exists():
        try:
            return json.loads(LEADERBOARD_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {"best_score": float("-inf"), "best_timestamp": None, "runs": []}


def _save_leaderboard(lb: dict) -> None:
    LEADERBOARD_PATH.write_text(json.dumps(lb, indent=2, default=str))


def _is_new_best(score: float, lb: dict) -> bool:
    return score > lb.get("best_score", float("-inf"))


def _cleanup_old_artifacts(lb: dict) -> None:
    """Retain only the top-K runs by prop-firm score; delete everything else.

    Ranking by score (not by recency) means the artifacts/ directory is a
    curated leaderboard of the K best runs ever produced — exactly what a
    portfolio reviewer wants to see, and what an EA author wants to ship.
    """
    runs = lb.get("runs", [])
    if not runs:
        return

    ranked = sorted(
        runs,
        key=lambda r: (r.get("score", float("-inf")), r.get("timestamp", "")),
        reverse=True,
    )
    keepers = ranked[:TOP_K_ARTIFACTS]
    keep_tss = {r.get("timestamp") for r in keepers}

    for run in ranked[TOP_K_ARTIFACTS:]:
        ts = run.get("timestamp", "")
        for pattern in (f"equity_{ts}.png", f"trades_{ts}.csv", f"metrics_{ts}.json"):
            p = ARTIFACTS_DIR / pattern
            if p.exists():
                p.unlink()
                print(f"      [cleanup] deleted {p.name}")

    lb["runs"] = [r for r in runs if r.get("timestamp") in keep_tss]


def _publish_latest_signal(
    *,
    sigs,
    bt,
    symbol: str,
    timeframe: str,
    horizon: int,
    risk_per_trade: float,
    atr_mult_tp: float,
    atr_mult_sl: float,
) -> None:
    """Persist the most recent bar's signal as artifacts/live_signal.json.

    The published snapshot is what the MT5 EA reads on each tick. We use the
    last bar that produced a non-NaN ATR — anything earlier wouldn't have
    valid SL/TP distances. If no actionable signal exists, the snapshot is
    still published with side=0 so the EA can confidently flatten.
    """
    bars_with_atr = bt.dropna(subset=["atr"])
    if bars_with_atr.empty:
        print("      [signal] no bar has a valid ATR yet; skipping snapshot.")
        return

    last = bars_with_atr.iloc[-1]
    bar_time = bars_with_atr.index[-1]
    sig_row = sigs.reindex([bar_time]).iloc[0] if bar_time in sigs.index else None

    side = int(sig_row["signal"]) if sig_row is not None and not np.isnan(sig_row["signal"]) else 0
    confidence = (
        float(sig_row["mc_confidence"])
        if sig_row is not None and not np.isnan(sig_row.get("mc_confidence", np.nan))
        else 0.0
    )
    atr_value = float(last["atr"])
    close_price = float(last["close"])
    if side != 0:
        sl = close_price - side * atr_mult_sl * atr_value
        tp = close_price + side * atr_mult_tp * atr_value
    else:
        sl = float("nan")
        tp = float("nan")

    snap = SignalSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        bar_time=bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time),
        side=side,
        sl_price=sl,
        tp_price=tp,
        risk_fraction=float(risk_per_trade),
        confidence=confidence,
        horizon_bars=int(horizon),
    )
    path = SignalExporter().publish(snap)
    print(f"\n      Live signal   -> {path}  (side={side:+d}, conf={confidence:.3f})")


def _record_run(lb: dict, ts: str, score: float, metrics_dict: dict) -> bool:
    """Record a run in the leaderboard. Returns True if it's a new best."""
    run_entry = {"timestamp": ts, "score": score, "sharpe": metrics_dict.get("sharpe"),
                 "max_drawdown": metrics_dict.get("max_drawdown"),
                 "profit_factor": metrics_dict.get("profit_factor"),
                 "win_rate": metrics_dict.get("win_rate"),
                 "n_trades": metrics_dict.get("n_trades")}
    lb.setdefault("runs", []).append(run_entry)

    is_best = _is_new_best(score, lb)
    if is_best:
        lb["best_score"] = score
        lb["best_timestamp"] = ts
    return is_best


@app.command()
def main(
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    years: float = 2.0,
    # ---- MC ensemble params ----
    horizon: int = 8,
    train_window: int = 2000,
    n_states: int = 5,
    refit_every: int = 96,
    n_mc_paths: int = 50_000,
    n_markov_paths: int = 50_000,
    confidence_high: float = 0.28,
    confidence_low: float = 0.18,
    min_prob_edge: float = 0.02,
    # ---- Engine ----
    tp_sl: bool = typer.Option(True, "--tp-sl/--no-tp-sl",
                                help="Use OHLC engine with ATR-based TP/SL exits"),
    atr_mult_tp: float = 2.0,
    atr_mult_sl: float = 2.0,
    initial_equity: float = typer.Option(10_000.0, "--initial-equity"),
    risk_per_trade: float = typer.Option(0.005, "--risk-per-trade",
                                         help="Fraction of equity per trade (0.5%% default)."),
    contract_size: float = typer.Option(100.0, "--contract-size"),
    max_lot_oz: float = typer.Option(1e9, "--max-lot-oz"),
    # ---- Scalping engine additions ----
    time_stop_bars: int = typer.Option(32, "--time-stop-bars",
                                       help="Force-close after N bars (0 = off)."),
    session_filter: str = typer.Option(
        "ny", "--session-filter",
        help="Comma list of allowed sessions. Empty = no filter.",
    ),
    cost_gating: bool = typer.Option(
        True, "--cost-gating/--no-cost-gating",
        help="Skip trades whose expected move doesn't beat friction.",
    ),
    cost_profile: str = typer.Option(
        "low_friction", "--cost-profile",
        help="Friction profile: 'low_friction' (ECN/raw-spread, prop-firm "
             "default) or 'standard_retail' (conservative).",
    ),
    min_edge_mult: float = typer.Option(1.2, "--min-edge-mult"),
    max_layers: int = typer.Option(1, "--max-layers",
                                    help="Max pyramid legs per idea. 1 = no layering."),
    add_at_atr_profit: float = typer.Option(0.5, "--add-at-atr-profit"),
    same_bar_tiebreak: str = typer.Option(
        "by_close", "--same-bar-tiebreak",
        help="When SL+TP both touched on a bar: by_close|sl_first|tp_first.",
    ),
    breakeven_at_atr: float = typer.Option(
        0.0, "--breakeven-at-atr",
        help="Move SL to entry once MFE >= N*ATR. 0 disables.",
    ),
    breakeven_buffer_atr: float = typer.Option(0.1, "--breakeven-buffer-atr"),
    trail_arm_atr: float = typer.Option(
        0.0, "--trail-arm-atr",
        help="Arm trailing stop after MFE >= N*ATR. 0 disables.",
    ),
    trail_distance_atr: float = typer.Option(1.0, "--trail-distance-atr"),
    # ---- Risk ----
    max_daily_loss: float = typer.Option(0.04, "--max-daily-loss"),
    max_total_drawdown: float = typer.Option(0.12, "--max-total-drawdown"),
    max_consecutive_losses: int = typer.Option(5, "--max-consecutive-losses"),
    cooldown_bars: int = typer.Option(16, "--cooldown-bars"),
    max_total_risk_per_idea: float = typer.Option(0.01, "--max-total-risk-per-idea"),
    min_atr_to_cost_ratio: float = typer.Option(3.0, "--min-atr-to-cost-ratio"),
    min_stop_to_cost_ratio: float = typer.Option(2.0, "--min-stop-to-cost-ratio"),
    # ---- Live playback ----
    live: bool = typer.Option(False, "--live/--no-live"),
    live_speed: int = typer.Option(2, "--live-speed"),
    live_interval_ms: int = typer.Option(20, "--live-interval-ms"),
    invert_signal: bool = typer.Option(False, "--invert-signal/--no-invert-signal"),
    # ---- Artifact management ----
    force_save: bool = typer.Option(False, "--force-save/--no-force-save",
                                     help="Save artifacts even if not a new best."),
    publish_signal: bool = typer.Option(
        True, "--publish-signal/--no-publish-signal",
        help="After the backtest, publish the latest bar's signal as "
             "artifacts/live_signal.json so the MT5 EA can consume it.",
    ),
    # ---- Data ----
    csv: str | None = typer.Option(None, help="Local CSV/parquet to use instead of MT5"),
    use_mt5: bool = True,
):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365 * years))

    print(f"[1/5] Loading {symbol} {timeframe} bars ({start.date()} -> {end.date()})...")
    bars = load_bars(symbol, timeframe, start, end, csv_path=csv, use_mt5=use_mt5)
    print(f"      {len(bars):,} bars loaded.")

    # ------------------------------------------------------------------
    # 2) Signal generation — MC ensemble only (no filter noise)
    # ------------------------------------------------------------------
    print(f"[2/5] Generating MC ensemble signals (h={horizon}, train={train_window})...")
    sig_cfg = EnsembleConfig(
        horizon=horizon, train_window=train_window, n_states=n_states,
        refit_every=refit_every,
        n_mc_paths=n_mc_paths, n_markov_paths=n_markov_paths,
        confidence_high=confidence_high,
        confidence_low=confidence_low,
        min_prob_edge=min_prob_edge,
    )
    sigs = generate_signals(bars["close"], sig_cfg)
    raw_signal = sigs["signal"]

    if invert_signal:
        raw_signal = -raw_signal
        print("      Signal inverted (--invert-signal).")
    print(f"      Raw signal changes: {int((raw_signal.diff().abs() > 0).sum())}")

    # Conviction breakdown
    high_conv = int((sigs["conviction"] == 2).sum())
    med_conv = int((sigs["conviction"] == 1).sum())
    total_signals = int((raw_signal != 0).sum())
    print(f"      Conviction: {high_conv} high, {med_conv} med, {total_signals} total signals")

    # Forward-return diagnostic
    fwd_ret = bars["close"].pct_change(horizon).shift(-horizon)
    sig_active = raw_signal[raw_signal != 0]
    if len(sig_active) > 50:
        corr = float(sig_active.corr(fwd_ret.reindex(sig_active.index)))
        hit = float((np.sign(sig_active) == np.sign(fwd_ret.reindex(sig_active.index))).mean())
        diag = (
            "ANTI-EDGE — consider --invert-signal" if corr < -0.01
            else "edge OK" if corr > 0.01 else "no edge"
        )
        print(f"      Signal vs {horizon}-bar fwd return: corr={corr:+.4f}, "
              f"hit-rate={hit:.3f}  ({diag})")

    # MC confidence diagnostics
    mc_conf = sigs["mc_confidence"].dropna()
    if not mc_conf.empty:
        print(f"      MC confidence: mean={mc_conf.mean():.3f}, "
              f"median={mc_conf.median():.3f}, "
              f"std={mc_conf.std():.3f}")

    # Per-conviction hit-rate diagnostic (guides conviction filter choice).
    if len(sig_active) > 50:
        for tier, tier_name in [(2, "HIGH"), (1, "MED")]:
            mask = sigs["conviction"] == tier
            t_sig = raw_signal[mask & (raw_signal != 0)]
            t_fwd = fwd_ret.reindex(t_sig.index).dropna()
            t_sig = t_sig.reindex(t_fwd.index)
            if len(t_sig) > 20:
                t_hit = float((np.sign(t_sig) == np.sign(t_fwd)).mean())
                t_corr = float(t_sig.corr(t_fwd))
                print(f"        {tier_name} conviction: n={len(t_sig)}, "
                      f"hit={t_hit:.3f}, corr={t_corr:+.4f}")

    # Only trade HIGH conviction signals — they have a measurable directional
    # edge (~57% hit rate) while MED signals are ~51% (coin flip noise).
    signal = raw_signal.where(sigs["conviction"] >= 2, 0).astype(int)

    # ------------------------------------------------------------------
    # 3) Backtest
    # ------------------------------------------------------------------
    print("[3/5] Running backtest...")

    # Bars-per-day for swap accrual.
    bars_per_day = max(1, int(round(86400 / (
        (bars.index[-1] - bars.index[0]).total_seconds() / max(1, len(bars) - 1)
    ))))

    if cost_profile == "low_friction":
        cost = CostModel.low_friction(bars_per_day=bars_per_day)
        cost.min_edge_cost_multiple = min_edge_mult
    elif cost_profile == "standard_retail":
        cost = CostModel(
            bars_per_day=bars_per_day,
            min_edge_cost_multiple=min_edge_mult,
        )
    else:
        raise typer.BadParameter(
            f"unknown --cost-profile {cost_profile!r}; "
            "expected 'low_friction' or 'standard_retail'."
        )
    risk_cfg = RiskConfig(
        risk_per_trade=risk_per_trade,
        max_total_risk_per_idea=max_total_risk_per_idea,
        max_daily_loss=max_daily_loss,
        max_total_drawdown=max_total_drawdown,
        max_consecutive_losses=max_consecutive_losses,
        cooldown_bars=cooldown_bars,
        min_atr_to_cost_ratio=min_atr_to_cost_ratio,
        min_stop_to_cost_ratio=min_stop_to_cost_ratio,
        max_lot_oz=max_lot_oz,
        contract_size=contract_size,
    )
    allowed = tuple(s.strip() for s in session_filter.split(",") if s.strip()) if session_filter else ()
    bt_cfg = OHLCBacktestConfig(
        initial_equity=initial_equity,
        contract_size=contract_size,
        risk_per_trade=risk_per_trade,
        max_lot_oz=max_lot_oz,
        atr_mult_tp=atr_mult_tp,
        atr_mult_sl=atr_mult_sl,
        time_stop_bars=time_stop_bars,
        allowed_sessions=allowed,
        cost_gating=cost_gating,
        max_layers=max_layers,
        add_at_atr_profit=add_at_atr_profit,
        same_bar_tiebreak=same_bar_tiebreak,
        breakeven_at_atr=breakeven_at_atr,
        breakeven_buffer_atr=breakeven_buffer_atr,
        trail_arm_atr=trail_arm_atr,
        trail_distance_atr=trail_distance_atr,
        cost=cost,
        risk=risk_cfg,
    )

    # Friction diagnostic
    from mcmc_cuda.features.strength import atr as _atr
    atr_now = float(_atr(bars["high"], bars["low"], bars["close"]).iloc[-200:].median())
    sl_pts = atr_mult_sl * atr_now
    rt_cost_pts = cost.round_trip_cost_price("overlap")
    friction_pct_sl = rt_cost_pts / sl_pts if sl_pts > 0 else float("inf")
    print(f"      [friction] ATR(median, last 200) = {atr_now:.2f}  "
          f"SL = {sl_pts:.2f}  RT cost (overlap) = {rt_cost_pts:.2f}  "
          f"friction/SL = {friction_pct_sl:.0%}")
    if friction_pct_sl > 0.5:
        print(f"      [WARN] friction is {friction_pct_sl:.0%} of SL distance — "
              f"this timeframe is cost-dominated.")

    if live:
        from mcmc_cuda.ui.live_chart import LiveChartConfig, play_live
        print("      [live] opening playback window — close it to continue.")
        live_cfg = LiveChartConfig(
            interval_ms=live_interval_ms,
            bars_per_frame=live_speed,
            title=f"{symbol} {timeframe} | risk={risk_per_trade*100:.2f}% | "
                  f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR | "
                  f"start ${initial_equity:,.0f}",
        )
        bt = play_live(bars[["open", "high", "low", "close"]], signal, bt_cfg, live_cfg)
    else:
        bt = run_backtest_ohlc(bars[["open", "high", "low", "close"]], signal, bt_cfg)

    trades = trade_log_ohlc(bt)
    metrics = compute_metrics(bt, trades)
    extended = compute_extended(bt, trades)

    # Entry funnel diagnostics
    diag = bt.attrs.get("_diag", {})
    if diag:
        print(f"      [funnel] signal_bars={diag['signal_bars']}  "
              f"already_in={diag['already_in_trade']}  "
              f"session={diag['session_blocked']}  "
              f"risk={diag['risk_blocked']}  "
              f"stop={diag['stop_blocked']}  "
              f"edge={diag['edge_blocked']}  "
              f"size0={diag['size_zero']}  "
              f"entries={diag['entries']}")

    # Exit reason breakdown
    if not trades.empty and "exit_reason" in trades.columns:
        reason_counts = trades["exit_reason"].value_counts().to_dict()
        same_bar_pct = float((trades["bars"] == 0).mean()) if "bars" in trades.columns else 0.0
        print(f"      [exits] reasons={reason_counts}  same-bar exits={same_bar_pct:.1%}")

    # ------------------------------------------------------------------
    # 4) Prop Firm Score
    # ------------------------------------------------------------------
    pf_score = extended.propfirm_score
    print(f"\n[4/5] Prop Firm Score: {pf_score:+.4f}")
    if pf_score > 0.5:
        print("      [PASS] PASSING range -- review equity curve for consistency")
    elif pf_score > 0.0:
        print("      [~] MARGINAL -- needs improvement but showing promise")
    else:
        print("      [FAIL] FAILING -- strategy is losing money or drawdown too deep")

    # ------------------------------------------------------------------
    # 5) Smart Artifact Management
    # ------------------------------------------------------------------
    print("[5/5] Checking leaderboard...")
    lb = _load_leaderboard()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dict = extended.to_dict()
    is_best = _record_run(lb, ts, pf_score, metrics_dict)

    should_save = is_best or force_save

    if is_best:
        print(f"      [*] NEW BEST! Score {pf_score:+.4f} beats previous "
              f"{lb.get('best_score', float('-inf')):+.4f}")
    else:
        prev_best = lb.get("best_score", float("-inf"))
        print(f"      Score {pf_score:+.4f} does not beat best {prev_best:+.4f}")
        if force_save:
            print("      --force-save: saving anyway.")
        else:
            print("      Skipping artifact save (use --force-save to override).")

    if should_save:
        eq_path = ARTIFACTS_DIR / f"equity_{ts}.png"
        trades_path = ARTIFACTS_DIR / f"trades_{ts}.csv"
        metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(bt.index, bt["equity"])
        ax[0].set_ylabel("Equity (USD)")
        title_bits = [
            f"{symbol} {timeframe}", f"h={horizon}",
            f"TP/SL={atr_mult_tp}/{atr_mult_sl} ATR",
            f"risk={risk_per_trade*100:.2f}%/trade",
        ]
        if time_stop_bars > 0:
            title_bits.append(f"ts={time_stop_bars}b")
        if session_filter:
            title_bits.append(f"sess={session_filter}")
        title_bits.append(f"PF-score={pf_score:+.3f}")
        ax[0].set_title(" | ".join(title_bits))
        ax[0].grid(alpha=0.3)
        ax[1].fill_between(bt.index, bt["drawdown"] * 100, 0, color="tab:red", alpha=0.4)
        ax[1].set_ylabel("Drawdown (%)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(eq_path, dpi=110)
        plt.close(fig)

        trades.to_csv(trades_path, index=False)
        metrics_path.write_text(json.dumps(metrics_dict, indent=2, default=str))

        print(f"\n      Equity curve  -> {eq_path}")
        print(f"      Trade log     -> {trades_path}")
        print(f"      Metrics JSON  -> {metrics_path}")

    # Cleanup old artifacts
    _cleanup_old_artifacts(lb)
    _save_leaderboard(lb)

    # ------------------------------------------------------------------
    # 6) EA handoff: publish the most recent bar's signal snapshot.
    # ------------------------------------------------------------------
    if publish_signal:
        _publish_latest_signal(
            sigs=sigs, bt=bt, symbol=symbol, timeframe=timeframe,
            horizon=horizon, risk_per_trade=risk_per_trade,
            atr_mult_tp=atr_mult_tp, atr_mult_sl=atr_mult_sl,
        )

    # Always print metrics to stdout
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(metrics_dict, indent=2, default=str))


if __name__ == "__main__":
    app()
