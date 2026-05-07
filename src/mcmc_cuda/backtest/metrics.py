"""Performance metrics for a completed backtest.

Annualization assumes the bar-return series, not the trade-level series, so
metrics like Sharpe stay comparable across timeframes / strategies.

Two layers of API:
- `compute(bt, trades)`             : headline metrics (back-compat)
- `compute_extended(bt, trades, ...)`: headline + per-session, per-regime,
                                       costs/gross, R-multiple stats
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_bars_in_trade: float
    bankrupt: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtendedMetrics:
    headline: Metrics
    costs_pct_gross: float = float("nan")
    avg_r_multiple: float = float("nan")
    median_r_multiple: float = float("nan")
    by_session: dict[str, dict] = field(default_factory=dict)
    by_regime: dict[str, dict] = field(default_factory=dict)
    trade_frequency_per_day: float = float("nan")
    propfirm_score: float = float("nan")

    def to_dict(self) -> dict:
        return {
            **self.headline.to_dict(),
            "costs_pct_gross": self.costs_pct_gross,
            "avg_r_multiple": self.avg_r_multiple,
            "median_r_multiple": self.median_r_multiple,
            "trade_frequency_per_day": self.trade_frequency_per_day,
            "propfirm_score": self.propfirm_score,
            "by_session": self.by_session,
            "by_regime": self.by_regime,
        }


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    median_spacing = pd.Series(index).diff().median()
    if pd.isna(median_spacing) or median_spacing.total_seconds() <= 0:
        return 1.0
    bars_per_year = pd.Timedelta(days=365).total_seconds() / median_spacing.total_seconds()
    return float(bars_per_year)


def compute(bt: pd.DataFrame, trades: pd.DataFrame) -> Metrics:
    eq = bt["equity"]
    rets = eq.pct_change().fillna(0.0)

    ann = _annualization_factor(bt.index)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((bt.index[-1] - bt.index[0]).total_seconds() / (365 * 86400), 1e-9)
    if eq.iloc[-1] > 0 and eq.iloc[0] > 0:
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    std = rets.std(ddof=0)
    sharpe = float(np.sqrt(ann) * rets.mean() / std) if std > 0 else 0.0

    downside = rets[rets < 0]
    dstd = downside.std(ddof=0) if not downside.empty else 0.0
    sortino = float(np.sqrt(ann) * rets.mean() / dstd) if dstd > 0 else 0.0

    mdd = float(bt["drawdown"].min())
    if mdd < 0 and not np.isnan(cagr):
        calmar = float(cagr / abs(mdd))
    else:
        calmar = float("nan")
    bankrupt = bool((eq <= 0).any())

    n = len(trades)
    if n == 0:
        return Metrics(
            total_return, cagr, sharpe, sortino, mdd, calmar,
            0, 0.0, 0.0, 0.0, 0.0, bankrupt,
        )

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    win_rate = float(len(wins) / n)
    gross_win = float(wins["net_pnl"].sum())
    gross_loss = float(-losses["net_pnl"].sum())
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = float(trades["net_pnl"].mean())
    avg_bars = float(trades["bars"].mean())

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        n_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_bars_in_trade=avg_bars,
        bankrupt=bankrupt,
    )


def _trade_summary(trades: pd.DataFrame) -> dict:
    """Reusable per-bucket trade summary (n, win_rate, expectancy, pf, net_pnl)."""
    if trades.empty:
        return dict(
            n=0, net_pnl=0.0, win_rate=0.0,
            expectancy=0.0, profit_factor=float("nan"),
            avg_bars=0.0,
        )
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    gw = float(wins["net_pnl"].sum())
    gl = float(-losses["net_pnl"].sum())
    return dict(
        n=int(len(trades)),
        net_pnl=float(trades["net_pnl"].sum()),
        win_rate=float(len(wins) / len(trades)),
        expectancy=float(trades["net_pnl"].mean()),
        profit_factor=float(gw / gl) if gl > 0 else float("inf"),
        avg_bars=float(trades["bars"].mean()),
    )


def compute_propfirm_score(m: Metrics) -> float:
    """Composite score for ranking backtest runs by prop-firm suitability.

    Weights:
      Sharpe      (0.30) — risk-adjusted return, the gold standard
      Calmar      (0.20) — return relative to max drawdown
      PF          (0.20) — gross win / gross loss
      MaxDD pen   (0.20) — penalty for deep drawdowns (prop-firm killer)
      WinRate adj (0.10) — bonus for > 50% win rate (psychological edge)

    Returns a single float; higher = better.  Negative scores are common
    for losing strategies.  A passable prop-firm run typically scores > 0.5.
    """
    sharpe_component = min(max(m.sharpe, -2.0), 3.0) * 0.30

    calmar_val = m.calmar if np.isfinite(m.calmar) else 0.0
    calmar_component = min(max(calmar_val, -2.0), 5.0) * 0.20

    pf_val = min(m.profit_factor, 3.0) if np.isfinite(m.profit_factor) else 0.0
    pf_component = (pf_val - 1.0) * 0.20  # centered at 1.0 (breakeven)

    # MaxDD penalty: linearly penalize drawdowns beyond -2%.  At -10% the
    # penalty is max (-0.20).  This is the most important guard.
    dd_pct = abs(m.max_drawdown)  # 0.069 for 6.9% DD
    dd_penalty = -min(max(dd_pct - 0.02, 0.0), 0.10) * 2.0 * 0.20

    wr_bonus = (m.win_rate - 0.50) * 0.10

    score = sharpe_component + calmar_component + pf_component + dd_penalty + wr_bonus

    # Bankrupt strategies get a massive penalty.
    if m.bankrupt:
        score -= 5.0

    return float(score)


def compute_extended(
    bt: pd.DataFrame,
    trades: pd.DataFrame,
    regime: Optional[pd.Series] = None,
) -> ExtendedMetrics:
    """Headline metrics + per-session / per-regime breakdowns.

    `regime` (optional) is a per-bar label aligned to bt.index; trade buckets
    are assigned by the regime label at trade entry time.
    """
    headline = compute(bt, trades)

    gross = float(bt["gross_pnl"].abs().sum())
    costs = float(bt["costs"].sum()) if "costs" in bt.columns else 0.0
    costs_pct = costs / gross if gross > 0 else float("nan")

    avg_r = float("nan")
    med_r = float("nan")
    if not trades.empty and "r_multiple" in trades.columns:
        r_clean = trades["r_multiple"].replace([np.inf, -np.inf], np.nan).dropna()
        if not r_clean.empty:
            avg_r = float(r_clean.mean())
            med_r = float(r_clean.median())

    days = max((bt.index[-1] - bt.index[0]).total_seconds() / 86400, 1e-9)
    freq_per_day = float(len(trades) / days) if not trades.empty else 0.0

    by_session: dict[str, dict] = {}
    if not trades.empty and "session" in trades.columns:
        for sess, group in trades.groupby("session"):
            by_session[str(sess)] = _trade_summary(group)

    by_regime: dict[str, dict] = {}
    if regime is not None and not trades.empty:
        regime_at_entry = regime.reindex(trades["entry_time"]).values
        tr = trades.assign(_regime=regime_at_entry)
        for label, group in tr.groupby("_regime"):
            by_regime[str(label)] = _trade_summary(group.drop(columns=["_regime"]))

    pf_score = compute_propfirm_score(headline)

    return ExtendedMetrics(
        headline=headline,
        costs_pct_gross=costs_pct,
        avg_r_multiple=avg_r,
        median_r_multiple=med_r,
        trade_frequency_per_day=freq_per_day,
        propfirm_score=pf_score,
        by_session=by_session,
        by_regime=by_regime,
    )
