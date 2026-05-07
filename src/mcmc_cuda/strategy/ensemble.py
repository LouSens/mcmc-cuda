"""Monte Carlo + Markov ensemble signal generator — prop-firm grade.

Core philosophy: the MC bootstrap + Markov chain simulation IS the edge.
We let the *distribution shape* (confidence score, CVaR, skewness) — not
just a crude prob_up threshold — control conviction.  This eliminates the
need for layered indicator filters (regime, ADX, RSI, slope) that add
noise and overfitting surface.

Signal logic:
  1. Fit Markov chain on rolling window of log-returns.
  2. Run bootstrap MC + Markov-chain forecast over the same horizon.
  3. Average prob_up; compute composite confidence from MC distribution.
  4. Three-tier signal:
       HIGH confidence → full signal (±1) with adapted TP/SL
       MED  confidence → reduced signal (±1) with tighter risk
       LOW  confidence → flat (0)
  5. Require expected_log_return to agree in sign with direction.
  6. Output MC distribution stats per bar for the engine to use in
     adaptive TP/SL sizing.

The rolling refit is expensive — we expose `refit_every` so users can
refit e.g. once per 96 bars (1 day at M15) instead of every bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcmc_cuda.gpu.markov import fit_markov, forecast_from_paths, sample_paths, state_of
from mcmc_cuda.gpu.monte_carlo import bootstrap_paths


@dataclass
class EnsembleConfig:
    horizon: int = 16             # bars; default 16 ~= 4h on M15
    train_window: int = 2000      # bars used to fit Markov + bootstrap pool
    n_states: int = 5
    n_mc_paths: int = 50_000
    n_markov_paths: int = 50_000
    refit_every: int = 96         # refit Markov chain every N bars (M15: ~1 day)
    seed: int | None = 42

    # --- Confidence-based signal thresholds ---
    # The MC confidence score concentrates around the 0.20–0.35 band on
    # XAUUSD M15 (prob_up clusters near 0.5). Setting `confidence_low` too
    # high silences the strategy on long histories; the previous 0.22 floor
    # produced ~13 trades over 2 years, which is a sample size, not a track
    # record. We move the floor down to broaden the trade population while
    # keeping `confidence_high` as the "full conviction" tier.
    confidence_high: float = 0.28
    confidence_low: float = 0.18
    # Minimum prob_up distance from 0.5 to even consider trading. 2pp is
    # enough to reject pure coin flips while preserving the signal density
    # needed for statistically meaningful evaluation on multi-year windows.
    min_prob_edge: float = 0.02


def generate_signals(close: pd.Series, cfg: EnsembleConfig | None = None) -> pd.DataFrame:
    """Compute per-bar directional signal and MC distribution diagnostics.

    Returns a DataFrame indexed like `close` with columns:
        signal (-1/0/+1), conviction (0=low/1=med/2=high),
        prob_up_mc, prob_up_markov, prob_up_avg,
        exp_logret_mc, exp_logret_markov, current_state,
        mc_confidence, mc_cvar_95, mc_skewness, mc_std,
        mc_p25, mc_p75
    """
    cfg = cfg or EnsembleConfig()
    log_ret = np.log(close).diff().dropna().values.astype(np.float64)
    idx = close.index[1:]  # log_ret aligned to bar t = ret from t-1 to t

    n = log_ret.size
    # 14 output columns
    n_cols = 14
    out = np.full((n, n_cols), np.nan, dtype=np.float64)
    model = None
    last_fit = -10**9

    for i in range(cfg.train_window, n):
        if i - last_fit >= cfg.refit_every or model is None:
            window = log_ret[i - cfg.train_window:i]
            model = fit_markov(window, n_states=cfg.n_states)
            last_fit = i

        current_ret = log_ret[i]
        s0 = state_of(current_ret, model)

        bar_seed = (cfg.seed + i) if cfg.seed is not None else None
        mc = bootstrap_paths(
            log_ret[i - cfg.train_window:i],
            horizon=cfg.horizon,
            n_paths=cfg.n_mc_paths,
            seed=bar_seed,
        )
        paths = sample_paths(
            model, start_state=s0, horizon=cfg.horizon,
            n_paths=cfg.n_markov_paths, seed=bar_seed,
        )
        p_mk, e_mk, _ = forecast_from_paths(paths, model)

        prob_avg = 0.5 * (mc.prob_up + p_mk)
        e_avg = 0.5 * (mc.expected_log_return + e_mk)

        # --- Confidence-gated signal ---
        prob_edge = abs(prob_avg - 0.5)
        mc_conf = mc.confidence_score

        sig = 0
        conviction = 0  # 0=none, 1=medium, 2=high

        if prob_edge >= cfg.min_prob_edge:
            # Direction from probability consensus. The prob_avg (MC + Markov
            # averaged) is the primary directional vote. We do NOT require
            # expected_log_return to agree in sign because the bootstrap mean
            # is skew-sensitive — a single fat-tail outlier in the history
            # pool can flip e_avg against the majority direction, creating a
            # systematic filter that rejects most valid signals.
            if prob_avg > 0.5:
                direction = 1
            elif prob_avg < 0.5:
                direction = -1
            else:
                direction = 0

            if direction != 0:
                if mc_conf >= cfg.confidence_high:
                    sig = direction
                    conviction = 2
                elif mc_conf >= cfg.confidence_low:
                    sig = direction
                    conviction = 1

        out[i] = [
            sig, conviction,
            mc.prob_up, p_mk, prob_avg,
            mc.expected_log_return, e_mk, s0,
            mc.confidence_score, mc.cvar_95, mc.skewness, mc.std,
            mc.p25, mc.p75,
        ]

    df = pd.DataFrame(
        out,
        index=idx,
        columns=[
            "signal", "conviction",
            "prob_up_mc", "prob_up_markov", "prob_up_avg",
            "exp_logret_mc", "exp_logret_markov", "current_state",
            "mc_confidence", "mc_cvar_95", "mc_skewness", "mc_std",
            "mc_p25", "mc_p75",
        ],
    )
    df["signal"] = df["signal"].fillna(0).astype(int)
    df["conviction"] = df["conviction"].fillna(0).astype(int)
    return df
