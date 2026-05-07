"""Monte Carlo path simulator on GPU (CuPy).

Two simulators:
- `gbm_paths`: geometric Brownian motion baseline (drift mu, vol sigma).
  Closed-form vectorized — fast, useful as a sanity baseline.
- `bootstrap_paths`: empirical bootstrap of historical log-returns —
  no distributional assumption, captures fat tails and serial-correlation-
  free properties of the realized return distribution.

Both return arrays of shape (n_paths, horizon) of *log-returns* (not prices).
The strategy layer aggregates these into directional probability and
expected-return forecasts.

MCResult now carries full distribution diagnostics: CVaR, skewness,
kurtosis, and a composite confidence score.  The ensemble uses these to
gate signals by *conviction quality*, not just prob_up > threshold.

VRAM accounting: float32 paths cost 4 bytes per cell. On the RTX 4050
(~6 GB), a (n_paths=200_000, horizon=64) buffer is ~50 MB — fine. If the
caller asks for something that exceeds free VRAM, we chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_cuda.gpu.device import free_vram_bytes, get_xp, has_cuda


@dataclass
class MCResult:
    """Aggregate forecasts from a path simulation.

    Extended with distribution shape metrics for prop-firm-grade
    confidence scoring.  The ``confidence_score`` is a [0, 1] composite
    that rises when (a) prob_up is far from 0.5, (b) tail risk (CVaR)
    is moderate, and (c) the distribution isn't pathologically skewed
    against the predicted direction.
    """

    prob_up: float           # P(return at horizon > 0)
    expected_log_return: float
    p05: float               # 5th percentile of horizon log-return
    p95: float
    n_paths: int

    # --- distribution diagnostics (new) ---
    cvar_95: float = 0.0     # Conditional VaR: mean of worst 5% of paths
    skewness: float = 0.0    # distribution skew (negative = crash risk)
    kurtosis: float = 0.0    # excess kurtosis (fat tails)
    std: float = 0.0         # path return standard deviation
    p25: float = 0.0         # 25th percentile
    p75: float = 0.0         # 75th percentile
    confidence_score: float = 0.0  # composite [0, 1] conviction metric


def _compute_distribution_stats(flat: np.ndarray, prob_up: float) -> dict:
    """Compute rich distribution stats from the flat array of horizon log-returns."""
    if flat.size < 10:
        return dict(cvar_95=0.0, skewness=0.0, kurtosis=0.0, std=0.0,
                    p25=0.0, p75=0.0, confidence_score=0.0)

    std = float(np.std(flat))
    p05 = float(np.quantile(flat, 0.05))
    p25 = float(np.quantile(flat, 0.25))
    p75 = float(np.quantile(flat, 0.75))

    # CVaR (Conditional Value at Risk) — mean of the worst 5% of outcomes.
    # This is the tail-risk metric prop firms care most about.
    worst_5pct = flat[flat <= p05]
    cvar_95 = float(worst_5pct.mean()) if worst_5pct.size > 0 else p05

    # Skewness and excess kurtosis via numpy.
    if std > 1e-15:
        centered = flat - flat.mean()
        skewness = float(np.mean(centered ** 3) / (std ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    # --- Composite confidence score ---
    # Three factors, each in [0, 1]:
    #
    # 1) Directional clarity: how far prob_up is from 0.5 (coin flip).
    #    |prob_up - 0.5| / 0.5 → 0 at 50%, 1 at 0% or 100%.
    dir_clarity = min(1.0, abs(prob_up - 0.5) / 0.4)

    # 2) Tail safety: CVaR shouldn't be catastrophic.  We map the ratio
    #    |cvar| / std into [0, 1] where lower |cvar|/std is better.
    #    Typical CVaR/std ~ 2.0 for Gaussian; worse is > 3.
    cvar_ratio = abs(cvar_95) / max(std, 1e-15)
    tail_safety = max(0.0, 1.0 - cvar_ratio / 4.0)

    # 3) Skew alignment: if we're calling long (prob_up > 0.5), positive
    #    skew helps; negative skew hurts.  Vice versa for short.
    side = 1.0 if prob_up >= 0.5 else -1.0
    # skew * side > 0 means skew aligns with our bet.
    skew_factor = np.clip(0.5 + 0.25 * side * skewness, 0.0, 1.0)

    confidence_score = float(
        0.50 * dir_clarity +
        0.30 * tail_safety +
        0.20 * skew_factor
    )

    return dict(
        cvar_95=cvar_95,
        skewness=skewness,
        kurtosis=kurtosis,
        std=std,
        p25=p25,
        p75=p75,
        confidence_score=confidence_score,
    )


def _chunk_size(n_paths: int, horizon: int, dtype_bytes: int = 4) -> int:
    """Pick a chunk size that fits in ~25% of free VRAM (leave room for other buffers)."""
    free = free_vram_bytes()
    if free is None:
        return n_paths
    budget = int(free * 0.25)
    per_path = horizon * dtype_bytes
    return max(1024, min(n_paths, budget // max(per_path, 1)))


def gbm_paths(
    mu: float,
    sigma: float,
    horizon: int,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Simulate horizon-step GBM log-returns and return aggregate stats.

    mu, sigma are *per-step* (already scaled to the bar timeframe).
    """
    xp = get_xp()
    rng = xp.random.default_rng(seed)
    chunk = _chunk_size(n_paths, horizon)

    sum_up = 0
    sum_logr = 0.0
    samples_q = []  # collect per-chunk horizon log-returns for quantiles

    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        # Antithetic variates: pair each draw with its negation to halve variance.
        half = m // 2
        z_half = rng.standard_normal((half, horizon), dtype=xp.float32)
        z = xp.concatenate([z_half, -z_half], axis=0)
        if z.shape[0] < m:  # odd m: top up with one extra row
            extra = rng.standard_normal((m - z.shape[0], horizon), dtype=xp.float32)
            z = xp.concatenate([z, extra], axis=0)

        log_r_step = (mu - 0.5 * sigma * sigma) + sigma * z
        horizon_log_r = log_r_step.sum(axis=1)
        sum_up += int((horizon_log_r > 0).sum().get() if has_cuda() else (horizon_log_r > 0).sum())
        sum_logr += float(horizon_log_r.sum().get() if has_cuda() else horizon_log_r.sum())
        samples_q.append(_to_numpy(horizon_log_r))
        remaining -= m

    flat = np.concatenate(samples_q)
    prob_up = sum_up / n_paths
    stats = _compute_distribution_stats(flat, prob_up)

    return MCResult(
        prob_up=prob_up,
        expected_log_return=sum_logr / n_paths,
        p05=float(np.quantile(flat, 0.05)),
        p95=float(np.quantile(flat, 0.95)),
        n_paths=n_paths,
        **stats,
    )


def bootstrap_paths(
    historical_log_returns: np.ndarray,
    horizon: int,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Sample horizon-step paths by bootstrapping from `historical_log_returns`.

    Each path is a sum of `horizon` iid draws from the empirical distribution.
    """
    xp = get_xp()
    if historical_log_returns.size < 32:
        raise ValueError("Need at least 32 historical returns to bootstrap.")

    pool = xp.asarray(historical_log_returns, dtype=xp.float32)
    n_pool = pool.shape[0]
    rng = xp.random.default_rng(seed)
    chunk = _chunk_size(n_paths, horizon)

    sum_up = 0
    sum_logr = 0.0
    samples_q = []

    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        idx = rng.integers(0, n_pool, size=(m, horizon), dtype=xp.int64)
        draws = pool[idx]
        horizon_log_r = draws.sum(axis=1)
        sum_up += int((horizon_log_r > 0).sum().get() if has_cuda() else (horizon_log_r > 0).sum())
        sum_logr += float(horizon_log_r.sum().get() if has_cuda() else horizon_log_r.sum())
        samples_q.append(_to_numpy(horizon_log_r))
        remaining -= m

    flat = np.concatenate(samples_q)
    prob_up = sum_up / n_paths
    stats = _compute_distribution_stats(flat, prob_up)

    return MCResult(
        prob_up=prob_up,
        expected_log_return=sum_logr / n_paths,
        p05=float(np.quantile(flat, 0.05)),
        p95=float(np.quantile(flat, 0.95)),
        n_paths=n_paths,
        **stats,
    )


def _to_numpy(arr) -> np.ndarray:
    return arr.get() if has_cuda() else np.asarray(arr)
