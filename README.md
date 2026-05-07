# mcmc_cuda

GPU-accelerated **Monte Carlo + Markov chain** ensemble for **XAUUSD** with a
realistic OHLC backtester, session-aware costs, prop-firm-style risk
controls, and an **MT5 paper-trading bridge**.

> **Paper-trading only.** The MT5 bridge re-checks the demo-account flag
> on every order; non-demo accounts are refused. This guard is intentional.

```
historical bars  ──►  CUDA MC + Markov  ──►  confidence-gated signal
                                                       │
                                                       ▼
                                       OHLC engine (TP/SL/trail/BE)
                                                       │
                            ┌──────────────────────────┴──────────────────────────┐
                            ▼                                                     ▼
                metrics + leaderboard                              live_signal.json (atomic)
                (top-3 by prop-firm score)                                ▼
                                                            MT5 EA polls + executes (demo)
```

## Why this exists

This is a **research stack** for evaluating a non-trivial price-prediction
ensemble (CUDA-vectorized empirical bootstrap + GPU Markov-chain forecast)
under honest XAUUSD trading frictions. The deliverable is reproducible
backtests + a clean handoff to MT5 so the same signal can be paper-traded
without rewriting it in MQL.

## What's in the box (MVPs)

| MVP | What it proves | Entrypoint |
|---|---|---|
| **Backtest MVP** | The MC/Markov ensemble produces a non-degenerate signal under realistic OHLC + session-aware costs. | `python scripts/run_backtest.py --years 2` |
| **Robustness MVP** | The OHLC engine respects daily-loss / total-drawdown circuit breakers, break-even movers, trailing stops, time-stops, and same-bar TP/SL tiebreakers. | `pytest tests/test_engine_scalp.py tests/test_breakeven.py tests/test_tiebreak.py` |
| **EA handoff MVP** | A backtest emits an atomic `artifacts/live_signal.json` an MT5 EA can poll on each tick. | `python scripts/run_backtest.py --years 1 && cat artifacts/live_signal.json` |
| **Demo guard MVP** | The Python broker bridge refuses to route orders to a non-demo account, even mid-session. | `pytest tests/test_broker_bridge.py` |

## Hardware & stack

- NVIDIA RTX 4050 (Ada, compute 8.9, ~6 GB VRAM); CuPy + Numba CUDA
- Python 3.10–3.12, Windows 11
- `MetaTrader5` package against an Exness MT5 demo terminal (Windows-only)

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

# Sanity-check MT5 connectivity (optional — only needed for live data / EA handoff)
python scripts/sanity_mt5.py

# Two-year backtest with the low-friction (ECN) cost profile (default)
python scripts/run_backtest.py --years 2

# Conservative retail cost profile
python scripts/run_backtest.py --years 2 --cost-profile standard_retail
```

The first call pulls historical bars from MT5 and caches them as parquet
under `data/raw/` so subsequent runs are offline.

## Signal philosophy: less surface, more sample

The strategy uses the MC/Markov ensemble's **distribution shape** (confidence
score, CVaR, skew alignment) to gate trades — *not* a stack of indicator
filters. We deliberately removed the regime / ADX / slope / RSI overlay
because every additional indicator inflates the overfitting surface and
shrinks the trade population.

Default thresholds (`confidence_high=0.28`, `confidence_low=0.18`,
`min_prob_edge=0.02`) are calibrated so a 2-year M15 backtest produces
**hundreds**, not tens, of trades — which is the minimum sample size for any
performance metric to mean anything.

## Costs: low-friction by default

The default cost profile is `low_friction`:

| Session | Spread (points) |
|---|---|
| overlap | 6  |
| london  | 8  |
| ny      | 10 |
| asia    | 18 |
| dead    | 22 |

Plus 1.5-point slippage per side and asymmetric XAUUSD swap. Switch to
`--cost-profile standard_retail` to test the same strategy under a
conservative retail friction profile.

## Artifact retention: top-3 by score

`scripts/run_backtest.py` writes per-run artifacts only when a run is in the
top-3 by prop-firm score, **and** prunes everything else on every invocation.
The `artifacts/` directory therefore stays a curated leaderboard:

```
artifacts/
  leaderboard.json
  equity_<best_ts>.png      trades_<best_ts>.csv      metrics_<best_ts>.json
  equity_<2nd_ts>.png       trades_<2nd_ts>.csv       metrics_<2nd_ts>.json
  equity_<3rd_ts>.png       trades_<3rd_ts>.csv       metrics_<3rd_ts>.json
  live_signal.json          ◄── most recent snapshot the EA reads
```

Adjust the cap via `TOP_K_ARTIFACTS` in [`src/mcmc_cuda/config.py`](src/mcmc_cuda/config.py).

## MT5 EA handoff

The Python side ends each backtest by writing
[`artifacts/live_signal.json`](src/mcmc_cuda/broker/mt5_bridge.py) atomically.
A companion EA (Expert Advisor) polls this file on each tick and submits
market orders inside MT5. The snapshot schema:

```json
{
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "bar_time": "2026-05-06T11:45:00+00:00",
  "side": 1,
  "sl_price": 2345.20,
  "tp_price": 2391.80,
  "risk_fraction": 0.005,
  "confidence": 0.34,
  "horizon_bars": 3,
  "generated_at": "2026-05-06T11:45:30.812+00:00"
}
```

Two execution paths share the same demo guard:

```python
from mcmc_cuda.broker import MT5Bridge, OrderTicket

with MT5Bridge() as bridge:
    bridge.assert_demo()                     # raises NotDemoAccountError if not demo
    bridge.submit_market("XAUUSD", OrderTicket(
        side=1, volume_lots=0.01, sl_price=2345.20, tp_price=2391.80,
    ))
```

## Project layout

```
src/mcmc_cuda/
  config.py            paths, constants, env loading
  data/                MT5 ingestion + parquet cache + CSV loader
  gpu/                 CuPy/Numba CUDA kernels (MC bootstrap, Markov chain)
  features/            ATR, microstructure helpers (used by SMC variant)
  strategy/            MC+Markov ensemble (default), SMC, sessions
  backtest/            OHLC engine, costs, risk, metrics
  broker/              MT5 bridge + atomic signal exporter
  ui/                  Matplotlib live playback
scripts/               CLI entry points
tests/                 ~20 unit tests covering engine, costs, risk, signals, broker
artifacts/             top-3 runs + live_signal.json
data/                  parquet cache (gitignored)
```

## Roadmap

- ✅ Phase 1 — Data ingest, CUDA MC paths, Markov chain, baseline backtest
- ✅ Phase 2 — Confidence-gated ensemble, OHLC engine, session-aware costs, prop-firm risk module
- ✅ Phase 3 — MT5 broker bridge with demo guard + atomic signal exporter
- ⏳ Phase 4 — Companion `.mq5` EA, walk-forward validation harness, optional XGBoost meta-filter

## License

Proprietary — personal research project.
