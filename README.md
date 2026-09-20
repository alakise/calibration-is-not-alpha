# Calibration Is Not Alpha

Companion code and reproducibility materials for:

> **Calibration Is Not Alpha: A Transaction-Cost-Aware Evaluation of Jev in Short-Horizon Cryptocurrency Markets**

This repository is an academic research artifact, not a trading product. The
active trading project was closed after the experiments failed to demonstrate a
cost-surviving alpha candidate.

## Research question

Does a calibrated System-One decision model add incremental predictive value to
short-horizon cryptocurrency markets beyond simple quantitative baselines?

The central scientific distinction is:

> Probability calibration is not equivalent to economically exploitable alpha.

## Main findings

- Jev directional output did not demonstrate robust short-horizon BTC alpha.
- TERNARY V1 `P(FLAT)` tracked subsequent movement magnitude, but added almost
  no out-of-sample information beyond deterministic volatility features
  (reported ΔR² approximately `+0.00084`).
- On the independent 40,320-state holdout, `MOVE_15M_15BPS` achieved Brier
  `0.2131387`, log loss `0.6186983`, and ECE `0.0421875`.
- Adding Jev probability to deterministic volatility improved Brier by only
  `1.4058e-05` and log loss by `3.4719e-05`.
- Corrected classical breakout/volatility baselines also failed to survive the
  preregistered 15 bps round-trip cost model out of sample.
- The study's conclusion is **NO CURRENT ALPHA CANDIDATE** under the tested
  conditions.

Negative and inconclusive results are intentionally preserved.

## Offline reproduction

The published tables and figures can be rebuilt without a network, API key, or
paid inference:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,plots]'
.venv/bin/python scripts/build_paper_tables.py --offline
.venv/bin/python -m pytest -q
```

`--offline` reads only the small, published derived artifacts under `results/`.
It does not contact Kraken, TypeSafe, or any exchange endpoint. The resulting
tables are written to `paper/tables/` and figures to `paper/figures/`.

Reconstruction from public Kraken OHLCV or a newly collected public WebSocket
stream is a separate, explicitly networked workflow:

```bash
.venv/bin/python scripts/run_corrected_classical.py --db path/to/local/candles.sqlite3
.venv/bin/python scripts/run_quant_microstructure_phase.py --micro-root path/to/microstructure
```

Those commands require local data and are not needed for the published result.

## Experiments

### Jev directional baseline

The original BTC/PF_XBTUSD experiment used pinned model `jev-1.13.0`,
`market-state-v2`, and `exposure-choice-v1` over 10,080 completed one-minute
timestamps. V1 (raw candles) and V2 (candles plus technical features) were
evaluated under both TERNARY V1 and BINARY V1, for 40,320 expert states total.
Directional accuracy was near chance and no configuration survived realistic
short-horizon costs.

### Movement/opportunity holdout

The independent holdout spans 2026-08-15 through 2026-09-11 UTC and contains
40,320 states. The pinned `market-opportunity-v1` bundle used seven native Noul
questions. It consumed 194,848,794 input tokens in the original run; estimated
input cost was $8.1836. The machine-readable summary is
`results/summaries/jev_opportunity_analysis.json`.

### Deterministic baselines

The corrected classical study contains 86,357 BTC one-minute observations and
explicitly fixes the earlier S1 volatility-scale mismatch. The official cost
model is 5 bps fee + 2 bps slippage + 0.5 bps spread per side, or 15 bps
round-trip. Walk-forward validation starts flat and does not carry selection
state into validation.

### Microstructure exploration

The live public Kraken snapshot contained 26,398,119 raw events, 40,724 BTC
canonical states, 40,347 ETH canonical states, and four reconnects at the fixed
report cutoff. The collector persisted L5 depth only; OFI is therefore labelled
as a depth-change proxy, not true price-level OFI. The short validation target
was insufficiently variable, so the microstructure result is limited/inconclusive.

### Maker simulation

Optimistic touch Q0 produced +$20.13 and conservative queue-aware Q0 +$2.09,
but maker fees were set to zero and exact queue position was unavailable. These
are diagnostic simulations, not alpha claims. Q4 filters reduced fills to three.

## Data and provenance

The repository publishes compact summaries, manifests, tables, and scripts. It
does not publish the 26M-event raw collector log, local SQLite caches, private
Jev responses, or credential-bearing files. Raw microstructure collection was
ephemeral and exact event-level replication is not guaranteed. Public Kraken
OHLCV can be recollected with the networked scripts; the exact Jev outputs are
represented by derived summaries and manifests because fresh inference may cost
money and model behavior can change.

## Repository structure

```text
paper/                 manuscript, figures, tables, references
results/               published summaries, manifests, derived tables
scripts/               offline table builder and research reconstruction tools
src/jev_crypto_research/ public candle/data and deterministic state utilities
tests/                 research invariants and offline safety tests
docs/                  methodology, reproducibility, and data availability
```

No frontend, dashboard, account endpoint, private exchange signing, order
submission, or paper-session worker is included in this public export.

## Fresh inference

Fresh Jev inference is deliberately not part of ordinary reproduction. Any
future extension must require an explicit `--allow-paid-inference` flag, a
user-supplied `TYPESAFE_API_KEY`, a pinned model/schema, and a separate budget.
The offline command cannot launch paid inference by construction.

## Limitations

- BTC-focused, predominantly one-minute analysis; no claim of asset or horizon
  generalization.
- Forward labels overlap and are research labels, not trading-strategy PnL.
- Retrospective Jev queries may overlap model training data; this is not a
  prospective frozen deployment evaluation.
- Cost and slippage assumptions are approximations.
- Pinned model version reduces but does not eliminate external-service drift.
- Historical L2 reconstruction was unavailable; microstructure validation was
  short and underpowered.
- No prospective 30-day frozen replication was completed.
- No causal predictability claim is made.

## Citation and license

See `CITATION.cff`, `paper/manuscript.md`, and `paper/references.bib`. Author
metadata in `CITATION.cff` is marked for confirmation rather than guessed.
The original research code is released under the MIT License; see `LICENSE`.
Before public publication, review `LICENSE_REVIEW.md` and any account-specific
TypeSafe order terms.

## Disclaimer

This is research software. It is not investment advice, does not demonstrate
profitability, and must not be used to place real exchange orders.
