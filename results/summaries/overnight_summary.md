# Overnight research summary

Generated: 2026-09-20T00:18:25.246949+00:00

## Safety

- Real exchange orders: `False`
- Baseline DB unchanged: `True`

## Jev holdout

- READY expert states: `40320` / 40,320
- PENDING: `0`, FAILED: `0`
- Input tokens: `194848794`; estimated input cost: `$8.1836`
- Timestamp range currently ready: `2026-08-15 00:00:00.000000` → `2026-09-11 23:59:00.000000`
- Contract: `market-opportunity-v1` with seven independent native Noul questions; no `exposure-choice-v1` calls were added.
- MOVE_15M_15BPS calibration: Brier `0.2131386904761905`, log loss `0.6186983129774795`, ECE `0.04218749999999999`.
- Jev vs deterministic volatility OOS (fixed 70/30 chronological split): Brier improvement `1.4057663113131724e-05`, log-loss improvement `3.471850420599942e-05` when Jev is added; diagnostic only.

## Classical quant

- Strategies/cost cells: `24`
- Walk-forward folds: `3`
- Official cost: 15 bps round trip (5 bps fee + 2 bps slippage + 0.5 bps spread per side).
- Gross/net at 15 bps by strategy:
  - `MR0`: 3665 trades, gross PnL `$54.06`, net PnL `$-5443.44`.
  - `S0`: 3956 trades, gross PnL `$57.83`, net PnL `$-5876.17`.
  - `S1`: 1 trades, gross PnL `$11.36`, net PnL `$9.86`.
  - `S2`: 2951 trades, gross PnL `$42.14`, net PnL `$-4384.36`.
  - `S3`: 2881 trades, gross PnL `$-19.36`, net PnL `$-4340.86`.
  - `S4`: 3850 trades, gross PnL `$51.77`, net PnL `$-5723.23`.

## Fixed hybrid comparison

Three hypotheses were evaluated without threshold search or new Jev calls: S3 only, S3 plus P(move 15m) >= 0.30, and S3 plus direction-specific event probability >= 0.25.
- `H0_S3_only`: full `-1970.40` PnL / `1355` trades; untouched validation `-443.24` PnL / `321` trades.
- `H1_S3_plus_move15_p30`: full `-964.43` PnL / `699` trades; untouched validation `-221.15` PnL / `175` trades.
- `H2_S3_plus_direction_edge_p25`: full `-1938.27` PnL / `1339` trades; untouched validation `-444.49` PnL / `321` trades.

## Microstructure

- Raw JSONL files: `5`
- Derived JSONL files: `2`
- Raw/derived lines observed at report time: `1802094`; storage: `458894028` bytes.

## Executive interpretation

- No cost-surviving OOS alpha candidate was found. Positive gross pockets were sparse or were eliminated by the official cost model.
- Jev movement probabilities are measurable but only marginally improve the fixed deterministic volatility diagnostic in this holdout; native Noul confidence is not available.
- The next informative experiment is longer live microstructure collection followed by a preregistered order-flow confirmation test; do not mine more thresholds on this sample.

No alpha claim is made; overlapping minute labels and the small sparse S1 sample limit inference.
