# Calibration Is Not Alpha: A Transaction-Cost-Aware Evaluation of Jev in Short-Horizon Cryptocurrency Markets

**Status:** working-paper draft. The numerical claims below are generated from
the committed result summaries, but public redistribution of Jev benchmark or
performance information requires confirmation of the service terms.

## Abstract

Recent AI trading systems can conflate probabilistic confidence with tradable
predictive advantage. We evaluate Jev, a calibrated System-One decision model,
on short-horizon Bitcoin market states using both directional and explicitly
labelable movement questions. An independent 28-day holdout contains 40,320
one-minute market states. For the `MOVE_15M_15BPS` event, Jev achieves Brier
score 0.2131, log loss 0.6187, and ECE 0.0422. However, adding the Jev
probability to deterministic volatility features improves chronological
validation Brier by only 1.4e-5 and log loss by 3.5e-5. Directional experiments
and corrected deterministic baselines likewise fail to demonstrate robust
economic value after a 15 bps round-trip cost assumption. The result is not that
calibration is useless; rather, calibration and semantic market-state judgment
need not imply incremental or economically exploitable information.

## 1. Introduction

Short-horizon financial prediction is a particularly demanding setting for AI
systems. Labels overlap, market states evolve, and small gross effects can be
eliminated by fees, spread, and slippage. A model can therefore be calibrated
without supplying a tradeable edge. This study asks whether a calibrated
System-One decision model adds information beyond inexpensive deterministic
volatility and price features.

The paper deliberately treats a negative result as a useful boundary condition.
We do not claim that no financial market contains exploitable structure, nor do
we claim that Jev is unsuitable for every task. We report what was and was not
demonstrated in a pinned, BTC-focused, short-horizon experiment.

## 2. Related work

Calibration is a distinct property from classification accuracy and must be
measured separately [@guo2017]. Machine-learning asset-pricing work emphasizes
chronological out-of-sample evaluation and economic, rather than purely
statistical, objectives [@gu2020]. High-frequency order-book research shows that
order-flow history can contain predictive information in some settings
[@sirignano2019]. Bitcoin studies also report short-horizon statistical
predictability while finding that transaction costs can erase the resulting
strategy returns [@jaquart2021]. Our contribution is a negative, cost-aware
evaluation of a typed probabilistic System-One judgment against deterministic
baselines in this specific setting.

## 3. Jev / System-One decision model

Jev returns typed judgments rather than free-form trade prose. The directional
experiment used a native `Choice` distribution over `LONG`, `SHORT`, and `FLAT`
(`exposure-choice-v1`) and a binary variant over `LONG` and `SHORT`. The
movement experiment used seven native Noul questions, including
`MOVE_15M_15BPS`. The pinned model was `jev-1.13.0`; state representation was
`market-state-v2`.

The model output is treated as a probability forecast. It is not a position
size, leverage, risk limit, or order instruction. Fixed-horizon returns are
research labels and are never substituted for strategy PnL.

## 4. Data

The directional baseline covers 10,080 completed BTC/PF_XBTUSD one-minute
timestamps with V1 raw candles and V2 candles plus technical features, under
TERNARY V1 and BINARY V1 contracts (40,320 expert states). The independent
movement holdout covers 2026-08-15 through 2026-09-11 UTC and contains 40,320
states.

The corrected classical dataset contains 86,357 BTC one-minute observations.
The exploratory public microstructure snapshot contains 26,398,119 raw events,
40,724 BTC canonical states, 40,347 ETH canonical states, and four reconnects.
Only L5 depth was persisted; the OFI feature is therefore a depth-change proxy.

## 5. Experimental design

We distinguish calibration, discrimination, gross expectancy, net expectancy,
and strategy PnL. Directional strategy entries occur no earlier than the next
completed candle. Walk-forward selection and validation windows are chronological;
validation begins flat. The official directional cost model is 5 bps fee, 2 bps
slippage, and 0.5 bps spread per side (15 bps round trip).

The corrected classical family is nested: S0 breakout, S1 corrected volatility
expansion, S2 efficiency ratio, S3 volume confirmation, and S4 ETH confirmation.
MR0 is a small extreme-move reversal control. The earlier S1 implementation
used incompatible horizon-scaled volatilities and is not treated as evidence.

## 6. Directional prediction experiment

Across variants, directional accuracy is approximately chance when restricted to
non-flat opportunities, with V2 producing a materially different SHORT bias.
The probability vectors and exact raw-signal metrics are preserved in
`results/summaries/baseline_research_summary.json` and
`results/summaries/second_pass_summary.json`.

No tested directional configuration survived the official cost model. This is
not a claim that every threshold or market is unprofitable; it is the result of
the preregistered configurations and windows actually evaluated.

## 7. Movement / opportunity experiment

On the independent holdout, `MOVE_15M_15BPS` has Brier 0.2131387, log loss
0.6186983, and ECE 0.0421875. The deterministic volatility baseline on the
chronological validation slice has Brier 0.1662127 and log loss 0.5118111.
Adding Jev changes these to 0.1661986 and 0.5117764, respectively. The
improvements are measurable but economically and scientifically small relative
to the deterministic baseline.

The TERNARY V1 `P(FLAT)` output is negatively correlated with future movement
magnitude (5-minute Spearman approximately -0.112). Low `P(FLAT)` observations
move more, but adding `P(FLAT)` to deterministic volatility features changes
5-minute OOS R² by only approximately +0.00084.

## 8. Deterministic quantitative baselines

With a 15-minute maximum hold, corrected S1 produced 4,898 candidates and 2,733
trades, with gross PnL $4.82 and net PnL -$4,094.68. S4 produced gross $22.47
and net -$2,034.03. MR0 produced gross $14.15 and net -$7,491.85. Across three
walk-forward validation folds, the selected configurations sum to gross $8.35
and net -$575.15.

These dollar values are supplementary implementation metrics. The primary
interpretation is in basis points per trade, turnover, coverage, and cost burden
in the generated tables.

## 9. Transaction-cost-aware evaluation

The gross-to-net gap is central: an apparent positive gross edge of a few
hundredths of a basis point per trade cannot offset a 15 bps round trip. Cost
sensitivity at 5, 10, 15, and 20 bps is generated by
`scripts/build_paper_tables.py` and is not threshold-mined.

## 10. Microstructure exploratory analysis

OFI proxy, trade-flow imbalance, L5 imbalance, microprice displacement, and
spread were evaluated with rank correlations, quantile diagnostics, and simple
nested ridge models. No strong, stable incremental OOS relationship was
demonstrated. The late validation target had insufficient variation; this makes
the result inconclusive rather than evidence that microstructure cannot contain
signal.

## 11. Results

The publication tables and figures are generated under `paper/tables/` and
`paper/figures/`. The maker simulation is intentionally labelled diagnostic:
optimistic Q0 is +$20.13, conservative queue-aware Q0 is +$2.09, maker fees are
zero by assumption, and Q4 leaves only three fills. No alpha claim follows.

## 12. Discussion

The results support five cautious interpretations. First, calibration is not
financial exploitability. Second, a semantic movement judgment may recognize a
volatility-like state while adding little beyond a cheap volatility estimator.
Third, technical context can change a model's decision surface without improving
predictive utility. Fourth, realistic transaction costs are indispensable in
short-horizon evaluation. Fifth, if System-One models are useful in this domain,
a meta-gate over independently validated quantitative signals may be a more
appropriate future hypothesis than direct raw-price direction prediction. That
last point is future work, not a demonstrated result.

## 13. Limitations

The analysis is BTC-focused and predominantly one-minute. Forward labels
overlap. Retrospective model queries may overlap unknown model training data.
Transaction-cost assumptions are approximations; exact queue position, maker
rebates, and historical L2 reconstruction were unavailable. The microstructure
sample was short and its validation target was degenerate. No prospective
30-day frozen replication was completed. Model-service behavior may change even
with a pinned version. No causal claim or cross-asset generalization is made.

## 14. Reproducibility and data availability

Run `python scripts/build_paper_tables.py --offline` after installing the
development dependencies. This uses only committed derived artifacts and makes
no network or paid calls. Large raw logs, SQLite caches, and private Jev
responses are intentionally excluded. Exact live microstructure event
replication is not possible from this repository alone.

## 15. Conclusion

Under the tested BTC short-horizon conditions, neither Jev nor the corrected
classical families produced a cost-surviving alpha candidate. The most useful
finding is a boundary condition: a calibrated, typed probability can be
statistically meaningful while adding negligible incremental information and no
demonstrated economic utility beyond simple deterministic baselines.

## References

See `paper/references.bib`. Claims about Jev itself are limited to the
authoritative TypeSafe documentation and to the experiment manifests; future
public release must resolve any service-term restrictions on publishing model
benchmarks.
