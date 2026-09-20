# Methodology

The study separates four estimands that are often conflated:

1. **Calibration:** whether predicted probabilities agree with event frequencies.
2. **Discrimination:** whether predictions rank or classify future outcomes.
3. **Economic utility:** whether a signal has positive expectancy after costs.
4. **Strategy PnL:** the result of a specified entry, exit, holding, and sizing rule.

Jev outputs are evaluated as probabilistic judgments. A fixed-horizon event label
is never treated as a runtime trade instruction. Classical strategies enter no
earlier than the next completed candle and use only information available at the
decision timestamp. The official directional cost model is 15 bps round-trip:
5 bps fee, 2 bps slippage, and 0.5 bps spread per side.

The holdout and walk-forward analyses are chronological. Validation begins flat;
selection observations are never used as validation observations. Overlapping
forward labels are retained but are not treated as independent observations for
causal claims.
