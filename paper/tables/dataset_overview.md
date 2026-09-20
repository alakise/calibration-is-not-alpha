| experiment | period | states | contract | model | input_representation | jev_calls | input_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev directional baseline | 7-day development window | 40320 | TERNARY_V1 + BINARY_V1 | jev-1.13.0 | V1 raw candles; V2 candles + technicals | 40320 | not republished in directional manifest |
| Jev opportunity holdout | 2026-08-15 through 2026-09-11 UTC | 40320 | market-opportunity-v1 | jev-1.13.0 | market-state-v2 + seven Noul questions | 40320 | 194848794 |
| Corrected classical baseline | historical BTC 1-minute sample | 86357 | deterministic labels | local deterministic | OHLCV and derived volatility/volume features | 0 | 0 |
| Microstructure exploration | fixed live public-Kraken snapshot | 81071 | exploratory | local deterministic | L5 depth and trade-flow proxies | 0 | 0 |
