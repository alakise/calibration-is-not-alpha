# Reproducibility

## Core offline mode

`python scripts/build_paper_tables.py --offline` reads only committed derived
results. It performs no network access, no exchange call, and no paid Jev call.
It regenerates the paper tables and figures that are supported by the published
summaries.

## From-public-data mode

The classical and microstructure scripts can consume a user-supplied local
Kraken public-data cache. Raw logs and local SQLite files are intentionally not
committed. Live WebSocket collection is append-only and public-data-only, but
the exact ephemeral event stream cannot be reproduced byte-for-byte.

## Fresh inference mode

Fresh Jev inference is outside the default public workflow. Any future adapter
must require an explicit `--allow-paid-inference` switch and a user-provided
key. Tests and offline reproduction must remain unable to trigger it.

## Versioning

The published manifests retain `jev-1.13.0`, `market-state-v2`,
`exposure-choice-v1`, and `market-opportunity-v1`. Remote service behavior may
still change, so the committed derived summaries are the primary record of the
reported experiments.
