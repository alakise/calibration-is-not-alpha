# Data availability

The study used public Kraken Futures one-minute OHLCV and a live public Futures
WebSocket book/trade collector. The exact raw 26M-event log, local candle/cache
databases, and private Jev response cache are not redistributed in this
repository because of size, ephemeral collection, and redistribution concerns.

Small derived summaries and manifests are included under `results/`. Public
OHLCV can be recollected with the documented scripts. The microstructure stream
was collected live and therefore cannot be reconstructed exactly from the
repository alone. No credentials, private account data, or order data are
included.
