"""Public Kraken Futures OHLCV access for reproducibility.

This module intentionally contains no credentials, signing, private endpoints,
order methods, or account state. It is suitable for public-data reconstruction
only. The paper's published tables do not require a network call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import httpx

from .candle_cache import LocalCandleStore
from .domain import Candle


class PublicKrakenClient:
    def __init__(
        self,
        cache_path: str,
        base_url: str = "https://futures.kraken.com",
        client: httpx.AsyncClient | None = None,
    ):
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=20)
        self._owns_client = client is None
        self.store = LocalCandleStore(cache_path)

    async def aclose(self) -> None:
        self.store.close()
        if self._owns_client:
            await self.client.aclose()

    async def historical_candles_1m(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[Candle]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("historical candle range must be timezone-aware")
        start, end = start.astimezone(UTC), end.astimezone(UTC)
        if end < start:
            raise ValueError("historical candle end must not precede start")
        expected = int((end - start).total_seconds() // 60) + 1
        cached = self.store.between(symbol, start, end)
        if len(cached) == expected:
            return cached
        cursor, end_epoch = int(start.timestamp()), int(end.timestamp())
        fetched: list[Candle] = []
        while cursor <= end_epoch:
            response = await self.client.get(
                f"/api/charts/v1/trade/{symbol}/1m",
                params={"from": cursor, "to": end_epoch, "count": 1000},
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("candles", [])
            if not rows:
                break
            batch = self._candles_from_payload(symbol, rows)
            fetched.extend(batch)
            next_cursor = int(batch[-1].open_time.timestamp()) + 60
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if not payload.get("more_candles"):
                break
        if fetched:
            self.store.upsert(fetched)
        return self.store.between(symbol, start, end)

    @staticmethod
    def _candles_from_payload(symbol: str, rows: list[dict]) -> list[Candle]:
        now = datetime.now(UTC)
        candles = []
        for item in rows:
            start = datetime.fromtimestamp(item["time"] / 1000, UTC)
            candles.append(
                Candle(
                    symbol=symbol,
                    open_time=start,
                    close_time=start + timedelta(minutes=1),
                    open=Decimal(str(item["open"])),
                    high=Decimal(str(item["high"])),
                    low=Decimal(str(item["low"])),
                    close=Decimal(str(item["close"])),
                    volume=Decimal(str(item["volume"])),
                    completed=start + timedelta(minutes=1) <= now,
                )
            )
        return candles

    @staticmethod
    def is_contiguous(candles: list[Candle]) -> bool:
        return all(
            right.open_time - left.open_time == timedelta(minutes=1)
            for left, right in pairwise(candles)
        )
