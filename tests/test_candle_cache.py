from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from jev_crypto_research.candle_cache import LocalCandleStore
from jev_crypto_research.domain import Candle


def test_local_candle_store_upserts_and_returns_recent_candles(tmp_path):
    store = LocalCandleStore(str(tmp_path / "candles.sqlite3"))
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    candles = [
        Candle(
            symbol="PF_XBTUSD",
            open_time=now + timedelta(minutes=index),
            close_time=now + timedelta(minutes=index + 1),
            open=Decimal(100 + index),
            high=Decimal(101 + index),
            low=Decimal(99 + index),
            close=Decimal("100.5") + index,
            volume=Decimal(1),
            completed=True,
        )
        for index in range(3)
    ]
    store.upsert(candles)
    store.upsert([candles[-1].model_copy(update={"close": Decimal(123)})])
    recent = store.recent("PF_XBTUSD", 2)
    assert len(recent) == 2
    assert recent[-1].close == Decimal(123)
    store.close()


def test_interval_cache_does_not_turn_an_open_candle_into_completed_history(tmp_path):
    store = LocalCandleStore(str(tmp_path / "candles.sqlite3"))
    open_time = datetime.now(UTC).replace(second=0, microsecond=0)
    current = Candle(
        symbol="PF_XBTUSD",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1),
        open=Decimal(100),
        high=Decimal(101),
        low=Decimal(99),
        close=Decimal(100),
        volume=Decimal(1),
        completed=True,
    )
    store.upsert([current])
    cached = store.between("PF_XBTUSD", open_time, current.close_time)
    assert cached[0].completed is False
    assert len(cached) == 1
    store.close()


@pytest.mark.asyncio
async def test_candle_gap_recovery_after_downtime(tmp_path):
    import httpx

    from jev_crypto_research.public_data import PublicKrakenClient

    db_path = str(tmp_path / "candles.sqlite3")
    store = LocalCandleStore(db_path)
    old_start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(hours=5)
    stale_candles = [
        Candle(
            symbol="PF_XBTUSD",
            open_time=old_start + timedelta(minutes=i),
            close_time=old_start + timedelta(minutes=i + 1),
            open=Decimal(100 + i),
            high=Decimal(101 + i),
            low=Decimal(99 + i),
            close=Decimal(100.5 + i),
            volume=Decimal(1),
            completed=True,
        )
        for i in range(61)
    ]
    store.upsert(stale_candles)
    assert store.count("PF_XBTUSD") == 61
    store.close()

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    current_candles_payload = [
        {
            "time": int((now - timedelta(minutes=61 - i)).timestamp() * 1000),
            "open": 200 + i,
            "high": 201 + i,
            "low": 199 + i,
            "close": 200.5 + i,
            "volume": 5,
        }
        for i in range(61)
    ]

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            count_requested = int(request.url.params.get("count", 2))
            rows = (
                current_candles_payload
                if request.url.params.get("from") is not None
                else current_candles_payload[-count_requested:]
            )
            return httpx.Response(
                200,
                json={"candles": rows},
                request=request,
            )

    client = PublicKrakenClient(
        cache_path=db_path,
        client=httpx.AsyncClient(
            base_url="https://futures.kraken.com", transport=MockTransport()
        ),
    )
    try:
        result = await client.historical_candles_1m(
            "PF_XBTUSD", now - timedelta(minutes=61), now
        )
        assert len(result) == 61
        # All returned candles must be contiguous (1-minute steps without gap)
        for left, right in pairwise(result):
            assert right.open_time - left.open_time == timedelta(minutes=1)
        assert result[-1].close == Decimal("260.5")
    finally:
        await client.aclose()
