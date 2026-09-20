from datetime import UTC, datetime, timedelta
from decimal import Decimal

from jev_crypto_research.domain import Candle


def completed_candles(count: int = 60) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol="PI_XBTUSD",
            open_time=start + timedelta(minutes=i),
            close_time=start + timedelta(minutes=i + 1),
            open=Decimal(100 + i),
            high=Decimal(101 + i),
            low=Decimal(99 + i),
            close=Decimal(100.5 + i),
            volume=Decimal(10 + i),
            completed=True,
        )
        for i in range(count)
    ]
