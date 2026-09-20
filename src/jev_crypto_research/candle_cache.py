"""Small SQLite cache used only for public historical OHLCV retrieval."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .domain import Candle


class LocalCandleStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS candles_1m (
                symbol TEXT NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                open TEXT NOT NULL,
                high TEXT NOT NULL,
                low TEXT NOT NULL,
                close TEXT NOT NULL,
                volume TEXT NOT NULL,
                PRIMARY KEY (symbol, open_time)
            )"""
        )
        self.connection.commit()

    def count(self, symbol: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM candles_1m WHERE symbol = ?", (symbol,)
        ).fetchone()
        return int(row[0]) if row else 0

    def upsert(self, candles: list[Candle]) -> None:
        self.connection.executemany(
            """INSERT OR REPLACE INTO candles_1m
            (symbol, open_time, close_time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    candle.symbol,
                    candle.open_time.isoformat(),
                    candle.close_time.isoformat(),
                    str(candle.open),
                    str(candle.high),
                    str(candle.low),
                    str(candle.close),
                    str(candle.volume),
                )
                for candle in candles
            ],
        )
        self.connection.commit()

    def between(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        rows = self.connection.execute(
            """SELECT symbol, open_time, close_time, open, high, low, close, volume
            FROM candles_1m WHERE symbol = ? AND open_time >= ? AND open_time <= ?
            ORDER BY open_time ASC""",
            (symbol, start.isoformat(), end.isoformat()),
        ).fetchall()
        now = datetime.now(UTC)
        return [
            Candle(
                symbol=row[0],
                open_time=datetime.fromisoformat(row[1]),
                close_time=datetime.fromisoformat(row[2]),
                open=Decimal(row[3]),
                high=Decimal(row[4]),
                low=Decimal(row[5]),
                close=Decimal(row[6]),
                volume=Decimal(row[7]),
                completed=datetime.fromisoformat(row[2]) <= now,
            )
            for row in rows
        ]

    def recent(self, symbol: str, limit: int = 61) -> list[Candle]:
        rows = self.connection.execute(
            """SELECT symbol, open_time, close_time, open, high, low, close, volume
            FROM candles_1m WHERE symbol = ? ORDER BY open_time DESC LIMIT ?""",
            (symbol, limit),
        ).fetchall()
        now = datetime.now(UTC)
        return [
            Candle(
                symbol=row[0],
                open_time=datetime.fromisoformat(row[1]),
                close_time=datetime.fromisoformat(row[2]),
                open=Decimal(row[3]),
                high=Decimal(row[4]),
                low=Decimal(row[5]),
                close=Decimal(row[6]),
                volume=Decimal(row[7]),
                completed=datetime.fromisoformat(row[2]) <= now,
            )
            for row in reversed(rows)
        ]

    def close(self) -> None:
        self.connection.close()
