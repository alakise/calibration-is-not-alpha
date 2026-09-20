from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Direction(StrEnum):
    """Research-only realized forward-return label."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class Exposure(StrEnum):
    """The directional exposure to hold until the next one-minute evaluation."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Horizon(StrEnum):
    M3 = "3m"
    M5 = "5m"
    M10 = "10m"


class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    completed: bool


class Trade(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    side: Side


class BookLevel(BaseModel):
    price: Decimal
    quantity: Decimal


class OrderBookSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    bids: list[BookLevel]
    asks: list[BookLevel]


class DerivativesSnapshot(BaseModel):
    timestamp: datetime
    funding_rate: Decimal | None = None
    open_interest: Decimal | None = None
    open_interest_change: Decimal | None = None


class ExpertDecision(BaseModel):
    """Native-choice distribution: LONG, SHORT, or FLAT. No trade instruction."""

    long: float = Field(ge=0)
    short: float = Field(ge=0)
    flat: float = Field(ge=0)

    @field_validator("flat")
    @classmethod
    def total_is_positive(cls, _: float, info):
        values = info.data
        if values.get("long", 0) + values.get("short", 0) + _ <= 0:
            raise ValueError("at least one choice probability must be positive")
        return _

    def normalized(self) -> "ExpertDecision":
        total = self.long + self.short + self.flat
        return ExpertDecision(
            long=self.long / total, short=self.short / total, flat=self.flat / total
        )

    def winner(self) -> Exposure:
        return max(
            (
                (Exposure.LONG, self.long),
                (Exposure.SHORT, self.short),
                (Exposure.FLAT, self.flat),
            ),
            key=lambda x: x[1],
        )[0]
