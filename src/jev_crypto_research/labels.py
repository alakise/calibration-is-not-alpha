from collections.abc import Sequence
from decimal import Decimal

from .domain import Candle, Direction, Exposure, Horizon

HORIZON_MINUTES = {Horizon.M3: 3, Horizon.M5: 5, Horizon.M10: 10}


def label_direction(
    candles: Sequence[Candle],
    decision_index: int,
    horizon: Horizon,
    flat_threshold_bps: Decimal,
) -> Direction:
    """Labels future close vs completed decision close; rejects unavailable future data."""
    future_index = decision_index + HORIZON_MINUTES[horizon]
    if decision_index < 0 or future_index >= len(candles):
        raise ValueError("future completed candle is unavailable for this label")
    entry, exit_price = candles[decision_index].close, candles[future_index].close
    move_bps = (exit_price / entry - Decimal(1)) * Decimal(10_000)
    if move_bps > flat_threshold_bps:
        return Direction.UP
    if move_bps < -flat_threshold_bps:
        return Direction.DOWN
    return Direction.FLAT


def label_exposure(
    candles: Sequence[Candle],
    decision_index: int,
    horizon: Horizon,
    flat_threshold_bps: Decimal,
) -> Exposure:
    """Offline-only evaluation target derived after the decision time."""
    return {
        Direction.UP: Exposure.LONG,
        Direction.DOWN: Exposure.SHORT,
        Direction.FLAT: Exposure.FLAT,
    }[label_direction(candles, decision_index, horizon, flat_threshold_bps)]
