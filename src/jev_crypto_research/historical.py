"""Leakage-safe reconstruction of historical market-only JEV states."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import ClassVar

from .domain import Candle, DerivativesSnapshot
from .features import FeatureEngine
from .state import MarketState, StateBuilder, StateVariant

INSUFFICIENT_CANDLE_HISTORY = "INSUFFICIENT_CANDLE_HISTORY"
MISSING_HISTORICAL_TRADES = "MISSING_HISTORICAL_TRADES"
MISSING_HISTORICAL_ORDERBOOK = "MISSING_HISTORICAL_ORDERBOOK"
MISSING_HISTORICAL_FUNDING = "MISSING_HISTORICAL_FUNDING"
MISSING_HISTORICAL_OI = "MISSING_HISTORICAL_OI"
MISSING_HISTORICAL_SPREAD = "MISSING_HISTORICAL_SPREAD"


@dataclass(frozen=True)
class HistoricalStateAt:
    timestamp: datetime
    states: dict[StateVariant, MarketState]
    skipped: dict[StateVariant, str]


@dataclass(frozen=True)
class HistoricalDerivatives:
    """Public, timestamped analytics known before a historical decision point."""

    funding_rates: dict[datetime, Decimal]
    open_interest: dict[datetime, Decimal]

    @classmethod
    def empty(cls) -> "HistoricalDerivatives":
        return cls({}, {})

    def snapshot_before(self, timestamp: datetime) -> DerivativesSnapshot | None:
        """Use the prior completed analytics bucket, never the current one."""
        timestamp = _utc(timestamp)
        funding_points = [
            point for point in self.funding_rates if _utc(point) < timestamp
        ]
        oi_points = [point for point in self.open_interest if _utc(point) < timestamp]
        if not funding_points or len(oi_points) < 2:
            return None
        latest_funding = max(funding_points, key=_utc)
        prior_oi, latest_oi = sorted(oi_points, key=_utc)[-2:]
        return DerivativesSnapshot(
            timestamp=max(_utc(latest_funding), _utc(latest_oi)),
            funding_rate=self.funding_rates[latest_funding],
            open_interest=self.open_interest[latest_oi],
            open_interest_change=self.open_interest[latest_oi]
            - self.open_interest[prior_oi],
        )

    def has_funding_before(self, timestamp: datetime) -> bool:
        timestamp = _utc(timestamp)
        return any(_utc(point) < timestamp for point in self.funding_rates)

    def has_open_interest_pair_before(self, timestamp: datetime) -> bool:
        timestamp = _utc(timestamp)
        return sum(_utc(point) < timestamp for point in self.open_interest) >= 2


class HistoricalStateReconstructor:
    """Uses only 1m OHLCV currently available from Kraken's historical chart API.

    Raw historical trades and order-book snapshots are deliberately not
    synthesized.  Variants that require them are recorded as skips instead of
    inheriting FeatureEngine's live/default placeholders.
    """

    available_from_candles: ClassVar[frozenset[StateVariant]] = frozenset(
        {StateVariant.V1_CANDLES, StateVariant.V2_TECHNICAL}
    )

    @staticmethod
    def completed_timestamps(
        candles: Sequence[Candle], start: datetime, end: datetime
    ) -> list[datetime]:
        start, end = _utc(start), _utc(end)
        return [
            _utc(candle.close_time)
            for candle in candles
            if candle.completed and start <= _utc(candle.close_time) <= end
        ]

    def reconstruct(
        self,
        candles: Sequence[Candle],
        timestamp: datetime,
        variants: Iterable[StateVariant],
        derivatives: HistoricalDerivatives | None = None,
    ) -> HistoricalStateAt:
        timestamp = _utc(timestamp)
        selected = tuple(variants)
        visible = [
            candle
            for candle in candles
            if candle.completed and _utc(candle.close_time) <= timestamp
        ]
        if (
            not visible
            or _utc(visible[-1].close_time) != timestamp
            or len(visible) < 60
        ):
            return HistoricalStateAt(
                timestamp,
                {},
                {variant: INSUFFICIENT_CANDLE_HISTORY for variant in selected},
            )
        window = visible[-60:]
        if any(
            _utc(right.close_time) - _utc(left.close_time) != timedelta(minutes=1)
            for left, right in pairwise(window)
        ):
            return HistoricalStateAt(
                timestamp,
                {},
                {variant: INSUFFICIENT_CANDLE_HISTORY for variant in selected},
            )
        # StateBuilder and FeatureEngine see the same completed 60-candle window
        # which was available at T, never the full future download.
        snapshot = (derivatives or HistoricalDerivatives.empty()).snapshot_before(
            timestamp
        )
        features = FeatureEngine().compute(window, derivatives=snapshot)
        all_states = StateBuilder().build_all(window, features)
        states: dict[StateVariant, MarketState] = {}
        skipped: dict[StateVariant, str] = {}
        for variant in selected:
            if variant in self.available_from_candles or (
                variant is StateVariant.V5_DERIVATIVES and snapshot is not None
            ):
                states[variant] = all_states[variant]
            else:
                skipped[variant] = self._skip_reason(
                    variant, snapshot, derivatives, timestamp
                )
        return HistoricalStateAt(timestamp, states, skipped)

    @staticmethod
    def _skip_reason(
        variant: StateVariant,
        snapshot: DerivativesSnapshot | None,
        derivatives: HistoricalDerivatives | None,
        timestamp: datetime,
    ) -> str:
        if variant in {
            StateVariant.V3_VOLUME,
            StateVariant.V6_CLASSIC_QUANT,
            StateVariant.V8_FULL,
        }:
            return MISSING_HISTORICAL_TRADES
        if variant is StateVariant.V4_MICROSTRUCTURE:
            return MISSING_HISTORICAL_ORDERBOOK
        if variant is StateVariant.V5_DERIVATIVES:
            source = derivatives or HistoricalDerivatives.empty()
            if not source.has_funding_before(timestamp):
                return MISSING_HISTORICAL_FUNDING
            if not source.has_open_interest_pair_before(timestamp):
                return MISSING_HISTORICAL_OI
            if snapshot is None or snapshot.funding_rate is None:
                return MISSING_HISTORICAL_FUNDING
        if variant is StateVariant.V7_MECHANICS:
            return MISSING_HISTORICAL_ORDERBOOK
        raise ValueError(f"unsupported state variant: {variant}")

    @staticmethod
    def forward_returns_bps(
        candles: Sequence[Candle], timestamp: datetime
    ) -> dict[str, str]:
        """Offline labels derived after reconstruction; never included in MarketState."""
        timestamp = _utc(timestamp)
        index = next(
            (
                i
                for i, candle in enumerate(candles)
                if _utc(candle.close_time) == timestamp
            ),
            None,
        )
        if index is None:
            return {}
        current = candles[index].close
        labels: dict[str, str] = {}
        for minutes in (1, 3, 5, 10):
            future_index = index + minutes
            if future_index >= len(candles):
                continue
            future = candles[future_index]
            if _utc(future.close_time) != timestamp + timedelta(minutes=minutes):
                continue
            value = (future.close / current - Decimal(1)) * Decimal(10_000)
            labels[f"return_t_plus_{minutes}m_bps"] = str(value)
        return labels


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
