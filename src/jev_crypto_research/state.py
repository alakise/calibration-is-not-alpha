from enum import StrEnum

from pydantic import BaseModel

from .domain import Candle

MARKET_STATE_VERSION = "market-state-v2"


class StateVariant(StrEnum):
    V1_CANDLES = "v1_candles"
    V2_TECHNICAL = "v2_technical"
    V3_VOLUME = "v3_volume"
    V4_MICROSTRUCTURE = "v4_microstructure"
    V5_DERIVATIVES = "v5_derivatives"
    V6_CLASSIC_QUANT = "v6_classic_quant"
    V7_MECHANICS = "v7_mechanics"
    V8_FULL = "v8_full"


class MarketState(BaseModel):
    variant: StateVariant
    symbol: str
    completed_candles: list[dict[str, str]]
    features: dict[str, float | None]


GROUPS = {
    "technical": {
        "rsi_5m",
        "ema20_distance",
        "ema50_distance",
        "atr_5m",
        "realized_vol_5m",
        "realized_vol_15m",
    },
    "volume": {"volume_zscore", "buy_sell_volume_ratio"},
    "micro": {"orderbook_imbalance", "spread_bps"},
    "derivatives": {"funding_rate", "open_interest_change"},
}
VARIANT_GROUPS = {
    StateVariant.V1_CANDLES: set(),
    StateVariant.V2_TECHNICAL: GROUPS["technical"],
    StateVariant.V3_VOLUME: GROUPS["volume"],
    StateVariant.V4_MICROSTRUCTURE: GROUPS["micro"],
    StateVariant.V5_DERIVATIVES: GROUPS["derivatives"],
    StateVariant.V6_CLASSIC_QUANT: GROUPS["technical"] | GROUPS["volume"],
    StateVariant.V7_MECHANICS: GROUPS["micro"] | GROUPS["derivatives"],
    StateVariant.V8_FULL: set().union(*GROUPS.values()),
}


class StateBuilder:
    def build_all(
        self, candles: list[Candle], features: dict[str, float | None]
    ) -> dict[StateVariant, MarketState]:
        if len(candles) < 60 or not all(c.completed for c in candles[-60:]):
            raise ValueError("StateBuilder accepts 60 completed candles only")
        raw = [
            {
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
            }
            for c in candles[-60:]
        ]
        return {
            variant: MarketState(
                variant=variant,
                symbol=candles[-1].symbol,
                completed_candles=raw,
                features={key: features.get(key) for key in keys},
            )
            for variant, keys in VARIANT_GROUPS.items()
        }
