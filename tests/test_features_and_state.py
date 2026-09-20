import pytest

from jev_crypto_research.domain import Side, Trade
from jev_crypto_research.features import FeatureEngine
from jev_crypto_research.state import StateBuilder, StateVariant
from tests.conftest import completed_candles


def test_feature_engine_and_eight_states():
    candles = completed_candles()
    features = FeatureEngine().compute(candles)
    states = StateBuilder().build_all(candles, features)
    assert len(states) == 8
    assert len(states[StateVariant.V8_FULL].completed_candles) == 60
    assert "funding_rate" in states[StateVariant.V8_FULL].features
    assert states[StateVariant.V1_CANDLES].features == {}


def test_open_candle_is_rejected():
    candles = completed_candles()
    candles[-1] = candles[-1].model_copy(update={"completed": False})
    with pytest.raises(ValueError, match="completed"):
        FeatureEngine().compute(candles)


def test_flat_prices_have_neutral_rsi_and_volume_ratio_is_json_finite():
    candles = completed_candles()
    flat = [candle.model_copy(update={"close": candles[0].close}) for candle in candles]
    features = FeatureEngine().compute(
        flat,
        trades=[
            Trade(
                symbol="PI_XBTUSD",
                timestamp=flat[-1].close_time,
                price=flat[-1].close,
                quantity=1,
                side=Side.BUY,
            )
        ],
    )
    assert features["rsi_5m"] == 50.0
    assert features["buy_sell_volume_ratio"] == 1.0
    assert features["buy_sell_volume_ratio"] not in {float("inf"), float("-inf")}

    import json
    import math

    # Strict JSON serialization with allow_nan=False must succeed
    serialized = json.dumps(features, allow_nan=False)
    assert "Infinity" not in serialized
    assert "NaN" not in serialized

    # Test sell-only trades
    sell_features = FeatureEngine().compute(
        flat,
        trades=[
            Trade(
                symbol="PI_XBTUSD",
                timestamp=flat[-1].close_time,
                price=flat[-1].close,
                quantity=1,
                side=Side.SELL,
            )
        ],
    )
    assert sell_features["buy_sell_volume_ratio"] == 0.0
    json.dumps(sell_features, allow_nan=False)

    # Test empty trades
    empty_features = FeatureEngine().compute(flat, trades=[])
    assert empty_features["buy_sell_volume_ratio"] == 0.5
    json.dumps(empty_features, allow_nan=False)

    # Ensure all computed feature floats are finite
    for k, v in features.items():
        if isinstance(v, float):
            assert math.isfinite(v), f"Feature {k} is not finite: {v}"
