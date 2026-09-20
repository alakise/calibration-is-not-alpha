from __future__ import annotations

import inspect
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from run_corrected_classical import (
    add_corrected_features,
    candidate_layers,
    mean_reversion_control,
)
from run_quant_microstructure_phase import (
    maker_filter,
    simulate_maker,
)


def _candles(count: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC")
    close = 100.0 + np.arange(count, dtype=float) * 0.03
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100.0 + np.arange(count, dtype=float),
        },
        index=index,
    )


def _micro_states(count: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=count, freq="s", tz="UTC")
    mid = np.full(count, 100.0)
    return pd.DataFrame(
        {
            "timestamp": index,
            "mid": mid,
            "best_bid": mid - 0.001,
            "best_ask": mid + 0.001,
            "spread_bps": np.full(count, 0.2),
            "bid_depth_l5": np.full(count, 10.0),
            "ask_depth_l5": np.full(count, 10.0),
            "l5_imbalance": np.zeros(count),
            "book_valid": np.ones(count, dtype=bool),
            "buy_qty": np.full(count, 1.0),
            "sell_qty": np.full(count, 1.0),
            "ofi_proxy_10s": np.zeros(count),
            "trade_flow_imbalance_10s": np.zeros(count),
            "rv_60s": np.full(count, 0.001),
        }
    )


def test_same_frequency_volatility_has_no_horizon_sqrt_scaling() -> None:
    data = add_corrected_features(_candles())
    row = 80
    expected = np.sqrt(np.mean(data["r1"].iloc[row - 4 : row + 1].pow(2)))
    assert np.isclose(data["vol5"].iloc[row], expected)
    assert np.isclose(
        data["vol5"].iloc[row] / data["vol60"].iloc[row],
        data["vol_ratio_5_60"].iloc[row],
    )


def test_donchian_reference_excludes_current_candle() -> None:
    data = add_corrected_features(_candles())
    row = 80
    expected = data["close"].iloc[row - 10 : row].max()
    assert data["donchian_hi10"].iloc[row] == expected
    assert data["donchian_hi10"].iloc[row] != data["close"].iloc[row]


def test_candidate_layers_are_nested() -> None:
    data = add_corrected_features(_candles())
    layers = candidate_layers(data, n=20)
    for stronger, weaker in zip(
        (
            "S4_BREAKOUT_VOL_ER_VOLUME_ETH",
            "S3_BREAKOUT_VOL_ER_VOLUME",
            "S2_BREAKOUT_VOL_ER",
            "S1_BREAKOUT_VOL",
        ),
        (
            "S3_BREAKOUT_VOL_ER_VOLUME",
            "S2_BREAKOUT_VOL_ER",
            "S1_BREAKOUT_VOL",
            "S0_BREAKOUT",
        ),
    ):
        assert np.all(layers[stronger].abs() <= layers[weaker].abs())


def test_features_and_mr0_do_not_change_when_future_rows_are_appended() -> None:
    data = _candles()
    before = add_corrected_features(data)
    extended = pd.concat(
        [
            data,
            pd.DataFrame(
                {
                    "open": [1e6],
                    "high": [2e6],
                    "low": [1.0],
                    "close": [1.5e6],
                    "volume": [1e9],
                },
                index=[data.index[-1] + pd.Timedelta(value=1, unit="min")],
            ),
        ]
    )
    after = add_corrected_features(extended)
    assert before.iloc[:-1].equals(after.loc[before.index].iloc[:-1])
    assert (
        mean_reversion_control(before)
        .iloc[:-1]
        .equals(mean_reversion_control(after.loc[before.index]).iloc[:-1])
    )


def test_official_cost_model_is_per_side_5_plus_2_plus_half() -> None:
    assert 2 * (5.0 + 2.0 + 0.5) == 15.0


def test_maker_rejects_invalid_book_and_has_no_order_api() -> None:
    row = pd.Series({"book_valid": False, "spread_bps": 1.0})
    assert maker_filter(row, 0) == (False, False)
    source = inspect.getsource(simulate_maker)
    assert "send_order" not in source
    assert "create_order" not in source


def test_conservative_queue_is_not_more_permissive_than_touch() -> None:
    frame = _micro_states()
    optimistic = simulate_maker(frame, "optimistic_touch", 0)
    conservative = simulate_maker(frame, "conservative_queue", 0)
    assert conservative["fills"] <= optimistic["fills"]
    assert conservative["partial_fills"] <= optimistic["partial_fills"]
    assert np.isfinite(optimistic["max_drawdown"])
    assert np.isfinite(conservative["max_drawdown"])
