from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from scripts.run_second_pass_analysis import (
    HORIZONS,
    block_bootstrap,
    calibration_analysis,
    candle_features,
    directional_metrics,
    distribution,
    interaction_table,
    load_state_rows,
    movement_analysis,
    v1_v2_analysis,
)


def _candles(count: int = 80) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        close = 100.0 + index * 0.1
        rows.append(
            {
                "open_time": start + timedelta(minutes=index),
                "close_time": start + timedelta(minutes=index + 1),
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 10.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _analysis_rows(count: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        p_flat = 0.05 + (index % 10) * 0.08
        chosen = "LONG" if index % 2 == 0 else "SHORT"
        row = {
            "timestamp": timestamp,
            "contract": "ternary_v1",
            "variant": "v1_candles",
            "p_flat": p_flat,
            "p_long": 0.75 - p_flat / 2,
            "p_short": 0.25 - p_flat / 2,
            "top_probability": 0.75 - p_flat / 2,
            "native_confidence": 0.55 + (index % 4) * 0.1,
            "margin": 0.2 + (index % 3) * 0.1,
            "chosen_direction": chosen,
        }
        for horizon in HORIZONS:
            row[f"return_{horizon}"] = float(
                (index + 1) * (HORIZONS.index(horizon) + 1)
            )
            row[f"label_{horizon}"] = "LONG" if index % 3 else "FLAT"
            row[f"signed_{horizon}"] = (
                row[f"return_{horizon}"]
                if chosen == "LONG"
                else -row[f"return_{horizon}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_block_bootstrap_is_deterministic() -> None:
    values = np.arange(100, dtype=float)
    assert block_bootstrap(values, seed=123) == block_bootstrap(values, seed=123)


def test_features_at_t_ignore_future_candles() -> None:
    candles = _candles()
    timestamp = pd.Timestamp(candles.iloc[64].close_time)
    before = candle_features(candles, timestamp)
    future = candles.copy()
    future.loc[len(future), ["open_time", "close_time"]] = [
        timestamp + pd.Timedelta(value=1, unit="min"),
        timestamp + pd.Timedelta(value=2, unit="min"),
    ]
    future.loc[len(future), ["open", "high", "low", "close", "volume"]] = [
        1,
        10000,
        1,
        9999,
        1e9,
    ]
    assert candle_features(future, timestamp) == before


def test_directional_metrics_keep_binary_and_ternary_accuracy_separate() -> None:
    ternary = _analysis_rows(4)
    binary = ternary.assign(contract="binary_v1", p_flat=0.0)
    ternary_metrics = directional_metrics(ternary, "1m")
    binary_metrics = directional_metrics(binary, "1m")
    assert ternary_metrics["three_class_accuracy"] is not None
    assert ternary_metrics["binary_direction_accuracy"] is None
    assert binary_metrics["three_class_accuracy"] is None
    assert binary_metrics["binary_direction_accuracy"] is not None


def test_binary_calibration_excludes_realized_flat_labels() -> None:
    rows = _analysis_rows(4).assign(contract="binary_v1", p_flat=0.0)
    result = calibration_analysis(rows, "1m")
    assert result["N"] == 2
    assert result["brier_score"] is not None


def test_probability_bucket_counts_conserve_rows() -> None:
    rows = _analysis_rows()
    movement = movement_analysis(rows)
    assert sum(item["N"] for item in movement["approximate_bins"].values()) == len(rows)
    interactions = interaction_table(rows, "p_flat", "native_confidence")
    assert sum(item["N"] for item in interactions.values()) == len(rows)


def test_v1_v2_alignment_is_exact_timestamp_only() -> None:
    rows = _analysis_rows(3)
    v2 = rows.assign(variant="v2_technical")
    result = v1_v2_analysis(pd.concat([rows, v2], ignore_index=True))
    assert result["ternary_v1"]["matched_timestamps"] == 3


def test_published_results_are_offline_and_no_paid_calls() -> None:
    import json
    from pathlib import Path

    result = json.loads(
        Path("results/summaries/quant_microstructure_summary.json").read_text()
    )
    assert result["no_paid_jev_calls"] is True
    assert result["no_real_orders"] is True


def test_second_pass_has_no_inference_call_path() -> None:
    from pathlib import Path

    source = Path("scripts/run_second_pass_analysis.py").read_text()
    assert "get_or_run_expert_inference" not in source
    assert "TypeSafeClient" not in source


def test_nonfinite_values_are_dropped_from_distributions() -> None:
    result = distribution([float("nan"), float("inf"), 1.0, 2.0])
    assert result["N"] == 2
    assert result["mean"] == 1.5


def test_research_labels_are_added_after_identity(tmp_path) -> None:
    import json
    import sqlite3

    db_path = tmp_path / "baseline.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "create table expert_inferences (symbol text,state_timestamp text,variant text,state_version text,jev_model text,direction_contract text,question_schema_version text,input_hash text,probabilities_json text,jev_confidence real,status text)"
        )
        timestamp = "2026-01-01T01:00:00+00:00"
        db.execute(
            "insert into expert_inferences values (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "PF_XBTUSD",
                timestamp,
                "v1_candles",
                "market-state-v2",
                "jev-1.13.0",
                "ternary_v1",
                "exposure-choice-v1",
                "identity-hash",
                json.dumps({"long": 0.6, "short": 0.2, "flat": 0.2}),
                0.6,
                "READY",
            ),
        )
    candles = _candles(140)
    candles["close_time"] = pd.date_range(
        "2026-01-01", periods=len(candles), freq="min", tz="UTC"
    )
    result = load_state_rows(db_path, candles)
    assert result.loc[0, "input_hash"] == "identity-hash"
    assert result.loc[0, "return_15m"] == result.loc[0, "return_15m"]
    assert "return_15m" not in result.loc[0, "input_hash"]
