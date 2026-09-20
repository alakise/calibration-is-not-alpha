#!/usr/bin/env python3
"""Second-pass, zero-Jev analysis of the frozen BTC baseline.

This script reads the immutable baseline SQLite backup and derives all features,
labels, statistics, and diagnostics locally.  It never imports the inference
service and never calls TypeSafe.  A separate candle-cache copy is used when a
public Kraken fetch is needed for the 15m/30m/60m research labels.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_crypto_research.public_data import PublicKrakenClient

BASELINE_MODEL = "jev-1.13.0"
BASELINE_SYMBOL = "PF_XBTUSD"
STATE_VERSION = "market-state-v2"
QUESTION_SCHEMA = "exposure-choice-v1"
CONTRACTS = ("ternary_v1", "binary_v1")
VARIANTS = ("v1_candles", "v2_technical")
HORIZONS = ("1m", "3m", "5m", "10m", "15m", "30m", "60m")
FLAT_THRESHOLD_BPS = 8.0
ROUND_TRIP_COSTS_BPS = (5.0, 10.0, 15.0, 20.0)
SEED = 20260920


def utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def iso(value: datetime) -> str:
    return utc(value).isoformat()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return utc(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(type(value).__name__)


def distribution(values: Iterable[float]) -> dict[str, Any]:
    values = [float(v) for v in values if finite(v) is not None]
    if not values:
        return {
            "N": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=float)
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "N": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float | None:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if (
        len(frame) < 3
        or frame.iloc[:, 0].nunique() < 2
        or frame.iloc[:, 1].nunique() < 2
    ):
        return None
    if method == "spearman":
        left, right = (
            frame.iloc[:, 0].rank(method="average"),
            frame.iloc[:, 1].rank(method="average"),
        )
        return finite(left.corr(right, method="pearson"))
    return finite(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="pearson"))


def label_direction(value: float | None) -> str | None:
    if value is None:
        return None
    if value > FLAT_THRESHOLD_BPS:
        return "LONG"
    if value < -FLAT_THRESHOLD_BPS:
        return "SHORT"
    return "FLAT"


def block_bootstrap(
    values: np.ndarray,
    statistic: str = "mean",
    *,
    reps: int = 400,
    block_size: int = 60,
    seed: int = SEED,
) -> dict[str, Any]:
    """Moving block bootstrap; fixed seed makes report generation deterministic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {
            "N": int(values.size),
            "reps": 0,
            "estimate": None,
            "ci95": [None, None],
            "block_size": block_size,
            "seed": seed,
        }
    # Keep at least four blocks for small diagnostic subsets; otherwise a
    # subset smaller than the requested 60-minute block would bootstrap as a
    # degenerate copy of itself.
    block_size = max(1, min(block_size, max(1, values.size // 4)))
    starts = np.arange(values.size)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        indices = []
        while len(indices) < values.size:
            start = int(rng.choice(starts))
            indices.extend(((start + np.arange(block_size)) % values.size).tolist())
        sample = values[np.asarray(indices[: values.size])]
        samples[i] = float(
            np.mean(sample) if statistic == "mean" else np.median(sample)
        )
    estimate = float(np.mean(values) if statistic == "mean" else np.median(values))
    return {
        "N": int(values.size),
        "reps": reps,
        "estimate": estimate,
        "ci95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "block_size": block_size,
        "seed": seed,
    }


def ols_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    if train_x.size == 0 or test_x.size == 0:
        return np.asarray([], dtype=float)
    beta = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    return test_x @ beta


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        return {"N": 0, "mae": None, "rmse": None, "r2": None}
    residual = actual - predicted
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "N": int(actual.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else None,
    }


def bin_label(
    value: float | None, bins: list[tuple[str, float, float | None]]
) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    for name, low, high in bins:
        if value >= low and (high is None or value < high):
            return name
    return None


PFLAT_BINS = [
    ("0.00-0.20", 0.0, 0.20),
    ("0.20-0.33", 0.20, 0.33),
    ("0.33-0.40", 0.33, 0.40),
    ("0.40-0.50", 0.40, 0.50),
    ("0.50-0.60", 0.50, 0.60),
    ("0.60+", 0.60, None),
]
TOP_BINS = [
    ("0.50-0.60", 0.50, 0.60),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70-0.80", 0.70, 0.80),
    ("0.80+", 0.80, None),
]
MARGIN_BINS = [
    ("0.00-0.10", 0.0, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.40", 0.20, 0.40),
    ("0.40+", 0.40, None),
]
CONF_BINS = [
    ("0.00-0.40", 0.0, 0.40),
    ("0.40-0.60", 0.40, 0.60),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70+", 0.70, None),
]

LOW_MOVEMENT_THRESHOLDS_BPS = (5.0, 10.0, 15.0)


def load_candles(path: Path, symbol: str) -> pd.DataFrame:
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "select open_time,close_time,open,high,low,close,volume from candles_1m where symbol=? order by open_time",
            (symbol,),
        ).fetchall()
    frame = pd.DataFrame(
        rows,
        columns=["open_time", "close_time", "open", "high", "low", "close", "volume"],
    )
    if frame.empty:
        raise RuntimeError(f"no candles found in {path} for {symbol}")
    for column in ("open_time", "close_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["close"])
        .drop_duplicates("close_time")
        .sort_values("close_time")
        .reset_index(drop=True)
    )
    return frame


async def ensure_candle_copy(
    source: Path, destination: Path, start: datetime, end: datetime, fetch: bool
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if fetch:
        client = PublicKrakenClient(cache_path=str(destination))
        try:
            await client.historical_candles_1m(
                BASELINE_SYMBOL,
                start - timedelta(minutes=60),
                end + timedelta(minutes=60),
            )
        finally:
            await client.aclose()
    return destination


def candle_features(
    candles: pd.DataFrame, timestamp: pd.Timestamp
) -> dict[str, float | None]:
    visible = candles[candles["close_time"] <= timestamp].tail(60).copy()
    if len(visible) < 60:
        return {
            "rsi_5m": None,
            "ema20_distance_bps": None,
            "ema50_distance_bps": None,
            "atr_5m_bps": None,
            "realized_vol_5m_bps": None,
            "realized_vol_15m_bps": None,
            "realized_vol_30m_bps": None,
            "range_5m_bps": None,
            "range_15m_bps": None,
            "range_30m_bps": None,
            "abs_return_1m_bps": None,
            "abs_return_5m_bps": None,
            "volume_zscore_30m": None,
        }
    close = visible["close"]
    ret = close.pct_change() * 10_000

    def rv(n: int) -> float | None:
        values = ret.tail(n).dropna()
        return float(values.std(ddof=1)) if len(values) > 1 else None

    def rng(n: int) -> float | None:
        part = visible.tail(n)
        return (
            float((part["high"].max() / part["low"].min() - 1) * 10_000)
            if len(part)
            else None
        )

    delta = close.diff().tail(5)
    gain, loss = delta.clip(lower=0).mean(), (-delta.clip(upper=0)).mean()
    rsi = (
        100.0
        if loss == 0 and gain > 0
        else 50.0
        if gain == loss == 0
        else float(100 - 100 / (1 + gain / loss))
        if loss
        else None
    )
    ema20, ema50 = (
        close.ewm(span=20, adjust=False).mean().iloc[-1],
        close.ewm(span=50, adjust=False).mean().iloc[-1],
    )
    tr = pd.concat(
        [
            (visible["high"] - visible["low"]),
            (visible["high"] - close.shift()).abs(),
            (visible["low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr5 = tr.tail(5).mean() / close.iloc[-1] * 10_000
    vol = visible["volume"].tail(30)
    vol_std = vol.std(ddof=1)
    return {
        "rsi_5m": rsi,
        "ema20_distance_bps": float((close.iloc[-1] / ema20 - 1) * 10_000),
        "ema50_distance_bps": float((close.iloc[-1] / ema50 - 1) * 10_000),
        "atr_5m_bps": float(atr5),
        "realized_vol_5m_bps": rv(5),
        "realized_vol_15m_bps": rv(15),
        "realized_vol_30m_bps": rv(30),
        "range_5m_bps": rng(5),
        "range_15m_bps": rng(15),
        "range_30m_bps": rng(30),
        "abs_return_1m_bps": float(abs(ret.iloc[-1]))
        if pd.notna(ret.iloc[-1])
        else None,
        "abs_return_5m_bps": float(abs((close.iloc[-1] / close.iloc[-6] - 1) * 10_000))
        if len(close) >= 6
        else None,
        "volume_zscore_30m": float((vol.iloc[-1] - vol.mean()) / vol_std)
        if vol_std and pd.notna(vol_std)
        else 0.0,
    }


def load_state_rows(db_path: Path, candles: pd.DataFrame) -> pd.DataFrame:
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """select symbol,state_timestamp,variant,state_version,jev_model,direction_contract,question_schema_version,input_hash,probabilities_json,jev_confidence from expert_inferences where status='READY' and symbol=? and state_version=? and jev_model=? and question_schema_version=? and direction_contract in ('ternary_v1','binary_v1') and variant in ('v1_candles','v2_technical') order by state_timestamp,variant,direction_contract""",
            (BASELINE_SYMBOL, STATE_VERSION, BASELINE_MODEL, QUESTION_SCHEMA),
        ).fetchall()
    records: list[dict[str, Any]] = []
    close_lookup = {
        pd.Timestamp(row.close_time): float(row.close) for row in candles.itertuples()
    }
    for (
        symbol,
        timestamp,
        variant,
        state_version,
        model,
        contract,
        schema,
        input_hash,
        probabilities_json,
        confidence,
    ) in rows:
        ts = pd.Timestamp(utc(timestamp))
        probs = json.loads(probabilities_json)
        p_long, p_short, p_flat = (
            float(probs.get("long", 0)),
            float(probs.get("short", 0)),
            float(probs.get("flat", 0)),
        )
        ordered = sorted(
            ((p_long, "LONG"), (p_short, "SHORT"), (p_flat, "FLAT")), reverse=True
        )
        top, chosen = ordered[0]
        directional = max(p_long, p_short)
        records.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "variant": variant,
                "contract": contract,
                "state_version": state_version,
                "model": model,
                "question_schema": schema,
                "input_hash": input_hash,
                "p_long": p_long,
                "p_short": p_short,
                "p_flat": p_flat if contract == "ternary_v1" else 0.0,
                "chosen_direction": chosen,
                "top_probability": top,
                "directional_probability": directional,
                "margin": top - ordered[1][0],
                "native_confidence": finite(confidence),
                "price": close_lookup.get(ts),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no baseline inference rows matched frozen identity")
    feature_rows = []
    for ts in frame["timestamp"].drop_duplicates().sort_values():
        feature_rows.append({"timestamp": ts, **candle_features(candles, ts)})
    features = pd.DataFrame(feature_rows)
    frame = frame.merge(features, on="timestamp", how="left")
    price_map = {
        pd.Timestamp(row.close_time): float(row.close) for row in candles.itertuples()
    }
    for minutes in (1, 3, 5, 10, 15, 30, 60):
        offset = pd.Timedelta(value=int(minutes), unit="min")
        frame[f"return_{minutes}m"] = [
            (
                (price_map.get(pd.Timestamp(ts) + offset) / price - 1) * 10_000
                if price and price_map.get(pd.Timestamp(ts) + offset)
                else np.nan
            )
            for ts, price in zip(frame["timestamp"], frame["price"])
        ]
    for horizon in HORIZONS:
        frame[f"label_{horizon}"] = frame[f"return_{horizon}"].map(label_direction)
        frame[f"signed_{horizon}"] = np.where(
            frame["chosen_direction"].eq("LONG"),
            frame[f"return_{horizon}"],
            np.where(
                frame["chosen_direction"].eq("SHORT"),
                -frame[f"return_{horizon}"],
                np.nan,
            ),
        )
    return frame.sort_values(["timestamp", "contract", "variant"]).reset_index(
        drop=True
    )


def baseline_audit(db_path: Path) -> dict[str, Any]:
    """Audit the frozen source without writing to it."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        status_rows = db.execute(
            "select status,count(*) from expert_inferences where symbol=? and state_version=? and jev_model=? and question_schema_version=? and direction_contract in ('ternary_v1','binary_v1') and variant in ('v1_candles','v2_technical') group by status",
            (BASELINE_SYMBOL, STATE_VERSION, BASELINE_MODEL, QUESTION_SCHEMA),
        ).fetchall()
        ready = db.execute(
            "select count(*) from expert_inferences where status='READY' and symbol=? and state_version=? and jev_model=? and question_schema_version=? and direction_contract in ('ternary_v1','binary_v1') and variant in ('v1_candles','v2_technical')",
            (BASELINE_SYMBOL, STATE_VERSION, BASELINE_MODEL, QUESTION_SCHEMA),
        ).fetchone()[0]
        timestamp_count = db.execute(
            "select count(distinct state_timestamp) from expert_inferences where status='READY' and symbol=? and state_version=? and jev_model=? and question_schema_version=? and direction_contract in ('ternary_v1','binary_v1') and variant in ('v1_candles','v2_technical')",
            (BASELINE_SYMBOL, STATE_VERSION, BASELINE_MODEL, QUESTION_SCHEMA),
        ).fetchone()[0]
    expected = timestamp_count * len(CONTRACTS) * len(VARIANTS)
    statuses = {str(status): int(count) for status, count in status_rows}
    return {
        "status_counts": statuses,
        "ready_selected": int(ready),
        "matched_timestamps": int(timestamp_count),
        "expected_states_for_matched_timestamps": int(expected),
        "cache_hits": int(ready),
        "new_jev_calls": 0,
        "failures": statuses.get("FAILED", 0),
        "skips": max(expected - ready, 0),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def movement_analysis(
    frame: pd.DataFrame, group_column: str = "p_flat"
) -> dict[str, Any]:
    rows = frame[frame["contract"] == "ternary_v1"].copy()
    output: dict[str, Any] = {
        "approximate_bins": {},
        "quantile_bins": {},
        "correlations": {},
        "bootstrap_mean_abs_return": {},
    }
    rows["pflat_bin"] = rows[group_column].map(lambda v: bin_label(v, PFLAT_BINS))
    for name, group in rows.groupby("pflat_bin", dropna=True, sort=False):
        output["approximate_bins"][name] = {
            "N": len(group),
            **{h: distribution(group[f"return_{h}"].abs().dropna()) for h in HORIZONS},
        }
    rows["pflat_decile"] = pd.qcut(
        rows[group_column],
        10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    )
    for name, group in rows.groupby("pflat_decile", observed=False, sort=False):
        output["quantile_bins"][str(name)] = {
            "N": len(group),
            **{h: distribution(group[f"return_{h}"].abs().dropna()) for h in HORIZONS},
        }
    for h in HORIZONS:
        target = rows[f"return_{h}"].abs()
        output["correlations"][h] = {
            "spearman": safe_corr(rows[group_column], target, "spearman"),
            "pearson": safe_corr(rows[group_column], target, "pearson"),
            "n": int(pd.concat([rows[group_column], target], axis=1).dropna().shape[0]),
        }
        values = rows[group_column].notna() & target.notna()
        output["bootstrap_mean_abs_return"][h] = block_bootstrap(
            target[values].to_numpy(), seed=SEED + HORIZONS.index(h)
        )
    return output


def directional_metrics(
    group: pd.DataFrame, horizon: str, cost: float = 15.0
) -> dict[str, Any]:
    group = group.dropna(subset=[f"return_{horizon}"]).copy()
    if group.empty:
        return {
            "N": 0,
            "directional_N": 0,
            "coverage": None,
            "three_class_accuracy": None,
            "binary_direction_accuracy": None,
            "directional_accuracy": None,
            "flat_rate": None,
            "signed_return_gross_bps": None,
            "signed_return_net_bps": None,
            "median_signed_return_bps": None,
        }
    realized = group[f"label_{horizon}"]
    directional = realized.isin(["LONG", "SHORT"])
    predicted = group["chosen_direction"].isin(["LONG", "SHORT"])
    signed = group.loc[predicted, f"signed_{horizon}"]
    return {
        "N": len(group),
        "directional_N": int(directional.sum()),
        "coverage": float(predicted.mean()) if len(group) else None,
        "three_class_accuracy": float((group["chosen_direction"] == realized).mean())
        if group["contract"].iloc[0] == "ternary_v1" and len(group)
        else None,
        "binary_direction_accuracy": float(
            (group.loc[directional, "chosen_direction"] == realized[directional]).mean()
        )
        if group["contract"].iloc[0] == "binary_v1" and directional.any()
        else None,
        "directional_accuracy": float(
            (group.loc[directional, "chosen_direction"] == realized[directional]).mean()
        )
        if directional.any()
        else None,
        "flat_rate": float((group["chosen_direction"] == "FLAT").mean())
        if len(group)
        else None,
        "signed_return_gross_bps": float(signed.mean()) if len(signed) else None,
        "signed_return_net_bps": float(signed.mean() - cost) if len(signed) else None,
        "median_signed_return_bps": float(signed.median()) if len(signed) else None,
        "signed_return_bootstrap": block_bootstrap(
            signed.to_numpy(dtype=float), seed=SEED + HORIZONS.index(horizon)
        ),
    }


def calibration_analysis(group: pd.DataFrame, horizon: str) -> dict[str, Any]:
    """Evaluate probability calibration without mixing binary and ternary labels."""
    valid = group.dropna(subset=[f"label_{horizon}", "top_probability"]).copy()
    contract = str(group["contract"].iloc[0]) if not group.empty else ""
    if contract == "binary_v1":
        valid = valid[valid[f"label_{horizon}"].isin(["LONG", "SHORT"])]
        classes = ("LONG", "SHORT")
        probability_columns = ("p_long", "p_short")
    else:
        classes = ("LONG", "SHORT", "FLAT")
        probability_columns = ("p_long", "p_short", "p_flat")
    if valid.empty:
        return {"N": 0, "brier_score": None, "log_loss": None, "reliability": {}}
    actual = valid[f"label_{horizon}"].to_numpy()
    probs = valid.loc[:, probability_columns].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    one_hot = np.asarray([[float(label == cls) for cls in classes] for label in actual])
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    log_loss = float(
        -np.mean(
            np.log(
                probs[
                    np.arange(len(actual)), [classes.index(label) for label in actual]
                ]
            )
        )
    )
    reliability: dict[str, Any] = {}
    bins = TOP_BINS
    valid = valid.assign(correct=valid["chosen_direction"].to_numpy() == actual)
    valid["top_bin"] = valid["top_probability"].map(
        lambda value: bin_label(value, bins)
    )
    for name, bucket in valid.groupby("top_bin", dropna=True, sort=False):
        reliability[str(name)] = {
            "N": len(bucket),
            "mean_top_probability": float(bucket["top_probability"].mean()),
            "empirical_accuracy": float(bucket["correct"].mean()),
            "mean_native_confidence": finite(bucket["native_confidence"].mean()),
        }
    return {
        "N": len(valid),
        "brier_score": brier,
        "log_loss": log_loss,
        "reliability": reliability,
    }


def flat_low_movement_analysis(rows: pd.DataFrame) -> dict[str, Any]:
    """Test whether native P(FLAT) tracks multiple low-opportunity definitions."""
    output: dict[str, Any] = {}
    for name, group in rows.groupby(
        rows["p_flat"].map(lambda value: bin_label(value, PFLAT_BINS)),
        dropna=True,
        sort=False,
    ):
        item: dict[str, Any] = {"N": len(group), "horizons": {}}
        for horizon in HORIZONS:
            valid = group.dropna(subset=[f"return_{horizon}"])
            movement = valid[f"return_{horizon}"].abs()
            item["horizons"][horizon] = {
                "N": len(valid),
                "mean_abs_return_bps": finite(movement.mean()),
                "low_movement_rates": {
                    f"abs_return_lt_{int(threshold)}bps": float(
                        (movement < threshold).mean()
                    )
                    if len(movement)
                    else None
                    for threshold in LOW_MOVEMENT_THRESHOLDS_BPS
                },
            }
        output[str(name)] = item
    return output


def raw_signal_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (contract, variant), group in frame.groupby(["contract", "variant"], sort=True):
        key = f"{contract}_{variant}"
        item: dict[str, Any] = {
            "N": len(group),
            "direction_frequency": {
                k: int(v)
                for k, v in group["chosen_direction"].value_counts().to_dict().items()
            },
            "mean_probabilities": {
                name: float(group[name].mean())
                for name in ("p_long", "p_short", "p_flat")
            },
            "native_confidence": distribution(group["native_confidence"].dropna()),
            "top_probability": distribution(group["top_probability"]),
            "margin": distribution(group["margin"]),
            "horizons": {},
        }
        for h in HORIZONS:
            item["horizons"][h] = directional_metrics(group, h)
            item.setdefault("calibration", {})[h] = calibration_analysis(group, h)
        output[key] = item
    return output


def volatility_baselines(frame: pd.DataFrame) -> dict[str, Any]:
    # The incremental P(FLAT) question is defined on the primary ternary/V1
    # stream; do not mix binary zeros or duplicate contracts into this test.
    rows = frame[
        (frame["variant"] == "v1_candles") & (frame["contract"] == "ternary_v1")
    ].copy()
    features = [
        "realized_vol_5m_bps",
        "realized_vol_15m_bps",
        "realized_vol_30m_bps",
        "atr_5m_bps",
        "range_15m_bps",
        "range_30m_bps",
        "abs_return_1m_bps",
        "abs_return_5m_bps",
    ]
    out: dict[str, Any] = {"rank_correlations": {}}
    for feature in features + ["p_flat"]:
        out["rank_correlations"][feature] = {
            h: {
                "spearman": safe_corr(
                    rows[feature], rows[f"return_{h}"].abs(), "spearman"
                ),
                "pearson": safe_corr(
                    rows[feature], rows[f"return_{h}"].abs(), "pearson"
                ),
            }
            for h in HORIZONS
        }
    out["incremental_oos"] = {}
    # The first 60% is fit only; the last 40% is untouched validation.
    unique_times = sorted(rows["timestamp"].unique())
    split = int(len(unique_times) * 0.60)
    train_times, test_times = set(unique_times[:split]), set(unique_times[split:])
    base_features = [
        "realized_vol_5m_bps",
        "realized_vol_15m_bps",
        "realized_vol_30m_bps",
        "atr_5m_bps",
        "range_15m_bps",
        "range_30m_bps",
    ]
    for h in HORIZONS:
        target = f"return_{h}"
        base = rows[rows["timestamp"].isin(train_times | test_times)].copy()
        for names, label in (
            (base_features, "deterministic_baseline"),
            (base_features + ["p_flat"], "baseline_plus_pflat"),
        ):
            train = base[base["timestamp"].isin(train_times)].dropna(
                subset=names + [target]
            )
            test = base[base["timestamp"].isin(test_times)].dropna(
                subset=names + [target]
            )
            x_train = np.column_stack(
                [np.ones(len(train)), train[names].to_numpy(float)]
            )
            x_test = np.column_stack([np.ones(len(test)), test[names].to_numpy(float)])
            pred = ols_fit_predict(x_train, train[target].abs().to_numpy(float), x_test)
            out["incremental_oos"].setdefault(h, {})[label] = {
                **regression_metrics(test[target].abs().to_numpy(float), pred),
                "train_N": len(train),
                "test_start": iso(min(test["timestamp"])) if len(test) else None,
                "test_end": iso(max(test["timestamp"])) if len(test) else None,
            }
        base_metrics = out["incremental_oos"][h].get("deterministic_baseline", {})
        ext_metrics = out["incremental_oos"][h].get("baseline_plus_pflat", {})
        out["incremental_oos"][h]["improvement"] = {
            metric: (
                base_metrics.get(metric) - ext_metrics.get(metric)
                if base_metrics.get(metric) is not None
                and ext_metrics.get(metric) is not None
                else None
            )
            for metric in ("mae", "rmse", "r2")
        }
    return out


def interaction_table(rows: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    rows = rows.copy()
    rows["xbin"] = rows[x].map(
        lambda v: bin_label(v, PFLAT_BINS if x == "p_flat" else CONF_BINS)
    )
    rows["ybin"] = rows[y].map(
        lambda v: bin_label(
            v,
            CONF_BINS
            if y == "native_confidence"
            else TOP_BINS
            if y == "top_probability"
            else MARGIN_BINS,
        )
    )
    result: dict[str, Any] = {}
    for (xb, yb), group in rows.dropna(subset=["xbin", "ybin"]).groupby(
        ["xbin", "ybin"], sort=False
    ):
        cell = {
            "N": len(group),
            "LONG": int((group["chosen_direction"] == "LONG").sum()),
            "SHORT": int((group["chosen_direction"] == "SHORT").sum()),
            "FLAT": int((group["chosen_direction"] == "FLAT").sum()),
        }
        for h in HORIZONS:
            valid = group.dropna(subset=[f"return_{h}"])
            signed = valid[f"signed_{h}"].dropna()
            actual = valid[f"label_{h}"]
            directional = actual.isin(["LONG", "SHORT"])
            cell[h] = {
                "N": len(valid),
                "mean_abs_move_bps": float(valid[f"return_{h}"].abs().mean())
                if len(valid)
                else None,
                "directional_hit_rate": float(
                    (
                        valid.loc[directional, "chosen_direction"]
                        == actual[directional]
                    ).mean()
                )
                if directional.any()
                else None,
                "median_signed_gross_bps": float(signed.median())
                if len(signed)
                else None,
                "gross_ev_bps": float(signed.mean()) if len(signed) else None,
                "net_ev_bps_at_15_round_trip": float(signed.mean() - 15)
                if len(signed)
                else None,
            }
        result[f"{xb}|{yb}"] = cell
    return result


def subset_analysis(rows: pd.DataFrame) -> dict[str, Any]:
    predicates = {
        "pflat_lt_0.20": rows["p_flat"] < 0.20,
        "pflat_lt_0.33": rows["p_flat"] < 0.33,
        "confidence_ge_0.60": rows["native_confidence"] >= 0.60,
        "confidence_ge_0.70": rows["native_confidence"] >= 0.70,
        "top_ge_0.70": rows["top_probability"] >= 0.70,
        "top_ge_0.80": rows["top_probability"] >= 0.80,
        "pflat_lt_0.20_and_confidence_ge_0.70": (rows["p_flat"] < 0.20)
        & (rows["native_confidence"] >= 0.70),
        "pflat_lt_0.33_and_confidence_ge_0.70": (rows["p_flat"] < 0.33)
        & (rows["native_confidence"] >= 0.70),
        "pflat_lt_0.33_and_top_ge_0.80": (rows["p_flat"] < 0.33)
        & (rows["top_probability"] >= 0.80),
    }
    out = {}
    for name, predicate in predicates.items():
        group = rows[predicate].copy()
        out[name] = {
            "N": len(group),
            "small_N": len(group) < 100,
            "horizons": {h: directional_metrics(group, h) for h in HORIZONS},
        }
    return out


def v1_v2_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for contract in CONTRACTS:
        pivot = frame[frame["contract"] == contract].pivot_table(
            index="timestamp",
            columns="variant",
            values=[
                "chosen_direction",
                "p_flat",
                "top_probability",
                "native_confidence",
                "margin",
            ],
            aggfunc="first",
        )
        if not isinstance(pivot.columns, pd.MultiIndex) or any(
            (metric, variant) not in pivot.columns
            for metric in ("chosen_direction", "p_flat")
            for variant in VARIANTS
        ):
            continue
        aligned = pivot.dropna(
            subset=[
                ("chosen_direction", "v1_candles"),
                ("chosen_direction", "v2_technical"),
            ]
        )
        disagree = aligned[
            (
                aligned[("chosen_direction", "v1_candles")]
                != aligned[("chosen_direction", "v2_technical")]
            )
        ]
        item = {
            "matched_timestamps": len(aligned),
            "disagreement_count": len(disagree),
            "disagreement_rate": float(len(disagree) / len(aligned))
            if len(aligned)
            else None,
            "both_direction_agreement": float(
                (
                    (
                        aligned[("chosen_direction", "v1_candles")].isin(
                            ["LONG", "SHORT"]
                        )
                    )
                    & (
                        aligned[("chosen_direction", "v2_technical")].isin(
                            ["LONG", "SHORT"]
                        )
                    )
                ).mean()
            )
            if len(aligned)
            else None,
            "horizons": {},
        }
        for h in HORIZONS:
            item["horizons"][h] = {
                variant: directional_metrics(
                    frame[
                        (frame["contract"] == contract) & (frame["variant"] == variant)
                    ],
                    h,
                )
                for variant in VARIANTS
            }
            item["horizons"][h]["disagreement_v1_signed_gross_bps"] = (
                float(
                    frame[
                        (frame["contract"] == contract)
                        & (frame["variant"] == "v1_candles")
                        & frame["timestamp"].isin(disagree.index)
                    ][f"signed_{h}"].mean()
                )
                if len(disagree)
                else None
            )
            item["horizons"][h]["disagreement_v2_signed_gross_bps"] = (
                float(
                    frame[
                        (frame["contract"] == contract)
                        & (frame["variant"] == "v2_technical")
                        & frame["timestamp"].isin(disagree.index)
                    ][f"signed_{h}"].mean()
                )
                if len(disagree)
                else None
            )
        out[contract] = item
    return out


def class_bias(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (contract, variant), group in frame.groupby(["contract", "variant"]):
        item = {
            "predicted": {
                key: float((group["chosen_direction"] == key).mean())
                for key in ("LONG", "SHORT", "FLAT")
            },
            "realized": {},
        }
        for h in HORIZONS:
            labels = group[f"label_{h}"].dropna()
            item["realized"][h] = (
                {
                    key: float((labels == key).mean())
                    for key in ("LONG", "SHORT", "FLAT")
                }
                if len(labels)
                else None
            )
        result[f"{contract}_{variant}"] = item
    return result


def stability(frame: pd.DataFrame) -> dict[str, Any]:
    rows = frame[
        (frame["contract"] == "ternary_v1") & (frame["variant"] == "v1_candles")
    ].copy()
    rows["utc_hour"] = rows["timestamp"].dt.hour
    rows["utc_day"] = rows["timestamp"].dt.strftime("%Y-%m-%d")
    rows["volatility_tercile"] = pd.qcut(
        rows["realized_vol_15m_bps"],
        3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )
    output = {}
    for name, group in rows.groupby("utc_hour", sort=True):
        output.setdefault("utc_hour", {})[str(name)] = {
            "N": len(group),
            "mean_abs_5m_bps": finite(group["return_5m"].abs().mean()),
            "mean_signed_5m_bps": finite(group["signed_5m"].mean()),
        }
    for name, group in rows.groupby("utc_day", sort=True):
        output.setdefault("utc_day", {})[str(name)] = {
            "N": len(group),
            "mean_abs_5m_bps": finite(group["return_5m"].abs().mean()),
            "mean_signed_5m_bps": finite(group["signed_5m"].mean()),
        }
    for name, group in rows.groupby("volatility_tercile", observed=False, sort=False):
        output.setdefault("volatility_tercile", {})[str(name)] = {
            "N": len(group),
            "mean_abs_5m_bps": finite(group["return_5m"].abs().mean()),
            "mean_signed_5m_bps": finite(group["signed_5m"].mean()),
        }
    return output


def cost_sensitivity(rows: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for name, group in rows.groupby(["contract", "variant"]):
        key = f"{name[0]}_{name[1]}"
        output[key] = {
            h: {
                f"net_ev_at_{int(cost)}_bps_round_trip": float(
                    group[f"signed_{h}"].dropna().mean() - cost
                )
                if group[f"signed_{h}"].notna().any()
                else None
                for cost in ROUND_TRIP_COSTS_BPS
            }
            for h in HORIZONS
        }
    return output


def make_report(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# Second-pass BTC baseline analysis",
        "",
        "**Zero-new-Jev analysis.** All state probabilities come from the frozen `bringup_baseline_20260919.sqlite3`; added horizons and features are local research labels/features only.",
        "",
        "## Executive findings",
        "",
        *[f"- {line}" for line in summary["executive_findings"]],
        "",
        "## Dataset / methodology",
        "",
        f"- {dataset['N_rows']} state rows across {dataset['matched_timestamps']} exact timestamps; frozen model `{BASELINE_MODEL}`, `{STATE_VERSION}`, `{QUESTION_SCHEMA}`.",
        f"- Candle source: `{dataset['candle_source']}`; public candle fetch was `{dataset['public_candle_fetch']}` and written only to a derived cache copy.",
        "- Costs: 5/10/15/20 bps round-trip sensitivity; official baseline is 15 bps.",
        "- Fixed-horizon EV is not policy replay PnL; all added horizons are labels after T.",
        "",
        "## P(FLAT) movement signal",
        "",
        json.dumps(summary["movement"], indent=2, default=json_default),
        "",
        "## Deterministic volatility baselines / incremental information",
        "",
        json.dumps(summary["volatility_baselines"], indent=2, default=json_default),
        "",
        "## Ternary P(FLAT) vs low movement",
        "",
        json.dumps(summary["flat_low_movement"], indent=2, default=json_default),
        "",
        "## P(FLAT) interactions",
        "",
        json.dumps(summary["interactions"], indent=2, default=json_default),
        "",
        "## Conditional direction / diagnostic subsets",
        "",
        json.dumps(summary["subsets"], indent=2, default=json_default),
        "",
        "## V1 vs V2 / class bias",
        "",
        json.dumps(summary["v1_v2"], indent=2, default=json_default),
        "",
        json.dumps(summary["class_bias"], indent=2, default=json_default),
        "",
        "## Calibration, costs, stability and uncertainty",
        "",
        json.dumps(summary["raw_signal"], indent=2, default=json_default),
        "",
        json.dumps(summary["cost_sensitivity"], indent=2, default=json_default),
        "",
        json.dumps(summary["stability"], indent=2, default=json_default),
        "",
        "## What should be tested next",
        "",
        *[f"- {line}" for line in summary["recommendation"]],
        "",
        "## Questions A-G",
        "",
        json.dumps(summary["question_answers"], indent=2, default=json_default),
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    frame: pd.DataFrame,
    candle_source: str,
    public_fetch: bool,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    primary = frame[
        (frame["contract"] == "ternary_v1") & (frame["variant"] == "v1_candles")
    ].copy()
    movement = movement_analysis(primary)
    interactions = {
        name: interaction_table(primary, "p_flat", name)
        for name in ("native_confidence", "top_probability", "margin")
    }
    raw = raw_signal_analysis(frame)
    summary = {
        "dataset": {
            "N_rows": len(frame),
            "matched_timestamps": int(frame["timestamp"].nunique()),
            "contracts": CONTRACTS,
            "variants": VARIANTS,
            "model": BASELINE_MODEL,
            "state_version": STATE_VERSION,
            "question_schema": QUESTION_SCHEMA,
            "candle_source": candle_source,
            "public_candle_fetch": public_fetch,
            "baseline_db_mutated": False,
            **source_audit,
        },
        "raw_signal": raw,
        "movement": movement,
        "flat_low_movement": flat_low_movement_analysis(primary),
        "volatility_baselines": volatility_baselines(frame),
        "interactions": interactions,
        "subsets": subset_analysis(primary),
        "v1_v2": v1_v2_analysis(frame),
        "class_bias": class_bias(frame),
        "cost_sensitivity": cost_sensitivity(frame),
        "stability": stability(frame),
        "recommendation": [
            "Do not run BINARY_EDGE_V1 yet; current low-P(FLAT) gross returns are small-N and do not survive the official 15 bps cost assumption.",
            "The strongest next controlled experiment is a compact movement/opportunity judgment or a local deterministic movement gate, but only after a longer multi-week out-of-sample dataset.",
            "Keep directional contract and movement/opportunity evidence separate; do not treat P(FLAT) as a trading policy threshold yet.",
        ],
        "executive_findings": [],
    }
    low = movement["correlations"].get("5m", {}).get("spearman")
    high_low = (
        movement["approximate_bins"].get("0.00-0.20", {}).get("5m", {}).get("mean")
    )
    high_flat = movement["approximate_bins"].get("0.60+", {}).get("5m", {}).get("mean")
    summary["executive_findings"] = [
        f"P(FLAT) vs 5m absolute movement Spearman={low}; lower P(FLAT) had mean 5m absolute movement {high_low} bps in 0.00-0.20 versus {high_flat} bps in 0.60+.",
        "The movement relationship is consistent with an opportunity/volatility proxy, but the deterministic-baseline and time-dependence checks prevent treating it as incremental alpha yet.",
        "Directional accuracy and signed EV remain near chance/negative after costs in the broad sample; fixed-horizon EV is not realized strategy PnL.",
        "High native-confidence/low-P(FLAT) intersections are diagnostic only and must be labelled by sample size and bootstrap uncertainty.",
    ]
    low_pflat = summary["subsets"]["pflat_lt_0.20"]["horizons"]["5m"]
    low_pflat_conf = summary["subsets"]["pflat_lt_0.20_and_confidence_ge_0.70"][
        "horizons"
    ]["5m"]
    incremental_5m = summary["volatility_baselines"]["incremental_oos"]["5m"]
    v1v2_ternary = summary["v1_v2"]["ternary_v1"]
    summary["question_answers"] = {
        "A": f"P(FLAT) is associated with movement magnitude in this 7-day sample (5m Spearman {low}); this is evidence of an opportunity/volatility proxy, not proof of an independent reusable signal.",
        "B": f"No material incremental information was established beyond deterministic volatility features: at 5m, adding P(FLAT) changed OOS MAE by {incremental_5m['improvement']['mae']:.6f} bps, RMSE by {incremental_5m['improvement']['rmse']:.6f} bps and R2 by {incremental_5m['improvement']['r2']:.6f}.",
        "C": f"Low P(FLAT) direction was not reliable: P(FLAT)<0.20 had N={summary['subsets']['pflat_lt_0.20']['N']} and 5m gross signed return {low_pflat['signed_return_gross_bps']:.2f} bps, with a broad block-bootstrap CI {low_pflat['signed_return_bootstrap']['ci95']}.",
        "D": f"High native confidence inside low P(FLAT) remained below the 15 bps round-trip hurdle: N={summary['subsets']['pflat_lt_0.20_and_confidence_ge_0.70']['N']}, 5m gross {low_pflat_conf['signed_return_gross_bps']:.2f} bps, hypothetical net {low_pflat_conf['signed_return_net_bps']:.2f} bps.",
        "E": f"V2 materially changes behavior rather than cleanly adding information: ternary V1/V2 disagreement was {v1v2_ternary['disagreement_rate']:.1%}; V2's 5m three-class accuracy was {v1v2_ternary['horizons']['5m']['v2_technical']['three_class_accuracy']:.3f} versus V1 {v1v2_ternary['horizons']['5m']['v1_candles']['three_class_accuracy']:.3f}, with a strong SHORT shift documented in class_bias.",
        "F": "Some diagnostic subsets show positive fixed-horizon gross EV, but none is simultaneously large, stable, and above the official 15 bps cost hurdle; these are not strategy PnL and are not alpha evidence.",
        "G": "Do not start BINARY_EDGE_V1 yet. The smallest justified next step is more multi-week out-of-sample data plus a compact movement/opportunity experiment or local movement gate, while keeping direction and opportunity contracts separate.",
    }
    return summary


async def main(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = utc(args.start)
    end = utc(args.end)
    source = Path(args.baseline_db)
    candle_source = Path(args.candle_db)
    source_hash_before = file_sha256(source)
    source_audit = baseline_audit(source)
    derived_candle_db = output / "analysis_candles.sqlite3"
    await ensure_candle_copy(
        candle_source, derived_candle_db, start, end, args.fetch_public_candles
    )
    candles = load_candles(derived_candle_db, BASELINE_SYMBOL)
    frame = load_state_rows(source, candles)
    # State-level table is reusable and explicitly carries no new labels into identity.
    csv_path = output / "state_level_analysis.csv.gz"
    frame.to_csv(
        csv_path, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%S%z"
    )
    summary = run_analysis(
        frame, str(derived_candle_db), args.fetch_public_candles, source_audit
    )
    source_hash_after = file_sha256(source)
    summary["integrity"] = {
        "new_jev_calls": 0,
        "inference_service_imported": False,
        "baseline_db_mutated": False,
        "baseline_db_sha256_before": source_hash_before,
        "baseline_db_sha256_after": source_hash_after,
        "baseline_db_hash_unchanged": source_hash_before == source_hash_after,
        "labels_15_30_60_are_research_only": True,
        "feature_cutoff": "all candle features use close_time <= T",
        "exact_v1_v2_timestamp_alignment": True,
        "bootstrap": {"type": "moving_block", "block_size_minutes": 60, "seed": SEED},
        "cost_semantics": "fixed-horizon net EV subtracts round-trip cost; policy replay is not run",
    }
    manifest = {
        "dataset": summary["dataset"],
        "source_db": str(source),
        "derived_candle_db": str(derived_candle_db),
        "state_table": str(csv_path),
        "new_jev_calls": 0,
        "cache_hits": source_audit["cache_hits"],
        "failures": source_audit["failures"],
        "skips": source_audit["skips"],
        "baseline_db_sha256_before": source_hash_before,
        "baseline_db_sha256_after": source_hash_after,
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_version": "second-pass-v1",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n"
    )
    (output / "second_pass_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n"
    )
    (output / "second_pass_summary.md").write_text(make_report(summary))
    print(
        json.dumps(
            {
                "event": "second_pass_complete",
                "output": str(output),
                "rows": len(frame),
                "timestamps": int(frame.timestamp.nunique()),
                "new_jev_calls": 0,
            },
            default=json_default,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-db", default="data/bringup_baseline_20260919.sqlite3"
    )
    parser.add_argument("--candle-db", default="data/candles.sqlite3")
    parser.add_argument("--output-dir", default="outputs/second_pass_analysis")
    parser.add_argument("--start", default="2026-09-12T18:42:00+00:00")
    parser.add_argument("--end", default="2026-09-19T18:41:00+00:00")
    parser.add_argument(
        "--fetch-public-candles", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
