#!/usr/bin/env python3
"""Analyze persisted market-opportunity-v1 Jev outputs without new calls."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = (
    "move_5m_15bps",
    "move_15m_15bps",
    "move_30m_15bps",
    "long_15m_15bps",
    "short_15m_15bps",
    "long_30m_15bps",
    "short_30m_15bps",
)
HORIZONS = (1, 3, 5, 10, 15, 30)
ROUND_TRIP_COST_BPS = 15.0


def _bins(value: float) -> str:
    for label, lower, upper in (
        ("0.00-0.20", 0.00, 0.20),
        ("0.20-0.30", 0.20, 0.30),
        ("0.30-0.40", 0.30, 0.40),
        ("0.40-0.50", 0.40, 0.50),
        ("0.50-0.60", 0.50, 0.60),
        ("0.60+", 0.60, None),
    ):
        if value >= lower and (upper is None or value < upper):
            return label
    return "outside"


def _score(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
            "mean_probability": None,
            "observed_rate": None,
        }
    clipped = np.clip(values, 1e-9, 1 - 1e-9)
    return {
        "n": len(values),
        "brier": float(np.mean((values - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        ),
        "mean_probability": float(values.mean()),
        "observed_rate": float(labels.mean()),
    }


def _reliability(values: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Fixed, predeclared calibration bins; no threshold search is performed."""
    edges = np.linspace(0.0, 1.0, 11)
    bins = []
    ece = 0.0
    for lower, upper in pairwise(edges):
        mask = (values >= lower) & (values < upper if upper < 1.0 else values <= upper)
        n = int(mask.sum())
        if n:
            mean_prediction = float(values[mask].mean())
            observed_rate = float(labels[mask].mean())
            ece += n / len(values) * abs(mean_prediction - observed_rate)
        else:
            mean_prediction = None
            observed_rate = None
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": n,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
            }
        )
    return {"ece": float(ece), "bins": bins}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _fit_logistic_oos(
    features: np.ndarray, labels: np.ndarray, jev_probability: np.ndarray
) -> dict[str, object]:
    """Small chronological logistic comparison using only the frozen records.

    The first 70% is the selection/training window and the final 30% is untouched
    validation.  This is deliberately a diagnostic model, not a policy search.
    """
    finite = (
        np.isfinite(features).all(axis=1)
        & np.isfinite(labels)
        & np.isfinite(jev_probability)
    )
    features, labels, jev_probability = (
        features[finite],
        labels[finite],
        jev_probability[finite],
    )
    split = int(len(labels) * 0.70)
    if split < 100 or len(labels) - split < 100:
        return {
            "available": False,
            "reason": "insufficient finite chronological samples",
        }

    def fit_predict(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
    ) -> np.ndarray:
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        x_train = (train_x - mean) / scale
        x_test = (test_x - mean) / scale
        x_train = np.c_[np.ones(len(x_train)), x_train]
        x_test = np.c_[np.ones(len(x_test)), x_test]
        weights = np.zeros(x_train.shape[1], dtype=float)
        # Conservative gradient descent with deterministic regularisation.
        for _ in range(1200):
            probability = _sigmoid(x_train @ weights)
            gradient = (x_train.T @ (probability - train_y)) / len(train_y)
            gradient[1:] += 0.01 * weights[1:]
            weights -= 0.25 * gradient
        return _sigmoid(x_test @ weights)

    train_y, valid_y = labels[:split], labels[split:]
    baseline = fit_predict(features[:split], train_y, features[split:])
    extended_features = np.c_[features, jev_probability]
    extended = fit_predict(
        extended_features[:split], train_y, extended_features[split:]
    )

    def score(probability: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        clipped = np.clip(probability, 1e-9, 1 - 1e-9)
        return {
            "n": len(actual),
            "brier": float(np.mean((probability - actual) ** 2)),
            "log_loss": float(
                -np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))
            ),
            "observed_rate": float(actual.mean()),
        }

    return {
        "available": True,
        "target": "move_15m_15bps",
        "selection_fraction": 0.70,
        "selection_n": int(split),
        "validation_n": len(valid_y),
        "baseline_features": ["rv5", "rv15", "rv60", "atr14_pct", "range_pct"],
        "baseline_validation": score(baseline, valid_y),
        "baseline_plus_jev_validation": score(extended, valid_y),
        "incremental_brier_improvement": float(
            score(baseline, valid_y)["brier"] - score(extended, valid_y)["brier"]
        ),
        "incremental_log_loss_improvement": float(
            score(baseline, valid_y)["log_loss"] - score(extended, valid_y)["log_loss"]
        ),
    }


def _top_margin(row: dict[str, object]) -> float:
    values = sorted((float(row[field]) for field in FIELDS), reverse=True)
    return values[0] - values[1]


def run(cache_db: Path, candle_db: Path, output_dir: Path) -> dict:
    with sqlite3.connect(cache_db) as connection:
        records = connection.execute(
            """select state_timestamp, probabilities_json, input_tokens, output_tokens
               from expert_inferences where symbol='PF_XBTUSD' and jev_model='jev-1.13.0'
               and direction_contract='opportunity_v1' and question_schema_version='market-opportunity-v1'
               and status='READY' order by state_timestamp"""
        ).fetchall()
    with sqlite3.connect(candle_db) as connection:
        candles = pd.read_sql_query(
            "select close_time, open, high, low, close, volume from candles_1m where symbol='PF_XBTUSD' order by close_time",
            connection,
        )
    candles["close_time"] = pd.to_datetime(candles["close_time"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        candles[column] = candles[column].astype(float)
    one_step = candles["close"].pct_change()
    candles["rv5"] = one_step.rolling(5).std() * np.sqrt(5)
    candles["rv15"] = one_step.rolling(15).std() * np.sqrt(15)
    candles["rv60"] = one_step.rolling(60).std() * np.sqrt(60)
    true_range = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - candles["close"].shift()).abs(),
            (candles["low"] - candles["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    candles["atr14_pct"] = true_range.rolling(14).mean() / candles["close"]
    candles["range_pct"] = (candles["high"] - candles["low"]) / candles["close"]
    prices = candles.set_index("close_time")["close"]
    candle_features = candles.set_index("close_time")
    rows = []
    for timestamp, raw, _, _ in records:
        ts = (
            pd.Timestamp(timestamp, tz="UTC")
            if not str(timestamp).endswith("+00:00")
            else pd.Timestamp(timestamp)
        )
        if ts not in prices.index:
            continue
        values = json.loads(raw)
        current = float(prices.loc[ts])
        returns = {
            h: float(prices.loc[ts + pd.to_timedelta(int(h), unit="m")] / current - 1)
            * 10_000
            if ts + pd.to_timedelta(int(h), unit="m") in prices.index
            else np.nan
            for h in HORIZONS
        }
        rows.append(
            {
                "timestamp": ts.isoformat(),
                **values,
                **{f"return_{h}m_bps": v for h, v in returns.items()},
                **{
                    feature: float(candle_features.loc[ts, feature])
                    for feature in ("rv5", "rv15", "rv60", "atr14_pct", "range_pct")
                },
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        summary = {"sample_count": 0, "note": "No complete persisted Jev records yet."}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "jev_opportunity_analysis.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    total_input = sum(int(row[2] or 0) for row in records)
    total_output = sum(int(row[3] or 0) for row in records)
    raw_scores = {}
    for field in FIELDS:
        labels = (
            np.array([1 if abs(r["return_5m_bps"]) >= 15 else 0 for r in rows])
            if field == "move_5m_15bps"
            else None
        )
        if field.startswith("move_"):
            horizon = int(field.split("_")[1].replace("m", ""))
            labels = np.array(
                [1 if abs(r[f"return_{horizon}m_bps"]) >= 15 else 0 for r in rows]
            )
        elif field.startswith("long_"):
            horizon = int(field.split("_")[1].replace("m", ""))
            labels = np.array(
                [1 if r[f"return_{horizon}m_bps"] >= 15 else 0 for r in rows]
            )
        else:
            horizon = int(field.split("_")[1].replace("m", ""))
            labels = np.array(
                [1 if r[f"return_{horizon}m_bps"] <= -15 else 0 for r in rows]
            )
        values = np.array([float(r[field]) for r in rows])
        raw_scores[field] = {
            **_score(values, labels),
            "min": float(values.min()),
            "max": float(values.max()),
            "reliability": _reliability(values, labels),
        }

    directional = {}
    for horizon in (15, 30):
        long_field, short_field = f"long_{horizon}m_15bps", f"short_{horizon}m_15bps"
        choices = np.array([1 if r[long_field] >= r[short_field] else -1 for r in rows])
        realized = np.array([r[f"return_{horizon}m_bps"] for r in rows], dtype=float)
        signed = choices * realized
        directional[str(horizon)] = {
            "n": len(signed),
            "direction_hit_rate_15bps": float(np.mean(signed >= 15)),
            "mean_gross_directional_return_bps": float(np.mean(signed)),
            "median_gross_directional_return_bps": float(np.median(signed)),
            "mean_net_directional_return_bps": float(
                np.mean(signed - ROUND_TRIP_COST_BPS)
            ),
            "cost_semantics": "15 bps round-trip",
        }

    top = np.array([max(float(r[field]) for field in FIELDS) for r in rows])
    margins = np.array([_top_margin(r) for r in rows])
    bins = {}
    for name, values in (("top_probability", top), ("top_second_margin", margins)):
        grouped = defaultdict(list)
        for row, value in zip(rows, values, strict=False):
            grouped[_bins(float(value))].append(row)
        bins[name] = {}
        for label, grouped_rows in grouped.items():
            chosen = np.array(
                [
                    max(float(r["long_15m_15bps"]), float(r["short_15m_15bps"]))
                    for r in grouped_rows
                ]
            )
            realized = np.array(
                [
                    (
                        1
                        if float(r["long_15m_15bps"]) >= float(r["short_15m_15bps"])
                        else -1
                    )
                    * r["return_15m_bps"]
                    for r in grouped_rows
                ]
            )
            bins[name][label] = {
                "n": len(grouped_rows),
                "mean_probability": float(chosen.mean()),
                "mean_gross_directional_return_bps": float(np.mean(realized)),
                "mean_net_directional_return_bps": float(
                    np.mean(realized - ROUND_TRIP_COST_BPS)
                ),
            }

    move_labels = np.array(
        [1 if abs(r["return_15m_bps"]) >= 15 else 0 for r in rows], dtype=float
    )
    baseline_features = np.array(
        [
            [r["rv5"], r["rv15"], r["rv60"], r["atr14_pct"], r["range_pct"]]
            for r in rows
        ],
        dtype=float,
    )
    incremental_value = _fit_logistic_oos(
        baseline_features,
        move_labels,
        np.array([float(r["move_15m_15bps"]) for r in rows], dtype=float),
    )

    summary = {
        "contract": "opportunity_v1",
        "question_schema_version": "market-opportunity-v1",
        "sample_count": len(rows),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "native_jev_confidence": {
            "available": False,
            "reason": "Noul answers return native probability only; no confidence field.",
        },
        "top_probability_margin_definition": "max and top-minus-second are computed across the seven independent Noul probabilities for diagnostic binning; they are not a normalized directional distribution.",
        "raw_question_scores": raw_scores,
        "directional": directional,
        "probability_margin_bins": bins,
        "incremental_value": incremental_value,
        "cost_semantics": "15 bps round-trip = two fee/slippage sides plus both half-spread sides in the configured model.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "jev_opportunity_states.csv.gz", index=False, compression="gzip"
    )
    (output_dir / "jev_opportunity_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Jev market-opportunity-v1 raw analysis",
        "",
        f"Samples: {len(rows)}",
        "",
        "| Question | N | Mean p | Observed | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, score in raw_scores.items():
        lines.append(
            f"| {field} | {score['n']} | {score['mean_probability']:.4f} | {score['observed_rate']:.4f} | {score['brier']:.4f} | {score['log_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Directional economic diagnostics",
            "",
            "| Horizon | N | Hit rate | Mean gross bps | Mean net bps |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for horizon, score in directional.items():
        lines.append(
            f"| {horizon}m | {score['n']} | {score['direction_hit_rate_15bps']:.4f} | {score['mean_gross_directional_return_bps']:.3f} | {score['mean_net_directional_return_bps']:.3f} |"
        )
    (output_dir / "jev_opportunity_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-db", default="data/overnight_research.sqlite3")
    parser.add_argument("--candle-db", default="data/overnight_candles.sqlite3")
    parser.add_argument("--output-dir", default="outputs/overnight_research")
    args = parser.parse_args()
    print(
        json.dumps(
            run(Path(args.cache_db), Path(args.candle_db), Path(args.output_dir)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
