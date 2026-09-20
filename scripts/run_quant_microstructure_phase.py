#!/usr/bin/env python3
"""Run the no-Jev quantitative/microstructure phase.

The collector stores raw book/trade events and a mixed derived stream.  This
script freezes a read boundary, audits both streams, builds a 1-second canonical
table, runs simple OOS ablations, and evaluates paper-only maker fills.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from run_classical_quant import load_frame
from run_corrected_classical import add_corrected_features, candidate_layers

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MICRO = ROOT / "outputs" / "overnight_research" / "microstructure"
DEFAULT_CANDLES = ROOT / "data" / "overnight_candles.sqlite3"
DEFAULT_OUT = ROOT / "outputs" / "quant_microstructure_phase"
SYMBOLS = ("PF_XBTUSD", "PF_ETHUSD")
HORIZONS = (1, 5, 10, 30, 60)
NOTIONAL = 1_000.0
OFFICIAL_COST_BPS = 15.0


def _ts(value: object) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        return parsed.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _event_timestamp(record: dict) -> pd.Timestamp | None:
    message = record.get("message", {})
    if message.get("feed") in {"trade", "trade_snapshot"} and message.get("time"):
        return _ts(float(message["time"]) / 1000.0)
    return _ts(record.get("received_at"))


def _received_second(value: object) -> str | None:
    if not isinstance(value, str) or len(value) < 19:
        return None
    return value[:19] + "+00:00"


def _trade_second(value: object) -> str | None:
    try:
        return (
            datetime.fromtimestamp(float(value) / 1000.0, UTC)
            .replace(microsecond=0)
            .isoformat()
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _raw_files(root: Path) -> list[Path]:
    return sorted((root / "raw").glob("**/*.jsonl"))


def _derived_files(root: Path) -> list[Path]:
    return sorted((root / "derived").glob("*.jsonl"))


def _snapshot_lines(path: Path, size: int):
    """Read exactly the bytes present at the snapshot boundary."""
    remaining = size
    with path.open("rb") as handle:
        while remaining > 0:
            raw = handle.readline()
            if not raw:
                break
            remaining -= len(raw)
            yield raw


def audit_raw(root: Path, cutoff: pd.Timestamp) -> dict[str, object]:
    counts: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    sides: Counter[str] = Counter()
    first: str | None = None
    last: str | None = None
    bad_json = 0
    lines = 0
    cutoff_second = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
    paths = _raw_files(root)
    sizes = {path: path.stat().st_size for path in paths}
    for path in paths:
        for raw_line in _snapshot_lines(path, sizes[path]):
            line = raw_line.decode("utf-8", errors="replace")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            received_text = _received_second(record.get("received_at"))
            if received_text is None or received_text[:19] > cutoff_second:
                continue
            lines += 1
            message = record.get("message", {})
            feed = str(message.get("feed") or message.get("event") or "unknown")
            symbol = str(record.get("symbol") or message.get("product_id") or "unknown")
            counts[feed] += 1
            symbols[symbol] += 1
            event_ts = _trade_second(message.get("time")) or received_text
            first = event_ts if first is None else min(first, event_ts)
            last = event_ts if last is None else max(last, event_ts)
            if feed == "trade":
                sides[str(message.get("side"))] += 1
            elif feed == "trade_snapshot":
                for trade in message.get("trades", []):
                    sides[str(trade.get("side"))] += 1
    events_path = root / "collector_events.jsonl"
    reconnects = 0
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            reconnects = sum(1 for line in handle if "reconnect" in line)
    return {
        "cutoff": cutoff.isoformat(),
        "files": [str(path.relative_to(ROOT)) for path in _raw_files(root)],
        "raw_lines": lines,
        "bad_json": bad_json,
        "symbols": dict(symbols),
        "feed_counts": dict(counts),
        "trade_side_counts": dict(sides),
        "first_event": first,
        "last_event": last,
        "reconnect_events": reconnects,
        "side_semantics": "Kraken trade side is preserved as taker-side metadata; no aggressor side is inferred locally.",
    }


def build_canonical(
    root: Path, cutoff: pd.Timestamp
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    buckets: dict[str, dict[str, dict[str, float | bool | str]]] = defaultdict(dict)
    derived_counts: Counter[str] = Counter()
    invalid_seconds: Counter[str] = Counter()
    first: str | None = None
    last: str | None = None
    cutoff_second = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
    paths = _derived_files(root)
    sizes = {path: path.stat().st_size for path in paths}
    for path in paths:
        for raw_line in _snapshot_lines(path, sizes[path]):
            line = raw_line.decode("utf-8", errors="replace")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            received_text = _received_second(record.get("received_at"))
            if received_text is None or received_text[:19] > cutoff_second:
                continue
            symbol = str(record.get("symbol", ""))
            if symbol not in SYMBOLS:
                continue
            source = str(record.get("source_feed", ""))
            derived_counts[source] += 1
            second = (
                _trade_second(record.get("trade_time_ms"))
                if source == "trade"
                else received_text
            ) or received_text
            first = second if first is None else min(first, second)
            last = second if last is None else max(last, second)
            row = buckets[symbol].setdefault(
                second,
                {
                    "book_valid": True,
                    "has_book": False,
                    "trade_count": 0.0,
                    "buy_qty": 0.0,
                    "sell_qty": 0.0,
                    "trade_notional": 0.0,
                },
            )
            if source in {"book", "book_snapshot"} and record.get("mid") is not None:
                for key in (
                    "best_bid",
                    "best_ask",
                    "mid",
                    "spread_bps",
                    "bid_depth_l5",
                    "ask_depth_l5",
                    "l5_imbalance",
                ):
                    if record.get(key) is not None:
                        row[key] = float(record[key])
                row["book_valid"] = bool(
                    row["book_valid"] and record.get("sequence_valid", True)
                )
                row["has_book"] = True
                if not row["book_valid"]:
                    invalid_seconds[symbol] += 1
            elif source == "trade":
                signed = float(record.get("signed_qty") or 0.0)
                row["trade_count"] = float(row["trade_count"]) + 1.0
                row["buy_qty"] = float(row["buy_qty"]) + max(signed, 0.0)
                row["sell_qty"] = float(row["sell_qty"]) + max(-signed, 0.0)
                row["trade_notional"] = float(row["trade_notional"]) + float(
                    record.get("vwap_notional") or 0.0
                )

    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in buckets.items():
        frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        if frame.empty:
            continue
        frame.index = pd.DatetimeIndex(frame.index)
        frame.index = (
            frame.index.tz_localize("UTC")
            if frame.index.tz is None
            else frame.index.tz_convert("UTC")
        )
        full_index = pd.date_range(
            frame.index.min(), frame.index.max(), freq="1s", tz="UTC"
        )
        frame = frame.reindex(full_index)
        for column in ("trade_count", "buy_qty", "sell_qty", "trade_notional"):
            frame[column] = frame[column].fillna(0.0)
        book_columns = [
            "best_bid",
            "best_ask",
            "mid",
            "spread_bps",
            "bid_depth_l5",
            "ask_depth_l5",
            "l5_imbalance",
        ]
        frame[book_columns] = frame[book_columns].ffill(limit=5)
        frame["book_valid"] = (
            frame["book_valid"].astype("boolean").fillna(True).astype(bool)
        )
        frame["has_book"] = (
            frame["has_book"].astype("boolean").fillna(False).astype(bool)
        )
        last_book = pd.Series(
            frame.index.where(frame["has_book"]), index=frame.index
        ).ffill()
        frame["book_age_seconds"] = (
            frame.index.to_series() - last_book
        ).dt.total_seconds()
        frame["book_valid"] = (
            frame["book_valid"] & frame["book_age_seconds"].le(5) & frame["mid"].notna()
        )
        depth_total = frame["bid_depth_l5"] + frame["ask_depth_l5"]
        frame["microprice"] = (
            frame["best_ask"] * frame["bid_depth_l5"]
            + frame["best_bid"] * frame["ask_depth_l5"]
        ) / depth_total.replace(0, np.nan)
        frame["microprice_displacement_bps"] = (
            frame["microprice"] / frame["mid"] - 1.0
        ) * 10_000
        frame["trade_flow_imbalance_1s"] = (frame["buy_qty"] - frame["sell_qty"]) / (
            frame["buy_qty"] + frame["sell_qty"]
        ).replace(0, np.nan)
        frame["ofi_proxy_1s"] = (
            frame["bid_depth_l5"].diff() - frame["ask_depth_l5"].diff()
        )
        for window in (1, 5, 10, 30, 60):
            frame[f"buy_qty_{window}s"] = (
                frame["buy_qty"].rolling(window, min_periods=1).sum()
            )
            frame[f"sell_qty_{window}s"] = (
                frame["sell_qty"].rolling(window, min_periods=1).sum()
            )
            total = frame[f"buy_qty_{window}s"] + frame[f"sell_qty_{window}s"]
            frame[f"trade_flow_imbalance_{window}s"] = (
                frame[f"buy_qty_{window}s"] - frame[f"sell_qty_{window}s"]
            ) / total.replace(0, np.nan)
            frame[f"ofi_proxy_{window}s"] = (
                frame["ofi_proxy_1s"].rolling(window, min_periods=1).sum()
            )
        frame["mid_return_1s"] = frame["mid"].pct_change(fill_method=None)
        frame["rv_60s"] = frame["mid_return_1s"].rolling(
            60, min_periods=20
        ).std() * math.sqrt(60)
        frame["spread_change_bps"] = frame["spread_bps"].diff()
        for horizon in HORIZONS:
            frame[f"future_mid_return_{horizon}s_bps"] = (
                frame["mid"].shift(-horizon) / frame["mid"] - 1.0
            ) * 10_000
        frame = frame.reset_index(names="timestamp")
        frame["symbol"] = symbol
        frames[symbol] = frame
    audit = {
        "cutoff": cutoff.isoformat(),
        "derived_files": [str(path.relative_to(ROOT)) for path in _derived_files(root)],
        "derived_counts": dict(derived_counts),
        "invalid_book_event_seconds": dict(invalid_seconds),
        "symbols": {symbol: len(frame) for symbol, frame in frames.items()},
        "valid_book_rows": {
            symbol: int(frame["book_valid"].sum()) for symbol, frame in frames.items()
        },
        "first_derived_event": first,
        "last_derived_event": last,
        "l1_l10_depth": "unavailable: collector persisted L5 depth only",
        "ofi_semantics": "depth-change OFI proxy only; true price-level OFI is not claimed from this feed snapshot.",
    }
    return frames, audit


def _ridge_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[scale < 1e-12] = 1.0
    x_train = np.nan_to_num((train_x - mean) / scale)
    x_test = np.nan_to_num((test_x - mean) / scale)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    regularizer = np.eye(x_train.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ train_y)
    return x_test @ weights


def microstructure_analysis(
    frames: dict[str, pd.DataFrame], candles: Path, out: Path
) -> dict[str, object]:
    frame = frames.get("PF_XBTUSD", pd.DataFrame()).copy()
    if frame.empty:
        return {"status": "no_valid_xbt_states"}
    btc = add_corrected_features(
        load_frame(candles, "PF_XBTUSD"), load_frame(candles, "PF_ETHUSD")
    )
    btc = btc.reindex(pd.DatetimeIndex(frame["timestamp"]), method="ffill")
    frame.index = pd.DatetimeIndex(frame["timestamp"])
    for column in ("r5", "vol60", "er20"):
        frame[column] = btc[column].to_numpy()
    frame["m0_score"] = frame["r5"]
    frame["m0_candidate"] = candidate_layers(btc, n=20)[
        "S3_BREAKOUT_VOL_ER_VOLUME"
    ].to_numpy()
    features = {
        "M0": ["m0_score", "vol60"],
        "M1": ["m0_score", "vol60", "ofi_proxy_10s"],
        "M2": ["m0_score", "vol60", "ofi_proxy_10s", "trade_flow_imbalance_10s"],
        "M3": [
            "m0_score",
            "vol60",
            "ofi_proxy_10s",
            "trade_flow_imbalance_10s",
            "l5_imbalance",
        ],
        "M4": [
            "m0_score",
            "vol60",
            "ofi_proxy_10s",
            "trade_flow_imbalance_10s",
            "l5_imbalance",
            "microprice_displacement_bps",
        ],
        "M5": [
            "m0_score",
            "vol60",
            "ofi_proxy_10s",
            "trade_flow_imbalance_10s",
            "l5_imbalance",
            "microprice_displacement_bps",
            "spread_bps",
            "bid_depth_l5",
            "ask_depth_l5",
        ],
    }
    target = frame["future_mid_return_30s_bps"]
    finite = target.notna() & frame["book_valid"]
    split = int(len(frame) * 0.70)
    results = []
    for model, columns in features.items():
        mask = finite & frame[columns].notna().all(axis=1)
        train_mask = mask & (np.arange(len(frame)) < split)
        test_mask = mask & (np.arange(len(frame)) >= split)
        train_x, train_y = (
            frame.loc[train_mask, columns].to_numpy(float),
            target[train_mask].to_numpy(float),
        )
        test_x, test_y = (
            frame.loc[test_mask, columns].to_numpy(float),
            target[test_mask].to_numpy(float),
        )
        if len(train_y) < 100 or len(test_y) < 100:
            continue
        prediction = _ridge_predict(train_x, train_y, test_x)
        target_std = float(np.std(test_y))
        rank_corr = (
            float(pd.Series(prediction).rank().corr(pd.Series(test_y).rank()))
            if target_std > 1e-12
            else None
        )
        results.append(
            {
                "model": model,
                "features": columns,
                "train_n": len(train_y),
                "test_n": len(test_y),
                "oos_target_std_bps": target_std,
                "oos_informative_target": bool(target_std > 1e-12),
                "oos_mae_bps": float(np.mean(np.abs(prediction - test_y))),
                "oos_rmse_bps": float(np.sqrt(np.mean((prediction - test_y) ** 2))),
                "oos_direction_hit_rate": float(
                    np.mean(np.sign(prediction) == np.sign(test_y))
                ),
                "oos_rank_corr": rank_corr,
                "mean_predicted_bps": float(prediction.mean()),
                "mean_realized_bps": float(test_y.mean()),
            }
        )
    diagnostics = {}
    for feature in (
        "ofi_proxy_10s",
        "trade_flow_imbalance_10s",
        "l5_imbalance",
        "microprice_displacement_bps",
        "spread_bps",
    ):
        mask = finite & frame[feature].notna()
        if mask.sum() < 100:
            continue
        x, y = frame.loc[mask, feature], target[mask]
        try:
            bins = pd.qcut(x, 5, labels=False, duplicates="drop")
            grouped = [
                {
                    "bin": int(group),
                    "n": int((bins == group).sum()),
                    "mean_feature": float(x[bins == group].mean()),
                    "mean_future_30s_bps": float(y[bins == group].mean()),
                }
                for group in sorted(bins.dropna().unique())
            ]
        except ValueError:
            grouped = []
        corr = float(x.rank().corr(y.rank()))
        diagnostics[feature] = {
            "full_sample_n": int(mask.sum()),
            "rank_corr_30s": corr if np.isfinite(corr) else None,
            "bins": grouped,
        }
    return {
        "symbol": "PF_XBTUSD",
        "split": {
            "selection_fraction": 0.70,
            "selection_n": split,
            "validation_n": len(frame) - split,
        },
        "hurdle_semantics": "mid-price diagnostic; 15 bps round-trip is not subtracted from every 30s label",
        "feature_diagnostics": diagnostics,
        "nested_models": results,
    }


def maker_filter(row: pd.Series, level: int) -> tuple[bool, bool]:
    # PF_XBTUSD was usually about 0.12 bps wide in this capture; this
    # predeclared 0.10 bps floor keeps the maker baseline observable.
    if (
        not bool(row.get("book_valid", False))
        or not np.isfinite(row.get("spread_bps", np.nan))
        or row["spread_bps"] < 0.10
    ):
        return False, False
    bid, ask = True, True
    if level >= 1:
        bid &= not (row.get("ofi_proxy_10s", 0.0) < 0)
        ask &= not (row.get("ofi_proxy_10s", 0.0) > 0)
    if level >= 2:
        bid &= not (row.get("trade_flow_imbalance_10s", 0.0) < -0.1)
        ask &= not (row.get("trade_flow_imbalance_10s", 0.0) > 0.1)
    if level >= 3:
        bid &= not (row.get("l5_imbalance", 0.0) < -0.1)
        ask &= not (row.get("l5_imbalance", 0.0) > 0.1)
    if level >= 4:
        bid &= bool(row.get("rv_60s", 0.0) <= row.get("rv_60s_p75", np.inf))
        ask &= bid
    return bid, ask


def simulate_maker(
    frame: pd.DataFrame, fill_model: str, toxicity_level: int
) -> dict[str, object]:
    data = frame.copy().reset_index(drop=True)
    data["rv_60s_p75"] = (
        data["rv_60s"].rolling(3600, min_periods=60).quantile(0.75).ffill()
    )
    cash = NOTIONAL * 10.0
    inventory = 0.0
    max_inventory = 0.0
    fees = 0.0
    spread_capture = 0.0
    fills = 0
    partial_fills = 0
    equity_curve = []
    adverse: dict[str, list[float]] = {f"{h}s": [] for h in HORIZONS}
    last_mark = float("nan")
    for i in range(len(data) - 61):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        if np.isfinite(row.get("mid", np.nan)) and float(row["mid"]) > 0:
            last_mark = float(row["mid"])
        bid_quote, ask_quote = maker_filter(row, toxicity_level)
        quote_qty = (
            NOTIONAL / float(row["mid"])
            if np.isfinite(row["mid"]) and row["mid"] > 0
            else 0.0
        )
        depth_bid = float(row.get("bid_depth_l5", 0.0) or 0.0)
        depth_ask = float(row.get("ask_depth_l5", 0.0) or 0.0)
        for side, enabled, demand, depth, price in (
            (
                "bid",
                bid_quote,
                next_row.get("sell_qty", 0.0),
                depth_bid,
                row.get("best_bid"),
            ),
            (
                "ask",
                ask_quote,
                next_row.get("buy_qty", 0.0),
                depth_ask,
                row.get("best_ask"),
            ),
        ):
            if not enabled or not np.isfinite(price) or quote_qty <= 0:
                continue
            if fill_model == "optimistic_touch":
                fraction = min(1.0, max(0.0, float(demand)) / quote_qty)
            else:
                queue_ahead = max(0.0, depth) * 0.5
                fraction = min(
                    1.0, max(0.0, (float(demand) - queue_ahead) * 0.5) / quote_qty
                )
            if fraction <= 0:
                continue
            qty = quote_qty * fraction
            if side == "bid" and inventory * float(row["mid"]) >= 2 * NOTIONAL:
                continue
            if side == "ask" and inventory <= 0:
                continue
            signed_qty = qty if side == "bid" else -qty
            cash -= float(price) * signed_qty
            inventory += signed_qty
            fills += 1
            partial_fills += int(fraction < 1.0)
            spread_capture += (float(row["mid"]) - float(price)) * signed_qty
            max_inventory = max(max_inventory, abs(inventory * float(row["mid"])))
            for horizon in HORIZONS:
                future_mid = data.iloc[i + horizon]["mid"]
                if np.isfinite(future_mid):
                    adverse[f"{horizon}s"].append(
                        float(
                            (future_mid / row["mid"] - 1)
                            * 10_000
                            * (1 if side == "bid" else -1)
                        )
                    )
        if np.isfinite(last_mark):
            equity_curve.append(cash + inventory * last_mark)
        elif equity_curve:
            equity_curve.append(equity_curve[-1])
    curve = np.asarray(equity_curve, dtype=float)
    curve = curve[np.isfinite(curve)]
    pnl = curve - NOTIONAL * 10.0
    drawdown = (
        float(np.max(np.maximum.accumulate(curve) - curve)) if len(curve) else 0.0
    )
    return {
        "fill_model": fill_model,
        "toxicity_level": toxicity_level,
        "states": len(data),
        "fills": fills,
        "partial_fills": partial_fills,
        "fill_rate_per_state": fills / len(data) if len(data) else 0.0,
        "gross_spread_capture": float(spread_capture),
        "maker_fee_bps_per_side": 0.0,
        "fees": fees,
        "net_pnl": float(pnl[-1]) if len(pnl) else 0.0,
        "max_inventory_usd": max_inventory,
        "max_drawdown": drawdown,
        "adverse_selection_bps": {
            key: (float(np.mean(value)) if value else None)
            for key, value in adverse.items()
        },
        "inventory_final_usd": float(inventory * data.iloc[-1]["mid"])
        if len(data) and np.isfinite(data.iloc[-1]["mid"])
        else None,
        "limitations": "Touch model uses next-second taker-side volume as a fill proxy; queue model applies 50% displayed depth ahead and a 50% residual fill haircut. No exact queue position is observable.",
    }


def run_maker(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    frame = frames.get("PF_XBTUSD", pd.DataFrame())
    results = (
        [
            simulate_maker(frame, model, level)
            for model in ("optimistic_touch", "conservative_queue")
            for level in range(5)
        ]
        if not frame.empty
        else []
    )
    return {
        "symbol": "PF_XBTUSD",
        "official_cost_note": "maker fee/rebate set to 0 because account tier is not assumed; replace with signed fee schedule before economics are used",
        "results": results,
    }


def write_jev_manifest(out: Path, cutoff: pd.Timestamp) -> dict[str, object]:
    manifest = {
        "manifest_version": 1,
        "status": "DESIGN_ONLY_NO_PAID_CALLS",
        "created_at": datetime.now(UTC).isoformat(),
        "data_cutoff": cutoff.isoformat(),
        "state_version": "market-state-v3-compact-proposed",
        "target_tokens": "500-800 where practical",
        "candidate_fields": [
            "symbol",
            "candidate_type",
            "breakout_strength",
            "momentum_normalized",
            "volatility_regime",
            "efficiency_ratio",
            "volume_confirmation",
            "ofi_10s",
            "ofi_30s",
            "ofi_60s",
            "trade_flow_imbalance",
            "book_imbalance_L1",
            "book_imbalance_L5",
            "microprice_displacement",
            "spread_bps",
            "depth",
            "liquidity_state",
        ],
        "questions": {
            "regime": ["TRENDING", "MEAN_REVERTING", "DISORDERED"],
            "signal_consistency": "native Noul",
            "liquidity_stressed": "native Noul",
            "toxic_bid": "native Noul",
            "toxic_ask": "native Noul",
            "quote_environment": [
                "BID_FAVORED",
                "ASK_FAVORED",
                "BOTH_ACCEPTABLE",
                "ABSTAIN",
            ],
        },
        "labels": [
            "future mid return 1s/5s/10s/30s/60s",
            "adverse selection after hypothetical passive bid/ask fill",
        ],
        "budget_usd": 0.0,
        "no_inference_launched": True,
    }
    (out / "jev_meta_experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run(micro_root: Path, candle_db: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp.now(tz="UTC")
    cached_states = {
        symbol: output_dir / f"microstructure_state_{symbol}.csv.gz"
        for symbol in SYMBOLS
    }
    if all(path.exists() for path in cached_states.values()):
        frames = {
            symbol: pd.read_csv(path, parse_dates=["timestamp"])
            for symbol, path in cached_states.items()
        }
        cutoff = max(
            pd.Timestamp(frame["timestamp"].max()) for frame in frames.values()
        )
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        raw_audit = audit_raw(micro_root, cutoff)
        derived_audit = {
            "cutoff": cutoff.isoformat(),
            "source": "cached canonical state files from the same bounded snapshot",
            "symbols": {symbol: len(frame) for symbol, frame in frames.items()},
            "valid_book_rows": {
                symbol: int(frame["book_valid"].sum())
                for symbol, frame in frames.items()
            },
            "derived_counts": raw_audit["feed_counts"],
            "invalid_book_event_seconds": {},
            "l1_l10_depth": "unavailable: collector persisted L5 depth only",
            "ofi_semantics": "depth-change OFI proxy only; true price-level OFI is not claimed from this feed snapshot.",
        }
    else:
        raw_audit = audit_raw(micro_root, cutoff)
        frames, derived_audit = build_canonical(micro_root, cutoff)
    state_paths = {}
    for symbol, frame in frames.items():
        path = output_dir / f"microstructure_state_{symbol}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        state_paths[symbol] = str(path.relative_to(ROOT))
    signal = microstructure_analysis(frames, candle_db, output_dir)
    maker = run_maker(frames)
    manifest = write_jev_manifest(output_dir, cutoff)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cutoff": cutoff.isoformat(),
        "raw_audit": raw_audit,
        "derived_audit": derived_audit,
        "canonical_state_files": state_paths,
        "signal_analysis": signal,
        "maker_simulation": maker,
        "jev_manifest": manifest,
        "no_paid_jev_calls": True,
        "no_real_orders": True,
    }
    (output_dir / "microstructure_data_audit.json").write_text(
        json.dumps({"raw": raw_audit, "derived": derived_audit}, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "microstructure_signal_analysis.json").write_text(
        json.dumps(signal, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "directional_ablation.json").write_text(
        json.dumps(signal.get("nested_models", []), indent=2), encoding="utf-8"
    )
    (output_dir / "maker_simulation.json").write_text(
        json.dumps(maker, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "maker_fill_model_comparison.json").write_text(
        json.dumps(maker, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "quant_microstructure_phase.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-root", default=str(DEFAULT_MICRO))
    parser.add_argument("--candle-db", default=str(DEFAULT_CANDLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    started = time.monotonic()
    summary = run(Path(args.micro_root), Path(args.candle_db), Path(args.output_dir))
    print(
        json.dumps(
            {
                "event": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "raw_lines": summary["raw_audit"]["raw_lines"],
                "symbols": summary["derived_audit"]["symbols"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
