#!/usr/bin/env python3
"""Leakage-safe classical BTC/ETH intraday research for the overnight run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "overnight_candles.sqlite3"
DEFAULT_OUT = ROOT / "outputs" / "overnight_research"
SYMBOL = "PF_XBTUSD"
INITIAL_EQUITY = 10_000.0
NOTIONAL = 1_000.0
HOLD_MINUTES = 15


def load_frame(path: Path, symbol: str) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query(
            "select open_time, close_time, open, high, low, close, volume from candles_1m where symbol=? order by open_time",
            connection,
            params=(symbol,),
        )
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    return frame.set_index("close_time")


def add_features(frame: pd.DataFrame, eth: pd.DataFrame | None = None) -> pd.DataFrame:
    data = frame.copy()
    close, high, low, volume = data["close"], data["high"], data["low"], data["volume"]
    data["r1"] = close.pct_change(1)
    for horizon in (3, 5, 10, 15, 20, 30, 60):
        data[f"r{horizon}"] = close.pct_change(horizon)
    data["rv5"] = data["r1"].rolling(5).std() * np.sqrt(5)
    data["rv15"] = data["r1"].rolling(15).std() * np.sqrt(15)
    data["rv60"] = data["r1"].rolling(60).std() * np.sqrt(60)
    data["ema20_distance"] = close / close.ewm(span=20, adjust=False).mean() - 1
    data["ema50_distance"] = close / close.ewm(span=50, adjust=False).mean() - 1
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    data["rsi14"] = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    data["atr14_pct"] = true_range.rolling(14).mean() / close
    data["volume_z"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std()
    for n in (10, 20, 30, 60):
        data[f"donchian_hi{n}"] = close.shift(1).rolling(n).max()
        data[f"donchian_lo{n}"] = close.shift(1).rolling(n).min()
    data["er20"] = data["r20"].abs() / data["r1"].abs().rolling(20).sum()
    if eth is not None:
        joined = eth["close"].pct_change(5).rename("eth_r5").reindex(data.index).ffill()
        data["eth_r5"] = joined
        data["btc_eth_agree"] = np.sign(data["r5"]) == np.sign(data["eth_r5"])
    else:
        data["eth_r5"] = np.nan
        data["btc_eth_agree"] = False
    return data


def signal_series(data: pd.DataFrame, strategy: str) -> pd.Series:
    breakout = data["close"] > data["donchian_hi20"]
    breakdown = data["close"] < data["donchian_lo20"]
    if strategy == "S0":
        return pd.Series(
            np.select([breakout, breakdown], [1, -1], default=0), index=data.index
        )
    filters = {
        "S1": data["rv5"] > data["rv60"],
        "S2": data["er20"] >= 0.25,
        "S3": data["volume_z"] >= 0,
        "S4": data["btc_eth_agree"],
    }
    if strategy in filters:
        valid = filters[strategy]
        return pd.Series(
            np.select([breakout & valid, breakdown & valid], [1, -1], default=0),
            index=data.index,
        )
    if strategy == "MR0":
        z = (data["close"] - data["close"].rolling(30).mean()) / data["close"].rolling(
            30
        ).std()
        return pd.Series(
            np.select([z <= -1.5, z >= 1.5], [1, -1], default=0), index=data.index
        )
    raise ValueError(strategy)


@dataclass(frozen=True)
class Trade:
    strategy: str
    entry_timestamp: str
    exit_timestamp: str
    direction: int
    entry_price: float
    exit_price: float
    gross_return_bps: float
    cost_bps: float
    net_return_bps: float
    net_pnl: float
    holding_minutes: int


def simulate(
    data: pd.DataFrame, signals: pd.Series, strategy: str, round_trip_cost_bps: float
) -> list[Trade]:
    trades: list[Trade] = []
    i = 0
    while i < len(data) - HOLD_MINUTES - 1:
        if signals.iloc[i] == 0 or not np.isfinite(data["close"].iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        exit_i = entry_i + HOLD_MINUTES
        direction = int(signals.iloc[i])
        entry, exit_price = (
            float(data["close"].iloc[entry_i]),
            float(data["close"].iloc[exit_i]),
        )
        gross = direction * (exit_price / entry - 1) * 10_000
        net = gross - round_trip_cost_bps
        trades.append(
            Trade(
                strategy,
                data.index[entry_i].isoformat(),
                data.index[exit_i].isoformat(),
                direction,
                entry,
                exit_price,
                gross,
                round_trip_cost_bps,
                net,
                NOTIONAL * net / 10_000,
                HOLD_MINUTES,
            )
        )
        i = exit_i
    return trades


def metrics(trades: list[Trade]) -> dict[str, float | int | None]:
    if not trades:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "net_return_bps": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "sharpe": None,
            "sortino": None,
            "turnover": 0.0,
            "fee_burden_bps": 0.0,
            "average_holding_minutes": 0.0,
        }
    returns = np.array([t.net_return_bps for t in trades], dtype=float)
    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    curve = np.r_[0.0, np.cumsum(pnl)]
    drawdown = float(np.max(np.maximum.accumulate(curve) - curve))
    wins, losses = returns[returns > 0], returns[returns < 0]
    downside = returns[returns < 0]
    return {
        "trades": len(trades),
        "net_pnl": float(pnl.sum()),
        "net_return_bps": float(returns.sum()),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else None,
        "max_drawdown": drawdown,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)))
        if len(returns) > 1 and returns.std(ddof=1)
        else None,
        "sortino": float(returns.mean() / downside.std(ddof=1) * np.sqrt(len(returns)))
        if len(downside) > 1 and downside.std(ddof=1)
        else None,
        "turnover": float(len(trades) * 2 * NOTIONAL),
        "fee_burden_bps": float(len(trades) * trades[0].cost_bps),
        "average_holding_minutes": float(np.mean([t.holding_minutes for t in trades])),
    }


def run(path: Path, output_dir: Path) -> dict:
    btc = add_features(load_frame(path, SYMBOL), load_frame(path, "PF_ETHUSD"))
    strategies = ("S0", "S1", "S2", "S3", "S4", "MR0")
    rows: list[dict] = []
    trades_out: list[dict] = []
    for strategy in strategies:
        signals = signal_series(btc, strategy)
        for cost in (5.0, 10.0, 15.0, 20.0):
            trades = simulate(btc, signals, strategy, cost)
            row = {"strategy": strategy, "round_trip_cost_bps": cost, **metrics(trades)}
            rows.append(row)
            trades_out.extend(asdict(t) | {"round_trip_cost_bps": cost} for t in trades)
    # A strict 14-day select / 7-day validate sequence; parameter choice is
    # made only inside each selection window, with validation starting flat.
    start = btc.index.min().ceil("min")
    folds = []
    cursor = max(start + timedelta(days=21), pd.Timestamp("2026-08-15", tz="UTC"))
    end = btc.index.max()
    while cursor + timedelta(days=21) <= end:
        select_start, select_end = cursor, cursor + timedelta(days=14)
        valid_end = select_end + timedelta(days=7)
        # Recompute on exact windows to avoid using validation timestamps for selection.
        candidates = []
        for strategy in strategies:
            for cost in (5.0, 10.0, 15.0, 20.0):
                subset = btc.loc[select_start:select_end]
                candidates.append(
                    (
                        strategy,
                        cost,
                        metrics(
                            simulate(
                                subset, signal_series(subset, strategy), strategy, cost
                            )
                        ),
                    )
                )
        chosen = max(candidates, key=lambda item: item[2]["net_pnl"])
        valid = btc.loc[select_end:valid_end]
        valid_metrics = metrics(
            simulate(valid, signal_series(valid, chosen[0]), chosen[0], chosen[1])
        )
        folds.append(
            {
                "selection_start": select_start.isoformat(),
                "selection_end": select_end.isoformat(),
                "validation_end": valid_end.isoformat(),
                "selected_strategy": chosen[0],
                "selected_cost_bps": chosen[1],
                "selection_metrics": chosen[2],
                "validation_metrics": valid_metrics,
            }
        )
        cursor = cursor + timedelta(days=7)
    summary = {
        "symbol": SYMBOL,
        "dataset_start": btc.index.min().isoformat(),
        "dataset_end": btc.index.max().isoformat(),
        "assumptions": {
            "notional_usd": NOTIONAL,
            "entry": "next completed 1m close after signal",
            "max_hold_minutes": HOLD_MINUTES,
            "costs": "round-trip, fee+spread+slippage represented by supplied bps grid",
            "lookahead": "rolling features and shifted breakouts; validation starts flat",
        },
        "strategies": rows,
        "walk_forward": folds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classical_quant_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(trades_out).to_csv(
        output_dir / "classical_trades.csv.gz", index=False, compression="gzip"
    )
    lines = [
        "# Classical quant overnight baseline",
        "",
        f"Dataset: {summary['dataset_start']} to {summary['dataset_end']}",
        "",
        "| Strategy | Cost bps RT | Trades | Net PnL | Win rate | PF | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['round_trip_cost_bps']:.0f} | {row['trades']} | {row['net_pnl']:.2f} | {row['win_rate'] if row['win_rate'] is not None else 'n/a'} | {row['profit_factor'] if row['profit_factor'] is not None else 'n/a'} | {row['max_drawdown']:.2f} |"
        )
    (output_dir / "classical_quant_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = run(Path(args.db), Path(args.output_dir))
    print(
        json.dumps(
            {
                "event": "complete",
                "strategies": len(summary["strategies"]),
                "folds": len(summary["walk_forward"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
