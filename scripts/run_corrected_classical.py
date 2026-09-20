#!/usr/bin/env python3
"""Corrected, leakage-safe classical breakout research for phase two.

This script intentionally writes only to the phase-two output namespace.  The
earlier overnight S1 result remains untouched and is explicitly marked
invalid because it compared horizon-scaled volatilities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from run_classical_quant import load_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "overnight_candles.sqlite3"
DEFAULT_OUT = ROOT / "outputs" / "quant_microstructure_phase"
SYMBOL = "PF_XBTUSD"
NOTIONAL = 1_000.0
COSTS = (5.0, 10.0, 15.0, 20.0)
N_VALUES = (10, 20, 30, 60)
VOL_THRESHOLDS = (1.0, 1.25, 1.5)
ER_WINDOW = 20
ER_THRESHOLD = 0.25
MAX_HOLDS = (15, 30)


def add_corrected_features(
    frame: pd.DataFrame, eth: pd.DataFrame | None = None
) -> pd.DataFrame:
    data = frame.copy()
    close, high, low, volume = data["close"], data["high"], data["low"], data["volume"]
    data["r1"] = close.pct_change()
    for horizon in (3, 5, 10, 15, 30, 60):
        data[f"r{horizon}"] = close.pct_change(horizon)
    # Same-frequency volatility: no sqrt(window) multiplier.
    squared = data["r1"].pow(2)
    for window in (5, 15, 30, 60):
        data[f"vol{window}"] = squared.rolling(window).mean().pow(0.5)
    data["vol_ratio_5_60"] = data["vol5"] / data["vol60"]
    data["vol_ratio_15_60"] = data["vol15"] / data["vol60"]
    data["vol_ratio_30_60"] = data["vol30"] / data["vol60"]
    one_step = close.diff().abs()
    for window in (10, 20, 30):
        data[f"er{window}"] = close.diff(window).abs() / one_step.rolling(window).sum()
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    data["atr14_pct"] = true_range.rolling(14).mean() / close
    data["volume_z60"] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
    data["volume_ratio60"] = volume / volume.rolling(60).mean()
    for n in N_VALUES:
        # Shift first: the current candle is never in its own reference range.
        data[f"donchian_hi{n}"] = close.shift(1).rolling(n).max()
        data[f"donchian_lo{n}"] = close.shift(1).rolling(n).min()
    if eth is not None:
        eth_r5 = eth["close"].pct_change(5).rename("eth_r5").reindex(data.index).ffill()
        data["eth_r5"] = eth_r5
        data["btc_eth_agree"] = np.sign(data["r5"]) == np.sign(data["eth_r5"])
    else:
        data["eth_r5"] = np.nan
        data["btc_eth_agree"] = False
    return data


def candidate_layers(
    data: pd.DataFrame,
    n: int = 20,
    vol_threshold: float = 1.25,
    er_window: int = ER_WINDOW,
    er_threshold: float = ER_THRESHOLD,
) -> dict[str, pd.Series]:
    breakout = data["close"] > data[f"donchian_hi{n}"]
    breakdown = data["close"] < data[f"donchian_lo{n}"]
    s0 = pd.Series(
        np.select([breakout, breakdown], [1, -1], default=0), index=data.index
    )
    vol_gate = data["vol_ratio_5_60"] >= vol_threshold
    er_gate = data[f"er{er_window}"] >= er_threshold
    volume_gate = data["volume_z60"] >= 0
    eth_gate = data["btc_eth_agree"]
    s1 = s0.where(vol_gate, 0)
    s2 = s1.where(er_gate, 0)
    s3 = s2.where(volume_gate, 0)
    s4 = s3.where(eth_gate, 0)
    return {
        "S0_BREAKOUT": s0,
        "S1_BREAKOUT_VOL": s1,
        "S2_BREAKOUT_VOL_ER": s2,
        "S3_BREAKOUT_VOL_ER_VOLUME": s3,
        "S4_BREAKOUT_VOL_ER_VOLUME_ETH": s4,
    }


def mean_reversion_control(
    data: pd.DataFrame,
    move_threshold: float = 2.0,
) -> pd.Series:
    """A small, preregistered extreme-move reversal control."""
    normalized = data["r5"] / data["vol60"].replace(0.0, np.nan)
    reversal = data["r1"]
    return pd.Series(
        np.select(
            [
                (normalized <= -move_threshold) & (reversal > 0),
                (normalized >= move_threshold) & (reversal < 0),
            ],
            [1, -1],
            default=0,
        ),
        index=data.index,
        name="MR0_MEAN_REVERSION",
    )


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
    exit_reason: str
    entry_hour_utc: int
    entry_day: str
    vol_quartile: int | None
    er_quartile: int | None


def _quartiles(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series, 4, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def simulate(
    data: pd.DataFrame,
    signals: pd.Series,
    strategy: str,
    round_trip_cost_bps: float,
    max_hold_minutes: int,
) -> list[Trade]:
    vol_q = _quartiles(data["vol60"])
    er_q = _quartiles(data["er20"])
    trades: list[Trade] = []
    i = 0
    last = len(data) - 1
    while i < last - 1:
        signal = int(signals.iloc[i])
        if signal == 0 or not np.isfinite(data["close"].iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= last:
            break
        direction = signal
        entry = float(data["close"].iloc[entry_i])
        end_i = min(last, entry_i + max_hold_minutes)
        exit_i = end_i
        exit_reason = "time_barrier"
        high_water = entry
        low_water = entry
        for k in range(entry_i + 1, end_i + 1):
            close_k = float(data["close"].iloc[k])
            high_water = max(high_water, close_k)
            low_water = min(low_water, close_k)
            atr = (
                float(data["atr14_pct"].iloc[k])
                if np.isfinite(data["atr14_pct"].iloc[k])
                else 0.0
            )
            if direction == 1 and close_k <= high_water * (1.0 - 1.5 * atr):
                exit_i, exit_reason = k, "volatility_trailing_stop"
                break
            if direction == -1 and close_k >= low_water * (1.0 + 1.5 * atr):
                exit_i, exit_reason = k, "volatility_trailing_stop"
                break
            # Invalidation closes only; an opposite signal is reconsidered later.
            if int(signals.iloc[k]) != direction:
                exit_i, exit_reason = k, "signal_invalidation"
                break
        exit_price = float(data["close"].iloc[exit_i])
        gross = direction * (exit_price / entry - 1.0) * 10_000
        net = gross - round_trip_cost_bps
        ts = data.index[entry_i]
        trades.append(
            Trade(
                strategy=strategy,
                entry_timestamp=ts.isoformat(),
                exit_timestamp=data.index[exit_i].isoformat(),
                direction=direction,
                entry_price=entry,
                exit_price=exit_price,
                gross_return_bps=float(gross),
                cost_bps=round_trip_cost_bps,
                net_return_bps=float(net),
                net_pnl=float(NOTIONAL * net / 10_000),
                holding_minutes=int(exit_i - entry_i),
                exit_reason=exit_reason,
                entry_hour_utc=int(ts.hour),
                entry_day=ts.date().isoformat(),
                vol_quartile=int(vol_q.iloc[entry_i])
                if np.isfinite(vol_q.iloc[entry_i])
                else None,
                er_quartile=int(er_q.iloc[entry_i])
                if np.isfinite(er_q.iloc[entry_i])
                else None,
            )
        )
        i = exit_i
    return trades


def metrics(trades: list[Trade], candidate_count: int) -> dict[str, object]:
    if not trades:
        return {
            "candidate_signals": candidate_count,
            "executed_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "gross_return_per_trade_bps": None,
            "net_return_per_trade_bps": None,
            "median_trade_bps": None,
            "win_rate": None,
            "average_win_bps": None,
            "average_loss_bps": None,
            "profit_factor": None,
            "turnover": 0.0,
            "cost_burden_bps": 0.0,
            "average_holding_minutes": None,
            "max_drawdown": 0.0,
            "sharpe": None,
            "sortino": None,
            "long_trades": 0,
            "short_trades": 0,
        }
    gross = np.array([t.gross_return_bps for t in trades], dtype=float)
    net = np.array([t.net_return_bps for t in trades], dtype=float)
    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    curve = np.r_[0.0, np.cumsum(pnl)]
    dd = float(np.max(np.maximum.accumulate(curve) - curve))
    wins, losses = net[net > 0], net[net < 0]
    downside = net[net < 0]
    return {
        "candidate_signals": candidate_count,
        "executed_trades": len(trades),
        "coverage": float(len(trades) / candidate_count) if candidate_count else 0.0,
        "long_trades": sum(t.direction == 1 for t in trades),
        "short_trades": sum(t.direction == -1 for t in trades),
        "gross_pnl": float(np.sum(gross) * NOTIONAL / 10_000),
        "net_pnl": float(np.sum(pnl)),
        "gross_return_per_trade_bps": float(gross.mean()),
        "net_return_per_trade_bps": float(net.mean()),
        "median_trade_bps": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "average_win_bps": float(wins.mean()) if len(wins) else None,
        "average_loss_bps": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else None,
        "turnover": float(len(trades) * 2 * NOTIONAL),
        "cost_burden_bps": float(len(trades) * trades[0].cost_bps),
        "average_holding_minutes": float(np.mean([t.holding_minutes for t in trades])),
        "max_drawdown": dd,
        "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(len(net)))
        if len(net) > 1 and net.std(ddof=1)
        else None,
        "sortino": float(net.mean() / downside.std(ddof=1) * np.sqrt(len(net)))
        if len(downside) > 1 and downside.std(ddof=1)
        else None,
        "by_day_net_pnl": _group_pnl(trades, "entry_day"),
        "by_utc_hour_net_pnl": _group_pnl(trades, "entry_hour_utc"),
        "by_volatility_quartile_net_pnl": _group_pnl(trades, "vol_quartile"),
        "by_efficiency_quartile_net_pnl": _group_pnl(trades, "er_quartile"),
    }


def _group_pnl(trades: list[Trade], attr: str) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for trade in trades:
        value = getattr(trade, attr)
        if value is not None:
            key = str(value)
            grouped[key] = grouped.get(key, 0.0) + trade.net_pnl
    return grouped


def _labels(data: pd.DataFrame) -> dict[str, object]:
    labels: dict[str, object] = {"horizons_minutes": {}}
    for horizon in (1, 3, 5, 10, 15, 30, 60):
        future = data["close"].shift(-horizon) / data["close"] - 1
        labels["horizons_minutes"][str(horizon)] = {
            "n": int(future.notna().sum()),
            "mean_return_bps": float((future.dropna() * 10_000).mean()),
            "median_return_bps": float((future.dropna() * 10_000).median()),
        }
    return labels


def run(db: Path, output_dir: Path) -> dict[str, object]:
    btc = add_corrected_features(load_frame(db, SYMBOL), load_frame(db, "PF_ETHUSD"))
    canonical = candidate_layers(btc)
    canonical["MR0_MEAN_REVERSION"] = mean_reversion_control(btc)
    config_rows: list[dict[str, object]] = []
    trades_out: list[dict[str, object]] = []
    for n in N_VALUES:
        layers = candidate_layers(btc, n=n)
        for name, signals in layers.items():
            for hold in MAX_HOLDS:
                trades = simulate(btc, signals, name, 15.0, hold)
                config_rows.append(
                    {
                        "n": n,
                        "max_hold_minutes": hold,
                        "strategy": name,
                        **metrics(trades, int((signals != 0).sum())),
                    }
                )
                trades_out.extend(
                    asdict(trade) | {"n": n, "max_hold_minutes": hold}
                    for trade in trades
                )
    mr0 = mean_reversion_control(btc)
    for hold in MAX_HOLDS:
        trades = simulate(btc, mr0, "MR0_MEAN_REVERSION", 15.0, hold)
        config_rows.append(
            {
                "n": None,
                "max_hold_minutes": hold,
                "strategy": "MR0_MEAN_REVERSION",
                "move_threshold_vol": 2.0,
                **metrics(trades, int((mr0 != 0).sum())),
            }
        )
        trades_out.extend(
            asdict(trade) | {"n": None, "max_hold_minutes": hold} for trade in trades
        )
    sensitivity = []
    for threshold in VOL_THRESHOLDS:
        layers = candidate_layers(btc, n=20, vol_threshold=threshold)
        for name in (
            "S0_BREAKOUT",
            "S1_BREAKOUT_VOL",
            "S2_BREAKOUT_VOL_ER",
            "S3_BREAKOUT_VOL_ER_VOLUME",
            "S4_BREAKOUT_VOL_ER_VOLUME_ETH",
        ):
            trades = simulate(btc, layers[name], name, 15.0, 15)
            sensitivity.append(
                {
                    "vol_ratio_threshold": threshold,
                    "strategy": name,
                    **metrics(trades, int((layers[name] != 0).sum())),
                }
            )

    folds = []
    start = btc.index.min().ceil("min")
    cursor = max(start + timedelta(days=21), pd.Timestamp("2026-08-15", tz="UTC"))
    while cursor + timedelta(days=21) <= btc.index.max():
        select_start, select_end = cursor, cursor + timedelta(days=14)
        valid_end = select_end + timedelta(days=7)
        selection = btc.loc[select_start:select_end]
        validation = btc.loc[select_end:valid_end]
        candidates = []
        for n in N_VALUES:
            layers = candidate_layers(selection, n=n)
            for name, signals in layers.items():
                trades = simulate(selection, signals, name, 15.0, 15)
                candidates.append((name, n, metrics(trades, int((signals != 0).sum()))))
        selection_mr0 = mean_reversion_control(selection)
        selection_mr0_trades = simulate(
            selection, selection_mr0, "MR0_MEAN_REVERSION", 15.0, 15
        )
        candidates.append(
            (
                "MR0_MEAN_REVERSION",
                None,
                metrics(selection_mr0_trades, int((selection_mr0 != 0).sum())),
            )
        )
        chosen = max(candidates, key=lambda item: item[2]["net_pnl"])
        if chosen[1] is None:
            valid_signal = mean_reversion_control(validation)
        else:
            valid_signal = candidate_layers(validation, n=chosen[1])[chosen[0]]
        valid_trades = simulate(validation, valid_signal, chosen[0], 15.0, 15)
        folds.append(
            {
                "selection_start": select_start.isoformat(),
                "selection_end": select_end.isoformat(),
                "validation_end": valid_end.isoformat(),
                "selected_strategy": chosen[0],
                "selected_n": chosen[1],
                "selection_metrics": chosen[2],
                "validation_metrics": metrics(
                    valid_trades, int((valid_signal != 0).sum())
                ),
            }
        )
        cursor += timedelta(days=7)

    result = {
        "dataset": {
            "symbol": SYMBOL,
            "start": btc.index.min().isoformat(),
            "end": btc.index.max().isoformat(),
            "rows": len(btc),
            "eth_confirmation_used": True,
            "lookahead": "features use current completed candle and past rolling windows; Donchian reference is shifted one candle",
        },
        "volatility_definition": {
            "vol5": "sqrt(mean(r_1m^2 over rolling 5 completed returns))",
            "vol60": "sqrt(mean(r_1m^2 over rolling 60 completed returns))",
            "ratio": "vol5/vol60",
            "previous_s1_invalid": True,
        },
        "canonical_config": {
            "donchian_n": 20,
            "vol_ratio_threshold": 1.25,
            "er_window": ER_WINDOW,
            "er_threshold": ER_THRESHOLD,
            "volume_gate": "volume_z60 >= 0",
            "mr0": {
                "definition": "abs(r5/vol60) >= 2 and r1 reverses the five-minute move",
                "move_threshold_vol": 2.0,
            },
            "official_cost_round_trip_bps": 15.0,
        },
        "canonical_candidate_counts": {
            name: int((signals != 0).sum()) for name, signals in canonical.items()
        },
        "canonical_n20_max_hold_results": [
            row for row in config_rows if row["n"] == 20
        ],
        "canonical_mr0_results": [
            row for row in config_rows if row["strategy"] == "MR0_MEAN_REVERSION"
        ],
        "all_predefined_n_results": config_rows,
        "volatility_threshold_sensitivity": sensitivity,
        "walk_forward": folds,
        "research_labels": _labels(btc),
        "trades": trades_out,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classical_corrected_results.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "classical_walk_forward.json").write_text(
        json.dumps(folds, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(trades_out).to_csv(
        output_dir / "classical_corrected_trades.csv.gz",
        index=False,
        compression="gzip",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    result = run(Path(args.db), Path(args.output_dir))
    print(
        json.dumps(
            {
                "event": "complete",
                "rows": result["dataset"]["rows"],
                "folds": len(result["walk_forward"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
