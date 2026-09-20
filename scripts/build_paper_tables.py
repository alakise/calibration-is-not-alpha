#!/usr/bin/env python3
"""Build paper tables/figures from committed derived artifacts.

The offline path is intentionally pure local I/O. It never imports an exchange
client or a Jev adapter, and it refuses to run if a required derived artifact
is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: str = "#245b8a",
) -> None:
    width, height = 900, 520
    left, bottom, top, right = 90, 80, 55, 35
    finite = [value for value in values if math.isfinite(value)]
    low = min(finite) if finite else 0.0
    high = max(finite) if finite else 1.0
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08
    x_step = (width - left - right) / max(1, len(values) - 1)

    def point(index: int, value: float) -> tuple[float, float]:
        return (
            left + index * x_step,
            top + (height - top - bottom) * (high - value) / (high - low),
        )

    points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(values))
    )
    labels_svg = "".join(
        f'<text x="{point(i, v)[0]:.1f}" y="{height - 45}" text-anchor="middle" font-size="12">{label}</text>'
        for i, (label, v) in enumerate(zip(labels, values, strict=True))
    )
    dots = "".join(
        f'<circle cx="{point(i, v)[0]:.1f}" cy="{point(i, v)[1]:.1f}" r="4" fill="{color}"/>'
        for i, v in enumerate(values)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>
<text x="20" y="{height / 2:.0f}" transform="rotate(-90 20 {height / 2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>
<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>
{dots}
{labels_svg}
<text x="{left}" y="{height - bottom + 28}" font-family="sans-serif" font-size="11">lower probability / bucket</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_offline() -> dict[str, object]:
    opportunity = load_json(RESULTS / "summaries" / "jev_opportunity_analysis.json")
    baseline = load_json(RESULTS / "summaries" / "baseline_research_summary.json")
    second_pass = load_json(RESULTS / "summaries" / "second_pass_summary.json")
    phase = load_json(RESULTS / "summaries" / "quant_microstructure_summary.json")

    raw = baseline["raw_signal"]
    dataset_rows = [
        {
            "experiment": "Jev directional baseline",
            "period": "7-day development window",
            "states": 40320,
            "contract": "TERNARY_V1 + BINARY_V1",
            "model": "jev-1.13.0",
            "input_representation": "V1 raw candles; V2 candles + technicals",
            "jev_calls": 40320,
            "input_tokens": "not republished in directional manifest",
        },
        {
            "experiment": "Jev opportunity holdout",
            "period": "2026-08-15 through 2026-09-11 UTC",
            "states": 40320,
            "contract": "market-opportunity-v1",
            "model": "jev-1.13.0",
            "input_representation": "market-state-v2 + seven Noul questions",
            "jev_calls": 40320,
            "input_tokens": 194848794,
        },
        {
            "experiment": "Corrected classical baseline",
            "period": "historical BTC 1-minute sample",
            "states": 86357,
            "contract": "deterministic labels",
            "model": "local deterministic",
            "input_representation": "OHLCV and derived volatility/volume features",
            "jev_calls": 0,
            "input_tokens": 0,
        },
        {
            "experiment": "Microstructure exploration",
            "period": "fixed live public-Kraken snapshot",
            "states": 81071,
            "contract": "exploratory",
            "model": "local deterministic",
            "input_representation": "L5 depth and trade-flow proxies",
            "jev_calls": 0,
            "input_tokens": 0,
        },
    ]
    write_csv(PAPER / "tables" / "dataset_overview.csv", dataset_rows)
    write_markdown(
        PAPER / "tables" / "dataset_overview.md",
        list(dataset_rows[0]),
        [list(row.values()) for row in dataset_rows],
    )

    directional_rows = []
    for key, item in sorted(raw.items()):
        five = item.get("horizons", {}).get("5m", {})
        directional_rows.append(
            {
                "contract_variant": key,
                "sample_count": item.get("sample_count"),
                "long": item.get("direction_frequency", {}).get(
                    "long", item.get("direction_frequency", {}).get("LONG")
                ),
                "short": item.get("direction_frequency", {}).get(
                    "short", item.get("direction_frequency", {}).get("SHORT")
                ),
                "flat": item.get("direction_frequency", {}).get(
                    "flat", item.get("direction_frequency", {}).get("FLAT")
                ),
                "5m_directional_hit_rate": five.get("directional_hit_rate"),
                "5m_gross_bps": five.get(
                    "mean_future_return_conditional_chosen_direction_bps",
                    five.get("signed_return_gross_bps"),
                ),
                "5m_net_bps": five.get(
                    "mean_net_directional_return_bps", five.get("signed_return_net_bps")
                ),
            }
        )
    write_csv(PAPER / "tables" / "directional_results.csv", directional_rows)
    write_markdown(
        PAPER / "tables" / "directional_results.md",
        list(directional_rows[0]),
        [list(row.values()) for row in directional_rows],
    )

    observed = opportunity["raw_question_scores"]["move_15m_15bps"]["observed_rate"]
    constant_brier = observed * (1 - observed)
    constant_log_loss = -(
        observed * math.log(observed) + (1 - observed) * math.log(1 - observed)
    )
    inc = opportunity["incremental_value"]
    calibration_rows = [
        {
            "model": "constant base rate",
            "N": 40320,
            "brier": constant_brier,
            "log_loss": constant_log_loss,
            "ECE": 0.0,
            "split": "holdout overall",
        },
        {
            "model": "Jev MOVE_15M_15BPS",
            "N": 40320,
            "brier": opportunity["raw_question_scores"]["move_15m_15bps"]["brier"],
            "log_loss": opportunity["raw_question_scores"]["move_15m_15bps"][
                "log_loss"
            ],
            "ECE": opportunity["raw_question_scores"]["move_15m_15bps"]["reliability"][
                "ece"
            ],
            "split": "holdout overall",
        },
        {
            "model": "deterministic volatility",
            "N": inc["baseline_validation"]["n"],
            "brier": inc["baseline_validation"]["brier"],
            "log_loss": inc["baseline_validation"]["log_loss"],
            "ECE": "NA",
            "split": "chronological validation",
        },
        {
            "model": "deterministic volatility + Jev",
            "N": inc["baseline_plus_jev_validation"]["n"],
            "brier": inc["baseline_plus_jev_validation"]["brier"],
            "log_loss": inc["baseline_plus_jev_validation"]["log_loss"],
            "ECE": "NA",
            "split": "chronological validation",
        },
    ]
    write_csv(PAPER / "tables" / "movement_calibration.csv", calibration_rows)
    write_markdown(
        PAPER / "tables" / "movement_calibration.md",
        list(calibration_rows[0]),
        [list(row.values()) for row in calibration_rows],
    )

    incremental_rows = [
        {
            "comparison": "deterministic volatility -> deterministic volatility + Jev",
            "validation_n": inc["baseline_plus_jev_validation"]["n"],
            "baseline_brier": inc["baseline_validation"]["brier"],
            "combined_brier": inc["baseline_plus_jev_validation"]["brier"],
            "brier_improvement": inc["incremental_brier_improvement"],
            "baseline_log_loss": inc["baseline_validation"]["log_loss"],
            "combined_log_loss": inc["baseline_plus_jev_validation"]["log_loss"],
            "log_loss_improvement": inc["incremental_log_loss_improvement"],
            "split": "fixed chronological 70/30 selection/validation",
        }
    ]
    write_csv(PAPER / "tables" / "incremental_predictive_value.csv", incremental_rows)
    write_markdown(
        PAPER / "tables" / "incremental_predictive_value.md",
        list(incremental_rows[0]),
        [list(row.values()) for row in incremental_rows],
    )

    classical = phase["corrected_classical"]["canonical_hold_15m"]
    cost_rows = []
    for strategy, item in classical.items():
        for cost, pnl in item.get("cost_sensitivity_net_pnl", {}).items():
            cost_rows.append(
                {"strategy": strategy, "round_trip_cost_bps": cost, "net_pnl_usd": pnl}
            )
    write_csv(PAPER / "tables" / "transaction_cost_sensitivity.csv", cost_rows)
    write_markdown(
        PAPER / "tables" / "transaction_cost_sensitivity.md",
        list(cost_rows[0]),
        [list(row.values()) for row in cost_rows],
    )

    classical_rows = []
    for strategy, item in classical.items():
        classical_rows.append(
            {
                "strategy": strategy,
                "candidate_signals": item.get("candidate_signals"),
                "executed_trades": item.get("executed_trades"),
                "gross_pnl_usd": item.get("gross_pnl"),
                "net_pnl_usd": item.get("net_pnl"),
                "gross_bps_per_trade": item.get("gross_return_per_trade_bps"),
                "net_bps_per_trade": item.get("net_return_per_trade_bps"),
                "win_rate": item.get("win_rate"),
                "max_drawdown_usd": item.get("max_drawdown"),
            }
        )
    write_csv(PAPER / "tables" / "classical_strategy_results.csv", classical_rows)
    write_markdown(
        PAPER / "tables" / "classical_strategy_results.md",
        list(classical_rows[0]),
        [list(row.values()) for row in classical_rows],
    )

    micro = phase["microstructure"]["signal_analysis"].get("feature_diagnostics", {})
    micro_rows = [
        {
            "feature": feature,
            "N": values.get("full_sample_n"),
            "rank_corr_30s": values.get("rank_corr_30s"),
        }
        for feature, values in micro.items()
    ]
    write_csv(PAPER / "tables" / "microstructure_exploratory.csv", micro_rows)
    write_markdown(
        PAPER / "tables" / "microstructure_exploratory.md",
        list(micro_rows[0]),
        [list(row.values()) for row in micro_rows],
    )

    flat = second_pass["flat_low_movement"]
    labels = list(flat)
    values = [flat[label]["horizons"]["5m"]["mean_abs_return_bps"] for label in labels]
    _svg_chart(
        PAPER / "figures" / "pflat_vs_5m_abs_movement.svg",
        "Ternary P(FLAT) bucket vs 5m movement",
        labels,
        values,
        "mean absolute return (bps)",
        "#8a3d62",
    )
    event = opportunity["raw_question_scores"]["move_15m_15bps"]
    rel = [item for item in event["reliability"]["bins"] if item["n"]]
    _svg_chart(
        PAPER / "figures" / "move_15m_reliability.svg",
        "MOVE_15M_15BPS reliability",
        [f"{item['lower']:.1f}" for item in rel],
        [item["mean_prediction"] for item in rel],
        "mean predicted probability",
        "#245b8a",
    )
    _svg_chart(
        PAPER / "figures" / "classical_net_pnl.svg",
        "Corrected classical net PnL at 15 bps",
        list(classical),
        [float(item.get("net_pnl") or 0) for item in classical.values()],
        "net PnL (USD)",
        "#9b2c2c",
    )

    manifest = {
        "offline": True,
        "paid_inference": False,
        "network": False,
        "source_artifacts": [
            "results/summaries/baseline_research_summary.json",
            "results/summaries/second_pass_summary.json",
            "results/summaries/jev_opportunity_analysis.json",
            "results/summaries/quant_microstructure_summary.json",
        ],
        "tables": sorted(
            str(path.relative_to(ROOT)) for path in (PAPER / "tables").glob("*")
        ),
        "figures": sorted(
            str(path.relative_to(ROOT)) for path in (PAPER / "figures").glob("*")
        ),
    }
    (PAPER / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="required safety switch for local-only generation",
    )
    args = parser.parse_args()
    if not args.offline:
        raise SystemExit(
            "Refusing to run without --offline; no network mode is implemented."
        )
    print(json.dumps(build_offline(), indent=2))


if __name__ == "__main__":
    main()
