#!/usr/bin/env python3
"""Validate manuscript headline numbers against committed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    opportunity = json.loads(
        (ROOT / "results/summaries/jev_opportunity_analysis.json").read_text()
    )
    phase = json.loads(
        (ROOT / "results/summaries/quant_microstructure_summary.json").read_text()
    )
    overnight = json.loads(
        (ROOT / "results/summaries/overnight_summary.json").read_text()
    )
    score = opportunity["raw_question_scores"]["move_15m_15bps"]
    inc = opportunity["incremental_value"]
    assert opportunity["sample_count"] == 40320
    assert opportunity["input_tokens"] == 194_848_794
    close(score["brier"], 0.2131386904761905)
    close(score["log_loss"], 0.6186983129774795)
    close(score["reliability"]["ece"], 0.0421875)
    close(inc["incremental_brier_improvement"], 1.4057663113131724e-05)
    close(inc["incremental_log_loss_improvement"], 3.471850420599942e-05)
    assert phase["dataset"]["classical"]["rows"] == 86_357
    assert overnight["safety"]["real_exchange_orders"] is False
    assert phase["no_paid_jev_calls"] is True
    print("paper headline claims match committed summaries")


if __name__ == "__main__":
    main()
