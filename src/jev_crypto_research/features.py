from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from itertools import pairwise

import numpy as np

from .domain import Candle, DerivativesSnapshot, OrderBookSnapshot, Side, Trade


def _ema(values: np.ndarray, span: int) -> float:
    alpha = 2 / (span + 1)
    current = values[0]
    for value in values[1:]:
        current = alpha * value + (1 - alpha) * current
    return float(current)


class FeatureEngine:
    """Computes model inputs from completed history only; caller provides no open candle."""

    def compute(
        self,
        candles: Sequence[Candle],
        trades: Sequence[Trade] = (),
        book: OrderBookSnapshot | None = None,
        derivatives: DerivativesSnapshot | None = None,
    ) -> dict[str, float | None]:
        if len(candles) < 60 or not all(c.completed for c in candles[-60:]):
            raise ValueError("exactly 60 or more completed 1m candles are required")
        window = candles[-60:]
        if any(
            right.open_time - left.open_time != timedelta(minutes=1)
            for left, right in pairwise(window)
        ):
            raise ValueError("1m candle sequence is discontinuous")
        closes = np.asarray([float(c.close) for c in window])
        highs = np.asarray([float(c.high) for c in window])
        lows = np.asarray([float(c.low) for c in window])
        volumes = np.asarray([float(c.volume) for c in window])
        returns = np.diff(np.log(closes))
        delta = np.diff(closes)
        gains, losses = np.maximum(delta[-5:], 0), np.maximum(-delta[-5:], 0)
        gain_mean, loss_mean = gains.mean(), losses.mean()
        if gain_mean == 0 and loss_mean == 0:
            rsi_5m = 50.0
        elif loss_mean == 0:
            rsi_5m = 100.0
        else:
            rsi_5m = float(100 - 100 / (1 + gain_mean / loss_mean))
        true_ranges = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])),
        )
        mean, std = volumes[:-1].mean(), volumes[:-1].std()
        buy = sum((t.quantity for t in trades if t.side is Side.BUY), Decimal(0))
        sell = sum((t.quantity for t in trades if t.side is Side.SELL), Decimal(0))
        result: dict[str, float | None] = {
            "volume_zscore": float((volumes[-1] - mean) / std) if std else 0.0,
            # A zero sell bucket is represented by a finite signed ratio.  The
            # feature schema is JSON-safe and retains direction without an
            # arbitrary cap.
            "buy_sell_volume_ratio": float(buy / (buy + sell)) if buy + sell else 0.5,
            "rsi_5m": rsi_5m,
            "ema20_distance": float((closes[-1] / _ema(closes[-20:], 20) - 1) * 10_000),
            "ema50_distance": float((closes[-1] / _ema(closes[-50:], 50) - 1) * 10_000),
            "atr_5m": float(true_ranges[-5:].mean()),
            "realized_vol_5m": float(returns[-5:].std(ddof=0) * np.sqrt(5)),
            "realized_vol_15m": float(returns[-15:].std(ddof=0) * np.sqrt(15)),
            "orderbook_imbalance": None,
            "spread_bps": None,
            "funding_rate": float(derivatives.funding_rate)
            if derivatives and derivatives.funding_rate is not None
            else None,
            "open_interest_change": float(derivatives.open_interest_change)
            if derivatives and derivatives.open_interest_change is not None
            else None,
        }
        if book and book.bids and book.asks:
            bid_qty, ask_qty = (
                sum((x.quantity for x in book.bids), Decimal(0)),
                sum((x.quantity for x in book.asks), Decimal(0)),
            )
            best_bid, best_ask = book.bids[0].price, book.asks[0].price
            mid = (best_bid + best_ask) / 2
            result["orderbook_imbalance"] = (
                float((bid_qty - ask_qty) / (bid_qty + ask_qty))
                if bid_qty + ask_qty
                else 0.0
            )
            result["spread_bps"] = (
                float((best_ask - best_bid) / mid * 10_000) if mid else None
            )
        return result
