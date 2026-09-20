#!/usr/bin/env python3
"""Append-only public Kraken Futures book/trade collector for research."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import websockets


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Collector:
    def __init__(self, output: Path, symbols: list[str]):
        self.output = output
        self.symbols = symbols
        self.books: dict[str, dict[str, dict[float, float]]] = defaultdict(
            lambda: {"bids": {}, "asks": {}}
        )
        self.last_seq: dict[tuple[str, str], int] = {}
        self.gaps = 0
        self.trades = 0

    def append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            )

    def raw(self, message: dict) -> None:
        symbol = message.get("product_id") or message.get("product_ids", ["unknown"])[0]
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        self.append(
            self.output / "raw" / day / f"{symbol}.jsonl",
            {"received_at": _now(), "symbol": symbol, "message": message},
        )

    def sequence(
        self, feed: str, symbol: str, seq: int | None, snapshot: bool = False
    ) -> bool:
        if seq is None:
            return True
        key = (feed, symbol)
        previous = self.last_seq.get(key)
        valid = snapshot or previous is None or seq == previous + 1
        if not valid:
            self.gaps += 1
        self.last_seq[key] = max(seq, previous or seq)
        return valid

    def derive_book(self, message: dict) -> None:
        feed = str(message.get("feed", ""))
        if feed not in {"book_snapshot", "book"}:
            return
        symbol = str(message.get("product_id", ""))
        book = self.books[symbol]
        snapshot = feed == "book_snapshot"
        if snapshot:
            book["bids"] = {
                float(row["price"]): float(row["qty"])
                for row in message.get("bids", [])
            }
            book["asks"] = {
                float(row["price"]): float(row["qty"])
                for row in message.get("asks", [])
            }
        else:
            side = str(message.get("side", ""))
            if side in book and "price" in message:
                price, qty = float(message["price"]), float(message.get("qty", 0))
                if qty <= 0:
                    book[side].pop(price, None)
                else:
                    book[side][price] = qty
        bid_prices = sorted(book["bids"], reverse=True)
        ask_prices = sorted(book["asks"])
        if not bid_prices or not ask_prices:
            return
        bids, asks = bid_prices[:5], ask_prices[:5]
        bid_qty = sum(book["bids"][price] for price in bids)
        ask_qty = sum(book["asks"][price] for price in asks)
        mid = (bids[0] + asks[0]) / 2
        derived = {
            "received_at": _now(),
            "symbol": symbol,
            "source_feed": feed,
            "source_seq": message.get("seq"),
            "sequence_valid": self.sequence(
                "book", symbol, message.get("seq"), snapshot
            ),
            "best_bid": bids[0],
            "best_ask": asks[0],
            "mid": mid,
            "spread_bps": (asks[0] - bids[0]) / mid * 10_000,
            "l5_imbalance": (bid_qty - ask_qty) / (bid_qty + ask_qty)
            if bid_qty + ask_qty
            else None,
            "bid_depth_l5": bid_qty,
            "ask_depth_l5": ask_qty,
        }
        self.append(self.output / "derived" / f"{symbol}.jsonl", derived)

    def derive_trade(self, message: dict) -> None:
        feed = str(message.get("feed", ""))
        if feed == "trade_snapshot":
            trades = message.get("trades", [])
            for trade in trades:
                self._trade_point(trade, snapshot=True)
        elif feed == "trade":
            self._trade_point(message)

    def _trade_point(self, trade: dict, snapshot: bool = False) -> None:
        symbol = str(trade.get("product_id", ""))
        try:
            qty, price = float(trade["qty"]), float(trade["price"])
        except (KeyError, TypeError, ValueError):
            return
        seq = trade.get("seq")
        valid = self.sequence(
            "trade", symbol, int(seq) if seq is not None else None, snapshot
        )
        signed_qty = qty if trade.get("side") == "buy" else -qty
        self.trades += 1
        self.append(
            self.output / "derived" / f"{symbol}.jsonl",
            {
                "received_at": _now(),
                "symbol": symbol,
                "source_feed": "trade",
                "trade_time_ms": trade.get("time"),
                "price": price,
                "qty": qty,
                "signed_qty": signed_qty,
                "vwap_notional": price * qty,
                "sequence_valid": valid,
                "source_seq": seq,
            },
        )

    async def run(self, ws_url: str, duration_seconds: float | None = None) -> None:
        started = time.monotonic()
        delay = 1.0
        while duration_seconds is None or time.monotonic() - started < duration_seconds:
            try:
                async with websockets.connect(
                    ws_url, open_timeout=15, ping_interval=20
                ) as socket:
                    await socket.send(
                        json.dumps(
                            {
                                "event": "subscribe",
                                "feed": "book",
                                "product_ids": self.symbols,
                            }
                        )
                    )
                    await socket.send(
                        json.dumps(
                            {
                                "event": "subscribe",
                                "feed": "trade",
                                "product_ids": self.symbols,
                            }
                        )
                    )
                    delay = 1.0
                    async for raw in socket:
                        message = json.loads(raw)
                        self.raw(message)
                        self.derive_book(message)
                        self.derive_trade(message)
                        if (
                            duration_seconds is not None
                            and time.monotonic() - started >= duration_seconds
                        ):
                            return
            except (OSError, TimeoutError, websockets.WebSocketException) as error:
                self.append(
                    self.output / "collector_events.jsonl",
                    {
                        "at": _now(),
                        "event": "reconnect",
                        "error": str(error)[:500],
                        "backoff_seconds": delay,
                    },
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)


async def main_async(args: argparse.Namespace) -> None:
    collector = Collector(Path(args.output_dir), args.symbol)
    await collector.run(args.ws_url, args.duration_seconds)
    print(
        json.dumps(
            {
                "event": "stopped",
                "trades": collector.trades,
                "sequence_gaps": collector.gaps,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="wss://futures.kraken.com/ws/v1")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument(
        "--output-dir", default="outputs/overnight_research/microstructure"
    )
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()
    if not args.symbol:
        args.symbol = ["PF_XBTUSD"]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
