#!/usr/bin/env python3
"""Fail-closed placeholder for optional, separately governed fresh inference."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional fresh Jev research entry point"
    )
    parser.add_argument(
        "--allow-paid-inference",
        action="store_true",
        help="explicit acknowledgement; this public artifact still has no paid adapter",
    )
    args = parser.parse_args()
    if not args.allow_paid_inference:
        raise SystemExit(
            "Fresh inference is disabled. Offline reproduction uses cached derived results. "
            "Pass --allow-paid-inference only in a separately reviewed private extension."
        )
    raise SystemExit(
        "No fresh-inference adapter is shipped in the public research artifact; "
        "no paid call was made."
    )


if __name__ == "__main__":
    main()
