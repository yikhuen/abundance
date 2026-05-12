#!/usr/bin/env python3
"""Sprint 8: Paper trading — simulate live trading with real-time data.

Runs the funding carry strategy in paper trading mode, fetching
live prices and funding rates via CCXT (no API keys needed for
public data). Simulates position management, risk checks, and
daily PnL reporting.

Usage:
  python scripts/run_paper_trade.py                    # Single cycle demo
  python scripts/run_paper_trade.py --cycles 3         # 3 funding cycles
  python scripts/run_paper_trade.py --pair ETHUSDT     # Trade ETH
  python scripts/run_paper_trade.py --daemon            # Run indefinitely
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.backtesting.costs import fetch_live_fees
from abundance.backtesting.metrics import MetricsCalculator
from abundance.paper_trading.engine import PaperTradingEngine

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def fetch_current_data(pair: str) -> dict | None:
    """Fetch current price and funding rate from Binance via CCXT."""
    try:
        import ccxt

        exchange = ccxt.binance({"enableRateLimit": True})
        ccxt_symbol = f"{pair[:3]}/{pair[3:]}:USDT"  # perp symbol

        # Fetch current ticker (price)
        ticker = exchange.fetch_ticker(ccxt_symbol)
        price = ticker["last"]

        # Fetch current funding rate
        funding = exchange.fetch_funding_rate(ccxt_symbol)
        rate_pct = funding["fundingRate"] * 100

        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        return {
            "price": price,
            "funding_rate_pct": rate_pct,
            "timestamp_ms": timestamp_ms,
            "mark_price": funding.get("markPrice", price),
        }
    except Exception as e:
        logger.error(f"CCXT fetch failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading simulation")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--cycles", type=int, default=1, help="Number of funding cycles to simulate")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--daemon", action="store_true", help="Run indefinitely (Ctrl+C to stop)")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Abundance · Sprint 8 · Paper Trading")
    logger.info(f"  Pair:     {args.pair}")
    logger.info(f"  Capital:  ${args.capital:,.0f}")
    logger.info(f"  Mode:     {'daemon' if args.daemon else f'{args.cycles} cycle(s)'}")
    logger.info("=" * 60)

    # ── Try fetching live fees ────────────────────────────
    cost_model = fetch_live_fees() or None  # falls back to default
    if cost_model:
        logger.info(
            f"Live fees loaded: maker {cost_model.perp_maker_fee*100:.2f}%, "
            f"taker {cost_model.perp_taker_fee*100:.2f}%"
        )

    # ── Initialise engine ─────────────────────────────────
    engine = PaperTradingEngine(
        pair=args.pair,
        initial_capital=args.capital,
        cost_model=cost_model,
    )
    engine.start()

    cycle = 0
    max_cycles = float("inf") if args.daemon else args.cycles

    while cycle < max_cycles:
        cycle += 1
        logger.info(f"\n--- Cycle {cycle} ---")

        # Fetch current data
        data = fetch_current_data(args.pair)
        if data is None:
            logger.warning("Data fetch failed — skipping cycle")
            time.sleep(args.interval)
            continue

        price = data["price"]
        rate = data["funding_rate_pct"]
        ts = data["timestamp_ms"]

        logger.info(
            f"  Price: ${price:,.2f} | "
            f"Funding: {rate:+.4f}% | "
            f"Capital: ${engine.capital:,.0f}"
        )

        # ── Strategy: simple funding carry ────────────────
        # Entry: rate > 0.01% (positive funding)
        # Exit: rate < 0.005% or negative
        entry_threshold = 0.010
        exit_threshold = 0.005

        # Check open positions for exit
        for pos in list(engine.positions):
            if rate < exit_threshold:
                engine.close_position(pos, price, rate, ts)
            else:
                # Accumulate funding
                pos.accumulated_funding += (rate / 100) * pos.position_size

        # Check for entry
        if not engine.positions and rate > entry_threshold:
            engine.open_position(price, rate, ts)

        # Update engine state
        status = engine.update(price, rate, ts)
        logger.info(
            f"  Equity: ${status['equity']:,.0f} | "
            f"DD: {status['drawdown_pct']:.2f}% | "
            f"Positions: {status['positions']}"
        )

        if engine.halted:
            break

        # Wait between cycles
        if cycle < max_cycles:
            time.sleep(args.interval)

    # ── Final report ──────────────────────────────────────
    engine.print_report()

    # Compute metrics from equity history
    if engine.equity_history:
        import polars as pl

        equity_df = pl.DataFrame(
            engine.equity_history,
            schema=["timestamp_ms", "equity"],
            orient="row",
        )
        trades_df = None
        if engine.trade_history:
            trades_df = pl.DataFrame(
                [{"pnl": t.net_pnl, "return_pct": t.net_pnl / t.position_size * 100}
                 for t in engine.trade_history]
            )
        report = MetricsCalculator.from_equity_curve(equity_df, trades_df)
        report.print()

    logger.info("=" * 60)
    logger.info("Sprint 8 — Paper Trading COMPLETE")
    logger.info("⚠️  Remember: 4+ weeks paper trading required before real capital")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
