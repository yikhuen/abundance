#!/usr/bin/env python3
"""Sprint 9: Execution bridge — deploy strategy to Binance testnet.

Usage:
  python scripts/deploy.py                           # dry-run signals
  python scripts/deploy.py --live                    # place real testnet orders
  python scripts/deploy.py --live --capital 500      # with \$500 capital
  python scripts/deploy.py --all --live              # trade all available pairs
  python scripts/deploy.py --daemon --interval 3600  # run hourly
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from abundance.deployment.bridge import OrderManager, SignalComputer
from abundance.paper_trading.testnet_client import get_testnet_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy strategy to Binance testnet")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--all", action="store_true", help="Trade all available pairs")
    parser.add_argument("--capital", type=float, default=100.0, help="Capital in USDT")
    parser.add_argument("--live", action="store_true", help="Place real orders (default: dry-run)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between signals (daemon)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Sprint 9 — Execution Bridge")
    logger.info(f"  Mode:    {'LIVE TRADING' if args.live else 'DRY-RUN'}")
    logger.info(f"  Capital: ${args.capital:,.0f}")
    logger.info(f"  Pair(s): {'ALL' if args.all else args.pair}")
    logger.info("=" * 60)

    # Connect to testnet
    client = get_testnet_client()
    if not client.ping():
        logger.error("Testnet connection failed")
        sys.exit(1)

    # Check balance
    bal = client.get_balance()
    usdt = bal.get("USDT", 0)
    logger.info(f"Testnet balance: ${usdt:,.2f} USDT")

    if args.live and usdt < args.capital * 0.5:
        logger.warning(f"Balance (${usdt:.0f}) < 50% of capital (${args.capital:.0f}) — orders may fail")
        if usdt < 10:
            logger.error("Insufficient balance for trading")
            sys.exit(1)

    # Determine pairs
    if args.all:
        from abundance.config.settings import settings

        pairs = sorted([
            d.name.replace("_1d", "").upper()
            for d in (settings.raw_dir / "klines").iterdir()
            if d.is_dir() and d.name.endswith("_1d")
        ])
        logger.info(f"Trading {len(pairs)} pairs")
    else:
        pairs = [args.pair]

    # Compute signals
    computer = SignalComputer(client)
    order_mgr = OrderManager(client)

    while True:
        signals = []
        for pair in pairs:
            try:
                sig = computer.compute(pair, args.capital / len(pairs))
                signals.append(sig)
                
                logger.info(
                    f"  {sig.pair:<10} Price ${sig.price:,.2f} | "
                    f"ADX {sig.adx:.0f} | "
                    f"Allocation {sig.allocation_pct*100:.0f}% | "
                    f"→ {sig.direction.upper()}"
                )
            except Exception as e:
                logger.error(f"Signal error {pair}: {e}")

        if not signals:
            logger.warning("No signals computed")
            if args.daemon:
                time.sleep(args.interval)
                continue
            break

        # Compute delta orders
        orders = order_mgr.compute_delta_orders(signals, args.capital)
        
        if not orders:
            logger.info("No orders needed — positions already aligned")
        else:
            logger.info(f"\nOrders ({len(orders)}):")
            for o in orders:
                logger.info(
                    f"  {o.side:>4} {o.quantity:.6f} {o.pair} @ MARKET "
                    f"{'[REDUCE]' if o.reduce_only else ''}"
                )

            if args.live:
                results = order_mgr.execute_orders(orders)
                for r in results:
                    if r["status"] == "filled":
                        logger.info(f"  ✅ {r['order'].side} {r['order'].pair} filled")
                    else:
                        logger.error(f"  ❌ {r['order'].pair}: {r.get('error', 'unknown')}")
            else:
                logger.info("  (DRY-RUN — no orders placed)")

        # Position summary
        pos = order_mgr.get_position_summary()
        if pos:
            logger.info(f"\nOpen positions:")
            for p, pdata in pos.items():
                pnl = "?"
                logger.info(f"  {p}: {pdata['size']:.6f} @ ${pdata['entry']:,.2f}")

        if not args.daemon:
            break

        logger.info(f"\nSleeping {args.interval}s until next signal...\n")
        time.sleep(args.interval)

    logger.info("=" * 60)
    logger.info("Sprint 9 — Execution Bridge COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
