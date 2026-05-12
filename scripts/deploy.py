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
from abundance.deployment.monitoring import AlertDispatcher, DrawdownTracker, TradeLogEntry, TradeLogger
from abundance.deployment.risk import RiskLimits, RiskManager
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

    # Initialise risk manager
    risk_mgr = RiskManager(
        limits=RiskLimits(
            max_position_pct=0.10,
            max_drawdown_pct=0.20,
            max_daily_loss_pct=0.05,
            cooldown_hours=24,
            leverage_cap=2.0,
        ),
        initial_equity=args.capital,
    )

    # Initialise monitoring
    trade_logger = TradeLogger()
    alerts = AlertDispatcher()
    dd_tracker = DrawdownTracker()
    dd_tracker.peak = args.capital

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

        # Apply risk validation to each order
        long_count = sum(1 for s in signals if s.direction == "long")
        validated_orders = []
        for o in orders:
            notional = o.quantity * next((s.price for s in signals if s.pair == o.pair), 0)
            approved, adjusted, reason = risk_mgr.validate_position(
                o.pair, notional, args.capital, long_count
            )
            if approved:
                if adjusted != notional:
                    o.quantity = adjusted / (notional / o.quantity) if notional > 0 else o.quantity
                validated_orders.append(o)
            else:
                logger.warning(f"  ⚠️  {o.pair} rejected: {reason}")
        orders = validated_orders
        
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

    # Risk report
    risk_report = risk_mgr.get_risk_report(args.capital)
    logger.info(f"\nRisk Status:")
    logger.info(f"  Drawdown: {risk_report['drawdown_pct']}% | Daily PnL: {risk_report['daily_pnl_pct']:+.2f}%")
    if risk_report["halted"]:
        logger.error(f"  HALTED: {risk_report['halt_reason']}")

    logger.info("=" * 60)
    logger.info("Sprints 9+10 — Execution Bridge + Risk Management COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
