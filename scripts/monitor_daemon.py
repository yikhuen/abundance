#!/usr/bin/env python3
"""Live monitoring daemon — polls tick data, triggers alarms.

Watches: prices, funding rates, positions, drawdown, connectivity.
Runs as a background service. Configured via HEARTBEAT.md or cron.

Usage:
  python scripts/monitor_daemon.py                    # run once, check + exit
  python scripts/monitor_daemon.py --daemon --interval 60  # run every 60s
  python scripts/monitor_daemon.py --once --alert     # single check with alerts
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from abundance.deployment.monitoring import AlertDispatcher, DrawdownTracker, TradeLogger
from abundance.deployment.risk import RiskLimits, RiskManager
from abundance.paper_trading.testnet_client import get_testnet_client


@dataclass
class MonitorState:
    """Persistent state across monitoring cycles."""

    last_price: dict[str, float] = field(default_factory=dict)
    last_funding: dict[str, float] = field(default_factory=dict)
    last_check_ms: int = 0
    consecutive_errors: int = 0
    anomaly_count: int = 0
    state_file: Path = Path("data/processed/monitor_state.json")

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_price": self.last_price,
            "last_funding": self.last_funding,
            "last_check_ms": self.last_check_ms,
            "consecutive_errors": self.consecutive_errors,
            "anomaly_count": self.anomaly_count,
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "MonitorState":
        p = path or Path("data/processed/monitor_state.json")
        if p.exists():
            data = json.loads(p.read_text())
            return cls(
                last_price=data.get("last_price", {}),
                last_funding=data.get("last_funding", {}),
                last_check_ms=data.get("last_check_ms", 0),
                consecutive_errors=data.get("consecutive_errors", 0),
                anomaly_count=data.get("anomaly_count", 0),
                state_file=p,
            )
        return cls(state_file=p)


PAIRS_TO_MONITOR = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def check_prices(client, pairs: list[str], state: MonitorState, alerts: AlertDispatcher) -> dict:
    """Check spot prices for anomalies.

    Flags: >5% move since last check, >50% vs 200d MA, zero/negative prices.
    """
    anomalies = []
    prices = {}

    for pair in pairs:
        try:
            price = client.get_price(pair)
            prices[pair] = price

            if price <= 0:
                anomalies.append(f"{pair}: zero/negative price ${price}")
                continue

            # Gap since last check
            prev = state.last_price.get(pair)
            if prev and prev > 0:
                change_pct = abs(price - prev) / prev * 100
                if change_pct > 5:
                    anomalies.append(f"{pair}: {change_pct:.1f}% gap since last check (${prev:,.0f} → ${price:,.0f})")

        except Exception as e:
            anomalies.append(f"{pair}: fetch failed — {e}")

    state.last_price = prices
    return {"prices": prices, "anomalies": anomalies}


def check_funding(client, pairs: list[str], state: MonitorState, alerts: AlertDispatcher) -> dict:
    """Check funding rates for anomalies.

    Flags: rate flips > 0.1%, extreme rates > 0.5%, sustained negative.
    """
    anomalies = []
    rates = {}

    for pair in pairs:
        try:
            rate = client.get_funding_rate(pair)  # raw decimal e.g. 0.000066
            rate_pct = rate * 100
            rates[pair] = rate_pct

            # Extreme rate
            if abs(rate_pct) > 0.5:
                anomalies.append(f"{pair}: extreme funding {rate_pct:+.4f}%")

            # Rapid flip since last check
            prev = state.last_funding.get(pair)
            if prev is not None:
                flip = rate_pct - prev
                if abs(flip) > 0.1:
                    anomalies.append(f"{pair}: funding flip {flip:+.4f}% ({prev:+.4f}% → {rate_pct:+.4f}%)")

        except Exception as e:
            anomalies.append(f"{pair}: funding fetch failed — {e}")

    state.last_funding = rates
    return {"rates": rates, "anomalies": anomalies}


def check_positions(client) -> dict:
    """Check open positions and unrealized PnL."""
    try:
        positions = client.get_positions()
        total_upnl = 0.0
        for p in positions:
            total_upnl += float(p.get("unRealizedProfit", 0))
        return {
            "count": len(positions),
            "total_upnl": round(total_upnl, 2),
            "details": [
                {
                    "symbol": p["symbol"],
                    "size": p.get("positionAmt", "0"),
                    "entry": p.get("entryPrice", "0"),
                    "upnl": round(float(p.get("unRealizedProfit", 0)), 2),
                }
                for p in positions
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def check_connection(client, state: MonitorState, alerts: AlertDispatcher) -> dict:
    """Check exchange connectivity."""
    try:
        client.get_price("BTCUSDT")
        if state.consecutive_errors > 0:
            alerts.send("info", f"Connection restored after {state.consecutive_errors} failures")
        state.consecutive_errors = 0
        return {"connected": True}
    except Exception as e:
        state.consecutive_errors += 1
        if state.consecutive_errors == 1:
            alerts.send("warning", f"Connection lost: {e}")
        elif state.consecutive_errors >= 5:
            alerts.send("critical", f"Connection down for {state.consecutive_errors} consecutive checks")
        return {"connected": False, "error": str(e), "consecutive_failures": state.consecutive_errors}


def run_check(alerts: AlertDispatcher, pairs: list[str] | None = None) -> dict:
    """Execute one full monitoring cycle."""
    client = get_testnet_client()
    state = MonitorState.load()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    results = {
        "timestamp_ms": now_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "connection": {},
        "prices": {},
        "funding": {},
        "positions": {},
        "anomalies": [],
    }

    # 1. Connection
    conn = check_connection(client, state, alerts)
    results["connection"] = conn
    if not conn["connected"]:
        state.save()
        return results

    # 2. Prices
    monitor_pairs = pairs if pairs else PAIRS_TO_MONITOR
    price_check = check_prices(client, monitor_pairs, state, alerts)
    results["prices"] = price_check["prices"]
    results["anomalies"].extend(price_check["anomalies"])

    # 3. Funding
    funding_check = check_funding(client, monitor_pairs, state, alerts)
    results["funding"] = funding_check["rates"]
    results["anomalies"].extend(funding_check["anomalies"])

    # 4. Positions
    results["positions"] = check_positions(client)

    # 5. Fire alerts for anomalies
    for anomaly in results["anomalies"]:
        alerts.send("warning", f"ANOMALY: {anomaly}")
        state.anomaly_count += 1

    state.last_check_ms = now_ms
    state.save()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Live monitoring daemon")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--once", action="store_true", help="Single check and exit")
    parser.add_argument("--alert", action="store_true", help="Enable alerts")
    parser.add_argument("--pairs", nargs="+", default=PAIRS_TO_MONITOR, help="Pairs to monitor")
    args = parser.parse_args()

    pairs_to_watch = args.pairs

    alerts = AlertDispatcher(telegram_enabled=args.alert)

    logger.info("=" * 50)
    logger.info("Abundance Live Monitor")
    logger.info(f"  Mode:     {'daemon' if args.daemon else 'once'}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  Pairs:    {', '.join(PAIRS_TO_MONITOR)}")
    logger.info("=" * 50)

    while True:
        try:
            results = run_check(alerts, pairs_to_watch)

            # Print summary
            prices_str = " | ".join(
                f"{p}: ${v:,.0f}" for p, v in results.get("prices", {}).items()
            )
            funding_str = " | ".join(
                f"{p}: {v:+.4f}%" for p, v in results.get("funding", {}).items()
            )
            pos = results.get("positions", {})
            pos_str = f"{pos.get('count', 0)} positions, uPnL ${pos.get('total_upnl', 0):+,.2f}"

            logger.info(f"  {prices_str}")
            logger.info(f"  {funding_str}")
            logger.info(f"  {pos_str}")

            if results["anomalies"]:
                logger.warning(f"  🚨 {len(results['anomalies'])} anomalies detected")

        except Exception as e:
            logger.error(f"Monitor cycle failed: {e}")

        if not args.daemon:
            break

        time.sleep(args.interval)

    logger.info("Monitor cycle complete")


if __name__ == "__main__":
    main()
