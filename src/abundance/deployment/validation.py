"""Sprint 12: Live deployment validation.

Compares live PnL against backtest expectations, measures slippage,
and tracks latency. Run after 48h of live trading.
"""

from datetime import datetime, timezone

from loguru import logger


def validate_slippage(
    expected_price: float, actual_price: float, pair: str
) -> dict:
    """Measure slippage between expected and actual fill price.

    Args:
        expected_price: Signal computation price (from klines).
        actual_price: Actual fill price from exchange.

    Returns:
        Dict with slippage metrics.
    """
    if expected_price <= 0:
        return {"slippage_pct": 0, "status": "no_data"}

    slip = (actual_price - expected_price) / expected_price * 100
    status = "ok" if abs(slip) < 0.1 else "warning" if abs(slip) < 0.5 else "high"

    return {
        "pair": pair,
        "expected": round(expected_price, 2),
        "actual": round(actual_price, 2),
        "slippage_pct": round(slip, 4),
        "status": status,
    }


def validate_pnl(
    live_pnl: float,
    backtest_pnl: float,
    pair: str,
    tolerance_pct: float = 20.0,
) -> dict:
    """Compare live PnL against backtest expectation.

    Args:
        live_pnl: Actual PnL from exchange.
        backtest_pnl: Expected PnL from backtest.
        tolerance_pct: Acceptable deviation percentage.

    Returns:
        Dict with validation result.
    """
    if abs(backtest_pnl) < 0.01:
        return {"deviation_pct": 0, "status": "no_data"}

    deviation = abs(live_pnl - backtest_pnl) / abs(backtest_pnl) * 100
    status = "ok" if deviation < tolerance_pct else "review"

    return {
        "pair": pair,
        "live_pnl": round(live_pnl, 2),
        "backtest_pnl": round(backtest_pnl, 2),
        "deviation_pct": round(deviation, 1),
        "status": status,
    }


def validate_latency(
    signal_ts_ms: int,
    order_ts_ms: int,
    fill_ts_ms: int,
) -> dict:
    """Measure end-to-end latency.

    Args:
        signal_ts_ms: When signal was computed.
        order_ts_ms: When order was placed.
        fill_ts_ms: When order was filled.

    Returns:
        Dict with latency breakdown.
    """
    signal_to_order = (order_ts_ms - signal_ts_ms) / 1000.0
    order_to_fill = (fill_ts_ms - order_ts_ms) / 1000.0
    total = (fill_ts_ms - signal_ts_ms) / 1000.0

    status = "ok" if total < 5.0 else "slow"

    return {
        "signal_to_order_s": round(signal_to_order, 2),
        "order_to_fill_s": round(order_to_fill, 2),
        "total_s": round(total, 2),
        "status": status,
    }


def deployment_readiness_check() -> dict:
    """Run a pre-deployment readiness check.

    Checks all systems are operational before going live.
    """
    checks = {}

    # 1. Testnet connection
    try:
        from abundance.paper_trading.testnet_client import get_testnet_client

        client = get_testnet_client()
        client.get_price("BTCUSDT")
        checks["testnet_connection"] = {"status": "ok", "btc_price": client.get_price("BTCUSDT")}
    except Exception as e:
        checks["testnet_connection"] = {"status": "fail", "error": str(e)}

    # 2. Data availability
    try:
        from abundance.config.settings import settings

        klines_dir = settings.raw_dir / "klines"
        pairs = [
            d.name.replace("_1d", "").upper()
            for d in klines_dir.iterdir()
            if d.is_dir() and d.name.endswith("_1d")
        ]
        checks["data_availability"] = {"status": "ok", "pairs": len(pairs)}
    except Exception as e:
        checks["data_availability"] = {"status": "fail", "error": str(e)}

    # 3. Signal computation
    try:
        from abundance.deployment.bridge import SignalComputer

        computer = SignalComputer(client)
        sig = computer.compute("BTCUSDT", 100)
        checks["signal_computation"] = {
            "status": "ok",
            "adx": round(sig.adx, 1),
            "allocation_pct": round(sig.allocation_pct * 100, 1),
        }
    except Exception as e:
        checks["signal_computation"] = {"status": "fail", "error": str(e)}

    # 4. Trade log
    try:
        from abundance.deployment.monitoring import TradeLogger

        logger_ = TradeLogger()
        count = logger_.get_trade_count()
        checks["trade_log"] = {"status": "ok", "total_trades": count}
    except Exception as e:
        checks["trade_log"] = {"status": "fail", "error": str(e)}

    # 5. Balance
    try:
        bal = client.get_balance()
        usdt = bal.get("USDT", 0)
        checks["balance"] = {"status": "ok" if usdt > 10 else "warning", "usdt": round(usdt, 2)}
    except Exception as e:
        checks["balance"] = {"status": "fail", "error": str(e)}

    # Overall
    failed = sum(1 for c in checks.values() if c["status"] == "fail")
    warnings = sum(1 for c in checks.values() if c["status"] == "warning")
    checks["overall"] = {
        "status": "go" if failed == 0 else "no_go",
        "checks_passed": len(checks) - 1 - failed,
        "checks_total": len(checks) - 1,
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return checks
