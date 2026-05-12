"""Execution bridge — converts strategy signals to testnet orders.

Computes the ADX-blended DGT+Trend signal for the current day,
maps it to position sizes, and issues delta orders to Binance testnet.
"""

from dataclasses import dataclass, field
from typing import Optional

import polars as pl
from loguru import logger

from abundance.backtesting.costs import CostModel
from abundance.config.settings import settings
from abundance.paper_trading.testnet_client import TestnetClient


@dataclass
class Signal:
    """Computed trading signal for one pair."""

    pair: str
    timestamp_ms: int
    price: float
    dgt_signal: float  # 0.0 to 0.5
    trend_signal: float  # 0.0 or 1.0
    adx: float  # 0-100
    adx_norm: float  # 0-1
    trend_weight: float  # 0-1 (high ADX → more trend)
    dgt_weight: float  # 0-1 (low ADX → more DGT)
    allocation_pct: float  # fraction of capital to allocate
    direction: str  # "long" or "flat"

    def target_notional(self, capital: float) -> float:
        """Notional value to allocate."""
        if self.direction == "flat":
            return 0.0
        return capital * self.allocation_pct


@dataclass
class OrderRequest:
    """An order to place on the exchange."""

    pair: str
    side: str  # BUY or SELL
    quantity: float
    order_type: str = "MARKET"
    reduce_only: bool = False


class SignalComputer:
    """Compute ADX-blended DGT+Trend signals from live market data."""

    def __init__(self, client: TestnetClient):
        self.client = client

    def compute(self, pair: str, capital: float = 100.0) -> Signal:
        """Compute the current trading signal for one pair.

        Uses 50 days of 1d klines to compute indicators.
        """
        plower = pair.lower()

        # Load recent daily klines for indicator computation
        df = (
            pl.scan_parquet(
                str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
            )
            .sort("timestamp_ms")
            .tail(200)  # last 200 days
            .collect()
        )

        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()
        timestamps = df["timestamp_ms"].to_list()
        n = len(close)

        if n < 50:
            return Signal(
                pair=pair,
                timestamp_ms=timestamps[-1] if timestamps else 0,
                price=close[-1] if close else 0,
                dgt_signal=0.0,
                trend_signal=0.0,
                adx=0.0,
                adx_norm=0.0,
                trend_weight=0.5,
                dgt_weight=0.5,
                allocation_pct=0.0,
                direction="flat",
            )

        # ATR
        atr_vals = [0.0] * n
        for i in range(n):
            tr = high[i] - low[i]
            if i > 0:
                tr = max(tr, abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            atr_vals[i] = tr
        atr_s = [
            sum(atr_vals[max(0, i - 13) : i + 1]) / min(i + 1, 14) for i in range(n)
        ]

        # DGT signal
        dgt_sig = [0.0] * n
        dgt_pos = 0
        ref = close[0] if close else 0
        for i in range(14, n):
            if dgt_pos == 0 and close[i] < ref - atr_s[i]:
                dgt_sig[i] = 0.5
                dgt_pos = 1
                ref = close[i]
            elif dgt_pos == 1 and close[i] > ref + atr_s[i]:
                dgt_sig[i] = 0
                dgt_pos = 0
                ref = close[i]
            elif dgt_pos == 1:
                dgt_sig[i] = 0.5

        # Trend signal
        fe = [close[0]] * n
        se = [close[0]] * n
        af = 2 / 21
        al = 2 / 51
        for i in range(1, n):
            fe[i] = close[i] * af + fe[i - 1] * (1 - af)
            se[i] = close[i] * al + se[i - 1] * (1 - al)
        trend_sig_vals = [1.0 if fe[i] > se[i] else 0.0 for i in range(n)]

        # ADX
        adx_vals = [0.0] * n
        ap = 14
        for i in range(ap * 2, n):
            pdm = []
            mdm = []
            trs = []
            for j in range(ap):
                idx = i - j
                up = high[idx] - high[idx - 1] if idx > 0 else 0
                down = low[idx - 1] - low[idx] if idx > 0 else 0
                pdm.append(up if up > down and up > 0 else 0)
                mdm.append(down if down > up and down > 0 else 0)
                tr = high[idx] - low[idx]
                if idx > 0:
                    tr = max(
                        tr,
                        abs(high[idx] - close[idx - 1]),
                        abs(low[idx] - close[idx - 1]),
                    )
                trs.append(tr)
            a14 = sum(trs) / ap
            pdi = (sum(pdm) / ap) / a14 * 100 if a14 > 0 else 0
            mdi = (sum(mdm) / ap) / a14 * 100 if a14 > 0 else 0
            dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0
            adx_vals[i] = dx

        # Current values
        current_adx = adx_vals[-1] if adx_vals[-1] > 0 else 25
        adx_norm = min(current_adx / 50.0, 1.0)
        trend_w = adx_norm
        dgt_w = 1.0 - adx_norm

        # Allocation: scale by signal strength
        signal_strength = trend_w * trend_sig_vals[-1] + dgt_w * dgt_sig[-1]
        allocation = signal_strength * 0.50  # max 50% allocation

        # Current price from testnet
        current_price = close[-1] if close else 0

        return Signal(
            pair=pair,
            timestamp_ms=timestamps[-1] if timestamps else 0,
            price=current_price,
            dgt_signal=dgt_sig[-1],
            trend_signal=trend_sig_vals[-1],
            adx=current_adx,
            adx_norm=adx_norm,
            trend_weight=trend_w,
            dgt_weight=dgt_w,
            allocation_pct=allocation,
            direction="long" if allocation > 0.05 else "flat",
        )


class OrderManager:
    """Convert signals to orders and manage position state."""

    def __init__(self, client: TestnetClient):
        self.client = client
        self.positions: dict[str, dict] = {}  # pair → {size, entry_price}

    def compute_delta_orders(
        self, signals: list[Signal], capital: float
    ) -> list[OrderRequest]:
        """Compute orders needed to reach target positions.

        Args:
            signals: Current trading signals for all pairs.
            capital: Available capital in quote currency.

        Returns:
            List of orders to place.
        """
        orders = []

        for sig in signals:
            target_notional = sig.target_notional(capital)
            current_pos = self.positions.get(sig.pair, {"size": 0.0})

            if target_notional > 0 and current_pos["size"] == 0:
                # Open new long position
                quantity = target_notional / sig.price if sig.price > 0 else 0
                if quantity > 0:
                    orders.append(
                        OrderRequest(pair=sig.pair, side="BUY", quantity=quantity)
                    )
            elif target_notional == 0 and current_pos["size"] > 0:
                # Close position
                orders.append(
                    OrderRequest(
                        pair=sig.pair,
                        side="SELL",
                        quantity=current_pos["size"],
                        reduce_only=True,
                    )
                )
            elif target_notional > 0 and current_pos["size"] > 0:
                # Rebalance: scale position
                current_notional = current_pos["size"] * sig.price
                delta_pct = (target_notional - current_notional) / max(
                    current_notional, 1
                )
                if abs(delta_pct) > 0.10:  # rebalance if >10% drift
                    if delta_pct > 0:
                        add_qty = (target_notional - current_notional) / sig.price
                        orders.append(
                            OrderRequest(pair=sig.pair, side="BUY", quantity=add_qty)
                        )
                    else:
                        reduce_qty = (current_notional - target_notional) / sig.price
                        orders.append(
                            OrderRequest(
                                pair=sig.pair,
                                side="SELL",
                                quantity=reduce_qty,
                                reduce_only=True,
                            )
                        )

        return orders

    def execute_orders(
        self, orders: list[OrderRequest]
    ) -> list[dict]:
        """Place orders on testnet and update position state.

        Returns list of order responses.
        """
        results = []
        for order in orders:
            try:
                symbol = order.pair
                resp = self.client.place_market_order(
                    symbol=symbol,
                    side=order.side,
                    quantity=order.quantity,
                    reduce_only=order.reduce_only,
                )
                results.append({"order": order, "response": resp, "status": "filled"})

                # Update position state
                if order.side == "BUY":
                    self.positions[symbol] = {
                        "size": self.positions.get(symbol, {}).get("size", 0)
                        + order.quantity,
                        "entry_price": float(
                            resp.get("avgPrice", resp.get("price", 0))
                        ),
                    }
                elif order.side == "SELL":
                    if symbol in self.positions:
                        self.positions[symbol]["size"] -= order.quantity
                        if self.positions[symbol]["size"] <= 0:
                            del self.positions[symbol]

            except Exception as e:
                logger.error(f"Order failed: {order} — {e}")
                results.append({"order": order, "status": "failed", "error": str(e)})

        return results

    def get_position_summary(self) -> dict:
        """Return current position summary."""
        return {
            pair: {"size": pos["size"], "entry": pos.get("entry_price", 0)}
            for pair, pos in self.positions.items()
        }
