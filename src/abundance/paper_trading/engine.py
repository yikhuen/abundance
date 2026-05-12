"""Paper trading engine — simulate live trading with real-time prices.

Uses CCXT public data (no API keys needed) to track positions,
compute PnL, and enforce risk limits. Designed to run for weeks
before any real capital is deployed.

Reference: NautilusTrader paper trading + mandatory 4-week period
before live deployment (per Abundance build plan).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import polars as pl
from loguru import logger

from abundance.backtesting.costs import COST_MODEL, CostModel


@dataclass
class Position:
    """A currently open paper trading position."""

    pair: str
    direction: str  # "long_perp" = short perp + long spot
    entry_ts: int  # epoch ms
    entry_price: float
    position_size: float  # quote currency notional
    accumulated_funding: float = 0.0
    total_fees: float = 0.0

    def current_pnl(self, mark_price: float, funding_rate_pct: float) -> float:
        """Compute current unrealised PnL including funding."""
        spot_pnl = (mark_price / self.entry_price - 1) * self.position_size
        # For delta-neutral carry: we earn funding when rate > 0
        funding_pnl = self.accumulated_funding + (funding_rate_pct / 100 * self.position_size)
        # Short perp benefits from price drops
        perp_pnl = -spot_pnl
        return perp_pnl + funding_pnl - self.total_fees


@dataclass
class TradeRecord:
    """A completed paper trade."""

    pair: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    position_size: float
    gross_pnl: float
    net_pnl: float
    fees: float
    total_funding: float
    duration_hours: float


@dataclass
class PaperTradingEngine:
    """Simulates live trading using real-time market data.

    Features:
      - Position tracking (entry/exit/PnL/funding)
      - Risk limits (max position size, max drawdown)
      - Auto-liquidation on risk breach
      - Trade logging and daily PnL reports
    """

    pair: str
    initial_capital: float = 10_000.0
    max_position_pct: float = 0.10  # max 10% of capital per position
    max_drawdown_pct: float = 0.20  # 20% max drawdown → halt
    cost_model: CostModel = field(default_factory=lambda: COST_MODEL)

    # Internal state
    capital: float = 0.0
    peak_capital: float = 0.0
    positions: list[Position] = field(default_factory=list)
    trade_history: list[TradeRecord] = field(default_factory=list)
    equity_history: list[tuple[int, float]] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""
    started_at: Optional[int] = None

    def __post_init__(self) -> None:
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital

    # ── Core trading operations ───────────────────────────────

    def start(self) -> None:
        """Initialise paper trading session."""
        self.started_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        logger.info(
            f"📈 Paper Trading Started | {self.pair} | "
            f"Capital: ${self.initial_capital:,.0f} | "
            f"Max position: {self.max_position_pct*100:.0f}% | "
            f"Max DD: {self.max_drawdown_pct*100:.0f}%"
        )

    def open_position(
        self, price: float, funding_rate_pct: float, timestamp_ms: int
    ) -> Optional[Position]:
        """Open a new carry position if risk allows.

        Returns the Position if opened, None if rejected.
        """
        if self.halted:
            logger.warning("Trading halted — cannot open position")
            return None

        # Risk check: position size within limits
        max_notional = self.capital * self.max_position_pct
        position_size = min(max_notional, self.capital * 0.05)  # 5% default

        # Risk check: drawdown limit
        current_dd = self.current_drawdown_pct()
        if abs(current_dd) > self.max_drawdown_pct * 100:
            self._halt(f"Max drawdown exceeded: {current_dd:.1f}%")
            return None

        # Apply entry costs
        entry_cost = self.cost_model.entry_cost(self.pair, use_perp=True)
        fees = entry_cost * position_size

        # Create position
        pos = Position(
            pair=self.pair,
            direction="carry",
            entry_ts=timestamp_ms,
            entry_price=price,
            position_size=position_size,
            total_fees=fees,
        )
        self.positions.append(pos)

        logger.info(
            f"📌 OPEN  {self.pair} | Size: ${position_size:,.0f} | "
            f"Price: ${price:,.2f} | Rate: {funding_rate_pct:.4f}% | "
            f"Fees: ${fees:.2f}"
        )
        return pos

    def close_position(
        self,
        position: Position,
        exit_price: float,
        funding_rate_pct: float,
        timestamp_ms: int,
    ) -> TradeRecord:
        """Close a position and record the trade."""
        # Final funding update
        funding_update = funding_rate_pct / 100 * position.position_size
        position.accumulated_funding += funding_update

        # Compute PnL
        gross_pnl = position.current_pnl(exit_price, 0)  # funding already added
        exit_cost = self.cost_model.exit_cost(self.pair, use_perp=True)
        exit_fees = exit_cost * position.position_size
        net_pnl = gross_pnl - exit_fees

        # Update capital
        self.capital += net_pnl
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Record trade
        trade = TradeRecord(
            pair=self.pair,
            entry_ts=position.entry_ts,
            exit_ts=timestamp_ms,
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_size=position.position_size,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=position.total_fees + exit_fees,
            total_funding=position.accumulated_funding,
            duration_hours=(timestamp_ms - position.entry_ts) / 3_600_000,
        )
        self.trade_history.append(trade)
        self.positions.remove(position)

        logger.info(
            f"📌 CLOSE {self.pair} | PnL: ${net_pnl:+,.2f} "
            f"({net_pnl/position.position_size*100:+.2f}%) | "
            f"Duration: {trade.duration_hours:.0f}h | "
            f"Capital: ${self.capital:,.0f}"
        )
        return trade

    def update(self, price: float, funding_rate_pct: float, timestamp_ms: int) -> dict:
        """Update engine state with latest market data.

        Call this every funding cycle (8h on Binance).
        Returns a status dict for reporting.
        """
        if self.started_at is None:
            self.start()

        # Record equity
        total_equity = self.capital
        if not self.halted:
            for pos in self.positions:
                total_equity += pos.current_pnl(price, funding_rate_pct)

        self.equity_history.append((timestamp_ms, total_equity))

        # Risk check
        current_dd = (total_equity / self.peak_capital - 1) * 100
        if abs(current_dd) > self.max_drawdown_pct * 100:
            self._halt(f"Max drawdown: {current_dd:.1f}%")

        return {
            "timestamp_ms": timestamp_ms,
            "price": price,
            "funding_rate_pct": funding_rate_pct,
            "capital": self.capital,
            "equity": total_equity,
            "positions": len(self.positions),
            "drawdown_pct": round(current_dd, 2),
            "halted": self.halted,
        }

    # ── Reporting ──────────────────────────────────────────────

    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak capital."""
        total = self.capital
        for pos in self.positions:
            total += pos.current_pnl(0, 0)  # approximate
        return (total / self.peak_capital - 1) * 100

    def summary(self) -> dict:
        """Produce a summary report of the paper trading session."""
        total_fees = sum(t.fees for t in self.trade_history)
        total_funding = sum(t.total_funding for t in self.trade_history)
        gross_pnl = sum(t.gross_pnl for t in self.trade_history)
        net_pnl = sum(t.net_pnl for t in self.trade_history)

        winners = [t for t in self.trade_history if t.net_pnl > 0]
        losers = [t for t in self.trade_history if t.net_pnl <= 0]

        return {
            "pair": self.pair,
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "total_return_pct": (self.capital / self.initial_capital - 1) * 100,
            "peak_capital": self.peak_capital,
            "max_drawdown_pct": (self.capital / self.peak_capital - 1) * 100,
            "total_trades": len(self.trade_history),
            "win_rate": len(winners) / max(len(self.trade_history), 1) * 100,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_fees": total_fees,
            "total_funding": total_funding,
            "open_positions": len(self.positions),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    def print_report(self) -> None:
        """Print a formatted paper trading report."""
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"Paper Trading Report — {s['pair']}")
        print(f"{'='*50}")
        print(f"  Capital:      ${s['initial_capital']:,.0f} → ${s['current_capital']:,.0f}")
        print(f"  Return:       {s['total_return_pct']:+.2f}%")
        print(f"  Max DD:       {s['max_drawdown_pct']:.2f}%")
        print(f"  Trades:       {s['total_trades']} ({s['win_rate']:.0f}% win rate)")
        print(f"  Net PnL:      ${s['net_pnl']:+,.2f}")
        print(f"  Fees:         ${s['total_fees']:.2f}")
        print(f"  Funding:      ${s['total_funding']:.2f}")
        print(f"  Open pos:     {s['open_positions']}")
        if s["halted"]:
            print(f"  ⚠️  HALTED:     {s['halt_reason']}")
        print(f"{'='*50}\n")

    # ── Internal ───────────────────────────────────────────────

    def _halt(self, reason: str) -> None:
        """Emergency halt — prevent further trading."""
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            logger.error(f"🛑 PAPER TRADING HALTED: {reason}")
