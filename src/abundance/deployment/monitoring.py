"""Monitoring and operational logging for live trading.

Features:
  - Trade log: SQLite database of every order
  - Daily PnL snapshots
  - Drawdown tracker with underwater duration
  - Alert dispatcher (Telegram via OpenClaw, console fallback)
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from abundance.config.settings import settings


@dataclass
class TradeLogEntry:
    """A single trade record."""

    timestamp_ms: int
    pair: str
    side: str
    quantity: float
    price: float
    notional: float
    order_id: str
    status: str
    pnl: float = 0.0


class TradeLogger:
    """Persistent trade log using SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or settings.processed_dir / "trades.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                notional REAL NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL,
                pnl REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date TEXT PRIMARY KEY,
                equity REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                daily_pnl REAL DEFAULT 0,
                running_sharpe REAL DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                regime_confidence REAL DEFAULT 50,
                halted INTEGER DEFAULT 0,
                halt_reason TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, entry: TradeLogEntry) -> None:
        """Record a trade to the database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO trades (timestamp_ms, pair, side, quantity, price, notional, order_id, status, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.timestamp_ms, entry.pair, entry.side,
                entry.quantity, entry.price, entry.notional,
                entry.order_id, entry.status, entry.pnl,
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Trade logged: {entry.side} {entry.quantity} {entry.pair}")

    def log_snapshot(
        self,
        equity: float,
        open_positions: int,
        daily_pnl: float,
        running_sharpe: float = 0.0,
        max_dd_pct: float = 0.0,
        regime_confidence: float = 50.0,
        halted: bool = False,
        halt_reason: str = "",
    ) -> None:
        """Write daily equity snapshot."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO daily_snapshots
               (date, equity, open_positions, daily_pnl, running_sharpe, max_drawdown_pct, regime_confidence, halted, halt_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                today, equity, open_positions, daily_pnl,
                running_sharpe, max_dd_pct, regime_confidence,
                1 if halted else 0, halt_reason,
            ),
        )
        conn.commit()
        conn.close()

    def get_trade_count(self) -> int:
        """Return total number of trades logged."""
        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        return count

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        """Return recent trades."""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp_ms DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {
                "ts": r[1], "pair": r[2], "side": r[3], "qty": r[4],
                "price": r[5], "notional": r[6], "order_id": r[7],
                "status": r[8], "pnl": r[9],
            }
            for r in rows
        ]


class AlertDispatcher:
    """Send alerts via available channels."""

    def __init__(self, telegram_enabled: bool = False):
        self.telegram_enabled = telegram_enabled

    def send(self, level: str, message: str) -> None:
        """Dispatch an alert.

        Args:
            level: 'info', 'warning', 'error', 'critical'.
            message: Alert content.
        """
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level.upper()}] {message}"

        # Console log
        if level == "critical":
            logger.critical(formatted)
        elif level == "error":
            logger.error(formatted)
        elif level == "warning":
            logger.warning(formatted)
        else:
            logger.info(formatted)

        # In production: send via OpenClaw Telegram bridge
        # if self.telegram_enabled: ...

    def position_opened(self, pair: str, side: str, size: float, price: float) -> None:
        self.send(
            "info",
            f"📌 {side} {size:.4f} {pair} @ ${price:,.2f}",
        )

    def position_closed(self, pair: str, pnl: float, pnl_pct: float) -> None:
        emoji = "🟢" if pnl > 0 else "🔴"
        self.send(
            "info",
            f"{emoji} Closed {pair}: PnL ${pnl:+.2f} ({pnl_pct:+.2f}%)",
        )

    def circuit_breaker(self, reason: str) -> None:
        self.send("critical", f"🛑 CIRCUIT BREAKER: {reason}")

    def daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        open_positions: int,
        drawdown_pct: float,
        regime: str = "unknown",
    ) -> None:
        emoji = "🟢" if daily_pnl > 0 else "🔴"
        self.send(
            "info",
            f"📊 Daily: {emoji} ${daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%) | "
            f"Equity ${equity:,.0f} | DD {drawdown_pct:.1f}% | "
            f"Positions: {open_positions} | Regime: {regime}",
        )


class DrawdownTracker:
    """Tracks drawdown depth and duration."""

    def __init__(self):
        self.peak = 0.0
        self.current_dd = 0.0
        self.underwater_start: int | None = None  # epoch ms
        self.max_underwater_days = 0

    def update(self, equity: float, timestamp_ms: int) -> dict:
        """Update drawdown state and return status."""
        if equity > self.peak:
            self.peak = equity
            if self.underwater_start is not None:
                duration = (timestamp_ms - self.underwater_start) / (24 * 3600 * 1000)
                if duration > self.max_underwater_days:
                    self.max_underwater_days = int(duration)
                self.underwater_start = None

        dd_pct = (self.peak - equity) / max(self.peak, 1) * 100
        self.current_dd = dd_pct

        if dd_pct > 0 and self.underwater_start is None:
            self.underwater_start = timestamp_ms

        underwater_days = 0
        if self.underwater_start is not None:
            underwater_days = int(
                (timestamp_ms - self.underwater_start) / (24 * 3600 * 1000)
            )

        return {
            "drawdown_pct": round(dd_pct, 2),
            "peak": self.peak,
            "underwater_days": underwater_days,
            "max_underwater_days": self.max_underwater_days,
        }
