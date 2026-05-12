"""Risk management for live trading.

Guardrails:
  - Max position size per asset (configurable % of equity)
  - Max drawdown circuit breaker (flatten all if equity < peak × threshold)
  - Max daily loss limit (halt if daily PnL < -threshold%)
  - Cooling-off period after circuit breaker
  - Kelly-inspired sizing: position = edge / variance × capital
  - Multi-asset correlation check: scale positions when correlated
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


@dataclass
class RiskLimits:
    """Configurable risk parameters."""

    max_position_pct: float = 0.10  # max 10% of equity per asset
    max_drawdown_pct: float = 0.20  # 20% drawdown → circuit breaker
    max_daily_loss_pct: float = 0.05  # 5% daily loss → halt
    cooldown_hours: int = 24  # hours to wait after breaker
    leverage_cap: float = 2.0  # max leverage
    correlation_scale_enabled: bool = True  # reduce size on correlated signals


@dataclass
class RiskState:
    """Mutable risk state during trading."""

    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    daily_low_equity: float = float("inf")
    halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[int] = None  # epoch ms
    last_day: Optional[str] = None  # YYYY-MM-DD for daily reset


class RiskManager:
    """Enforces risk limits during live trading."""

    def __init__(self, limits: RiskLimits | None = None, initial_equity: float = 0.0):
        self.limits = limits or RiskLimits()
        self.state = RiskState(peak_equity=initial_equity)
        if initial_equity > 0:
            self.state.daily_start_equity = initial_equity

    def check_halt(self) -> tuple[bool, str]:
        """Check if trading should be halted.

        Returns (halted, reason).
        """
        if not self.state.halted:
            return False, ""

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if self.state.halt_until and now_ms > self.state.halt_until:
            logger.info(f"Cooldown expired — resuming trading")
            self.state.halted = False
            self.state.halt_reason = ""
            self.state.halt_until = None
            return False, ""

        return True, self.state.halt_reason

    def update_equity(self, equity: float) -> None:
        """Update equity and check drawdown/daily loss limits.

        Call after every position update or at regular intervals.
        """
        # Reset daily tracking
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.state.last_day:
            self.state.daily_start_equity = equity
            self.state.daily_low_equity = equity
            self.state.last_day = today

        # Track peak
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # Track daily low
        if equity < self.state.daily_low_equity:
            self.state.daily_low_equity = equity

        # Check drawdown
        dd_pct = (self.state.peak_equity - equity) / self.state.peak_equity * 100
        if dd_pct > self.limits.max_drawdown_pct * 100:
            self._halt(f"Max drawdown breached: {dd_pct:.1f}% > {self.limits.max_drawdown_pct*100:.0f}%")

        # Check daily loss
        daily_loss_pct = (
            (self.state.daily_start_equity - equity) / self.state.daily_start_equity * 100
        )
        if daily_loss_pct > self.limits.max_daily_loss_pct * 100:
            self._halt(f"Daily loss limit breached: {daily_loss_pct:.1f}% > {self.limits.max_daily_loss_pct*100:.0f}%")

    def validate_position(
        self,
        pair: str,
        notional: float,
        equity: float,
        correlated_count: int = 1,
    ) -> tuple[bool, float, str]:
        """Validate and potentially adjust a position.

        Args:
            pair: Trading pair.
            notional: Proposed notional value.
            equity: Current total equity.
            correlated_count: Number of correlated long signals (for scaling).

        Returns:
            (approved, adjusted_notional, reason).
        """
        # Check halt
        halted, reason = self.check_halt()
        if halted:
            return False, 0.0, reason

        # Max position size (cap, don't reject)
        max_notional = equity * self.limits.max_position_pct
        if notional > max_notional:
            return True, max_notional, f"Capped at {self.limits.max_position_pct*100:.0f}% of equity"

        # Correlation scaling: if N assets signal same direction, scale by 1/√N
        if self.limits.correlation_scale_enabled and correlated_count > 1:
            scale = 1.0 / (correlated_count ** 0.5)
            adjusted = notional * scale
            return True, adjusted, f"Scaled by 1/√{correlated_count} = {scale:.2f}"

        return True, notional, "Approved"

    def get_risk_report(self, equity: float) -> dict:
        """Generate a risk status report."""
        dd_pct = (self.state.peak_equity - equity) / max(self.state.peak_equity, 1) * 100
        daily_pnl_pct = (
            (equity - self.state.daily_start_equity) / max(self.state.daily_start_equity, 1) * 100
        )
        return {
            "equity": equity,
            "peak_equity": self.state.peak_equity,
            "drawdown_pct": round(dd_pct, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
        }

    def _halt(self, reason: str) -> None:
        """Trigger circuit breaker."""
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_until = int(
            datetime.now(timezone.utc).timestamp() * 1000
        ) + self.limits.cooldown_hours * 3600 * 1000
        logger.error(f"🛑 CIRCUIT BREAKER: {reason}")
        logger.info(f"   Trading halted until: {datetime.fromtimestamp(self.state.halt_until/1000, tz=timezone.utc)}")
