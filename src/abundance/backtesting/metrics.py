"""Strategy evaluation metrics for backtesting.

Standalone metrics calculator — works with any Polars DataFrame
containing OHLCV + returns data. No dependency on NautilusTrader.
"""

import math
from dataclasses import dataclass, field

import polars as pl


@dataclass
class MetricsReport:
    """Container for all computed strategy metrics."""

    # Core metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    annualized_volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Trade-level metrics
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    # Metadata
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0

    # Extra for custom metrics
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert to flat dict for serialization."""
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "annualized_volatility_pct": round(self.annualized_volatility_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
            **self.extra,
        }

    def print(self) -> None:
        """Pretty-print the metrics report."""
        d = self.to_dict()
        print(f"\n{'='*50}")
        print("Strategy Metrics Report")
        print(f"{'='*50}")
        print(f"  Period:        {d['start_date'][:10]} → {d['end_date'][:10]}")
        print(f"  Trading Days:  {d['trading_days']}")
        print(f"  Total Return:  {d['total_return_pct']}%")
        print(f"  Ann. Return:   {d['annualized_return_pct']}%")
        print(f"  Ann. Vol:      {d['annualized_volatility_pct']}%")
        print(f"  Sharpe:        {d['sharpe_ratio']}")
        print(f"  Sortino:       {d['sortino_ratio']}")
        print(f"  Max DD:        {d['max_drawdown_pct']}%")
        print(f"  Calmar:        {d['calmar_ratio']}")
        if self.total_trades > 0:
            print(f"  Win Rate:      {d['win_rate_pct']}%")
            print(f"  Profit Factor: {d['profit_factor']}")
            print(f"  Total Trades:  {d['total_trades']}")
        print(f"{'='*50}")


class MetricsCalculator:
    """Calculate strategy performance metrics from equity curve data.

    Expects a Polars DataFrame with at minimum:
      - timestamp_ms: epoch milliseconds
      - equity: portfolio value at each timestamp
      - returns_pct: period returns (optional — computes from equity if absent)

    Trading days assumed: 365 (crypto, 24/7). Override with trading_days_per_year.
    """

    RISK_FREE_RATE = 0.04  # 4% annualized (approximate risk-free)

    @classmethod
    def _detect_periods_per_year(cls, df: pl.DataFrame) -> int:
        """Detect number of periods per year from timestamp spacing."""
        if len(df) < 2:
            return 365  # default: daily
        ts = df["timestamp_ms"]
        # Median interval between consecutive timestamps (ms)
        intervals = ts.diff().drop_nulls()
        median_interval_ms = intervals.median()
        if median_interval_ms is None or median_interval_ms <= 0:
            return 365
        # periods per year = ms_per_year / median_interval_ms
        ms_per_year = 365.25 * 24 * 3600 * 1000
        periods = int(ms_per_year / median_interval_ms)
        return max(1, periods)

    @classmethod
    def from_equity_curve(
        cls,
        equity_curve: pl.DataFrame,
        trades: pl.DataFrame | None = None,
    ) -> MetricsReport:
        """Compute all metrics from an equity curve DataFrame.

        Args:
            equity_curve: Must contain 'timestamp_ms' and 'equity' columns.
            trades: Optional trade log with 'pnl' column for trade-level metrics.

        Returns:
            MetricsReport with all computed metrics.
        """
        df = equity_curve.sort("timestamp_ms")
        report = MetricsReport()
        periods_per_year = cls._detect_periods_per_year(df)

        # ── Basic info ─────────────────────────────────────────
        ts_min = df["timestamp_ms"].min()
        ts_max = df["timestamp_ms"].max()
        from datetime import datetime, timezone

        report.start_date = datetime.fromtimestamp(
            ts_min / 1000, tz=timezone.utc
        ).isoformat()
        report.end_date = datetime.fromtimestamp(
            ts_max / 1000, tz=timezone.utc
        ).isoformat()
        report.trading_days = int((ts_max - ts_min) / (24 * 3600 * 1000))

        # ── Returns ────────────────────────────────────────────
        equity = df["equity"]
        if "returns_pct" in df.columns:
            returns = df["returns_pct"] / 100.0
        else:
            # Compute from equity changes
            returns = equity.diff() / equity.shift(1)
            returns = returns.fill_null(0.0)

        total_return = (equity[-1] / equity[0] - 1) if equity[0] > 0 else 0.0
        report.total_return_pct = total_return * 100

        # ── Annualized metrics ─────────────────────────────────
        years = report.trading_days / 365.25
        if years > 0 and total_return > -1:
            report.annualized_return_pct = (
                ((1 + total_return) ** (1 / years) - 1) * 100
            )

        # Annualized volatility (year-fraction adjusted)
        period_vol = returns.std()
        if period_vol is not None:
            report.annualized_volatility_pct = (
                period_vol * math.sqrt(periods_per_year) * 100
            )

        # ── Risk-adjusted metrics (frequency-aware) ───────────
        mean_ret = returns.mean()
        rf_per_period = cls.RISK_FREE_RATE / periods_per_year
        if period_vol and period_vol > 0:
            # Sharpe (annualized)
            excess = mean_ret - rf_per_period
            report.sharpe_ratio = float(
                (excess / period_vol) * math.sqrt(periods_per_year)
            )

            # Sortino (downside deviation only)
            downside = returns.filter(returns < 0)
            downside_std = downside.std() if len(downside) > 0 else period_vol
            if downside_std and downside_std > 0:
                report.sortino_ratio = float(
                    (excess / downside_std) * math.sqrt(periods_per_year)
                )

        # ── Drawdown ───────────────────────────────────────────
        cumulative_max = equity.cum_max()
        drawdowns = (equity - cumulative_max) / cumulative_max
        report.max_drawdown_pct = float(drawdowns.min() * 100)

        # Calmar ratio
        if abs(report.max_drawdown_pct) > 0:
            report.calmar_ratio = report.annualized_return_pct / abs(
                report.max_drawdown_pct
            )

        # ── Trade-level metrics ────────────────────────────────
        if trades is not None and "pnl" in trades.columns and len(trades) > 0:
            pnl = trades["pnl"]
            winners = pnl.filter(pnl > 0)
            losers = pnl.filter(pnl <= 0)

            report.total_trades = len(trades)
            report.win_rate_pct = (
                (len(winners) / len(pnl) * 100) if len(pnl) > 0 else 0.0
            )

            gross_profit = winners.sum() if len(winners) > 0 else 0.0
            gross_loss = abs(losers.sum()) if len(losers) > 0 else 1.0
            report.profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return report

    @classmethod
    def from_prices(
        cls,
        prices: pl.DataFrame,
        *,
        initial_capital: float = 10_000.0,
        allocation_pct: float = 1.0,
    ) -> tuple[pl.DataFrame, MetricsReport]:
        """Simple buy-and-hold benchmark from price data.

        Args:
            prices: DataFrame with 'timestamp_ms' and 'close' columns.
            initial_capital: Starting portfolio value.
            allocation_pct: Fraction of capital to allocate (1.0 = fully invested).

        Returns:
            Tuple of (equity_curve, metrics_report).
        """
        df = prices.sort("timestamp_ms")
        close = df["close"]

        # Buy-and-hold: equity = capital * (close / close[0])
        equity = initial_capital * (1 + allocation_pct * (close / close[0] - 1))

        equity_curve = df.select("timestamp_ms").with_columns(
            pl.lit(initial_capital).cast(pl.Float64).alias("equity")
        )

        # Build equity from close prices
        equity_curve = equity_curve.with_columns(
            equity.alias("equity")
        )

        return equity_curve, cls.from_equity_curve(equity_curve)
