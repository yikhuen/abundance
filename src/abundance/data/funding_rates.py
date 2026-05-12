"""Fetch perpetual funding rates via CCXT.

Funding rates are periodic payments between long and short positions
in perpetual futures markets. The funding rate mechanism is the core
driver of funding carry strategies:

  - Positive rate → longs pay shorts (short gets paid)
  - Negative rate → shorts pay longs (long gets paid)

Typical interval: 8 hours on Binance, 1 hour on Hyperliquid.

Reference: https://www.binance.com/en/support/faq/360033525271
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from abundance.data.fetcher import DataFetcher

# Perpetual symbol mapping (spot pair → perp symbol)
PERP_SYMBOLS = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
}

FUNDING_INTERVAL_HOURS = 8  # Binance USDT-margined perpetuals


class FundingRateFetcher(DataFetcher):
    """Fetch historical funding rates from perpetual futures markets.

    Data columns:
      - timestamp_ms: funding timestamp (epoch ms)
      - funding_rate: periodic rate (e.g. 0.0001 = 0.01%)
      - funding_rate_pct: funding_rate * 100
      - mark_price: mark price at funding time
    """

    # CCXT stores rates as raw decimals (0.0001 = 0.01%)
    # We multiply by 100 for percentage display
    RATE_COLUMNS = [
        "timestamp_ms",
        "funding_rate",
        "mark_price",
    ]

    def __init__(
        self,
        symbol: str,
        exchange: str = "binance",
        output_dir: Path | None = None,
    ) -> None:
        perp_symbol = PERP_SYMBOLS.get(symbol, f"{symbol[:3]}/{symbol[3:]}:USDT")
        super().__init__(symbol, output_dir or Path("data/raw/funding"))
        self.exchange_name = exchange
        self.perp_symbol = perp_symbol
        self._exchange: Any = None

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class(
                {"enableRateLimit": True, "rateLimit": 200}
            )
            self._exchange.load_markets()
        return self._exchange

    def _get_output_path(self) -> Path:
        return self.output_dir / self.symbol.lower()

    # ── Public API ──────────────────────────────────────────────

    def fetch(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch funding rate history for a date range.

        CCXT limits: typically 1000 entries per request, so we paginate.

        Args:
            start_date: ISO date string (e.g. '2020-01-01'), inclusive.
            end_date: ISO date string, inclusive. Defaults to now.

        Returns:
            Polars DataFrame with columns: timestamp_ms, funding_rate,
            funding_rate_pct, mark_price.
        """
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.now(timezone.utc).isoformat()[:10]

        since = int(
            datetime.strptime(start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        end_ms = int(
            datetime.strptime(end_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        logger.info(
            f"Fetching {self.symbol} funding rates ({self.perp_symbol}) "
            f"from {self.exchange_name}: {start_date} → {end_date}"
        )

        all_rows: list[list] = []
        current_since = since

        while current_since < end_ms:
            try:
                rates = self.exchange.fetch_funding_rate_history(
                    self.perp_symbol, since=current_since, limit=1000
                )
            except Exception as e:
                logger.error(f"CCXT error at {current_since}: {e}")
                break

            if not rates:
                break

            for r in rates:
                all_rows.append(
                    [
                        int(r["timestamp"]),
                        float(r["fundingRate"]),
                        float(r.get("markPrice", 0)),
                    ]
                )

            # Advance to after the last fetched timestamp
            last_ts = rates[-1]["timestamp"]
            if last_ts <= current_since:
                break  # prevent infinite loop
            current_since = last_ts + 1

        if not all_rows:
            logger.warning(f"No funding rates found for {self.symbol}")
            return pl.DataFrame(
                schema={
                    "timestamp_ms": pl.Int64,
                    "funding_rate": pl.Float64,
                    "funding_rate_pct": pl.Float64,
                    "mark_price": pl.Float64,
                }
            )

        df = pl.DataFrame(
            all_rows,
            schema=["timestamp_ms", "funding_rate", "mark_price"],
            orient="row",
        )

        # Add percentage column for readability
        df = df.with_columns((pl.col("funding_rate") * 100).alias("funding_rate_pct"))

        # Sort and deduplicate
        df = df.sort("timestamp_ms").unique(subset=["timestamp_ms"])

        logger.info(f"Fetched {len(df):,} funding rate records for {self.symbol}")
        return df

    def fetch_recent(self, lookback_days: int = 90) -> pl.DataFrame:
        """Fetch recent funding rates (convenience wrapper)."""
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()[:10]
        return self.fetch(start_date=start)
