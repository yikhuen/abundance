"""Live market data fetcher via CCXT unified exchange API.

Provides real-time and recent-historical kline data from any exchange
supported by CCXT (100+ exchanges).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from abundance.data.fetcher import DataFetcher


class CCXTFetcher(DataFetcher):
    """Fetch live and recent kline data via CCXT.

    Unlike BinanceVisionFetcher (bulk historical), this pulls recent data
    directly from exchange APIs. Useful for:
    - Filling gaps between last Binance Vision update and now
    - Exchanges not covered by Binance Vision (e.g. Hyperliquid)
    - Funding rates, order book snapshots, and other non-kline data
    """

    # CCXT-supported kline intervals (standardized)
    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(
        self,
        symbol: str,
        exchange: str = "binance",
        interval: str = "1h",
        output_dir: Path | None = None,
    ) -> None:
        """Initialise CCXT fetcher.

        Args:
            symbol: Trading pair (exchange-native format, e.g. 'BTC/USDT').
            exchange: CCXT exchange ID (e.g. 'binance', 'bybit', 'hyperliquid').
            interval: Kline interval.
            output_dir: Directory for saved Parquet files.
        """
        super().__init__(
            symbol.replace("/", ""), output_dir or Path("data/raw/klines")
        )
        self.exchange_name = exchange
        self.interval = self.INTERVAL_MAP.get(interval, interval)
        self._exchange: Any = None

    @property
    def exchange(self) -> Any:
        """Lazy-init CCXT exchange instance."""
        if self._exchange is None:
            import ccxt  # noqa: F811

            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class(
                {"enableRateLimit": True, "rateLimit": 200}
            )
            self._exchange.load_markets()
        return self._exchange

    def _get_output_path(self) -> Path:
        return self.output_dir / f"{self.symbol.lower()}_{self.interval}"

    # ── Public API ──────────────────────────────────────────────

    def fetch_recent(
        self, lookback_days: int = 7, limit: int = 1000
    ) -> pl.DataFrame:
        """Fetch recent klines from exchange API.

        Most exchanges limit to 500–1000 candles per request.
        For longer lookbacks, use fetch_range() or BinanceVisionFetcher.

        Args:
            lookback_days: How many days back to fetch.
            limit: Max candles per request (exchange-dependent).

        Returns:
            Polars DataFrame with standardized columns.
        """
        since = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
            * 1000
        )

        ccxt_symbol = f"{self.symbol[:3]}/{self.symbol[3:]}"  # BTC/USDT
        ohlcv = self.exchange.fetch_ohlcv(
            ccxt_symbol, timeframe=self.interval, since=since, limit=limit
        )

        df = pl.DataFrame(
            ohlcv,
            schema=[
                "open_time", "open", "high", "low", "close", "volume",
            ],
            orient="row",
        )

        df = self.standardize_schema(df)
        logger.info(
            f"CCXT: fetched {len(df)} {self.interval} candles "
            f"from {self.exchange_name} ({self.symbol})"
        )
        return df

    def fetch(
        self,
        lookback_days: int = 7,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Main fetch interface."""
        return self.fetch_recent(lookback_days=lookback_days, **kwargs)
