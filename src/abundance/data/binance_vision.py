"""Download historical klines from Binance Vision (S3 bulk data)."""

import hashlib
import tempfile
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from abundance.data.fetcher import DataFetcher


class BinanceVisionFetcher(DataFetcher):
    """Fetches historical kline data from Binance's public S3 bucket.

    Data is available at https://data.binance.vision/ in daily and monthly
    ZIP archives containing CSV files.

    Reference: https://github.com/binance/binance-public-data

    Key characteristics:
    - Spot monthly klines available back to 2017 for major pairs.
    - Daily files available the day after trading.
    - Spot data from 2025-01-01 onward uses microsecond timestamps.
    - Each ZIP contains a single CSV with columns:
      open_time, open, high, low, close, volume, close_time, quote_volume,
      trades, taker_buy_volume, taker_buy_quote_volume, ignore
    """

    BASE_URL = "https://data.binance.vision"

    # Kline column names (Binance convention, no header in CSV)
    KLINE_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ]

    def __init__(
        self,
        symbol: str,
        interval: str = "1h",
        market_type: str = "spot",
        output_dir: Path | None = None,
    ) -> None:
        """Initialise Binance Vision fetcher.

        Args:
            symbol: Trading pair (e.g. 'BTCUSDT').
            interval: Kline interval — '1m', '5m', '15m', '1h', '4h', '1d', etc.
            market_type: 'spot', 'futures/usd_m', or 'futures/coin_m'.
            output_dir: Directory for saved Parquet files.
        """
        super().__init__(symbol, output_dir or Path("data/raw/klines"))
        self.interval = interval
        self.market_type = market_type
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "abundance/0.1.0"})

    # ── Public API ──────────────────────────────────────────────────

    def fetch_monthly(
        self,
        year: int,
        month: int,
    ) -> pl.DataFrame:
        """Fetch a single month of kline data.

        Monthly files are compiled on the 1st of the following month.
        Example: BTCUSDT-1h-2025-01.zip for January 2025.
        """
        url = self._build_monthly_url(year, month)
        return self._download_and_parse(url)

    def fetch_daily(
        self,
        year: int,
        month: int,
        day: int,
    ) -> pl.DataFrame:
        """Fetch a single day of kline data.

        Daily files are available the day after trading.
        Example: BTCUSDT-1h-2025-01-15.zip
        """
        url = self._build_daily_url(year, month, day)
        return self._download_and_parse(url)

    def fetch(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        mode: str = "monthly",
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Main fetch interface — download a date range of klines.

        Args:
            start_date: ISO date string (e.g. '2024-01-01'), inclusive.
            end_date: ISO date string, inclusive. Defaults to yesterday.
            mode: 'monthly' (one ZIP per month) or 'daily' (one ZIP per day).

        Returns:
            Polars DataFrame with standardized columns, sorted by timestamp.
        """
        if start_date is None:
            start_date = "2017-08-17"  # BTCUSDT listed
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info(
            f"Fetching {self.symbol} {self.interval} {mode} klines: "
            f"{start.date()} → {end.date()}"
        )

        all_frames: list[pl.DataFrame] = []
        current = start

        while current <= end:
            try:
                if mode == "monthly":
                    df = self.fetch_monthly(current.year, current.month)
                    current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
                else:
                    df = self.fetch_daily(current.year, current.month, current.day)
                    current += timedelta(days=1)

                if not df.is_empty():
                    all_frames.append(df)
                else:
                    logger.warning(f"No data for {current.year}-{current.month:02d}")

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.debug(f"No file for {current.year}-{current.month:02d} (404)")
                else:
                    logger.error(f"HTTP error fetching {current}: {e}")
                if mode == "monthly":
                    current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
                else:
                    current += timedelta(days=1)
                    continue

        if not all_frames:
            logger.warning("No data fetched — empty date range or all 404s")
            return pl.DataFrame(schema=self.KLINE_COLUMNS)

        combined = pl.concat(all_frames, how="diagonal_relaxed")
        combined = self.standardize_schema(combined)
        combined = combined.sort("timestamp_ms").unique(subset=["timestamp_ms"])

        logger.info(f"Fetched {len(combined):,} rows total")
        return combined

    # ── URL construction ─────────────────────────────────────────────

    def _build_monthly_url(self, year: int, month: int) -> str:
        return (
            f"{self.BASE_URL}/data/{self.market_type}/monthly/klines/"
            f"{self.symbol}/{self.interval}/"
            f"{self.symbol}-{self.interval}-{year}-{month:02d}.zip"
        )

    def _build_daily_url(self, year: int, month: int, day: int) -> str:
        return (
            f"{self.BASE_URL}/data/{self.market_type}/daily/klines/"
            f"{self.symbol}/{self.interval}/"
            f"{self.symbol}-{self.interval}-{year:04d}-{month:02d}-{day:02d}.zip"
        )

    # ── Download + parse ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout)
        ),
    )
    def _download_and_parse(self, url: str) -> pl.DataFrame:
        """Download a single ZIP and parse its CSV into a DataFrame."""
        logger.debug(f"Downloading: {url}")
        response = self.session.get(url, timeout=120)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pl.read_csv(
                    f.read(),
                    has_header=False,
                    new_columns=self.KLINE_COLUMNS,
                    schema_overrides={
                        "open_time": pl.Int64,
                        "close_time": pl.Int64,
                        "ignore": pl.Float64,
                    },
                )

        # Drop unused 'ignore' column to prevent schema conflicts across months
        if "ignore" in df.columns:
            df = df.drop("ignore")

        return df
