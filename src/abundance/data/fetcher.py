"""Abstract base for data fetching strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


class DataFetcher(ABC):
    """Abstract base class for market data fetchers.

    All fetchers must implement fetch() and return a Polars DataFrame.
    Subclasses handle source-specific authentication, rate limiting,
    and data format parsing.
    """

    def __init__(self, symbol: str, output_dir: Path) -> None:
        self.symbol = symbol.upper()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, **kwargs: Any) -> pl.DataFrame:
        """Fetch data from the source and return as a Polars DataFrame.

        Returns:
            Polars DataFrame with standardized column schema:
            timestamp_ms, open, high, low, close, volume, ...
        """
        ...

    def save_parquet(self, df: pl.DataFrame, partition_cols: list[str] | None = None) -> Path:
        """Persist DataFrame to partitioned Parquet files.

        Args:
            df: Polars DataFrame to save.
            partition_cols: Columns to partition by (e.g. ['year', 'month']).

        Returns:
            Path to the saved Parquet directory.
        """
        output_path = self.output_dir / self.symbol.lower()
        output_path.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            output_path,
            partition_by=partition_cols,
            compression="zstd",
            statistics=True,
        )
        logger.info(
            f"Saved {len(df):,} rows → {output_path} "
            f"(partitions: {partition_cols or 'none'})"
        )
        return output_path

    @staticmethod
    def standardize_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Normalise kline columns to a consistent schema.

        Expected input columns (Binance convention):
            open_time, open, high, low, close, volume,
            close_time, quote_volume, trades, taker_buy_volume,
            taker_buy_quote_volume, ignore
        """
        rename_map = {
            "open_time": "timestamp_ms",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "close_time": "close_timestamp_ms",
            "quote_volume": "quote_volume",
            "trades": "trade_count",
            "taker_buy_volume": "taker_buy_volume",
            "taker_buy_quote_volume": "taker_buy_quote_volume",
        }

        existing = [c for c in rename_map if c in df.columns]
        df = df.rename({c: rename_map[c] for c in existing})

        timestamp_cols = [
            c for c in ("timestamp_ms", "close_timestamp_ms")
            if c in df.columns
        ]
        for col in timestamp_cols:
            df = df.with_columns(pl.col(col).cast(pl.Int64))

        return df
