"""NautilusTrader Parquet data catalog adapter.

Converts our kline Parquet data to NautilusTrader's catalog format
for use with the high-level BacktestNode API.
"""

from pathlib import Path

import polars as pl
from loguru import logger


class CatalogWriter:
    """Write kline data to NautilusTrader-compatible Parquet catalog.

    NautilusTrader expects bar data with these columns:
      - open, high, low, close, volume
      - ts_event (event timestamp, nanoseconds)
      - ts_init (initialization timestamp, nanoseconds)

    Reference: https://nautilustrader.io/docs/latest/concepts/backtesting/
    """

    def __init__(self, catalog_path: Path | str) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)

    def write_bars(
        self,
        df: pl.DataFrame,
        instrument_id: str,
        bar_type: str,
    ) -> Path:
        """Convert our kline Parquet to NautilusTrader bar Parquet.

        Args:
            df: Our standardized kline DataFrame (columns: timestamp_ms,
                open, high, low, close, volume, ...).
            instrument_id: NautilusTrader instrument ID (e.g. 'BTCUSDT.BINANCE').
            bar_type: Bar type spec (e.g. 'BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL').

        Returns:
            Path to the written bar Parquet file.
        """
        # Convert ms timestamps to ns (NautilusTrader convention)
        df = df.with_columns(
            (pl.col("timestamp_ms") * 1_000_000).cast(pl.Int64).alias("ts_event"),
            (pl.col("timestamp_ms") * 1_000_000).cast(pl.Int64).alias("ts_init"),
        )

        # Select only the columns NautilusTrader needs
        bar_df = df.select(
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ts_event",
                "ts_init",
            ]
        )

        # NautilusTrader catalog path: catalog/bars/{instrument_id}/{bar_type}.parquet
        output_path = (
            self.catalog_path / "bars" / instrument_id / f"{bar_type}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bar_df.write_parquet(output_path, compression="zstd")
        logger.info(
            f"Wrote {len(bar_df):,} bars → {output_path} "
            f"(instrument={instrument_id}, bar_type={bar_type})"
        )

        return output_path

    def write_instruments(self, instruments: list[dict]) -> Path:
        """Write instrument definitions to Parquet.

        Args:
            instruments: List of instrument dicts with at minimum:
                id, raw_symbol, asset_class, price_precision, size_precision, ...

        Returns:
            Path to the written instruments Parquet file.
        """
        df = pl.DataFrame(instruments)
        output_path = self.catalog_path / "instruments.parquet"
        df.write_parquet(output_path)
        logger.info(f"Wrote {len(instruments)} instrument definitions → {output_path}")
        return output_path

    def write_from_our_catalog(
        self,
        our_parquet_path: Path | str,
        instrument_id: str,
        bar_type: str,
    ) -> Path:
        """Convenience: read from our partitioned Parquet and write to NautilusTrader catalog.

        Args:
            our_parquet_path: Path to our partitioned kline Parquet directory
                (e.g. 'data/raw/klines/btcusdt_1h').
            instrument_id: NautilusTrader instrument ID.
            bar_type: NautilusTrader bar type string.
        """
        glob = str(Path(our_parquet_path) / "**" / "*.parquet")
        df = pl.scan_parquet(glob).sort("timestamp_ms").collect()
        return self.write_bars(df, instrument_id, bar_type)
