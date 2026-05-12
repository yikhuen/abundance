"""DuckDB-backed storage layer for market data.

Provides a thin abstraction over DuckDB for efficient analytical queries
on partitioned Parquet datasets.
"""

from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from loguru import logger


class MarketDataStore:
    """DuckDB-based storage for market data.

    Reads from partitioned Parquet files on disk and provides
    a query interface for backtesting and analysis.
    """

    def __init__(self, db_path: Path | str = "data/processed/market_data.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        return self._conn

    def register_parquet_glob(
        self,
        table_name: str,
        parquet_glob: str,
        *,
        hive_partitioning: bool = True,
    ) -> None:
        """Register a Parquet glob as a DuckDB view.

        Uses hive-style partitioning by default (year=YYYY/month=MM).
        The view reads directly from Parquet files — no ingest needed.

        Note: Scanning large partitioned datasets on WSL2 /mnt/c/ mounts
        can be slow due to filesystem overhead. Consider running on a
        native Linux filesystem for production use.
        """
        self.conn.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning={hive_partitioning})"
        )
        logger.info(f"Registered view '{table_name}' → {parquet_glob}")

    def query(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query and return results as a Polars DataFrame."""
        result = self.conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return pl.DataFrame(rows, schema=columns, orient="row")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MarketDataStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
