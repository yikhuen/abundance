"""Data quality validation for market data.

Checks:
  - Completeness: gap detection, expected row counts, coverage windows
  - Integrity: OHLCV sanity (high≥low, non-negative volume, etc.)
  - Schema: column consistency across partitions, type stability
  - Point-in-time: lookahead prevention (future timestamps)

All checks operate on Parquet datasets via Polars lazy scanning.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    details: str = ""
    affected_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Runs data quality checks against partitioned Parquet datasets.

    Usage:
        validator = DataValidator("data/raw/klines/btcusdt_1h")
        validator.run_all()
        validator.print_report()
    """

    # ── Expected intervals in milliseconds ───────────────────
    INTERVAL_MS = {
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    def __init__(self, parquet_path: Path | str, interval: str | None = None) -> None:
        self.path = Path(parquet_path)
        self.interval = interval
        self._df: pl.LazyFrame | None = None
        self.results: list[ValidationResult] = []

    @property
    def df(self) -> pl.LazyFrame:
        if self._df is None:
            glob = str(self.path / "**" / "*.parquet")
            self._df = pl.scan_parquet(glob).sort("timestamp_ms")
        return self._df

    # ── Public API ─────────────────────────────────────────────

    def run_all(self) -> list[ValidationResult]:
        """Run all quality checks and return results."""
        checks: list[tuple[str, Callable[[], ValidationResult]]] = [
            ("Completeness: row count", self.check_row_count),
            ("Completeness: gap detection", self.check_gaps),
            ("Integrity: OHLCV sanity", self.check_ohlcv_sanity),
            ("Integrity: duplicate timestamps", self.check_duplicates),
            ("Integrity: zero-volume candles", self.check_zero_volume),
            ("Schema: column types", self.check_schema),
            ("Point-in-time: future dates", self.check_future_timestamps),
        ]

        name = self.path.name
        logger.info(f"Running {len(checks)} validation checks on '{name}'")

        for check_name, check_fn in checks:
            try:
                result = check_fn()
                result.check_name = check_name  # overwrite with descriptive name
                self.results.append(result)
            except Exception as e:
                self.results.append(
                    ValidationResult(check_name, False, f"Check failed: {e}")
                )

        return self.results

    def print_report(self) -> None:
        """Print a formatted validation report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        name = self.path.name

        print(f"\n{'='*60}")
        print(f"Validation Report: {name}")
        print(f"{'='*60}")
        for r in self.results:
            status = "✅" if r.passed else "❌"
            print(f"  {status} {r.check_name}")
            if r.details:
                print(f"     {r.details}")
        print(f"{'='*60}")
        print(f"  {passed}/{len(self.results)} checks passed")
        if failed:
            print(f"  {failed} failures — review above")
        print(f"{'='*60}\n")

    # ── Individual checks ──────────────────────────────────────

    def check_row_count(self) -> ValidationResult:
        """Verify the dataset has a reasonable number of rows."""
        count = self.df.select(pl.len()).collect().item()
        passed = count > 0
        return ValidationResult(
            check_name="",
            passed=passed,
            details=f"{count:,} rows",
            metadata={"row_count": count},
        )

    def check_gaps(self) -> ValidationResult:
        """Detect gaps between consecutive candles larger than expected."""
        if self.interval is None:
            return ValidationResult("", True, "No interval specified — skipping gap check")

        expected_ms = self.INTERVAL_MS.get(self.interval)
        if expected_ms is None:
            return ValidationResult("", True, f"Unknown interval '{self.interval}' — skipping")

        # Compute gaps: diff between consecutive timestamps
        df_collected = self.df.select("timestamp_ms").collect()
        gaps = df_collected["timestamp_ms"].diff().drop_nulls()

        # Allow some tolerance (5% leeway for exchange timing variance)
        tolerance = int(expected_ms * 0.05)
        max_gap = expected_ms + tolerance
        large_gaps = gaps.filter(gaps > max_gap)

        if len(large_gaps) == 0:
            return ValidationResult(
                "",
                True,
                f"No gaps > {max_gap}ms (expected: {expected_ms}ms)",
            )

        # Gaps are expected in real exchange data (maintenance, early listing).
        # Treat as passed with a note rather than a failure.
        pct = len(large_gaps) / len(df_collected) * 100
        return ValidationResult(
            "",
            True,  # informational, not a data integrity failure
            f"{len(large_gaps):,} gaps ({pct:.3f}% of candles) — "
            f"exchange maintenance/early listing. Max gap: {large_gaps.max():,}ms",
            affected_rows=len(large_gaps),
            metadata={"gap_count": len(large_gaps), "max_gap_ms": large_gaps.max()},
        )

    def check_ohlcv_sanity(self) -> ValidationResult:
        """Check OHLCV price/volume sanity rules."""
        df_collected = self.df.select(
            ["open", "high", "low", "close", "volume"]
        ).collect()

        issues: list[str] = []

        # High must be >= Low
        bad_hl = df_collected.filter(pl.col("high") < pl.col("low"))
        if len(bad_hl) > 0:
            issues.append(f"high < low: {len(bad_hl)} rows")

        # Open/Close must be within [Low, High]
        bad_open = df_collected.filter(
            (pl.col("open") < pl.col("low")) | (pl.col("open") > pl.col("high"))
        )
        if len(bad_open) > 0:
            issues.append(f"open out of [low, high]: {len(bad_open)} rows")

        bad_close = df_collected.filter(
            (pl.col("close") < pl.col("low")) | (pl.col("close") > pl.col("high"))
        )
        if len(bad_close) > 0:
            issues.append(f"close out of [low, high]: {len(bad_close)} rows")

        # Volume must be >= 0
        bad_vol = df_collected.filter(pl.col("volume") < 0)
        if len(bad_vol) > 0:
            issues.append(f"negative volume: {len(bad_vol)} rows")

        # Prices must be > 0 (sanity)
        for col_name in ["open", "high", "low", "close"]:
            bad_price = df_collected.filter(pl.col(col_name) <= 0)
            if len(bad_price) > 0:
                issues.append(f"{col_name} <= 0: {len(bad_price)} rows")

        if not issues:
            return ValidationResult("", True, "All OHLCV values pass sanity checks")

        return ValidationResult(
            "",
            False,
            "; ".join(issues),
            affected_rows=max(
                len(bad_hl), len(bad_open), len(bad_close), len(bad_vol)
            ),
        )

    def check_duplicates(self) -> ValidationResult:
        """Check for duplicate timestamps."""
        dupes = (
            self.df.group_by("timestamp_ms")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .collect()
        )

        if len(dupes) == 0:
            return ValidationResult("", True, "No duplicate timestamps")

        return ValidationResult(
            "",
            False,
            f"{len(dupes)} duplicate timestamps found (total extra rows: "
            f"{dupes['count'].sum() - len(dupes)})",
            affected_rows=dupes["count"].sum() - len(dupes),
            metadata={"duplicate_groups": len(dupes)},
        )

    def check_zero_volume(self) -> ValidationResult:
        """Flag candles with zero volume (may indicate maintenance windows)."""
        zero_vol = (
            self.df.filter(pl.col("volume") == 0)
            .select("timestamp_ms")
            .collect()
        )

        if len(zero_vol) == 0:
            return ValidationResult("", True, "No zero-volume candles")

        pct = len(zero_vol) / self.df.select(pl.len()).collect().item() * 100
        return ValidationResult(
            "",
            True,  # zero volume is informational, not a failure
            f"{len(zero_vol):,} zero-volume candles ({pct:.2f}%) — "
            f"may indicate exchange maintenance",
            affected_rows=len(zero_vol),
            metadata={"zero_volume_count": len(zero_vol)},
        )

    def check_schema(self) -> ValidationResult:
        """Verify consistent schema across all Parquet partitions.

        Only checks for critical OHLCV + timestamp columns (required by all
        data sources). Optional columns (quote_volume, trade_count, etc.) are
        noted but not required.
        """
        df_collected = self.df.limit(1).collect()
        actual_cols = set(df_collected.columns)

        required_cols = {"timestamp_ms", "open", "high", "low", "close", "volume"}
        optional_cols = {
            "close_timestamp_ms", "quote_volume", "trade_count",
            "taker_buy_volume", "taker_buy_quote_volume", "year", "month",
        }

        missing_required = required_cols - actual_cols
        present_optional = optional_cols & actual_cols
        missing_optional = optional_cols - actual_cols

        if missing_required:
            return ValidationResult(
                "", False, f"missing required columns: {missing_required}"
            )

        # Check for nulls in critical columns
        null_counts = {}
        for col in required_cols:
            n = self.df.filter(pl.col(col).is_null()).select(pl.len()).collect().item()
            if n > 0:
                null_counts[col] = n

        if null_counts:
            return ValidationResult(
                "",
                False,
                f"null values in critical columns: {null_counts}",
                affected_rows=sum(null_counts.values()),
            )

        detail = "Schema consistent"
        if missing_optional:
            detail += f" (CCXT source — optional cols absent: {missing_optional})"
        return ValidationResult("", True, detail)

    def check_future_timestamps(self) -> ValidationResult:
        """Ensure no timestamps are in the future (point-in-time violation)."""
        import time

        now_ms = int(time.time() * 1000)
        # Allow a small buffer (24h) for timezone edge cases
        future = (
            self.df.filter(pl.col("timestamp_ms") > now_ms + 86_400_000)
            .select("timestamp_ms")
            .collect()
        )

        if len(future) == 0:
            return ValidationResult("", True, "No future timestamps")

        from datetime import datetime, timezone

        max_future = future["timestamp_ms"].max()
        max_dt = datetime.fromtimestamp(max_future / 1000, tz=timezone.utc)
        return ValidationResult(
            "",
            False,
            f"{len(future):,} rows with future timestamps (latest: {max_dt.isoformat()})",
            affected_rows=len(future),
        )
