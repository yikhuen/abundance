#!/usr/bin/env python3
"""Sprint 1 · Story 1.2: Data quality validation across all datasets.

Runs completeness, integrity, schema, and point-in-time checks on
all 30 DuckDB tables (15 historical + 15 live) and produces a report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.config.settings import settings
from abundance.data.validation import DataValidator


# All datasets to validate: (path, interval) tuples
# Historical datasets
HISTORICAL = [
    ("data/raw/klines/btcusdt_1h", "1h"),
    ("data/raw/klines/btcusdt_4h", "4h"),
    ("data/raw/klines/btcusdt_1d", "1d"),
    ("data/raw/klines/btcusdt_5m", "5m"),
    ("data/raw/klines/btcusdt_15m", "15m"),
    ("data/raw/klines/ethusdt_1h", "1h"),
    ("data/raw/klines/ethusdt_4h", "4h"),
    ("data/raw/klines/ethusdt_1d", "1d"),
    ("data/raw/klines/ethusdt_5m", "5m"),
    ("data/raw/klines/ethusdt_15m", "15m"),
    ("data/raw/klines/solusdt_1h", "1h"),
    ("data/raw/klines/solusdt_4h", "4h"),
    ("data/raw/klines/solusdt_1d", "1d"),
    ("data/raw/klines/solusdt_5m", "5m"),
    ("data/raw/klines/solusdt_15m", "15m"),
]

# Live datasets (smaller — only gap check is meaningful for 7-day windows)
LIVE = [
    ("data/raw/live/btcusdt_1h", "1h"),
    ("data/raw/live/btcusdt_4h", "4h"),
    ("data/raw/live/btcusdt_1d", "1d"),
    ("data/raw/live/btcusdt_5m", "5m"),
    ("data/raw/live/btcusdt_15m", "15m"),
    ("data/raw/live/ethusdt_1h", "1h"),
    ("data/raw/live/ethusdt_4h", "4h"),
    ("data/raw/live/ethusdt_1d", "1d"),
    ("data/raw/live/ethusdt_5m", "5m"),
    ("data/raw/live/ethusdt_15m", "15m"),
    ("data/raw/live/solusdt_1h", "1h"),
    ("data/raw/live/solusdt_4h", "4h"),
    ("data/raw/live/solusdt_1d", "1d"),
    ("data/raw/live/solusdt_5m", "5m"),
    ("data/raw/live/solusdt_15m", "15m"),
]


def validate_datasets(
    datasets: list[tuple[str, str]], label: str
) -> dict[str, int]:
    """Run validation on a set of datasets, return summary stats."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Validating {label} datasets ({len(datasets)} total)")
    logger.info(f"{'─'*60}")

    summary = {"total": 0, "passed": 0, "failed": 0, "warnings": 0}

    for path_str, interval in datasets:
        dataset_path = Path(path_str)
        if not dataset_path.exists():
            logger.warning(f"  SKIP: {path_str} — directory not found")
            continue

        validator = DataValidator(dataset_path, interval)
        results = validator.run_all()
        validator.print_report()

        for r in results:
            summary["total"] += 1
            if r.passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1

    return summary


def main() -> None:
    """Run validation on all datasets and print final summary."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 1 · Story 1.2")
    logger.info("Data Quality Validation & Monitoring")
    logger.info("=" * 60)

    all_summaries: list[tuple[str, dict[str, int]]] = []

    # ── Historical ───────────────────────────────────────────
    hist_summary = validate_datasets(HISTORICAL, "Historical")
    all_summaries.append(("Historical", hist_summary))

    # ── Live ─────────────────────────────────────────────────
    live_summary = validate_datasets(LIVE, "Live (CCXT)")
    all_summaries.append(("Live", live_summary))

    # ── Final summary ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Data Quality Summary")
    logger.info(f"{'='*60}")
    logger.info(f"{'Category':<15} {'Total':>7} {'Passed':>7} {'Failed':>7}")
    logger.info(f"{'-'*15} {'-'*7} {'-'*7} {'-'*7}")

    grand = {"total": 0, "passed": 0, "failed": 0}
    for cat, s in all_summaries:
        logger.info(
            f"{cat:<15} {s['total']:>7} {s['passed']:>7} {s['failed']:>7}"
        )
        for k in grand:
            grand[k] += s[k]

    logger.info(f"{'-'*15} {'-'*7} {'-'*7} {'-'*7}")
    logger.info(
        f"{'TOTAL':<15} {grand['total']:>7} {grand['passed']:>7} {grand['failed']:>7}"
    )
    logger.info(f"{'='*60}")

    if grand["failed"] > 0:
        logger.error(f"{grand['failed']} checks failed — review details above")
    else:
        logger.info("All checks passed!")

    logger.info("Sprint 1 · Story 1.2 — COMPLETE")


if __name__ == "__main__":
    main()
