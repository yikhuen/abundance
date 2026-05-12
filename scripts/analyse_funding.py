#!/usr/bin/env python3
"""Sprint 3 · Stage 3.2: Funding rate analysis for carry threshold selection.

Analyses funding rate distributions, persistence, and optimal
carry thresholds across BTC, ETH, SOL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings


def analyse_pair(pair: str) -> dict:
    """Analyse funding rates for a single pair."""
    glob = str(settings.raw_dir / "funding" / pair.lower() / "**" / "*.parquet")
    df = pl.scan_parquet(glob).sort("timestamp_ms").collect()

    if df.is_empty():
        return {}

    rate = df["funding_rate_pct"]
    n = len(df)

    # Basic statistics
    results = {
        "pair": pair,
        "records": n,
        "mean_pct": rate.mean(),
        "median_pct": rate.median(),
        "std_pct": rate.std(),
        "min_pct": rate.min(),
        "max_pct": rate.max(),
        "skew": rate.skew(),
    }

    # Percentile thresholds
    for pct in [75, 80, 85, 90, 95]:
        results[f"p{pct}_threshold"] = rate.quantile(pct / 100)

    # Positive funding frequency
    positive_pct = (rate > 0).sum() / n * 100
    results["positive_pct"] = positive_pct

    # Persistence: autocorrelation at lag 1
    shifted = rate.shift(1)
    if n > 2:
        autocorr = (rate - rate.mean()).dot(shifted - shifted.mean())
        autocorr /= (n - 1) * rate.std() * shifted.std()
        results["autocorr_lag1"] = autocorr
    else:
        results["autocorr_lag1"] = 0.0

    # Extreme events (> 0.1% or < -0.1%)
    extreme_positive = (rate > 0.1).sum()
    extreme_negative = (rate < -0.1).sum()
    results["extreme_positive_events"] = extreme_positive
    results["extreme_negative_events"] = extreme_negative

    # Duration analysis: consecutive positive periods
    is_positive = rate > 0
    pos_runs = []
    current_run = 0
    for v in is_positive.to_list():
        if v:
            current_run += 1
        else:
            if current_run > 0:
                pos_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        pos_runs.append(current_run)

    if pos_runs:
        results["avg_positive_run"] = sum(pos_runs) / len(pos_runs)
        results["max_positive_run"] = max(pos_runs)
    else:
        results["avg_positive_run"] = 0
        results["max_positive_run"] = 0

    return results


def main() -> None:
    """Run funding rate analysis and recommend carry thresholds."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 3 · Stage 3.2")
    logger.info("Funding Rate Analysis & Threshold Selection")
    logger.info("=" * 60)

    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_results = []

    for pair in pairs:
        results = analyse_pair(pair)
        if results:
            all_results.append(results)

            logger.info(f"\n{'─'*50}")
            logger.info(f"  {pair} Funding Rate Analysis")
            logger.info(f"{'─'*50}")
            logger.info(f"  Records:         {results['records']:,}")
            logger.info(f"  Mean rate:       {results['mean_pct']:.4f}%")
            logger.info(f"  Median rate:     {results['median_pct']:.4f}%")
            logger.info(f"  Std rate:        {results['std_pct']:.4f}%")
            logger.info(f"  Range:           [{results['min_pct']:.4f}%, {results['max_pct']:.4f}%]")
            logger.info(f"  Positive freq:   {results['positive_pct']:.1f}%")
            logger.info(f"  Autocorr (lag1): {results['autocorr_lag1']:.3f}")
            logger.info(f"  Avg pos. run:    {results['avg_positive_run']:.1f} periods")
            logger.info(f"  Max pos. run:    {results['max_positive_run']} periods")
            logger.info(f"  Extreme +:       {results['extreme_positive_events']} | -: {results['extreme_negative_events']}")
            logger.info(f"\n  Percentile thresholds:")
            for pct in [75, 80, 85, 90, 95]:
                logger.info(f"    P{pct}: {results[f'p{pct}_threshold']:.4f}%")

    # ── Threshold recommendation ────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Carry Threshold Recommendations")
    logger.info(f"{'='*60}")
    logger.info("Strategy: short when funding_rate > threshold,")
    logger.info("         close when funding_rate reverts below threshold/2.")
    logger.info("")
    for r in all_results:
        pair = r["pair"]
        # Recommend P85 as entry threshold (captures top 15% of funding rates)
        entry = r["p85_threshold"]
        logger.info(
            f"  {pair:<10} Entry:  {entry:.4f}% (P85)"
        )
    logger.info("")
    logger.info("Rationale: P85 balances signal frequency with premium capture.")
    logger.info("  Too low  (< P70) → noise, too many false entries")
    logger.info("  Too high (> P90) → rare signals, insufficient trades")
    logger.info(f"{'='*60}")

    logger.info("Stage 3.2 — COMPLETE")


if __name__ == "__main__":
    main()
