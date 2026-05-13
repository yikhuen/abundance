"""Real lookahead bias assertion.

For each strategy, runs it on the full dataset to get signal arrays,
then recomputes signals under progressive data truncation at multiple
points t. If sig[t] computed with df[:t+1] differs from sig[t] computed
with df[:t], the strategy peeked at bar t's data.

This actually inspects the strategy's signal output — unlike the
previous version which re-derived formulas inside the audit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.config.settings import settings


def check_strategy_signals(
    strategy,
    pair: str = "BTCUSDT",
    sample_points: int = 10,
) -> tuple[bool, list[dict]]:
    """Verify strategy signals have no lookahead.

    Args:
        strategy: Strategy instance implementing signals(df).
        pair: Trading pair to load.
        sample_points: Number of random bars to spot-check.

    Returns:
        (passed, violations) where violations list has {bar, signal_full, signal_truncated}
    """
    plower = pair.lower()
    df_full = pl.scan_parquet(
        str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
    ).sort("timestamp_ms").collect()

    N = len(df_full)

    # Get the "ground truth" signals from full dataset
    full_signals = strategy.signals(df_full)

    # Spot-check: at bar t, compute signal using only data[:t]
    # The signal at position t-1 (which governs position during bar t)
    # should be computable from df[:t] only.
    violations = []

    # Check at regular intervals
    check_points = list(range(100, min(N, 800), 50)) + list(range(N - 200, N - 10, 15))
    check_points = [c for c in check_points if 0 <= c < N]

    for t in check_points[:sample_points]:
        df_truncated = df_full[:t]
        truncated_signals = strategy.signals(df_truncated)

        # Signal at bar t-1 (position held during bar t) should match
        idx = t - 1
        if idx < 0 or idx >= len(truncated_signals):
            continue

        sig_full = full_signals[idx]
        sig_trunc = truncated_signals[idx]

        if abs(sig_full - sig_trunc) > 1e-9:
            violations.append({
                "bar": t,
                "idx": idx,
                "signal_full": sig_full,
                "signal_truncated": sig_trunc,
                "close_at_bar": df_full["close"][t] if t < N else None,
            })

    return len(violations) == 0, violations


def check_all():
    """Run lookahead check on all active strategies."""
    from abundance.strategies.momentum.grayscale_ma50 import MA50Strategy
    from abundance.strategies.ema_crossover import EMA20Strategy
    from abundance.strategies.donchian_breakout import DonchianStrategy
    from abundance.strategies.composite.adx_blend import ADXBlendStrategy

    strategies = [
        ("MA50", MA50Strategy()),
        ("EMA20", EMA20Strategy()),
        ("Donchian", DonchianStrategy()),
        ("ADX-blend", ADXBlendStrategy()),
    ]

    results = {}
    for name, strat in strategies:
        passed, violations = check_strategy_signals(strat, pair="BTCUSDT", sample_points=15)
        results[name] = {"passed": passed, "violations": len(violations)}
        status = "✅ PASS" if passed else f"❌ FAIL ({len(violations)} violations)"
        print(f"  {name:15s}: {status}")
        if violations:
            for v in violations[:3]:
                print(f"    bar={v['bar']} idx={v['idx']}: full={v['signal_full']:.3f} trunc={v['signal_truncated']:.3f}")

    all_pass = all(r["passed"] for r in results.values())
    print(f"\nLookahead: {'✅ ALL CLEAN' if all_pass else '❌ VIOLATIONS FOUND'}")
    return all_pass


if __name__ == "__main__":
    ok = check_all()
    sys.exit(0 if ok else 1)
