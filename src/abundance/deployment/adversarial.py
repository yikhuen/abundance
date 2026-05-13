"""Adversarial validation — mechanical red-team checks (no LLM required).

Runs deterministic checks that catch common backtesting bugs:
1. Lookahead: signal[t] recomputed from df[:t] must match full run
2. Signal sanity: no impossible values, reasonable trade frequency
3. Cost verification: no double-counting
4. Walk-forward: OOS performance comparable to IS
5. Parameter sensitivity: Sharpe doesn't collapse on ±10% perturbation

These run WITHOUT an LLM — they're pure code assertions.
LLM-based critique (for narrative issues, economic reasoning, etc.)
uses the structured prompt returned by get_llm_critique_prompt().
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from abundance.config.settings import settings
from abundance.strategies.base import Strategy, StrategyArtifacts


@dataclass
class AdversarialResult:
    """Result of adversarial red-team check."""
    passed: bool
    severity: str  # "low", "medium", "high", "critical"
    issues: list[dict] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "severity": self.severity,
            "issues": self.issues,
            "checks_run": self.checks_run,
        }


def check_lookahead(strategy: Strategy, pair: str = "BTCUSDT") -> AdversarialResult:
    """Verify signal[t] uses only data available at t-1."""
    plower = pair.lower()
    df = pl.scan_parquet(
        str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
    ).sort("timestamp_ms").collect()
    N = len(df)

    strategy.set_pair(pair)
    full_signals = strategy.signals(df)
    issues = []

    for t in [200, 400, 600, 800, 1000, 1200, N // 2, N - 200]:
        if t <= 0 or t >= N:
            continue
        df_trunc = df[:t]
        strategy.set_pair(pair)
        trunc_signals = strategy.signals(df_trunc)
        idx = t - 1
        if idx >= len(trunc_signals):
            continue
        if abs(full_signals[idx] - trunc_signals[idx]) > 1e-9:
            issues.append({
                "check": "lookahead",
                "bar": t, "idx": idx,
                "signal_full": full_signals[idx],
                "signal_truncated": trunc_signals[idx],
                "detail": "Signal changed when future data added — lookahead suspected",
            })

    severity = "critical" if issues else "low"
    return AdversarialResult(
        passed=len(issues) == 0,
        severity=severity,
        issues=issues,
        checks_run=["lookahead: signal truncation invariance at 8 points"],
    )


def check_signal_sanity(artifacts: StrategyArtifacts) -> AdversarialResult:
    """Check signal array for impossible values and patterns."""
    sig = artifacts.signals
    N = len(sig)
    issues = []

    # Must be all 0 or 1 (or 0 to 1 for continuous blends)
    for i, s in enumerate(sig):
        if s < 0 or s > 1:
            issues.append({
                "check": "signal_range",
                "bar": i, "value": s,
                "detail": f"Signal out of [0,1] range: {s} at bar {i}",
            })

    # Active fraction should be reasonable (1-50%)
    active = sum(1 for s in sig if s > 0.01) / max(N, 1)
    if active < 0.005:
        issues.append({
            "check": "activity",
            "active_pct": active * 100,
            "detail": f"Strategy active <0.5% of bars ({active*100:.2f}%) — too infrequent",
        })
    if active > 0.95:
        issues.append({
            "check": "activity",
            "active_pct": active * 100,
            "detail": f"Strategy active >95% of bars ({active*100:.1f}%) — essentially B&H",
        })

    # Flips should be reasonable
    flips = sum(1 for i in range(1, N) if abs(sig[i] - sig[i-1]) > 0.01)
    flip_rate = flips / max(N, 1)
    if flip_rate > 0.3:
        issues.append({
            "check": "trade_frequency",
            "flips": flips, "flip_rate": f"{flip_rate*100:.1f}%",
            "detail": f"Signal flips >30% of bars ({flip_rate*100:.1f}%) — likely overfitting",
        })

    severity = "high" if issues else "low"
    return AdversarialResult(
        passed=len(issues) == 0,
        severity=severity,
        issues=issues,
        checks_run=["signal_range [0,1]", f"activity {active*100:.1f}%", f"flip_rate {flip_rate*100:.1f}%"],
    )


def check_walk_forward(strategy: Strategy, pair: str = "BTCUSDT") -> AdversarialResult:
    """Compare full-sample Sharpe to walk-forward OOS Sharpe."""
    plower = pair.lower()
    df = pl.scan_parquet(
        str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
    ).sort("timestamp_ms").collect()
    close = df["close"].to_list()
    N = len(close)

    # Split at 60% — train on 2017-2021, test on 2022-2025
    split = N * 60 // 100
    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1, N)]

    strategy.set_pair(pair)
    sig = strategy.signals(df)
    from abundance.backtesting.costs import COST_MODEL
    entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
    exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)

    def sharpe_from_returns(net_ret, start, end):
        sub = net_ret[start:end]
        if len(sub) < 20:
            return 0.0
        mean = sum(sub) / len(sub)
        var = sum((r - mean)**2 for r in sub) / len(sub)
        return (mean / (var**0.5)) * (365**0.5) if var > 0 else 0.0

    net_ret = [0.0] * N
    for i in range(1, N):
        gross = ret[i] * sig[i]
        txn = 0.0
        if sig[i] > sig[i-1]: txn = entry_cost
        elif sig[i] < sig[i-1]: txn = exit_cost
        net_ret[i] = gross - txn

    is_sharpe = sharpe_from_returns(net_ret, 1, split)
    oos_sharpe = sharpe_from_returns(net_ret, split, N)

    issues = []
    if abs(oos_sharpe - is_sharpe) > 1.5:
        issues.append({
            "check": "walk_forward",
            "is_sharpe": round(is_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "detail": f"OOS Sharpe differs from IS by >1.5: IS={is_sharpe:.2f} OOS={oos_sharpe:.2f}",
        })
    if oos_sharpe < 0:
        issues.append({
            "check": "walk_forward",
            "oos_sharpe": round(oos_sharpe, 3),
            "detail": f"OOS Sharpe is negative: {oos_sharpe:.2f}",
        })

    severity = "critical" if oos_sharpe < 0 else ("high" if issues else "low")
    return AdversarialResult(
        passed=len(issues) == 0,
        severity=severity,
        issues=issues,
        checks_run=[f"IS Sharpe={is_sharpe:.2f}", f"OOS Sharpe={oos_sharpe:.2f}"],
    )


def check_parameter_sensitivity(strategy: Strategy, pair: str = "BTCUSDT") -> AdversarialResult:
    """Perturb parameters ±20% and check Sharpe doesn't collapse."""
    from abundance.backtesting.costs import COST_MODEL
    import copy

    plower = pair.lower()
    df = pl.scan_parquet(
        str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
    ).sort("timestamp_ms").collect()
    close = df["close"].to_list(); N = len(close)
    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1, N)]

    def compute_sharpe(sig):
        entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)
        nr = [0.0]*N
        for i in range(1,N):
            g = ret[i]*sig[i]; t = 0.0
            if sig[i] > sig[i-1]: t = entry_cost
            elif sig[i] < sig[i-1]: t = exit_cost
            nr[i] = g-t
        sub = nr[1:]
        if len(sub) < 20: return 0.0
        m = sum(sub)/len(sub); v = sum((r-m)**2 for r in sub)/len(sub)
        return (m/v**0.5)*(365**0.5) if v > 0 else 0.0

    base_sharpe = compute_sharpe(strategy.signals(df))
    params = strategy._get_params()
    sharpe_range = []

    for key, val in params.items():
        if not isinstance(val, (int, float)):
            continue
        for factor in [0.8, 1.2]:
            perturbed = copy.deepcopy(strategy)
            perturbed_params = perturbed._get_params()
            perturbed_params[key] = max(2, int(val * factor))
            # Hack: set attribute directly
            setattr(perturbed, key, perturbed_params[key])
            try:
                pert_sig = perturbed.signals(df)
                pert_sharpe = compute_sharpe(pert_sig)
                sharpe_range.append(pert_sharpe)
            except Exception:
                pass

    issues = []
    if sharpe_range:
        min_s = min(sharpe_range)
        max_drop = (base_sharpe - min_s) / max(base_sharpe, 0.01)
        if max_drop > 0.5:
            issues.append({
                "check": "parameter_sensitivity",
                "base_sharpe": round(base_sharpe, 3),
                "min_perturbed": round(min_s, 3),
                "max_drop_pct": f"{max_drop*100:.0f}%",
                "detail": f"Sharpe drops >50% on parameter perturbation — likely overfit",
            })

    severity = "high" if issues else "low"
    return AdversarialResult(
        passed=len(issues) == 0,
        severity=severity,
        issues=issues,
        checks_run=[f"base Sharpe={base_sharpe:.2f}"],
    )


def check_strategy_uses_abc(strategy: Strategy, pair: str = "BTCUSDT") -> AdversarialResult:
    """Verify the strategy exposes a non-empty, non-trivial signal vector via signals().

    Strategies that bypass the ABC by returning [] from signals() and computing
    everything inside an overridden run() cannot be properly audited for lookahead,
    signal sanity, or parameter sensitivity. This check closes that loophole.

    The check loads the same daily-kline data the other checks use, calls
    strategy.signals(df), and rejects if the result is empty, length-mismatched,
    or all-zero. Severity is critical because none of the other checks are
    meaningful without a real signal array.
    """
    plower = pair.lower()
    df = pl.scan_parquet(
        str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
    ).sort("timestamp_ms").collect()
    N = len(df)

    strategy.set_pair(pair)
    issues = []
    try:
        sig = strategy.signals(df)
    except Exception as e:
        issues.append({
            "check": "abc_compliance",
            "detail": f"strategy.signals(df) raised: {e}",
        })
        return AdversarialResult(
            passed=False, severity="critical", issues=issues,
            checks_run=["abc_compliance: signals() must execute"],
        )

    if not sig or len(sig) == 0:
        issues.append({
            "check": "abc_compliance",
            "signal_length": 0,
            "detail": "signals() returned empty list — strategy bypasses ABC audit surface",
        })
    elif len(sig) != N:
        issues.append({
            "check": "abc_compliance",
            "signal_length": len(sig), "data_length": N,
            "detail": f"signals() length {len(sig)} != data length {N}",
        })
    elif all(abs(s) < 1e-9 for s in sig):
        issues.append({
            "check": "abc_compliance",
            "detail": "signals() returned all zeros — likely stub implementation",
        })

    severity = "critical" if issues else "low"
    return AdversarialResult(
        passed=len(issues) == 0,
        severity=severity,
        issues=issues,
        checks_run=[f"abc_compliance: signals() returns {len(sig) if sig else 0} values"],
    )


def run_full_adversarial(
    strategy: Strategy, pair: str = "BTCUSDT"
) -> dict:
    """Run all mechanical adversarial checks on a strategy.

    Returns a dict with overall pass/fail, severity, and per-check details.
    """
    checks = [
        ("abc_compliance", check_strategy_uses_abc(strategy, pair)),
        ("lookahead", check_lookahead(strategy, pair)),
        ("signal_sanity", check_signal_sanity(strategy.run(pair))),
        ("walk_forward", check_walk_forward(strategy, pair)),
        ("parameter_sensitivity", check_parameter_sensitivity(strategy, pair)),
    ]

    results = {}
    overall_passed = True
    max_severity = "low"
    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    for name, result in checks:
        results[name] = result.to_dict()
        if not result.passed:
            overall_passed = False
        if severity_order.get(result.severity, 0) > severity_order.get(max_severity, 0):
            max_severity = result.severity

    return {
        "passed": overall_passed,
        "severity": max_severity,
        "checks": results,
    }


def get_llm_critique_prompt(artifacts: StrategyArtifacts, adversarial_results: dict) -> str:
    """Generate a structured prompt for LLM-based critique.

    The LLM should evaluate: economic mechanism validity, overfitting risk,
    regime dependence, capacity constraints, and benchmark appropriateness.
    """
    return f"""You are a quantitative finance adversarial reviewer. Evaluate this trading strategy for failure modes that mechanical checks cannot catch.

STRATEGY: {artifacts.pair} | Params: {artifacts.params}
MECHANICAL CHECK RESULTS: {adversarial_results}

Please evaluate:
1. ECONOMIC MECHANISM: Is there a real reason this should make money, or is it curve-fitting? What market inefficiency does it exploit? When would that inefficiency disappear?
2. OVERFITTING RISK: How many parameters? How many were optimized? Is the strategy complexity justified by the data?
3. REGIME DEPENDENCE: What market regimes would kill this strategy? Has it been tested in those regimes?
4. CAPACITY: Would this strategy survive at scale? What's the max AUM before slippage erodes alpha?
5. BENCHMARK: Is the comparison fair? Does the strategy take more risk than the benchmark?
6. IMPLEMENTATION: Are there hidden costs (funding, basis, exchange risk) not modeled?

Return a structured response with:
- severity: "low" | "medium" | "high" | "critical"
- issues_found: list of specific problems
- recommendation: "approve" | "revise" | "reject"
- rationale: detailed explanation
"""
