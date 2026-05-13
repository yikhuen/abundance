"""Quick verification: check_strategy_uses_abc flags carry/arb on BTC.

FundingCarry should PASS now that signals() actually returns a non-empty array
from funding-state lookup. HeArbitrage will PASS on BTC (perp data exists) and
FAIL on assets without perp_klines (correct outcome — flags un-auditable pairs).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abundance.deployment.adversarial import check_strategy_uses_abc
from abundance.strategies.momentum.grayscale_ma50 import MA50Strategy
from abundance.strategies.ema_crossover import EMA20Strategy
from abundance.strategies.donchian_breakout import DonchianStrategy
from abundance.strategies.composite.adx_blend import ADXBlendStrategy
from abundance.strategies.funding_momentum_btcusdt import FundingCarryStrategy
from abundance.strategies.he_arbitrage import HeArbitrageStrategy

strategies = [
    ("MA50", MA50Strategy()),
    ("EMA20", EMA20Strategy()),
    ("Donchian", DonchianStrategy()),
    ("ADX-blend", ADXBlendStrategy()),
    ("FundingCarry", FundingCarryStrategy()),
    ("HeArbitrage", HeArbitrageStrategy()),
]

print(f"{'Strategy':15s} {'Result':8s} {'Severity':10s} Notes")
print("-" * 70)
for name, s in strategies:
    r = check_strategy_uses_abc(s, "BTCUSDT")
    status = "PASS" if r.passed else "FAIL"
    note = r.checks_run[0] if r.checks_run else ""
    print(f"{name:15s} {status:8s} {r.severity:10s} {note}")
    for issue in r.issues:
        d = issue.get("detail", "")
        print(f"  issue: {d}")
