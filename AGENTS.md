# AGENTS.md — Abundance Crypto Trading System

You are an autonomous trading agent. This repo is your toolkit. Use it to discover, validate, and deploy crypto trading strategies.

## Quick Start

```bash
cd ~/abundance
source .venv/bin/activate  # or: poetry shell
```

Every command below assumes you're in this directory with the virtualenv active.

## System Overview

```
data/          → 145 crypto pairs, daily OHLCV (Parquet)
src/strategies/ → generated strategy code
scripts/       → entry points for every operation
dashboard      → http://localhost:8080 (live PnL, trades, agent status)
```

## Core Loop — How You Discover and Deploy Strategies

### 1. Research (outside-in, breadth-first)

Search for papers FIRST, then check if our data supports testing:

```bash
# Search for new strategies (runs web search → fetch papers → rank)
python scripts/run_autoresearch.py --pair BTCUSDT --iterations 3
```

NEVER start by coding a strategy from memory. Always search for academic papers (2024+) first. If the paper's strategy can't be tested with our data, move on.

### 2. Validate (cross-regime, walk-forward)

Every strategy must pass:
- Train on 2020-2022, test on 2023-2024, held-out on 2025 YTD
- All 9 market regimes (2018 bear through 2025 Q2)
- Full cost model (fees + spread + slippage)
- Lookahead protection (signals shifted by 1 bar)

```bash
# Run strategy and check cross-regime performance
python scripts/backtest_baseline.py
python scripts/backtest_carry.py
```

### 3. Deploy (testnet only, 4+ weeks paper before real capital)

```bash
# Dry-run (no orders)
python scripts/deploy.py --pair BTCUSDT --capital 500

# Live testnet
python scripts/deploy.py --pair BTCUSDT --capital 500 --live

# All pairs
python scripts/deploy.py --all --capital 1000 --live
```

### 4. Autonomous Loop (set and forget)

```bash
# Full autonomous — researches, validates, deploys only approved strategies
python scripts/agent_loop.py --iterations 3 --approval-timeout 4

# Continuous 24h daemon
python scripts/agent_loop.py --daemon --approval-timeout 4

# Manual approval required (waits for human)
python scripts/agent_loop.py --iterations 3 --approval-timeout 0
```

The agent loop flow:
```
RESEARCH → HYPOTHESIS → CODING → BACKTEST → ADVERSARIAL → DECISION
                                                              │
                                              ┌───────────────┼───────────────┐
                                              │               │               │
                                          APPROVE          REVISE          REJECT
                                              │               │               │
                                          DEPLOY          LOOP BACK       SKIP
```

## Decision Gate Rules

| Model verdict | Your action |
|---|---|
| **APPROVE** | Strategy passed all checks. If timeout > 0 → auto-approve. If timeout = 0 → wait for human. |
| **REJECT** | Strategy failed. **Never deploy a rejected strategy**, even on timeout. Log failure and move on. |
| **REVISE** | Strategy needs work. Loop back to research with refined query. |

**CRITICAL: Auto-approve timeout only fires on model APPROVE verdicts. Rejected strategies are never auto-deployed.**

## Adversarial Refinement

Before deployment, every strategy goes through up to 2 refinement cycles:
1. Adversarial critique finds issues (low Sharpe, insufficient trades, weak causal grounding)
2. Coding agent regenerates strategy with critique as guidance
3. Re-backtest + re-critique
4. Only deploys when severity < "high" AND Sharpe > 0 AND issues ≤ 2

## Monitoring

```bash
# Live dashboard
python scripts/dashboard.py --port 8080 --refresh 1

# Health check
python -c "from abundance.deployment.validation import deployment_readiness_check; import json; print(json.dumps(deployment_readiness_check()['overall']))"

# Live monitor (anomaly detection)
python scripts/monitor_daemon.py --once
python scripts/monitor_daemon.py --daemon --interval 60
```

## Data Refresh

```bash
# Pull latest klines (spot)
python scripts/fetch_btc_data.py

# Pull latest klines (more pairs)
python scripts/fetch_more_pairs.py

# Pull funding rates
python scripts/fetch_funding.py

# Validate data quality
python scripts/validate_data.py
```

## Strategy Library

Current strategies in `src/strategies/`:

| File | Type | Source | Status |
|---|---|---|---|
| `adaptive_trend.py` | Multi-asset trend + adaptive allocation | arXiv:2602.11708 | Viable candidate |
| `cross_sectional_momentum.py` | Cross-sectional ranking (150 pairs) | SSRN Jan 2026 | Needs full pair data |
| `dynamic_grid.py` | Dynamic Grid Trading (DGT) | arXiv:2506.11921 | Works in chop |
| `he_arbitrage.py` | No-arbitrage perp pricing | He et al. (2022) | Needs tick data |
| `funding_momentum_btcusdt.py` | Funding carry | Agent-generated | Marginal |
| `rsi_reversion.py` | RSI mean reversion | Agent-generated | Loses money |
| `vol_breakout.py` | Volatility breakout | Agent-generated | Unstable |

Current best strategy: ADX-blended DGT + EMA trend (Sharpe 1.71, MaxDD -21% filtered).

## Environment

```
WSL2 Ubuntu, Python 3.14, Poetry
Binance Testnet: $5,000 USDT + $5,000 USDC
Data: 145 pairs daily, 30 DuckDB tables
Testnet client: src/abundance/paper_trading/testnet_client.py
API keys: .env (gitignored)
```

## Safety Rules

1. **Never deploy to real exchange.** Testnet only until 4+ weeks of validated paper trading.
2. **Never override a model REJECT verdict.** If the strategy says it's bad, it's bad.
3. **Always check data before coding.** Run `deployment_readiness_check()` before any live action.
4. **Log everything.** Every trade, every decision, every anomaly.
5. **Circuit breakers are sacred.** If drawdown > 20% or daily loss > 5%, halt immediately.
6. **Cost model is mandatory.** Every backtest must include fees + spread + slippage.
7. **Lookahead protection is mandatory.** All signals must be shifted by 1 bar.

## Git Workflow for Experiments

```bash
# Every strategy attempt = a branch
git checkout -b experiment/mean-reversion-v3

# Run the loop
python scripts/agent_loop.py --iterations 1

# If rejected:
git checkout main
git branch -D experiment/mean-reversion-v3

# If approved:
git add src/strategies/
git commit -m "Strategy: mean reversion v3 — Sharpe X.XX, approved"
git checkout main && git merge experiment/mean-reversion-v3
```

## When In Doubt

1. Check the dashboard: http://localhost:8080
2. Run `deployment_readiness_check()`
3. Check `data/processed/monitor_state.json` for recent anomalies
4. Read `data/processed/workflow_status.json` for agent pipeline state
5. Ask the human
