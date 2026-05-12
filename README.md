# Abundance

**Agentic crypto trading system** — integrating mature open-source components
with OpenClaw orchestration and human-in-the-loop gates.

**Status:** Sprint 1 — Data Pipeline Foundation

## Architecture

```
openclaw (orchestrator)
  └─ langgraph (workflow)
       ├─ research agent    → Exa/Perplexity → arXiv
       ├─ hypothesis agent  → RD-Agent patterns
       ├─ coding agent      → Claude Code
       └─ adversarial agent → TradeTrap patterns

data layer
  ├─ binance vision (historical klines)
  ├─ ccxt (live unified API)
  ├─ hyperliquid (L2 order books)
  └─ parquet + duckdb/polars (storage + query)

execution
  └─ nautilustrader (backtest + live, rust core)
```

## Quick Start

```bash
# Install dependencies
poetry install

# Fetch first batch of BTCUSDT data (Sprint 1, Task 1)
poetry run python scripts/fetch_btc_data.py
```

## Project Structure

```
abundance/
├── src/abundance/
│   ├── data/           # Data fetching + storage
│   │   ├── fetcher.py          # Abstract base class
│   │   ├── binance_vision.py   # Binance Vision downloads
│   │   └── storage.py          # DuckDB query layer
│   ├── config/
│   │   └── settings.py         # Env-based configuration
│   ├── backtesting/    # NautilusTrader integration (Sprint 2)
│   └── orchestration/  # LangGraph + agent workflows (Sprint 4+)
├── scripts/            # One-shot data scripts
├── tests/              # pytest suite
├── notebooks/          # Exploratory analysis
└── data/               # Local Parquet + DuckDB (gitignored)
```

## Guiding Principles

1. **Integrate, don't reinvent.** Every component is a battle-tested OSS project.
2. **Causal grounding over data mining.** Every hypothesis cites a published mechanism.
3. **Evaluation infrastructure before automation.** Backtest harness before agent loop.

## Epics

| Epic | Sprint | Focus |
|------|--------|-------|
| 1 — Foundation | 1–2 | Data pipeline + evaluation harness |
| 2 — Manual Calibration | 3 | Hand-built funding carry strategy |
| 3 — Agentic Layer | 4–6 | LangGraph + research/coding/adversarial agents |
| 4 — Autoresearch | 7–8 | Overnight loop + paper trading |

## Requirements

- Python ≥3.10 (recommended: 3.12 LTS)
- Poetry for dependency management
- Docker for NautilusTrader (Sprint 2+)

## License

MIT
