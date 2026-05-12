#!/usr/bin/env python3
"""Autoresearch loop — agent-first, no mocks.

Called BY the OpenClaw agent. The agent provides REAL tools.
This is the infrastructure that runs the research → backtest → decide loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger


def print_usage() -> None:
    """Print usage for the OpenClaw agent."""
    print("""
Autoresearch Loop — Agent-First
================================
Called by the OpenClaw agent after reading AGENTS.md.

Functions available:
  agent_research(pair, tools)     → run one iteration
  agent_compare(results)          → rank strategies by Sharpe
  agent_deploy(pair, strategy_file) → deploy to testnet

The agent (OpenClaw) handles:
  - web search (real web_search tool)
  - code generation (LLM writes strategy code)
  - adversarial review (LLM critiques)
  - decision making (agent decides approve/reject/revise)

This script provides: infrastructure, state tracking, workflow status.
""")


if __name__ == "__main__":
    print_usage()
