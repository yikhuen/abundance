#!/usr/bin/env python3
"""Agent-first workflow runner — expects REAL tools from OpenClaw agent.

This script is called BY an OpenClaw agent. The agent provides tools.
No mocks. No stubs. No standalone mode.

The agent reads AGENTS.md, loads this script, and calls run_iteration()
with its web_search, web_fetch, write, and read tools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.orchestration.spawner import TOOL_SCOPES


def print_usage() -> None:
    """Print usage instructions for the OpenClaw agent."""
    print("""
Agent-First Workflow Runner
============================
This script is NOT run standalone. An OpenClaw agent calls these functions:

  1. Read AGENTS.md for full instructions
  2. agent_research(pair, tools)  → web search → hypothesis → backtest → decide
  3. agent_deploy(pair, capital)  → compute signal → ready for testnet orders

Required tools (provided by OpenClaw agent):
  web_search(query: str) → str
  web_fetch(url: str) → str
  write(path: str, content: str) → None
  read(path: str) → str
  exec(command: str) → str

Tool scoping per node:
  research:    web_search, web_fetch
  hypothesis:  read, write
  coding:      read, write, exec
  backtest:    exec, read
  adversarial: read, write
""")


def get_tool_scope(agent_type: str) -> list[str]:
    """Return the tool scope for a given agent type."""
    return TOOL_SCOPES.get(agent_type, {}).get("tools", [])


if __name__ == "__main__":
    print_usage()
