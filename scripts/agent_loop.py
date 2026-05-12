#!/usr/bin/env python3
"""Autonomous agent loop — agent-first, no mocks, no stubs, no templates.

This is INFRASTRUCTURE ONLY. The intelligence comes from the OpenClaw agent
that reads AGENTS.md and calls these functions with REAL tools.

Usage by an OpenClaw agent:
  1. Read AGENTS.md
  2. Call run_iteration() with real web_search/web_fetch/write tools
  3. Agent fills hypothesis, writes code, critiques, decides
  4. Deploy only approved strategies to testnet
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger


def update_workflow_status(node: str, status: str, details: str = "") -> None:
    """Update the workflow status file for the dashboard."""
    status_path = Path("data/processed/workflow_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)

    current = {}
    if status_path.exists():
        current = json.loads(status_path.read_text())

    current[node] = status
    if details:
        current[f"{node}_details"] = details
    current["last_updated"] = datetime.now(timezone.utc).isoformat()

    status_path.write_text(json.dumps(current, indent=2))


def run_iteration(
    pair: str,
    query: str,
    iteration: int,
    tools: dict,
    previous_critique: str = "",
) -> dict:
    """Run one full iteration of the research pipeline.

    Args:
        pair: Trading pair.
        query: Research query.
        iteration: Iteration number.
        tools: Dict of REAL tool functions from the OpenClaw agent.
            Required keys: web_search, web_fetch, write, read.
            The agent provides these — no mocks, no stubs.
        previous_critique: Critique from previous iteration.

    Returns dict with results + decision.
    """
    from abundance.orchestration.agents import (
        adversarial_node,
        backtest_node,
        coding_node,
        decision_node,
        hypothesis_node,
        research_node,
    )

    task_id = f"agent-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-iter{iteration}"

    # Validate required tools
    required = ["web_search", "web_fetch", "write"]
    missing = [t for t in required if t not in tools]
    if missing:
        return {"status": "error", "error": f"Missing required tools: {missing}"}

    state = {
        "task_id": task_id,
        "pair": pair,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_query": query,
        "research_findings": "",
        "papers_cited": [],
        "hypothesis": "",
        "hypothesis_rationale": "",
        "causal_mechanism": "",
        "strategy_code": "",
        "strategy_file": "",
        "backtest_results": {},
        "metrics_summary": "",
        "critique": "",
        "issues_found": [],
        "severity": "low",
        "decision": "",
        "decision_rationale": "",
        "human_approved": False,
    }

    # ── Run pipeline ──────────────────────────────────────
    nodes = [
        ("research", research_node),
        ("hypothesis", hypothesis_node),
        ("coding", coding_node),
        ("backtest", backtest_node),
        ("adversarial", adversarial_node),
        ("decision", decision_node),
    ]

    for node_name, node_fn in nodes:
        update_workflow_status(node_name, "active")
        try:
            result = node_fn(state, tools)
            if isinstance(result, dict):
                state.update(result)
            update_workflow_status(node_name, "completed")
        except Exception as e:
            logger.error(f"Node {node_name} failed: {e}")
            update_workflow_status(node_name, "error", str(e))
            return {"status": "error", "error": str(e), "node": node_name}

    return {
        "status": "ok",
        "task_id": task_id,
        "pair": pair,
        "iteration": iteration,
        "hypothesis": state.get("hypothesis", ""),
        "backtest_results": state.get("backtest_results", {}),
        "metrics_summary": state.get("metrics_summary", ""),
        "critique": state.get("critique", ""),
        "severity": state.get("severity", "low"),
        "issues_found": state.get("issues_found", []),
        "decision": state.get("decision", "reject"),
        "decision_rationale": state.get("decision_rationale", ""),
        "strategy_file": state.get("strategy_file", ""),
    }


def deploy_signal(pair: str, capital: float) -> dict:
    """Deploy current signal to testnet (dry-run safe).

    Called by the agent when a strategy is approved.
    """
    from abundance.deployment.bridge import SignalComputer
    from abundance.paper_trading.testnet_client import get_testnet_client

    client = get_testnet_client()
    computer = SignalComputer(client)
    sig = computer.compute(pair, capital)

    return {
        "pair": sig.pair,
        "direction": sig.direction,
        "allocation_pct": round(sig.allocation_pct * 100, 1),
        "price": sig.price,
        "target_notional": round(sig.target_notional(capital), 2),
    }


# ── Agent-facing entry points ──────────────────────────────────
# These are the functions an OpenClaw agent would call.
# The agent reads AGENTS.md, then invokes these using its native tools.

def agent_research(pair: str, tools: dict) -> dict:
    """Run a single research cycle and return findings."""
    return run_iteration(pair, "profitable crypto trading strategies", 1, tools)


def agent_deploy_check(pair: str, capital: float = 500) -> dict:
    """Check what would be deployed (dry-run)."""
    return deploy_signal(pair, capital)


def agent_status() -> dict:
    """Return full system status for the dashboard."""
    from abundance.deployment.validation import deployment_readiness_check

    return {
        "readiness": deployment_readiness_check(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
