#!/usr/bin/env python3
"""Sprint 4: Run the agentic research-to-trading LangGraph workflow.

Orchestrates: RESEARCH → HYPOTHESIS → CODING → BACKTEST → ADVERSARIAL → DECISION

In dev mode, uses stub agents. In production, spawns real OpenClaw
sub-agents with scoped tools.

Usage:
  python scripts/run_workflow.py              # dev mode, auto-approve
  python scripts/run_workflow.py --pair ETH    # target ETHUSDT
  python scripts/run_workflow.py --live        # production mode
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.orchestration.spawner import get_spawner
from abundance.orchestration.workflow import WorkflowBuilder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agentic research-to-trading workflow"
    )
    parser.add_argument(
        "--pair", default="BTCUSDT", help="Trading pair to research"
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Research query (default: auto-generated from pair)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use production spawner (requires OpenClaw)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=True,
        help="Auto-approve decisions (skip HITL in dev)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Abundance · Sprint 4 · Agentic Workflow Runner")
    logger.info(f"  Pair:    {args.pair}")
    logger.info(f"  Mode:    {'production' if args.live else 'development'}")
    logger.info(f"  HITL:    {'auto-approve' if args.auto_approve else 'manual'}")
    logger.info("=" * 60)

    # ── Initialise workflow ─────────────────────────────────
    builder = WorkflowBuilder()
    env = "production" if args.live else "dev"
    spawner = get_spawner(env)

    # ── Build initial state ─────────────────────────────────
    initial_state = {
        "task_id": f"research-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "pair": args.pair,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_query": args.query or f"profitable trading signals for {args.pair} crypto perpetuals",
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

    # ── Compile with HITL at decision node ──────────────────
    interrupt_before = ["decision"] if not args.auto_approve else None
    app = builder.compile(interrupt_before=interrupt_before)
    config = {"configurable": {"thread_id": initial_state["task_id"]}}

    # ── Run workflow ────────────────────────────────────────
    logger.info("\nStarting workflow...")
    logger.info(f"Thread: {initial_state['task_id']}\n")

    # Stream through each node
    for event in app.stream(initial_state, config):
        node_name = list(event.keys())[0]
        node_state = event[node_name]

        logger.info(f"✅ Node: {node_name}")
        if node_name == "research":
            logger.info(f"   Query: {node_state.get('research_query', '')[:100]}")
        elif node_name == "hypothesis":
            logger.info(f"   Hypothesis: {node_state.get('hypothesis', '')[:100]}")
        elif node_name == "coding":
            logger.info(f"   File: {node_state.get('strategy_file', '')}")
        elif node_name == "backtest":
            logger.info(f"   Results: {node_state.get('metrics_summary', '')[:100]}")
        elif node_name == "adversarial":
            severity = node_state.get("severity", "low")
            issues = node_state.get("issues_found", [])
            logger.info(f"   Severity: {severity}, Issues: {len(issues)}")
        elif node_name == "decision":
            decision = node_state.get("decision", "reject")
            logger.info(f"   Decision: {decision}")
            if decision == "approve":
                logger.info("   ✅ Strategy approved for paper trading")
            elif decision == "revise":
                logger.info("   🔄 Revision requested — looping back to research")

    # ── Final state ─────────────────────────────────────────
    final_state = app.get_state(config)
    logger.info(f"\n{'='*60}")
    logger.info("Workflow Complete")
    logger.info(f"  Decision:  {final_state.values.get('decision', 'unknown')}")
    logger.info(f"  Rationale: {final_state.values.get('decision_rationale', '')}")
    logger.info(f"{'='*60}")

    logger.info("Sprint 4 — Agentic Workflow COMPLETE")


if __name__ == "__main__":
    main()
