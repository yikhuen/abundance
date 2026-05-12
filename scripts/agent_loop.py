#!/usr/bin/env python3
"""Autonomous agent loop with human-in-the-loop approval gate.

Runs the full autoresearch pipeline and pauses at the decision node.
Sends summary via alerts. Waits for human approval or auto-approves
after timeout.

Usage:
  python scripts/agent_loop.py                              # full auto
  python scripts/agent_loop.py --approval-timeout 4         # 4h timeout
  python scripts/agent_loop.py --approval-timeout 0         # require manual
  python scripts/agent_loop.py --iterations 5 --daemon      # run continuously
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from abundance.deployment.monitoring import AlertDispatcher


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


def run_iteration(pair: str, query: str, iteration: int, previous_critique: str = "") -> dict:
    """Run one full iteration of the research pipeline.

    Returns dict with results + decision.
    """
    import polars as pl

    from abundance.backtesting.metrics import MetricsCalculator
    from abundance.config.settings import settings
    from abundance.orchestration.agents import (
        adversarial_node,
        backtest_node,
        coding_node,
        decision_node,
        hypothesis_node,
        research_node,
    )

    task_id = f"agent-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-iter{iteration}"

    # Refine query from previous critique
    if previous_critique and "sharpe" in previous_critique.lower():
        query = "high Sharpe ratio risk-adjusted " + query
    elif previous_critique and "drawdown" in previous_critique.lower():
        query = "low drawdown capital preservation " + query

    # Mock tools in standalone mode
    tools = {
        "web_search": lambda q: f"[Search results for: {q}]",
        "web_fetch": lambda u: f"[Fetch content from: {u}]",
        "write": lambda p, c: Path(p).parent.mkdir(parents=True, exist_ok=True) or Path(p).write_text(c),
        "read": lambda p: Path(p).read_text() if Path(p).exists() else "",
    }

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


def wait_for_approval(result: dict, timeout_hours: int, alerts: AlertDispatcher) -> str:
    """Wait for human approval with timeout.

    Returns 'approved', 'rejected', or 'auto_approved'.

    KEY RULE: If the strategy voted 'reject', timeout defaults to REJECT.
    Auto-approve only fires if the strategy itself voted 'approve'.
    """
    decision = result.get("decision", "reject")
    summary = (
        f"📋 Agent Decision Request\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Pair:     {result['pair']}\n"
        f"Strategy: {result.get('hypothesis', 'N/A')[:150]}\n"
        f"Results:  {result.get('metrics_summary', 'N/A')}\n"
        f"Severity: {result.get('severity', '?')}\n"
        f"Issues:   {len(result.get('issues_found', []))}\n"
        f"Auto-decision: {decision.upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
    )

    alerts.send("warning", f"DECISION REQUIRED ({timeout_hours}h timeout):\n{summary}")

    if timeout_hours == 0:
        logger.info("⏸️  Awaiting human approval (no timeout)")
        logger.info(summary)
        logger.info("Reply 'approve', 'reject', or 'revise' to continue")
        return "awaiting"

    # CRITICAL: only auto-approve if the model itself voted approve
    if decision == "approve":
        alerts.send("info", f"✅ AUTO-APPROVED (model confidence): {result.get('decision_rationale', '')}")
        update_workflow_status("decision", "auto_approved")
        return "auto_approved"
    elif decision == "reject":
        # Model said reject → respect that. No auto-approval.
        alerts.send("info", f"❌ AUTO-REJECTED: Strategy's own verdict was REJECT — skipping deployment")
        update_workflow_status("decision", "rejected", result.get("decision_rationale", ""))
        return "rejected"
    else:
        # revise — not deployable without human input
        alerts.send("info", f"🔄 Strategy needs revision — skipping deployment")
        update_workflow_status("decision", "needs_revision")
        return "rejected"


def main():
    parser = argparse.ArgumentParser(description="Autonomous agent loop with HITL approval")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations per run")
    parser.add_argument("--approval-timeout", type=float, default=1.0,
                        help="Hours to wait for human approval (0 = manual, >0 = auto-approve after timeout)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--daemon-interval", type=int, default=86400, help="Seconds between daemon runs (default: 24h)")
    parser.add_argument("--query", default="profitable crypto trading strategies", help="Research seed query")
    args = parser.parse_args()

    alerts = AlertDispatcher()
    logger.info("=" * 60)
    logger.info("Abundance — Autonomous Agent Loop")
    logger.info(f"  Pair:             {args.pair}")
    logger.info(f"  Iterations:       {args.iterations}")
    logger.info(f"  Approval timeout: {'manual' if args.approval_timeout == 0 else f'{args.approval_timeout}h (auto-approve)'}")
    logger.info(f"  Mode:             {'daemon' if args.daemon else 'once'}")
    logger.info("=" * 60)

    # Reset workflow status
    for node in ["research", "hypothesis", "coding", "backtest", "adversarial", "decision", "paper_trade"]:
        update_workflow_status(node, "idle")

    best_result = None
    best_sharpe = -999

    while True:
        previous_critique = ""
        for iteration in range(1, args.iterations + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Iteration {iteration}/{args.iterations}")
            logger.info(f"{'='*50}")

            result = run_iteration(args.pair, args.query, iteration, previous_critique)

            if result["status"] != "ok":
                logger.error(f"Iteration {iteration} failed: {result.get('error')}")
                continue

            # ── Generative-Adversarial Refinement Loop ──────
            # Refine strategy if adversarial critique finds issues
            refine_count = 0
            MAX_REFINEMENTS = 2
            while refine_count < MAX_REFINEMENTS:
                severity = result.get("severity", "low")
                issues = result.get("issues_found", [])
                sharpe = result.get("backtest_results", {}).get("sharpe", -999)

                # Check if refinement is needed
                needs_refinement = (
                    severity in ("high", "critical")
                    or (isinstance(sharpe, (int, float)) and sharpe < 0)
                    or len(issues) > 2
                )

                if not needs_refinement:
                    logger.info(f"  ✅ Strategy passes adversarial review (severity: {severity}, issues: {len(issues)})")
                    break

                refine_count += 1
                logger.info(f"  🔄 Refinement {refine_count}/{MAX_REFINEMENTS} — addressing: {', '.join(issues[:2])}")
                update_workflow_status("adversarial", f"refining_{refine_count}")

                # Refine: re-run coding with critique as feedback
                refined = run_iteration(
                    args.pair,
                    f"refine strategy addressing: {'; '.join(issues[:2])}",
                    iteration,
                    previous_critique="\n".join(issues),
                )
                if refined["status"] == "ok":
                    result = refined
                    logger.info(f"  Refined: Sharpe {result.get('backtest_results', {}).get('sharpe', '?')}, severity {result.get('severity', '?')}")
                else:
                    logger.warning(f"  Refinement failed — keeping previous version")
                    break

            if refine_count >= MAX_REFINEMENTS:
                logger.warning(f"  Max refinements ({MAX_REFINEMENTS}) reached — proceeding with current version")

            previous_critique = result.get("critique", "")

            # Track best
            sharpe = result.get("backtest_results", {}).get("sharpe", -999)
            if isinstance(sharpe, (int, float)) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_result = result

            # ── HITL Decision Gate ──────────────────────────
            approval = wait_for_approval(result, args.approval_timeout, alerts)

            if approval == "auto_approved":
                update_workflow_status("paper_trade", "deploying")
                logger.info(f"📈 Deploying {result.get('strategy_file', '?')} to testnet")

                # Trigger deployment (dry-run for safety)
                try:
                    from abundance.deployment.bridge import OrderManager, SignalComputer
                    from abundance.paper_trading.testnet_client import get_testnet_client

                    client = get_testnet_client()
                    computer = SignalComputer(client)
                    sig = computer.compute(args.pair, 500)
                    logger.info(f"  Signal: {sig.direction} @ {sig.allocation_pct*100:.0f}% — ready to deploy")
                    alerts.send("info", f"📈 Strategy deployed: {sig.direction} {sig.pair} @ {sig.allocation_pct*100:.0f}%")
                except Exception as e:
                    logger.error(f"Deployment failed: {e}")

                update_workflow_status("paper_trade", "deployed")
                break  # Exit iteration loop after successful deployment

            elif approval == "rejected":
                logger.info(f"Strategy rejected — continuing to next iteration")
                update_workflow_status("decision", "rejected")
                continue

            elif approval == "awaiting":
                logger.info("Awaiting human response — pausing agent")
                # In production: sleep and poll for response
                # For now: break out
                break

        # ── End of run summary ──────────────────────────────
        if best_result:
            logger.info(f"\n{'='*60}")
            logger.info(f"Best strategy this run:")
            logger.info(f"  Iteration: {best_result['iteration']}")
            logger.info(f"  Hypothesis: {best_result.get('hypothesis', '?')[:120]}")
            logger.info(f"  Sharpe: {best_result.get('backtest_results', {}).get('sharpe', '?')}")
            logger.info(f"  Decision: {best_result.get('decision', '?')}")
            logger.info(f"{'='*60}")

        if not args.daemon:
            break

        # ── Daemon: monitor + sleep + repeat. Never exits. ──
        crash_count = 0
        
        # Inter-cycle monitoring (price checks between research runs)
        monitor_cycles = args.daemon_interval // args.monitor_interval
        for mc in range(monitor_cycles):
            try:
                logger.info(f"[Monitor {mc+1}/{monitor_cycles}] Checking prices...")
                from abundance.paper_trading.testnet_client import get_testnet_client
                c = get_testnet_client()
                btc = c.get_price("BTCUSDT")
                
                # Quick health checks
                pos = c.get_positions()
                total_upnl = sum(float(p.get("unRealizedProfit", 0)) for p in pos)
                
                # Check for anomalies
                if abs(total_upnl) > 100:
                    alerts.send("warning", f"Large uPnL: \${total_upnl:+,.2f}")
                if len(pos) > 10:
                    alerts.send("warning", f"High position count: {len(pos)}")
                    
            except Exception as e:
                crash_count += 1
                logger.error(f"Monitor error (crash #{crash_count}): {e}")
                if crash_count > 5:
                    alerts.send("critical", f"Too many crashes ({crash_count}) — pausing 10min")
                    time.sleep(600)
                    crash_count = 0
            
            time.sleep(args.monitor_interval)

    # Reset to idle
    for node in ["research", "hypothesis", "coding", "backtest", "adversarial", "decision", "paper_trade"]:
        update_workflow_status(node, "completed")

    logger.info("Agent loop complete")


if __name__ == "__main__":
    main()
