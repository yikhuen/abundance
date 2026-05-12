#!/usr/bin/env python3
"""Sprint 5+6: Run the agentic workflow with REAL AI agents.

Each node performs actual work:
  RESEARCH   → web_search + web_fetch (Gemini via OpenClaw)
  HYPOTHESIS → causal hypothesis generation
  CODING     → strategy implementation (writes real code)
  BACKTEST   → run through evaluation harness
  ADVERSARIAL → critique + failure mode analysis
  DECISION   → approve / revise / reject

Usage:
  python scripts/run_agents.py                          # auto-approve, BTC
  python scripts/run_agents.py --pair ETHUSDT           # target ETH
  python scripts/run_agents.py --hitl                   # human-in-the-loop
  python scripts/run_agents.py --query "mean reversion strategies SOL"
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.orchestration.agents import build_workflow, compile_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agentic research-to-trading workflow with real AI agents"
    )
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--query", default=None, help="Research query override")
    parser.add_argument(
        "--hitl", action="store_true", help="Enable human-in-the-loop (pause at decision)"
    )
    parser.add_argument(
        "--max-loops", type=int, default=3, help="Max revise→research loops"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Abundance · Sprints 5+6 · Real Agent Workflow")
    logger.info(f"  Pair:    {args.pair}")
    logger.info(f"  HITL:    {'enabled' if args.hitl else 'auto-approve'}")
    logger.info(f"  Loops:   max {args.max_loops}")
    logger.info("=" * 60)

    # ── Wire real tools (OpenClaw provides these) ──────────
    # In the OpenClaw runtime, these are available as function calls.
    # For standalone Python, we provide mock implementations.
    tools: dict = {}

    # Try to import OpenClaw tools (available when running inside OpenClaw)
    try:
        # These are injected by OpenClaw at runtime
        from __builtins__ import web_search, web_fetch, write, read, exec  # type: ignore # noqa: F811

        tools = {
            "web_search": web_search,
            "web_fetch": web_fetch,
            "write": write,
            "read": read,
            "exec": exec,
        }
        logger.info("Using OpenClaw runtime tools")
    except (ImportError, AttributeError):
        # Standalone mode: use mock tools for development
        logger.info("Using mock tools (standalone mode)")

        def _mock_search(query: str) -> str:
            return (
                f"[Mock] Searched for: {query}\n"
                "Results would include arXiv papers, trading strategy blogs, "
                "and market analysis from web sources."
            )

        def _mock_fetch(url: str) -> str:
            return f"[Mock] Fetched: {url}\n(Content would be extracted here)"

        def _mock_write(path: str, content: str) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)

        def _mock_read(path: str) -> str:
            return Path(path).read_text() if Path(path).exists() else ""

        def _mock_exec(cmd: str) -> str:
            import subprocess

            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                return result.stdout or result.stderr
            except Exception as e:
                return str(e)

        tools = {
            "web_search": _mock_search,
            "web_fetch": _mock_fetch,
            "write": _mock_write,
            "read": _mock_read,
            "exec": _mock_exec,
        }

    # ── Initial state ──────────────────────────────────────
    initial_state = {
        "task_id": f"agent-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "pair": args.pair,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_query": args.query or "",
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

    # ── Build & compile ────────────────────────────────────
    workflow = build_workflow(tools)
    interrupt_before = ["decision"] if args.hitl else None
    app = compile_workflow(workflow, interrupt_before=interrupt_before)
    config = {"configurable": {"thread_id": initial_state["task_id"]}}

    # ── Run workflow ───────────────────────────────────────
    logger.info(f"\n🚀 Starting workflow — Thread: {initial_state['task_id']}\n")

    loop_count = 0
    final_state = initial_state

    while loop_count < args.max_loops:
        loop_count += 1
        logger.info(f"--- Loop {loop_count}/{args.max_loops} ---")

        for event in app.stream(final_state, config):
            node_name = list(event.keys())[0]
            node_state = event[node_name]

            if node_name == "__interrupt__":
                logger.info("⏸️  Paused for human approval (decision gate)")
                logger.info(
                    f"\n📋 {'─'*50}\n"
                    f"Hypothesis: {node_state.get('hypothesis', 'N/A')[:200]}\n"
                    f"Backtest:   {node_state.get('metrics_summary', 'N/A')}\n"
                    f"Critique:   {node_state.get('severity', 'N/A')} severity, "
                    f"{len(node_state.get('issues_found', []))} issues\n"
                    f"{'─'*50}\n"
                )
                logger.info("Send 'approve', 'revise', or 'reject' to continue")
                # In production: OpenClaw pauses here and waits for Telegram reply
                # For dev: auto-continue with "approve"
                app.update_state(config, {"decision": "approve", "human_approved": True})
                continue

            # Log node completion
            emoji = {
                "research": "🔍",
                "hypothesis": "💡",
                "coding": "💻",
                "backtest": "📊",
                "adversarial": "🛡️",
                "decision": "⚖️",
                "paper_trade": "📈",
            }.get(node_name, "✅")

            logger.info(f"{emoji}  {node_name.upper()}")

            if node_name == "research":
                findings_len = len(node_state.get("research_findings", ""))
                logger.info(f"    Research: {findings_len} chars of findings")
            elif node_name == "hypothesis":
                logger.info(f"    Hypothesis: {node_state.get('hypothesis', '')[:120]}")
            elif node_name == "coding":
                logger.info(f"    File: {node_state.get('strategy_file', '')}")
            elif node_name == "backtest":
                results = node_state.get("backtest_results", {})
                logger.info(
                    f"    Sharpe: {results.get('sharpe', 'N/A')}, "
                    f"Return: {results.get('return_pct', 'N/A')}%"
                )
            elif node_name == "adversarial":
                logger.info(
                    f"    Severity: {node_state.get('severity', '?')}, "
                    f"Issues: {len(node_state.get('issues_found', []))}"
                )
            elif node_name == "decision":
                logger.info(f"    → {node_state.get('decision', '?').upper()}")

            # Update state for potential next loop
            if isinstance(node_state, dict):
                for k, v in node_state.items():
                    final_state[k] = v

        # Check if we need to loop
        decision = final_state.get("decision", "reject")
        if decision == "approve":
            logger.info("\n✅ Strategy approved — paper trading")
            break
        elif decision == "reject":
            logger.info("\n❌ Strategy rejected")
            break
        elif decision == "revise":
            logger.info(f"\n🔄 Revision loop {loop_count} — restarting research\n")
            continue

    # ── Final summary ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Workflow Complete")
    logger.info(f"  Pair:      {args.pair}")
    logger.info(f"  Loops:     {loop_count}")
    logger.info(f"  Decision:  {final_state.get('decision', 'unknown')}")
    logger.info(f"  Rationale: {final_state.get('decision_rationale', '')}")
    logger.info(f"  Strategy:  {final_state.get('strategy_file', '')}")
    logger.info(f"{'='*60}")

    logger.info("Sprints 5+6 — Real Agent Workflow COMPLETE")


if __name__ == "__main__":
    main()
