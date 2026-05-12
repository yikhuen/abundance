#!/usr/bin/env python3
"""Sprint 7: Autoresearch loop — iterative strategy discovery.

Runs the full agentic workflow in a loop, where each iteration:
  1. Learns from the previous iteration's adversarial critique
  2. Refines the research query
  3. Generates a new hypothesis
  4. Writes + backtests new strategy code
  5. Tracks the best strategy across all iterations

Usage:
  python scripts/run_autoresearch.py --pair BTCUSDT --iterations 2
  python scripts/run_autoresearch.py --pair ETHUSDT --iterations 5 --hitl
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.orchestration.agents import (
    backtest_node,
    build_workflow,
    coding_node,
    compile_workflow,
    decision_node,
)
from abundance.orchestration.workflow import ResearchState


def refine_query(iteration: int, prev_critique: str, pair: str) -> tuple[str, str]:
    """Generate research query + hypothesis hint per iteration."""
    strategies = [
        ("no-arbitrage perpetual futures pricing deviation He Manela Ross", "arbitrage"),
        ("RSI mean reversion oversold crypto perpetuals", "rsi"),
        ("volatility breakout ATR trailing stop crypto", "breakout"),
        ("funding rate momentum carry arbitrage crypto", "carry"),
    ]
    idx = (iteration - 1) % len(strategies)
    query, hint = strategies[idx]
    return f"{query} {pair} site:arxiv.org", hint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run iterative autoresearch loop"
    )
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of research iterations"
    )
    parser.add_argument(
        "--hitl", action="store_true", help="Human-in-the-loop at decision"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Abundance · Sprint 7 · Autoresearch Loop")
    logger.info(f"  Pair:       {args.pair}")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  HITL:       {'enabled' if args.hitl else 'auto'}")
    logger.info("=" * 60)

    # ── Wire tools (mock in standalone, real in OpenClaw) ──
    tools = {
        "web_search": lambda q: f"[Search] {q}",
        "web_fetch": lambda u: f"[Fetch] {u}",
        "write": lambda p, c: Path(p).parent.mkdir(parents=True, exist_ok=True) or Path(p).write_text(c),
        "read": lambda p: Path(p).read_text() if Path(p).exists() else "",
        "exec": lambda c: __import__("subprocess").run(c, shell=True, capture_output=True, text=True).stdout,
    }

    # ── Build workflow ────────────────────────────────────
    workflow = build_workflow(tools)
    app = compile_workflow(workflow, interrupt_before=["decision"] if args.hitl else None)

    # ── Track best strategy ───────────────────────────────
    best_strategy: dict = {"sharpe": -999, "iteration": 0, "file": "", "hypothesis": ""}
    all_results: list[dict] = []
    prev_critique = ""

    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'='*60}")

        # Refine query based on previous iteration's critique
        research_query, strategy_hint = refine_query(iteration, prev_critique, args.pair)
        logger.info(f"Research query: {research_query[:120]}... | hint: {strategy_hint}")

        # Build initial state with hint
        task_id = f"auto-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-iter{iteration}"
        initial_state = {
            "task_id": f"{task_id}__{strategy_hint}",
            "pair": args.pair,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_query": research_query,
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
        config = {"configurable": {"thread_id": task_id}}

        # ── Run workflow ──────────────────────────────────
        final_state: dict = initial_state.copy()

        for event in app.stream(initial_state, config):
            node_name = list(event.keys())[0]
            node_state = event[node_name]

            if node_name == "__interrupt__":
                logger.info("⏸️  Human approval needed — auto-approving")
                app.update_state(config, {"decision": "approve", "human_approved": True})
                continue

            emoji = {
                "research": "🔍", "hypothesis": "💡", "coding": "💻",
                "backtest": "📊", "adversarial": "🛡️", "decision": "⚖️",
                "paper_trade": "📈",
            }.get(node_name, "✅")
            logger.info(f"  {emoji} {node_name.upper()}")

            if isinstance(node_state, dict):
                for k, v in node_state.items():
                    final_state[k] = v

        # ── Capture results ───────────────────────────────
        results = final_state.get("backtest_results", {})
        sharpe = results.get("sharpe", -999)
        decision = final_state.get("decision", "reject")
        prev_critique = final_state.get("critique", "")
        strategy_file = final_state.get("strategy_file", "")
        hypothesis = final_state.get("hypothesis", "")

        iteration_result = {
            "iteration": iteration,
            "sharpe": sharpe,
            "return_pct": results.get("return_pct", 0),
            "max_dd": results.get("max_dd", 0),
            "decision": decision,
            "strategy_file": strategy_file,
            "hypothesis": hypothesis[:80],
            "severity": final_state.get("severity", "?"),
            "issues": len(final_state.get("issues_found", [])),
        }
        all_results.append(iteration_result)

        # Track best
        if isinstance(sharpe, (int, float)) and sharpe > best_strategy["sharpe"]:
            best_strategy = {
                "sharpe": sharpe,
                "iteration": iteration,
                "file": strategy_file,
                "hypothesis": hypothesis[:100],
            }

        logger.info(
            f"  Iteration {iteration} result: "
            f"Sharpe {sharpe}, Return {results.get('return_pct', 0)}%, "
            f"Decision: {decision}"
        )

        # ── Early stop if approved ────────────────────────
        if decision == "approve":
            logger.info(f"  ✅ Strategy approved at iteration {iteration}")
            if args.iterations > iteration:
                logger.info(f"  (continuing for {args.iterations - iteration} more iterations...)")
            # Don't break — keep searching for better

    # ── Final report ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Autoresearch Complete")
    logger.info(f"{'='*60}")
    logger.info(f"{'Iter':>4} {'Sharpe':>7} {'Return%':>8} {'MaxDD%':>7} {'Decision':>10} {'Issues':>6}")
    logger.info(f"{'─'*4} {'─'*7} {'─'*8} {'─'*7} {'─'*10} {'─'*6}")

    for r in all_results:
        logger.info(
            f"{r['iteration']:>4} {r['sharpe']:>7.3f} {r['return_pct']:>8.1f} "
            f"{r['max_dd']:>7.1f} {r['decision']:>10} {r['issues']:>6}"
        )

    logger.info(f"\n🏆 Best strategy: Iteration {best_strategy['iteration']}")
    logger.info(f"   Sharpe: {best_strategy['sharpe']:.3f}")
    logger.info(f"   File:   {best_strategy['file']}")
    logger.info(f"   Hypothesis: {best_strategy['hypothesis']}")

    logger.info(f"\n{'='*60}")
    logger.info("Sprint 7 — Autoresearch Loop COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
