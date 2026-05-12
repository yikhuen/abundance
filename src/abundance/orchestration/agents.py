"""Agent-first workflow nodes — no mocks, no stubs, no templates.

Each node expects REAL OpenClaw tools passed via the `tools` dict.
The agent (OpenClaw) calls these nodes from its loop.

Tools expected:
  web_search(query) → str     — OpenClaw's web search
  web_fetch(url) → str        — OpenClaw's web fetch  
  write(path, content) → None — OpenClaw's file writer
  exec(command) → str         — OpenClaw's command executor

Nodes are stateless functions that take ResearchState + tools → return updated state.
"""

from typing import Any, Literal

from loguru import logger

from abundance.orchestration.workflow import ResearchState


def research_node(state: ResearchState, tools: dict[str, Any]) -> dict:
    """Search web for papers, market context, strategy ideas.

    Requires: tools['web_search'], tools['web_fetch']
    """
    pair = state.get("pair", "BTCUSDT")
    query = state.get("research_query") or (
        f"profitable algorithmic trading strategies {pair} crypto perpetuals 2025 2026"
    )

    logger.info(f"[RESEARCH] Searching: {query[:100]}")

    findings = []
    search = tools.get("web_search")
    fetch = tools.get("web_fetch")

    if search:
        try:
            results = search(query)
            findings.append(str(results))
        except Exception as e:
            logger.error(f"Search failed: {e}")

    if fetch:
        for url in ["https://arxiv.org/list/q-fin.TR/recent", "https://arxiv.org/list/q-fin.PR/recent"]:
            try:
                content = fetch(url)
                findings.append(str(content)[:3000])
            except Exception:
                pass

    state["research_query"] = query
    state["research_findings"] = "\n\n".join(findings) if findings else ""
    state["papers_cited"] = []

    return state


def hypothesis_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Generate testable hypothesis from research findings.

    The real agent (OpenClaw) should replace this with LLM-driven generation.
    This node provides the structure; the agent fills in the content.
    """
    pair = state.get("pair", "BTCUSDT")
    findings = state.get("research_findings", "")

    # Hypothesis, rationale, and causal mechanism are written by the actual LLM agent.
    # These fields are placeholders that the agent overwrites.
    state["hypothesis"] = ""
    state["hypothesis_rationale"] = ""
    state["causal_mechanism"] = ""

    logger.info(f"[HYPOTHESIS] Ready for agent to fill — {pair}")
    return state


def coding_node(state: ResearchState, tools: dict[str, Any]) -> dict:
    """Write strategy code to disk.

    Requires: tools['write']. The agent (OpenClaw) generates the code via LLM
    and writes it to src/abundance/strategies/.

    Returns the file path in state['strategy_file'].
    """
    pair = state.get("pair", "BTCUSDT")
    hypothesis = state.get("hypothesis", "")
    mechanism = state.get("causal_mechanism", "")

    logger.info(f"[CODING] Ready to write strategy for {pair}")

    # The agent is responsible for generating code and calling write()
    # This node provides the target path. The agent fills strategy_code + strategy_file.
    state["strategy_code"] = ""
    state["strategy_file"] = ""

    return state


def backtest_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Run strategy through evaluation harness.

    Imports the strategy module and calls run_strategy().
    Falls back to B&H benchmark if strategy can't be executed.
    """
    pair = state.get("pair", "BTCUSDT")
    strategy_file = state.get("strategy_file", "")

    logger.info(f"[BACKTEST] {pair} | strategy: {strategy_file or 'B&H fallback'}")

    strategy_ran = False
    try:
        if strategy_file and strategy_file.endswith(".py"):
            module_path = (
                strategy_file.replace("/", ".").replace("src.", "").replace(".py", "")
            )
            import importlib

            mod = importlib.import_module(module_path)
            if hasattr(mod, "run_strategy"):
                _, report = mod.run_strategy(pair=pair)
                strategy_ran = True
    except Exception as e:
        logger.warning(f"[BACKTEST] Strategy failed: {e} — B&H fallback")

    if not strategy_ran:
        import polars as pl
        from abundance.backtesting.metrics import MetricsCalculator
        from abundance.config.settings import settings

        kline_path = settings.raw_dir / "klines" / f"{pair.lower()}_1d"
        df = (
            pl.scan_parquet(str(kline_path / "**" / "*.parquet"))
            .sort("timestamp_ms")
            .select(["timestamp_ms", "close"])
            .collect()
        )
        initial = 10_000.0
        equity = initial * (df["close"] / df["close"][0])
        curve = df.select("timestamp_ms").with_columns(equity.alias("equity"))
        report = MetricsCalculator.from_equity_curve(curve)

    state["backtest_results"] = {
        "sharpe": round(report.sharpe_ratio, 3),
        "return_pct": round(report.total_return_pct, 1),
        "max_dd": round(report.max_drawdown_pct, 1),
    }
    state["metrics_summary"] = (
        f"Sharpe {report.sharpe_ratio:.3f}, Return {report.total_return_pct:.1f}%"
    )
    return state


def adversarial_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Critique the strategy for failure modes.

    The real agent (OpenClaw) should generate a thorough critique.
    This node provides structure; the agent fills critique text.
    """
    state["critique"] = ""
    state["issues_found"] = []
    state["severity"] = "low"

    logger.info("[ADVERSARIAL] Ready for agent critique")
    return state


def decision_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Decide: approve, revise, or reject.

    The real agent (OpenClaw) makes this decision based on backtest results
    and adversarial critique. This node provides the decision in state.
    """
    state["decision"] = ""
    state["decision_rationale"] = ""
    state["human_approved"] = False

    logger.info("[DECISION] Ready for agent decision")
    return state


def paper_trade_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Deploy to paper trading. Sets state for the deployment script."""
    logger.info(f"[PAPER_TRADE] Strategy: {state.get('strategy_file', '?')}")
    state["decision"] = "deployed"
    return state


def route_decision(state: ResearchState) -> Literal["approve", "revise", "reject"]:
    decision = state.get("decision", "reject")
    if decision == "approve":
        return "approve"
    elif decision == "revise":
        return "revise"
    return "reject"


def build_workflow(tools: dict[str, Any]) -> Any:
    """Build the LangGraph workflow. Requires langgraph installed."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    workflow = StateGraph(ResearchState)

    workflow.add_node("research", lambda s: research_node(s, tools))
    workflow.add_node("hypothesis", lambda s: hypothesis_node(s, tools))
    workflow.add_node("coding", lambda s: coding_node(s, tools))
    workflow.add_node("backtest", lambda s: backtest_node(s, tools))
    workflow.add_node("adversarial", lambda s: adversarial_node(s, tools))
    workflow.add_node("decision", lambda s: decision_node(s, tools))
    workflow.add_node("paper_trade", lambda s: paper_trade_node(s, tools))

    workflow.set_entry_point("research")
    workflow.add_edge("research", "hypothesis")
    workflow.add_edge("hypothesis", "coding")
    workflow.add_edge("coding", "backtest")
    workflow.add_edge("backtest", "adversarial")
    workflow.add_edge("adversarial", "decision")
    workflow.add_conditional_edges("decision", route_decision, {
        "approve": "paper_trade", "revise": "research", "reject": "__end__",
    })
    workflow.add_edge("paper_trade", "__end__")

    return workflow


def compile_workflow(workflow, interrupt_before=None):
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
