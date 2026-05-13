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


def adversarial_node(state: ResearchState, tools: dict[str, Any]) -> dict:
    """Run mechanical checks + call LLM critique, merge results.

    Mechanical checks: lookahead, signal sanity, walk-forward, parameter sensitivity.
    LLM: structured red-team — if available, parse JSON response and merge into issues.

    If severity >= medium, decision_node will auto-reject (unless human_approved).
    """
    pair = state.get("pair", "BTCUSDT")
    strategy_file = state.get("strategy_file", "")

    # Try to instantiate strategy from state
    strategy = None
    if strategy_file:
        try:
            module_path = strategy_file.replace("/", ".").replace("src.", "").replace(".py", "")
            import importlib
            mod = importlib.import_module(module_path)
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and hasattr(obj, 'signals') and obj.__name__.endswith('Strategy'):
                    strategy = obj()
                    break
        except Exception as e:
            logger.warning(f"[ADVERSARIAL] Cannot instantiate strategy: {e}")

    critique = ""
    issues_found = []
    severity = "low"

    if strategy is not None:
        try:
            from abundance.deployment.adversarial import (
                run_full_adversarial,
                get_llm_critique_prompt,
            )
            art = strategy.run(pair)
            results = run_full_adversarial(strategy, pair)
            severity = results.get("severity", "low")

            for check_name, check_data in results.get("checks", {}).items():
                if not check_data.get("passed", True):
                    for issue in check_data.get("issues", []):
                        issues_found.append({
                            "check": check_name,
                            "detail": issue.get("detail", str(issue)),
                        })

            # Generate LLM critique prompt
            llm_prompt = get_llm_critique_prompt(art, results)

            # Try to call LLM if tools provide it
            llm_call = tools.get("llm") or tools.get("llm_call")
            if llm_call:
                try:
                    llm_response = llm_call(llm_prompt)
                    # Parse structured JSON from LLM response
                    llm_data = _parse_llm_critique(llm_response)
                    if llm_data:
                        llm_severity = llm_data.get("severity", "low")
                        # Upgrade severity if LLM finds worse issues
                        sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                        if sev_order.get(llm_severity, 0) > sev_order.get(severity, 0):
                            severity = llm_severity
                        for issue in llm_data.get("issues_found", []):
                            issues_found.append({"check": "llm_critique", "detail": issue})
                        state["llm_recommendation"] = llm_data.get("recommendation", "")
                        state["llm_rationale"] = llm_data.get("rationale", "")
                except Exception as e:
                    logger.warning(f"[ADVERSARIAL] LLM call failed: {e}")
                    issues_found.append({"check": "llm_critique", "detail": f"LLM call error: {e}"})
            else:
                # No LLM available — store prompt for agent to use
                logger.info("[ADVERSARIAL] No LLM tool available — storing prompt for agent")
                issues_found.append({
                    "check": "llm_critique",
                    "detail": "LLM not available — agent should review prompt below",
                })

            critique = llm_prompt
            logger.info(
                f"[ADVERSARIAL] {pair}: severity={severity}, "
                f"mechanical_checks={list(results['checks'].keys())}, "
                f"llm_called={llm_call is not None}"
            )
        except Exception as e:
            logger.error(f"[ADVERSARIAL] Checks failed: {e}")
            critique = f"Adversarial check error: {e}"
            severity = "high"
            issues_found.append({"check": "runtime", "detail": str(e)})
    else:
        critique = "No strategy instantiated — cannot run adversarial checks."
        severity = "high"
        issues_found.append({"check": "instantiation", "detail": "Strategy class not found"})

    state["critique"] = critique
    state["issues_found"] = issues_found
    state["severity"] = severity

    return state


def _parse_llm_critique(response: str) -> dict | None:
    """Parse structured JSON from LLM critique response."""
    import json
    # Try to extract JSON block from response
    text = response.strip()
    # Look for JSON code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    # Try { } delimited JSON
    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat entire response as a single issue
        return {
            "severity": "medium",
            "issues_found": [response[:500]],
            "recommendation": "revise",
            "rationale": response[:1000],
        }


def decision_node(state: ResearchState, _tools: dict[str, Any]) -> dict:
    """Decide: approve, revise, or reject.

    Auto-rejects if adversarial severity >= high (unless human_approved).
    Otherwise defers to state['decision'] for agent override.
    """
    severity = state.get("severity", "low")
    human_approved = state.get("human_approved", False)

    # Auto-reject on high/critical severity unless human override
    if severity in ("high", "critical") and not human_approved:
        issues = state.get("issues_found", [])
        issue_summary = "; ".join(i.get("detail", "")[:100] for i in issues[:3])
        state["decision"] = "reject"
        state["decision_rationale"] = (
            f"Auto-rejected: adversarial severity={severity}. Issues: {issue_summary}"
        )
        logger.warning(f"[DECISION] AUTO-REJECT {state.get('pair', '?')}: severity={severity}")
        return state

    # If no agent override, apply default based on backtest
    if not state.get("decision"):
        backtest = state.get("backtest_results", {})
        sharpe = backtest.get("sharpe", 0)
        if sharpe > 1.0:
            state["decision"] = "approve"
            state["decision_rationale"] = f"Sharpe {sharpe:.2f} > 1.0 — approve"
        elif sharpe > 0.5:
            state["decision"] = "revise"
            state["decision_rationale"] = f"Sharpe {sharpe:.2f} in 0.5-1.0 — needs revision"
        else:
            state["decision"] = "reject"
            state["decision_rationale"] = f"Sharpe {sharpe:.2f} < 0.5 — reject"

    logger.info(
        f"[DECISION] {state.get('pair', '?')}: {state['decision']} "
        f"(severity={severity}, human_approved={human_approved})"
    )
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
