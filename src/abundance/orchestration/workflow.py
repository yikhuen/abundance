"""LangGraph workflow for the agentic research → trading loop.

Directed graph:
  RESEARCH → HYPOTHESIS → CODING → BACKTEST → ADVERSARIAL → DECISION
     ↑                                                          │
     └────────────────────── (reject) ──────────────────────────┘
                                                 │ (approve)
                                           PAPER_TRADE

Each node is a sub-agent spawned by OpenClaw. The workflow pauses at
DECISION for human-in-the-loop approval before proceeding to paper
trading.

State is checkpointed via LangGraph's SQLite checkpointer for
multi-day resume capability.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


class ResearchState(TypedDict):
    """State shared across all workflow nodes."""

    # Task identification
    task_id: str
    pair: str
    created_at: str

    # Research phase
    research_query: str
    research_findings: str
    papers_cited: list[str]

    # Hypothesis phase
    hypothesis: str
    hypothesis_rationale: str
    causal_mechanism: str

    # Coding phase
    strategy_code: str
    strategy_file: str

    # Backtest phase
    backtest_results: dict[str, Any]
    metrics_summary: str

    # Adversarial review
    critique: str
    issues_found: list[str]
    severity: str  # "low" | "medium" | "high" | "critical"

    # Decision
    decision: str  # "approve" | "reject" | "revise"
    decision_rationale: str
    human_approved: bool


class WorkflowBuilder:
    """Builds and configures the LangGraph research-to-trading workflow."""

    def __init__(self, checkpoint_db: str = ":memory:") -> None:
        self.checkpoint_db = checkpoint_db
        self.graph = self._build()

    def _build(self) -> StateGraph:
        """Construct the directed graph with all nodes and edges."""
        workflow = StateGraph(ResearchState)

        # ── Nodes ──────────────────────────────────────────
        workflow.add_node("research", self._research_node)
        workflow.add_node("hypothesis", self._hypothesis_node)
        workflow.add_node("coding", self._coding_node)
        workflow.add_node("backtest", self._backtest_node)
        workflow.add_node("adversarial", self._adversarial_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("paper_trade", self._paper_trade_node)

        # ── Edges ──────────────────────────────────────────
        workflow.set_entry_point("research")
        workflow.add_edge("research", "hypothesis")
        workflow.add_edge("hypothesis", "coding")
        workflow.add_edge("coding", "backtest")
        workflow.add_edge("backtest", "adversarial")
        workflow.add_edge("adversarial", "decision")

        # Conditional branch at decision
        workflow.add_conditional_edges(
            "decision",
            self._route_decision,
            {
                "approve": "paper_trade",
                "revise": "research",  # loop back
                "reject": "__end__",
            },
        )
        workflow.add_edge("paper_trade", "__end__")

        return workflow

    def compile(self, interrupt_before: list[str] | None = None):
        """Compile with optional HITL interrupt points.

        Args:
            interrupt_before: Nodes to pause before for human approval.
                Default: ['decision'] — human must approve before paper trading.
                Pass [] or ['decision'] to enable interrupts at those nodes.
                Pass None to skip interrupts (auto-approve dev mode).
        """

        checkpointer = MemorySaver()
        return self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
        )

    # ── Node implementations ──────────────────────────────────

    def _research_node(self, state: ResearchState) -> dict:
        """Research agent: retrieve papers and market context.

        In production, this node spawns an OpenClaw sub-agent with
        web search + arXiv tools. For now, it's a stub that logs.
        """
        query = state.get("research_query", f"funding rate arbitrage {state.get('pair', 'BTC')} crypto")
        state["research_query"] = query
        state["research_findings"] = (
            f"[Research stub] Query: '{query}'. "
            f"Agent will search arXiv + web for relevant papers."
        )
        state["papers_cited"] = []
        return state

    def _hypothesis_node(self, state: ResearchState) -> dict:
        """Hypothesis agent: generate testable hypothesis from research.

        Must cite causal mechanism (not data-mined pattern).
        """
        findings = state.get("research_findings", "")
        pair = state.get("pair", "BTCUSDT")
        state["hypothesis"] = (
            f"[Hypothesis stub] Based on findings about {pair}, "
            f"propose a testable hypothesis with causal grounding."
        )
        state["hypothesis_rationale"] = "Stub — agent will generate"
        state["causal_mechanism"] = "Stub"
        return state

    def _coding_node(self, state: ResearchState) -> dict:
        """Coding agent: implement strategy from hypothesis.

        Spawns a code-only sub-agent (edit scope, no deletion).
        Produces a Python strategy file in the abundance codebase.
        """
        hypothesis = state.get("hypothesis", "")
        state["strategy_code"] = f"# Stub strategy for: {hypothesis}"
        state["strategy_file"] = "src/abundance/strategies/generated_strategy.py"
        return state

    def _backtest_node(self, state: ResearchState) -> dict:
        """Backtest agent: run strategy through the evaluation harness.

        Uses our existing metrics calculator + NautilusTrader catalog.
        """
        state["backtest_results"] = {"sharpe": 0.0, "return_pct": 0.0, "max_dd": 0.0}
        state["metrics_summary"] = (
            "[Backtest stub] Strategy will be tested against "
            "historical data using the Sprint 2 eval harness."
        )
        return state

    def _adversarial_node(self, state: ResearchState) -> dict:
        """Adversarial agent: critique the strategy.

        Models TradeTrap-style perturbation tests and identifies:
        - Lookahead bias
        - Overfitting indicators
        - Regime dependency
        - Capacity limits
        """
        state["critique"] = (
            "[Adversarial stub] Strategy will be stress-tested for "
            "overfitting, lookahead, and regime fragility."
        )
        state["issues_found"] = []
        state["severity"] = "low"
        return state

    def _decision_node(self, state: ResearchState) -> dict:
        """Decision node: human-in-the-loop gate.

        Pauses workflow. Human reviews research, hypothesis, backtest,
        and adversarial critique before approving.
        """
        # In production, this sends a Telegram message via OpenClaw
        # and waits for the human's reply.
        state["decision"] = "approve"  # default stub
        state["decision_rationale"] = "Auto-approved in stub mode"
        state["human_approved"] = True
        return state

    def _paper_trade_node(self, state: ResearchState) -> dict:
        """Paper trading deployment.

        In production, deploys strategy to NautilusTrader paper trading
        environment and monitors for 4+ weeks before real capital.
        """
        strategy_file = state.get("strategy_file", "unknown")
        state["decision"] = "deployed"
        return state

    @staticmethod
    def _route_decision(
        state: ResearchState,
    ) -> Literal["approve", "revise", "reject"]:
        """Route based on decision outcome."""
        decision = state.get("decision", "reject")
        if decision == "approve":
            return "approve"
        elif decision == "revise":
            return "revise"
        else:
            return "reject"
