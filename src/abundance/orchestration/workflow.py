"""Research state type definition for the agent workflow.

This is the shared state schema used by all workflow nodes.
The agent (OpenClaw) fills this state as it progresses through
the research → hypothesis → coding → backtest → adversarial → decision pipeline.
"""

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """State shared across all workflow nodes."""

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
