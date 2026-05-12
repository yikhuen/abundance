"""OpenClaw sub-agent tool scopes for LangGraph workflow nodes.

Each node in the research→trading pipeline spawns a fresh OpenClaw
sub-agent with scoped tools. This module defines the tool scoping.

The agent (OpenClaw) provides the actual tools. No mocks, no stubs.
"""

from logging import getLogger

logger = getLogger(__name__)

# ── Tool scoping definitions ──────────────────────────────────

TOOL_SCOPES = {
    "research": {
        "tools": ["web_search", "web_fetch"],
        "description": "Paper retrieval and market research agent — search arXiv, SSRN, web",
    },
    "hypothesis": {
        "tools": ["read", "write"],
        "description": "Hypothesis generation — reads research findings, writes hypothesis proposals",
        "workspace": "restricted to abundance/research/",
    },
    "coding": {
        "tools": ["read", "write", "edit", "exec"],
        "description": "Strategy implementation — writes code, runs tests",
        "workspace": "restricted to abundance/src/abundance/strategies/",
        "exec": {"allowed_commands": ["poetry run python scripts/backtest_*.py"]},
    },
    "backtest": {
        "tools": ["exec", "read"],
        "description": "Backtest execution — runs eval harness, reads results",
        "exec": {"allowed_commands": ["poetry run python scripts/backtest_*.py"]},
    },
    "adversarial": {
        "tools": ["read", "write"],
        "description": "Adversarial review — critiques strategy, writes findings",
        "workspace": "restricted to abundance/research/",
    },
}


def get_tool_scope(agent_type: str) -> dict:
    """Return the tool scope for a given agent type."""
    return TOOL_SCOPES.get(agent_type, {"tools": []})
