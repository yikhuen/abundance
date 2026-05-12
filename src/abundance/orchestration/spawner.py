"""OpenClaw sub-agent spawner for LangGraph workflow nodes.

Each node in the research→trading pipeline spawns a fresh OpenClaw
sub-agent with scoped tools. This module provides the spawn logic
and tool-scope definitions.

In production, this communicates with OpenClaw's TaskFlow system.
In dev mode, it uses mock agents that simulate the workflow.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger


class SubAgentSpawner(Protocol):
    """Protocol for spawning OpenClaw sub-agents."""

    def spawn(
        self,
        agent_type: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Spawn a sub-agent and return its result."""
        ...


# ── Tool scoping definitions ──────────────────────────────────
# Each agent type gets a restricted set of tools

TOOL_SCOPES = {
    "research": {
        "tools": ["web_search", "web_fetch"],
        "description": "Paper retrieval and market research agent",
        "env": {"EXA_API_KEY": "optional"},
    },
    "hypothesis": {
        "tools": ["read", "write"],
        "description": "Hypothesis generation agent — reads research, writes proposals",
        "workspace": "restricted to abundance/research/",
    },
    "coding": {
        "tools": ["read", "write", "edit", "exec"],
        "description": "Strategy implementation agent — edit-only on strategy code",
        "workspace": "restricted to abundance/src/abundance/strategies/",
        "exec": {"allowed_commands": ["poetry run python scripts/backtest_*.py"]},
    },
    "backtest": {
        "tools": ["exec", "read"],
        "description": "Backtest execution agent — runs eval harness, reads results",
        "exec": {"allowed_commands": ["poetry run python scripts/backtest_*.py"]},
    },
    "adversarial": {
        "tools": ["read", "write"],
        "description": "Adversarial review agent — critiques strategy, writes report",
        "workspace": "restricted to abundance/research/",
    },
}


@dataclass
class DevSubAgentSpawner:
    """Development-mode spawner that simulates agent responses.

    In production, replace with OpenClaw TaskFlow spawner that creates
    real sub-agents with the defined tool scopes.
    """

    mock_responses: dict[str, str] = field(default_factory=dict)

    def spawn(
        self,
        agent_type: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate spawning a sub-agent (dev mode)."""
        scope = TOOL_SCOPES.get(agent_type, {"tools": []})
        logger.info(
            f"[DEV] Spawning {agent_type} agent | task: {task[:80]}... | "
            f"tools: {scope['tools']}"
        )
        return {
            "agent_type": agent_type,
            "task": task,
            "result": self.mock_responses.get(agent_type, f"[{agent_type}] stub result"),
            "tools_used": scope["tools"],
            "status": "completed",
        }


def get_spawner(env: str = "dev") -> SubAgentSpawner:
    """Factory: return agent spawner for the current environment.

    Args:
        env: 'dev' for simulated agents, 'production' for real OpenClaw spawns.
    """
    if env == "production":
        # TODO: Implement OpenClaw TaskFlow spawner
        # from openclaw import TaskFlowSpawner
        # return TaskFlowSpawner(tool_scopes=TOOL_SCOPES)
        logger.warning("Production spawner not implemented — using dev mode")
    return DevSubAgentSpawner()
